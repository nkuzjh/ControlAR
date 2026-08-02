#!/usr/bin/env python
"""ControlAR fast engine: official generate() restructured for CUDA graphs,
with optional cfg_interval and INT8 weight-only quantization.

Key differences vs official autoregressive/models/generate.py:
  1. condition_token is CLONED after prefill so reduce-overhead CUDA graph
     replays can safely read it (fixes "output overwritten by subsequent run").
  2. sampled tokens are cloned out of the graph pool each step.
  3. --cfg-interval N: after N decode steps, drop the uncond branch (2x fewer
     sequences) for the rest, at a small quality cost.
  4. --quant int8wo: torchao int8_weight_only on all Linears (decode is
     weight-bandwidth-bound on RTX 3090).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from safetensors.torch import load_file  # noqa: E402
from torchvision.utils import save_image  # noqa: E402
from tokenizer.tokenizer_image.vq_model import VQ_models  # noqa: E402
from language.t5 import T5Embedder  # noqa: E402
from autoregressive.models.gpt_t2i import GPT_models  # noqa: E402
from autoregressive.models.generate import sample as of_sample  # noqa: E402
from condition.hed import HEDdetector  # noqa: E402


def prefill(model, cond_idx, input_pos, cfg_scale, condition,
            control_strength, **sampling_kwargs):
    if cfg_scale > 1.0:
        logits, _ = model(None, cond_idx, input_pos, condition=condition,
                          control_strength=control_strength)
        cond_logits, uncond_logits = torch.split(logits,
                                                 len(logits) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = model(None, cond_idx, input_pos, condition=condition)
    return of_sample(logits, **sampling_kwargs)[0]


def decode_one_token(model, x, input_pos, cfg_scale, cfg_flag, condition,
                     **sampling_kwargs):
    if cfg_scale > 1.0 and cfg_flag:
        x_combined = torch.cat([x, x])
        logits, _ = model(x_combined, cond_idx=None, input_pos=input_pos,
                          condition=condition)
        cond_logits, uncond_logits = torch.split(logits,
                                                 len(logits) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        # cond-only branch (after cfg_interval or cfg_scale==1)
        logits, _ = model(x, cond_idx=None, input_pos=input_pos,
                          condition=condition)
    return of_sample(logits, **sampling_kwargs)


def _decode_step_e2e(model, x, input_pos, condition, cfg_scale, top_k,
                     temperature):
    """Whole decode step in ONE compiled graph: forward + CFG + top-k +
    multinomial (temperature/top-k passed as scalars, specialized on)."""
    x_combined = torch.cat([x, x])
    logits, _ = model(x_combined, cond_idx=None, input_pos=input_pos,
                      condition=condition)
    cond_logits, uncond_logits = torch.split(logits, len(logits) // 2, dim=0)
    logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        thresh = torch.topk(logits, k)[0][..., -1, None]
        logits = torch.where(logits < thresh,
                             torch.full_like(logits, -float("inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


@torch.no_grad()
def generate_fast(model, cond, max_new_tokens, emb_masks=None, cfg_scale=1.0,
                  cfg_interval=-1, condition=None, control_strength=1.0,
                  e2e_step=None, **sampling_kwargs):
    # condition chain (same as official)
    condition = model.adapter(condition)
    condition = model.adapter_mlp(condition)
    cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
    cond_combined = (torch.cat([cond, cond_null]) if cfg_scale > 1.0 else cond)
    condition_null = torch.zeros_like(condition)
    condition_combined = (torch.cat([condition, condition_null])
                          if cfg_scale > 1.0 else condition)

    T = cond.shape[1]
    T_new = T + max_new_tokens
    max_batch_size = cond.shape[0]
    device = cond.device
    with torch.device(device):
        mb = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=mb, max_seq_length=T_new,
                           dtype=model.tok_embeddings.weight.dtype)
    if emb_masks is not None:
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = (model.causal_mask[:, :, :T] *
                                           torch.cat([emb_masks,
                                                      emb_masks]).unsqueeze(1))
        else:
            model.causal_mask[:, :, :T] = (model.causal_mask[:, :, :T] *
                                           emb_masks.unsqueeze(1))
    eye = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2),
                    device=device)
    model.causal_mask[:] = model.causal_mask * (1 - eye) + eye

    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)
    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, cond_combined, input_pos, cfg_scale,
                         condition_combined, control_strength,
                         **sampling_kwargs)
    # --- CUDA-graph safety: pull condition_token out of the prefill pool ---
    if model.condition_token is not None:
        model.condition_token = [t.clone() for t in model.condition_token]
    seq[:, T:T + 1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    cur_token = next_token.view(-1, 1)
    cfg_flag = True
    for i in range(max_new_tokens - 1):
        if cfg_interval > -1 and i > cfg_interval and cfg_flag:
            cfg_flag = False
            # drop the uncond branch for real: shrink KV caches, mask and
            # condition tokens from 2B to B rows (one-time eager work)
            B = cur_token.shape[0]
            for blk in model.layers:
                kc = blk.attention.kv_cache
                kc.k_cache = kc.k_cache[:B].clone()
                kc.v_cache = kc.v_cache[:B].clone()
            model.max_batch_size = B
            model.causal_mask = model.causal_mask[:B].clone()
            if model.condition_token is not None:
                model.condition_token = [t[:B]
                                         for t in model.condition_token]
        if e2e_step is not None and cfg_flag and cfg_scale > 1.0 and \
           sampling_kwargs.get("sample_logits", True):
            next_token = e2e_step(model, cur_token, input_pos,
                                  condition_combined, cfg_scale,
                                  sampling_kwargs.get("top_k", 2000),
                                  sampling_kwargs.get("temperature", 1.0))
            next_token = next_token.clone()
        else:
            next_token, _ = decode_one_token(model, cur_token, input_pos,
                                             cfg_scale, cfg_flag,
                                             condition_combined,
                                             **sampling_kwargs)
            next_token = next_token.clone()  # out of the graph pool
        seq[:, T + 1 + i:T + 2 + i] = next_token
        input_pos += 1
        cur_token = next_token.view(-1, 1)
    return seq[:, T:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt-ckpt", default="checkpoints/t2i/hed.safetensors")
    ap.add_argument("--vq-ckpt", default="checkpoints/vq/vq_ds16_t2i.pt")
    ap.add_argument("--t5-path", default="checkpoints/t5-ckpt")
    ap.add_argument("--condition-path",
                    default="condition/example/t2i/multigen/eye.png")
    ap.add_argument("--prompt", default="a beautiful blue eye, ultra detailed")
    ap.add_argument("--num-images", type=int, default=1)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--cfg-interval", type=float, default=-1)
    ap.add_argument("--top-k", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile-mode", type=str, default=None,
                    choices=[None, "default", "reduce-overhead",
                             "max-autotune-no-cudagraphs", "max-autotune"])
    ap.add_argument("--e2e", action="store_true",
                    help="compile whole decode step (fwd+CFG+sample) into one graph")
    ap.add_argument("--quant", type=str, default=None,
                    choices=[None, "int8wo", "int4wo"])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default="/tmp/fast.png")
    ap.add_argument("--tokens-out", default="/tmp/fast.npy")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda"
    precision = torch.bfloat16
    n_tokens = 1024

    vq_model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
    vq_model.to(device).eval()
    vq_model.load_state_dict(
        torch.load(args.vq_ckpt, map_location="cpu")["model"])

    gpt = GPT_models["GPT-XL"](block_size=n_tokens, cls_token_num=120,
                               model_type="t2i", condition_type="hed",
                               adapter_size="small").to(device=device,
                                                        dtype=precision)
    gpt.load_state_dict(load_file(args.gpt_ckpt), strict=False)
    gpt.eval()

    if args.quant in ("int8wo", "int4wo"):
        from torchao.quantization import (Int4WeightOnlyConfig,
                                          Int8WeightOnlyConfig, quantize_)
        cfg = Int8WeightOnlyConfig() if args.quant == "int8wo"             else Int4WeightOnlyConfig()
        quantize_(gpt, cfg,
                  filter_fn=lambda m, n: isinstance(m, torch.nn.Linear) and
                  (n.startswith("layers.") or n == "output"))
        print(f"[fast] {args.quant} quantized")

    e2e_step = None
    if args.e2e:
        mode = args.compile_mode or "reduce-overhead"
        if mode == "default":
            mode = None
        e2e_step = torch.compile(_decode_step_e2e, mode=mode, fullgraph=True)
        print(f"[fast] e2e decode step compiled ({mode})")
    elif args.compile_mode:
        mode = None if args.compile_mode == "default" else args.compile_mode
        gpt = torch.compile(gpt, mode=mode, fullgraph=True)
        print(f"[fast] compiled ({args.compile_mode})")

    t5 = T5Embedder(device=device, local_cache=True, cache_dir=args.t5_path,
                    dir_or_name="flan-t5-xl", torch_dtype=precision,
                    model_max_length=120)
    caption_embs, emb_masks = t5.get_text_embeddings([args.prompt] *
                                                     args.num_images)
    new_masks = torch.flip(emb_masks, dims=[-1])
    new_embs = []
    for emb, m in zip(caption_embs, emb_masks):
        v = int(m.sum().item())
        new_embs.append(torch.cat([emb[v:], emb[:v]]))
    c_indices = torch.stack(new_embs) * new_masks[:, :, None]

    hed = HEDdetector().to(device).eval()
    img = torch.from_numpy(np.array(Image.open(args.condition_path))).permute(
        2, 0, 1).unsqueeze(0).to(device)
    hed_map = hed(img)
    condition_img = hed_map.unsqueeze(1).repeat(args.num_images, 3, 1, 1)
    condition_img = 2 * (condition_img / 255 - 0.5)

    def run():
        return generate_fast(
            gpt, c_indices, n_tokens, new_masks,
            condition=condition_img.to(precision), cfg_scale=args.cfg_scale,
            cfg_interval=args.cfg_interval, temperature=args.temperature,
            top_k=args.top_k, top_p=1.0, sample_logits=args.top_k != 1,
            control_strength=1.0, e2e_step=e2e_step)

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    t0 = time.time()
    index_sample = run()
    torch.cuda.synchronize()
    dt = time.time() - t0
    total = args.num_images * n_tokens
    print(f"[fast] sampling: {dt:.2f}s for {total} tokens "
          f"({total / dt:.1f} tok/s) cfg_interval={args.cfg_interval} "
          f"compile={args.compile_mode} quant={args.quant}")

    np.save(args.tokens_out, index_sample.cpu().numpy())
    samples = vq_model.decode_code(index_sample,
                                   [args.num_images, 8, 32, 32])
    hed_vis = 2 * (hed_map.unsqueeze(1).repeat(1, 3, 1, 1) / 255.0 - 0.5)
    save_image(torch.cat([hed_vis, samples], 0), args.out,
               nrow=args.num_images + 1, normalize=True, value_range=(-1, 1))
    print(f"[fast] saved {args.out}")


if __name__ == "__main__":
    main()
