#!/usr/bin/env python
"""Generate MultiGen-20M val images with the fast engine (cudagraph),
following the official test_t2i.py protocol (hed, cfg=4, top_k=2000)."""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from safetensors.torch import load_file  # noqa: E402
from torchvision.utils import save_image  # noqa: E402
from tokenizer.tokenizer_image.vq_model import VQ_models  # noqa: E402
from language.t5 import T5Embedder  # noqa: E402
from autoregressive.models.gpt_t2i import GPT_models  # noqa: E402
from condition.hed import HEDdetector  # noqa: E402
from fast_inference.fast_engine import generate_fast  # noqa: E402

VAL = "data/MultiGen20M/val"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt-ckpt", default="checkpoints/t2i/hed.safetensors")
    ap.add_argument("--vq-ckpt", default="checkpoints/vq/vq_ds16_t2i.pt")
    ap.add_argument("--t5-path", default="checkpoints/t5-ckpt")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--top-k", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile-mode", default="max-autotune")
    ap.add_argument("--e2e", action="store_true", default=True)
    ap.add_argument("--out", default="sample/multigen/hed_fast500")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda"
    precision = torch.bfloat16
    n_tokens = 1024

    vis_dir = f"{args.out}/visualization"
    ann_dir = f"{args.out}/annotations"
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    todo = [i for i in range(args.start, args.end)
            if not os.path.exists(f"{vis_dir}/{i:06d}.png")]
    print(f"[fasteval] {len(todo)} to generate")
    if not todo:
        return

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
    from fast_inference.fast_engine import _decode_step_e2e
    e2e_step = None
    if args.e2e:
        e2e_step = torch.compile(_decode_step_e2e, mode=args.compile_mode,
                                 fullgraph=True)
    else:
        gpt = torch.compile(gpt, mode=args.compile_mode, fullgraph=True)

    t5 = T5Embedder(device=device, local_cache=True, cache_dir=args.t5_path,
                    dir_or_name="flan-t5-xl", torch_dtype=precision,
                    model_max_length=120)
    hed = HEDdetector().to(device).eval()

    t_start = time.time()
    done = 0
    for cs in range(0, len(todo), args.batch):
        chunk = todo[cs:cs + args.batch]
        B = len(chunk)
        prompts, imgs = [], []
        for i in chunk:
            prompts.append(
                str(np.load(f"{VAL}/caption_emb/{i}.npz")["prompt"][0]))
            imgs.append(
                np.array(Image.open(f"{VAL}/image/{i}.png").convert("RGB")))
        img_t = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to(device)
        hed_map = hed(img_t.float())
        condition = hed_map.unsqueeze(1) / 255.0
        condition = condition.repeat(1, 3, 1, 1)
        condition = 2 * (condition - 0.5)

        caption_embs, emb_masks = t5.get_text_embeddings(prompts)
        new_masks = torch.flip(emb_masks, dims=[-1])
        new_embs = []
        for emb, m in zip(caption_embs, emb_masks):
            v = int(m.sum().item())
            new_embs.append(torch.cat([emb[v:], emb[:v]]))
        c_indices = torch.stack(new_embs) * new_masks[:, :, None]

        index_sample = generate_fast(
            gpt, c_indices, n_tokens, new_masks,
            condition=condition.to(precision), cfg_scale=args.cfg_scale,
            cfg_interval=-1, temperature=1.0, top_k=args.top_k, top_p=1.0,
            sample_logits=True, control_strength=1.0, e2e_step=e2e_step)
        samples = vq_model.decode_code(index_sample, [B, 8, 32, 32])
        for j, i in enumerate(chunk):
            save_image(samples[j], f"{vis_dir}/{i:06d}.png", nrow=1,
                       normalize=True, value_range=(-1, 1))
            save_image(condition[j, 0], f"{ann_dir}/{i:06d}.png", nrow=1,
                       normalize=True, value_range=(-1, 1))
        done += B
        el = time.time() - t_start
        print(f"[fasteval] {done}/{len(todo)} "
              f"{done * n_tokens / el:.0f} tok/s", flush=True)
    print(f"[fasteval] DONE {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
