"""Compiled, fixed-shape inference helpers for the CSGO Seen-10 runner.

This module is safe to import on CPU.  CUDA work begins only when the caller
invokes ``generate_compiled`` with CUDA tensors.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping, Sequence

import torch

from autoregressive.models.generate import prefill as official_prefill


COMPILE_MODE = "reduce-overhead"


def stable_batch_seed(
    base_seed: int,
    task: str,
    batch_index: int,
    real_sample_ids: Sequence[str],
) -> int:
    """Hash the identity of one manifest block into a deterministic RNG seed."""

    payload = json.dumps(
        {
            "base_seed": int(base_seed),
            "task": str(task),
            "batch_index": int(batch_index),
            "real_sample_ids": [str(sample_id) for sample_id in real_sample_ids],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def pad_batch_to_fixed_size(
    batch: Mapping[str, Any], batch_size: int
) -> dict[str, Any]:
    """Pad the final DataLoader batch by repeating its last tensor/list item."""

    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    values = dict(batch)
    sizes = {int(value.shape[0]) for value in values.values() if isinstance(value, torch.Tensor)}
    sizes.update(len(value) for value in values.values() if isinstance(value, (list, tuple)))
    if len(sizes) != 1:
        raise ValueError(f"Batch fields have inconsistent leading sizes: {sorted(sizes)}")
    real_count = sizes.pop()
    if real_count < 1:
        raise ValueError("Cannot pad an empty batch")
    if real_count > batch_size:
        raise ValueError(f"Batch has {real_count} items, larger than fixed size {batch_size}")
    padding = batch_size - real_count
    if padding == 0:
        return values

    padded: dict[str, Any] = {}
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            tail = value[-1:].expand((padding,) + tuple(value.shape[1:]))
            padded[name] = torch.cat((value, tail), dim=0)
        elif isinstance(value, list):
            padded[name] = value + [value[-1]] * padding
        elif isinstance(value, tuple):
            padded[name] = value + (value[-1],) * padding
        else:
            raise TypeError(
                f"Unsupported collated batch field {name!r}: {type(value).__name__}"
            )
    return padded


def _decode_step_logits(
    model: torch.nn.Module,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    condition: torch.Tensor,
    cfg_scale: float,
    top_k: int,
    temperature: float,
) -> torch.Tensor:
    """Compute classifier-free guidance and top-k filtered logits."""

    x_combined = torch.cat([x, x])
    logits, _ = model(
        x_combined,
        cond_idx=None,
        input_pos=input_pos,
        condition=condition,
    )
    cond_logits, uncond_logits = torch.split(logits, len(logits) // 2, dim=0)
    logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k)[0][..., -1, None]
        logits = torch.where(
            logits < threshold, torch.full_like(logits, -float("inf")), logits
        )
    return logits


def _decode_step_e2e(
    model: torch.nn.Module,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    condition: torch.Tensor,
    cfg_scale: float,
    top_k: int,
    temperature: float,
) -> torch.Tensor:
    """Run one full CFG/top-k/sample decode step in one compiled graph."""

    logits = _decode_step_logits(
        model, x, input_pos, condition, cfg_scale, top_k, temperature
    )
    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, 1)


class CompiledCachePool:
    """Reuse one KV allocation per decode shape and clear it before each batch."""

    def __init__(self) -> None:
        self.signature: tuple[int, int, torch.dtype] | None = None
        self.reuse_count = 0
        self.allocate_count = 0
        self.first_e2e_step_logged = False

    def prepare(
        self,
        model: torch.nn.Module,
        *,
        batch_size: int,
        sequence_length: int,
        dtype: torch.dtype,
        cfg_scale: float,
    ) -> None:
        cache_batch_size = batch_size * 2 if cfg_scale > 1.0 else batch_size
        signature = (cache_batch_size, sequence_length, dtype)
        if self.signature != signature:
            with torch.device(next(model.parameters()).device):
                model.setup_caches(
                    max_batch_size=cache_batch_size,
                    max_seq_length=sequence_length,
                    dtype=dtype,
                )
            self.signature = signature
            self.allocate_count += 1
            return

        # Every decode reads the cache tensors; stale K/V values must not leak
        # from the preceding fixed-shape batch.
        for layer in model.layers:
            cache = layer.attention.kv_cache
            if cache is None:
                raise RuntimeError("Compiled cache pool found an uninitialized KV cache")
            cache.k_cache.zero_()
            cache.v_cache.zero_()
        self.reuse_count += 1


def _keep_condition_tokens(
    model: torch.nn.Module, fresh_tokens: list[torch.Tensor] | None
) -> None:
    """Keep condition tokens at stable addresses for reduce-overhead graphs."""

    if fresh_tokens is None:
        return
    persistent = getattr(model, "_csgo_infer_condition_tokens", None)
    if (
        persistent is None
        or len(persistent) != len(fresh_tokens)
        or any(old.shape != new.shape for old, new in zip(persistent, fresh_tokens))
    ):
        persistent = [token.detach().clone() for token in fresh_tokens]
        model._csgo_infer_condition_tokens = persistent
    else:
        for old, new in zip(persistent, fresh_tokens):
            old.copy_(new)
    model.condition_token = persistent


@torch.no_grad()
def generate_compiled(
    model: torch.nn.Module,
    caption: torch.Tensor,
    *,
    max_new_tokens: int,
    condition: torch.Tensor,
    cfg_scale: float,
    temperature: float,
    top_k: int,
    top_p: float,
    e2e_step: Any,
    cache_pool: CompiledCachePool,
) -> torch.Tensor:
    """Generate a fixed-size batch using the benchmark-validated decode path."""

    if top_p != 1.0:
        raise ValueError("Compiled inference requires top_p=1.0")
    if cfg_scale <= 1.0:
        raise ValueError("Compiled inference requires cfg_scale > 1")
    if caption.shape[0] < 1:
        raise ValueError("Compiled inference cannot generate an empty batch")

    condition = model.adapter(condition)
    condition = model.adapter_mlp(condition)
    cond_null = torch.zeros_like(caption) + model.cls_embedding.uncond_embedding
    cond_combined = torch.cat([caption, cond_null])
    condition_null = torch.zeros_like(condition)
    condition_combined = torch.cat([condition, condition_null])

    batch_size = caption.shape[0]
    sequence_length = caption.shape[1] + max_new_tokens
    device = caption.device
    cache_pool.prepare(
        model,
        batch_size=batch_size,
        sequence_length=sequence_length,
        dtype=model.tok_embeddings.weight.dtype,
        cfg_scale=cfg_scale,
    )

    # CSGO has no embedding masks, so keep the model's native bool mask.
    input_pos = torch.arange(caption.shape[1], device=device)
    next_token = official_prefill(
        model,
        cond_combined,
        input_pos,
        cfg_scale,
        condition_combined,
        control_strength=1.0,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        sample_logits=True,
    )
    _keep_condition_tokens(model, model.condition_token)

    sequence = torch.empty(
        (batch_size, sequence_length), dtype=torch.int, device=device
    )
    sequence[:, caption.shape[1] : caption.shape[1] + 1] = next_token
    input_pos = torch.tensor([caption.shape[1]], device=device, dtype=torch.int)
    current_token = next_token.view(-1, 1)
    for index in range(max_new_tokens - 1):
        step_started = time.perf_counter()
        next_token = e2e_step(
            model,
            current_token,
            input_pos,
            condition_combined,
            cfg_scale,
            top_k,
            temperature,
        ).clone()
        if not cache_pool.first_e2e_step_logged:
            torch.cuda.synchronize(device)
            print(
                json.dumps(
                    {
                        "event": "compiled_first_e2e_step",
                        "engine": "compiled",
                        "batch_size": batch_size,
                        "phase": "first_e2e_step",
                        "time": time.time(),
                        "elapsed_s": time.perf_counter() - step_started,
                        "cache_reused": cache_pool.reuse_count > 0,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            cache_pool.first_e2e_step_logged = True
        sequence[:, caption.shape[1] + 1 + index : caption.shape[1] + 2 + index] = next_token
        input_pos += 1
        current_token = next_token.view(-1, 1)
    return sequence[:, caption.shape[1] :]


def prepare_compiled_step() -> Any:
    """Create the benchmark-validated fullgraph reduce-overhead decode graph."""

    return torch.compile(_decode_step_e2e, mode=COMPILE_MODE, fullgraph=True)
