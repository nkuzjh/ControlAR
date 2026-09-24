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
STATELESS_SEED_POLICY = (
    "stateless-sample-v1: SHA256(UTF-8 bytes of str(base_seed) + NUL + "
    "sample_id + NUL + decimal token index), first 64 bits mapped to (0, 1)"
)


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


def stateless_uniforms_for_sample(
    base_seed: int, sample_id: str, count: int
) -> list[float]:
    """Create a batch-independent uniform stream for one manifest sample.

    The stream is generated on the CPU from the sample identity and token
    position.  It is therefore unchanged when a batch is resumed with some
    output files already present, or when the same manifest block is executed
    with a different neighbouring sample.
    """

    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    values: list[float] = []
    for token_index in range(count):
        payload = f"{int(base_seed)}\0{sample_id}\0{token_index}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        # The half-open interval is kept away from both endpoints.  This
        # avoids selecting a zero-probability token at exactly u=0 and keeps
        # inverse-CDF sampling finite at u=1.
        integer = int.from_bytes(digest[:8], "big")
        values.append((integer + 0.5) / float(1 << 64))
    return values


def stateless_uniforms_for_batch(
    base_seed: int,
    sample_ids: Sequence[str],
    count: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``[batch, count]`` uniforms in manifest order."""

    if not sample_ids:
        raise ValueError("Cannot build stateless uniforms for an empty batch")
    rows = [stateless_uniforms_for_sample(base_seed, str(sample_id), count) for sample_id in sample_ids]
    return torch.tensor(rows, dtype=torch.float32, device=device)


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


def _sample_inverse_cdf(logits: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    """Sample one token per row from already filtered logits."""

    if uniforms.ndim != 1 or uniforms.shape[0] != logits.shape[0]:
        raise ValueError(
            f"uniforms must have shape [{logits.shape[0]}], got {tuple(uniforms.shape)}"
        )
    # Accumulate the inverse CDF in FP32.  BF16 cumulative sums can collapse
    # many small top-k probabilities into the same bucket and bias sampling.
    probabilities = torch.softmax(logits.float(), dim=-1)
    cdf = torch.cumsum(probabilities, dim=-1)
    # ``cdf < u`` selects the first bucket whose cumulative mass reaches u.
    # Clamp for the unlikely case of accumulated floating-point mass below u.
    token = torch.sum(cdf < uniforms.float().unsqueeze(-1), dim=-1)
    return token.clamp_max(logits.shape[-1] - 1).to(dtype=torch.long).unsqueeze(-1)


def _prefill_stateless(
    model: torch.nn.Module,
    cond_idx: torch.Tensor,
    input_pos: torch.Tensor,
    cfg_scale: float,
    condition: torch.Tensor,
    control_strength: float,
    temperature: float,
    top_k: int,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    """Prefill and sample without touching torch's mutable RNG state."""

    if cfg_scale > 1.0:
        logits, _ = model(
            None,
            cond_idx,
            input_pos,
            condition=condition,
            control_strength=control_strength,
        )
        cond_logits, uncond_logits = torch.split(logits, len(logits) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = model(None, cond_idx, input_pos, condition=condition)
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k)[0][..., -1, None]
        logits = torch.where(
            logits < threshold, torch.full_like(logits, -float("inf")), logits
        )
    return _sample_inverse_cdf(logits, uniforms)


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


def _decode_step_stateless(
    model: torch.nn.Module,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    condition: torch.Tensor,
    cfg_scale: float,
    top_k: int,
    temperature: float,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    """Compiled decode step using caller-provided per-sample uniforms."""

    logits = _decode_step_logits(
        model, x, input_pos, condition, cfg_scale, top_k, temperature
    )
    return _sample_inverse_cdf(logits, uniforms)


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
    stateless_uniforms: torch.Tensor | None = None,
    stateless_step: Any = None,
    stateless_logits_step: Any = None,
) -> torch.Tensor:
    """Generate a fixed-size batch using the benchmark-validated decode path."""

    if top_p != 1.0:
        raise ValueError("Compiled inference requires top_p=1.0")
    if cfg_scale <= 1.0:
        raise ValueError("Compiled inference requires cfg_scale > 1")
    if caption.shape[0] < 1:
        raise ValueError("Compiled inference cannot generate an empty batch")
    if stateless_uniforms is None and e2e_step is None:
        raise ValueError("e2e_step is required for legacy compiled inference")
    if stateless_uniforms is not None:
        if stateless_uniforms.ndim != 2:
            raise ValueError(
                f"stateless_uniforms must have shape [batch, tokens], got {tuple(stateless_uniforms.shape)}"
            )
        if stateless_uniforms.shape != (caption.shape[0], max_new_tokens):
            raise ValueError(
                "stateless_uniforms shape must equal "
                f"({caption.shape[0]}, {max_new_tokens}), got {tuple(stateless_uniforms.shape)}"
            )
        if (stateless_step is None) == (stateless_logits_step is None):
            raise ValueError("Exactly one stateless decode step is required with stateless_uniforms")
        stateless_uniforms = stateless_uniforms.to(device=caption.device, dtype=torch.float32)
    elif stateless_step is not None or stateless_logits_step is not None:
        raise ValueError("Stateless decode step requires stateless_uniforms")

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
    if stateless_uniforms is None:
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
    else:
        next_token = _prefill_stateless(
            model,
            cond_combined,
            input_pos,
            cfg_scale,
            condition_combined,
            1.0,
            temperature,
            top_k,
            stateless_uniforms[:, 0],
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
        if stateless_uniforms is None:
            next_token = e2e_step(
                model,
                current_token,
                input_pos,
                condition_combined,
                cfg_scale,
                top_k,
                temperature,
            ).clone()
        elif stateless_logits_step is not None:
            # PEFT opt-in: compile the costly Transformer/CFG/top-k path, but
            # keep inverse-CDF scan in native CUDA ATen.  Some Inductor builds
            # fail to lower the fused [16, 16384] cumsum in the full graph.
            logits = stateless_logits_step(
                model,
                current_token,
                input_pos,
                condition_combined,
                cfg_scale,
                top_k,
                temperature,
            )
            next_token = _sample_inverse_cdf(
                logits, stateless_uniforms[:, index + 1]
            ).clone()
        else:
            next_token = stateless_step(
                model,
                current_token,
                input_pos,
                condition_combined,
                cfg_scale,
                top_k,
                temperature,
                stateless_uniforms[:, index + 1],
            ).clone()
        if not cache_pool.first_e2e_step_logged:
            torch.cuda.synchronize(device)
            print(
                json.dumps(
                    {
                        "event": (
                            "compiled_first_logits_then_aten_cdf_step"
                            if stateless_logits_step is not None
                            else "compiled_first_e2e_step"
                        ),
                        "engine": "compiled",
                        "batch_size": batch_size,
                        "phase": (
                            "compiled_logits_then_aten_inverse_cdf"
                            if stateless_logits_step is not None
                            else "first_e2e_step"
                        ),
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


def prepare_compiled_stateless_step() -> Any:
    """Create the aligned decode graph with an explicit uniform input."""

    return torch.compile(_decode_step_stateless, mode=COMPILE_MODE, fullgraph=True)
