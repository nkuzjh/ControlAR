#!/usr/bin/env python3
"""Standalone CSGO Seen-10 inference microbenchmark.

This intentionally lives beside (and does not modify) ``infer_seen10.py``.
It loads the same validation-selected checkpoint, BF16 autoregressive model,
FP32 VQ decoder, numeric pose/map condition and radar condition.  The
benchmark uses a small deterministic, map-balanced sample from each task so
that eager and compiled runs have the same amount of work.

The timed path includes DataLoader collation, host-to-device copies, pose
embedding, all 784 autoregressive tokens, FP32 VQ decoding and JPEG writes.
Model/checkpoint loading and the optional global warmup are reported
separately.  No target image is opened by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoregressive.models.generate import prefill as official_prefill  # noqa: E402
from autoregressive.models.generate import generate as official_generate  # noqa: E402
from csgo_seen10.data import (  # noqa: E402
    MAP_ORDER,
    Seen10GenerationDataset,
    read_benchmark_rows,
)
from csgo_seen10.model import (  # noqa: E402
    Seen10GenerationModel,
    build_gpt,
    load_checkpoint,
)
from tokenizer.tokenizer_image.vq_model import VQ_models  # noqa: E402


TASK_TO_SPLIT = {
    "discrete": "seen_discrete_test",
    "continuous": "seen_continuous",
}
DEFAULT_CONFIG = ROOT / "configs" / "csgo_seen10.json"
IMAGE_SIZE = 448
DOWNSAMPLE_SIZE = 16
TOKEN_COUNT = 120
CODE_LENGTH = (IMAGE_SIZE // DOWNSAMPLE_SIZE) ** 2
CFG_SCALE = 4.0
TEMPERATURE = 1.0
TOP_K = 2000
TOP_P = 1.0


def _config_from_cli() -> tuple[argparse.Namespace, dict[str, Any], Path]:
    """Parse config first, then expose benchmark-specific overrides."""

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_CONFIG))
    pre_args, _ = pre.parse_known_args()
    config_path = Path(pre_args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(
        description="CSGO Seen-10 inference benchmark (eager or compiled decode step)."
    )
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--mode", choices=("eager", "compiled"), default="eager")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples-per-task", type=int, default=16)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--vq-checkpoint", default=config.get("vq_checkpoint"))
    parser.add_argument("--data-root", default=config.get("data_root"))
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow an existing output directory (use only for an intentional rerun).",
    )
    args = parser.parse_args()
    return args, config, config_path


def _resolve_path(value: str | os.PathLike[str] | None, *, base: Path = ROOT) -> Path:
    if value is None:
        raise ValueError("A required path is missing from the command line/config")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _checkpoint_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_seed(seed: int, sample_id: str) -> int:
    payload = f"{seed}\0{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _select_uniform_map_balanced(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Select uniformly spaced rows from the manifest's fixed map ordering.

    The shared benchmark rows are ordered by the ten fixed maps.  A global
    ``linspace`` therefore gives deterministic map coverage for the documented
    16-row sample while keeping the exact same identities for every mode and
    batch size.
    """

    if count < 1:
        raise ValueError(f"samples-per-task must be positive, got {count}")
    if count > len(rows):
        raise ValueError(f"Requested {count} samples but split contains only {len(rows)} rows")
    for row in rows:
        if str(row["map_name"]) not in MAP_ORDER:
            raise ValueError(f"Unexpected map in manifest rows: {row['map_name']!r}")
    positions = np.linspace(0, len(rows) - 1, count, dtype=int).tolist()
    return [rows[position] for position in positions]


def _load_vq_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    vq_model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8).to(
        device=device, dtype=torch.float32
    )
    checkpoint = load_checkpoint(checkpoint_path)
    state = checkpoint.get("model", checkpoint)
    vq_model.load_state_dict(state, strict=True)
    vq_model.eval()
    vq_model.requires_grad_(False)
    return vq_model


def _validate_model_config(
    checkpoint: dict[str, Any],
    *,
    gpt_model: str,
    caption_dim: int,
    adapter_size: str,
    condition_type: str,
) -> None:
    model_config = checkpoint.get("model_config", {})
    expected = {
        "gpt_model": gpt_model,
        "image_size": IMAGE_SIZE,
        "downsample_size": DOWNSAMPLE_SIZE,
        "token_count": TOKEN_COUNT,
        "caption_dim": caption_dim,
        "adapter_size": adapter_size,
        "condition_type": condition_type,
    }
    if model_config:
        for key, value in expected.items():
            if model_config.get(key) != value:
                raise ValueError(
                    f"Checkpoint {key}={model_config.get(key)!r}, requested {value!r}"
                )


def _save_unilip_style(image_tensor: torch.Tensor, path: Path) -> None:
    pixels = (
        (image_tensor.detach().float().clamp(-1, 1) + 1.0) / 2.0 * 255.0
    ).round().clamp(0, 255).to(torch.uint8)
    array = pixels.permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(array, mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _read_proc_rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        return None
    return None


def _gpu_snapshot() -> dict[str, Any]:
    """Collect out-of-band GPU/process samples for competition diagnostics."""

    snapshot: dict[str, Any] = {
        "pid": os.getpid(),
        "process_rss_bytes": _read_proc_rss_bytes(),
        "nvidia_smi_available": False,
        "gpus": [],
        "compute_processes": [],
    }
    query_gpu = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    query_proc = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu_output = subprocess.check_output(query_gpu, text=True, stderr=subprocess.DEVNULL, timeout=5)
        proc_output = subprocess.check_output(query_proc, text=True, stderr=subprocess.DEVNULL, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return snapshot
    snapshot["nvidia_smi_available"] = True
    for line in gpu_output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            snapshot["gpus"].append(
                {
                    "index": int(fields[0]),
                    "utilization_gpu_percent": float(fields[1]),
                    "memory_used_mib": float(fields[2]),
                    "memory_total_mib": float(fields[3]),
                }
            )
        except ValueError:
            continue
    for line in proc_output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            snapshot["compute_processes"].append(
                {"pid": int(fields[0]), "used_memory_mib": float(fields[1])}
            )
        except ValueError:
            continue
    return snapshot


def _decode_step_logits(
    model: torch.nn.Module,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    condition: torch.Tensor,
    cfg_scale: float,
    top_k: int,
    temperature: float,
) -> torch.Tensor:
    """Compute CFG and top-k filtered logits for one compiled decode step."""

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
        logits = torch.where(logits < threshold, torch.full_like(logits, -float("inf")), logits)
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
    """One full CFG/top-k/sample decode step, equivalent to fast_engine's helper."""

    logits = _decode_step_logits(model, x, input_pos, condition, cfg_scale, top_k, temperature)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


class _CompiledCachePool:
    """Reuse one cache allocation per decode shape while clearing every call."""

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

        # The decoder reads the complete cache tensor for every decode step;
        # stale keys/values from a prior sample would otherwise affect output.
        for layer in model.layers:
            cache = layer.attention.kv_cache
            if cache is None:
                raise RuntimeError("Compiled cache pool found an uninitialized KV cache")
            cache.k_cache.zero_()
            cache.v_cache.zero_()
        self.reuse_count += 1


def _keep_condition_tokens(
    model: torch.nn.Module,
    fresh_tokens: list[torch.Tensor] | None,
) -> None:
    """Copy new condition tokens into a persistent list used by compiled graphs."""

    if fresh_tokens is None:
        return
    persistent = getattr(model, "_csgo_bench_condition_tokens", None)
    if (
        persistent is None
        or len(persistent) != len(fresh_tokens)
        or any(old.shape != new.shape for old, new in zip(persistent, fresh_tokens))
    ):
        persistent = [token.detach().clone() for token in fresh_tokens]
        model._csgo_bench_condition_tokens = persistent
    else:
        for old, new in zip(persistent, fresh_tokens):
            old.copy_(new)
    model.condition_token = persistent


@torch.no_grad()
def _generate_compiled(
    model: torch.nn.Module,
    cond: torch.Tensor,
    *,
    max_new_tokens: int,
    condition: torch.Tensor,
    cfg_scale: float,
    temperature: float,
    top_k: int,
    top_p: float,
    e2e_step: Any,
    cache_pool: _CompiledCachePool,
) -> torch.Tensor:
    """Fast-style generation with a compiled step and reused/reset KV caches."""

    if top_p != 1.0:
        raise ValueError("Compiled microbenchmark requires the fixed top_p=1.0")
    condition = model.adapter(condition)
    condition = model.adapter_mlp(condition)
    cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
    cond_combined = torch.cat([cond, cond_null]) if cfg_scale > 1.0 else cond
    condition_null = torch.zeros_like(condition)
    condition_combined = (
        torch.cat([condition, condition_null]) if cfg_scale > 1.0 else condition
    )

    batch_size = cond.shape[0]
    sequence_length = cond.shape[1] + max_new_tokens
    device = cond.device
    cache_pool.prepare(
        model,
        batch_size=batch_size,
        sequence_length=sequence_length,
        dtype=model.tok_embeddings.weight.dtype,
        cfg_scale=cfg_scale,
    )
    # CSGO has no embedding masks.  In particular, leave the model's native
    # bool causal mask untouched.
    input_pos = torch.arange(cond.shape[1], device=device)
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
    sequence[:, cond.shape[1] : cond.shape[1] + 1] = next_token
    input_pos = torch.tensor([cond.shape[1]], device=device, dtype=torch.int)
    cur_token = next_token.view(-1, 1)
    for index in range(max_new_tokens - 1):
        step_started = time.perf_counter()
        next_token = e2e_step(
            model,
            cur_token,
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
                        "mode": "compiled",
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
        sequence[:, cond.shape[1] + 1 + index : cond.shape[1] + 2 + index] = next_token
        input_pos += 1
        cur_token = next_token.view(-1, 1)
    return sequence[:, cond.shape[1] :]


def _batch_to_device(
    batch: dict[str, Any], device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    radar = batch["radar"].to(device=device, dtype=dtype, non_blocking=True)
    pose = batch["pose"].to(device=device, dtype=dtype, non_blocking=True)
    map_id = batch["map_id"].to(device=device, non_blocking=True)
    return radar, pose, map_id


def _run_one_batch(
    *,
    batch: dict[str, Any],
    model: Seen10GenerationModel,
    vq_model: torch.nn.Module,
    device: torch.device,
    mode: str,
    compiled_step: Any | None,
    compiled_cache: _CompiledCachePool | None,
    output_root: Path | None,
    task: str,
    batch_index: int,
    seed: int,
    write_images: bool,
    cpu_collate_s: float = 0.0,
) -> dict[str, Any]:
    """Run one batch and return timing/memory/identity information."""

    if not write_images:
        # Warmups are still synchronized and measured, but never write formal
        # benchmark outputs.
        output_root_for_batch = None
    else:
        output_root_for_batch = output_root
    sample_ids = [str(value) for value in batch["sample_id"]]
    map_names = [str(value) for value in batch["map_name"]]
    file_frames = [str(value) for value in batch["file_frame"]]
    batch_started = time.perf_counter() - cpu_collate_s
    stage: dict[str, float] = {"cpu_collate_s": cpu_collate_s}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    h2d_started = time.perf_counter()
    radar, pose, map_id = _batch_to_device(batch, device, torch.bfloat16)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    stage["h2d_s"] = time.perf_counter() - h2d_started

    pose_started = time.perf_counter()
    with torch.inference_mode():
        caption = model.pose_map_embedder(pose, map_id)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    stage["pose_s"] = time.perf_counter() - pose_started

    generation_started = time.perf_counter()
    with torch.inference_mode():
        if mode == "eager":
            codes = official_generate(
                model.gpt,
                caption,
                max_new_tokens=CODE_LENGTH,
                condition=radar,
                cfg_scale=CFG_SCALE,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                sample_logits=True,
            )
        else:
            if compiled_step is None:
                raise RuntimeError("Compiled mode was selected without a compiled decode step")
            if compiled_cache is None:
                raise RuntimeError("Compiled mode was selected without a cache pool")
            codes = _generate_compiled(
                model.gpt,
                caption,
                max_new_tokens=CODE_LENGTH,
                condition=radar,
                cfg_scale=CFG_SCALE,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                e2e_step=compiled_step,
                cache_pool=compiled_cache,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    stage["generation_s"] = time.perf_counter() - generation_started

    # Decode each image independently in FP32.  This keeps batch-16 peak
    # memory bounded and applies identically to eager and compiled modes.
    vq_started = time.perf_counter()
    images: list[torch.Tensor] = []
    shape = [1, 8, IMAGE_SIZE // DOWNSAMPLE_SIZE, IMAGE_SIZE // DOWNSAMPLE_SIZE]
    with torch.inference_mode():
        for index in range(codes.shape[0]):
            image = vq_model.decode_code(codes[index : index + 1], shape)[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            images.append(image)
    stage["vq_decode_s"] = time.perf_counter() - vq_started

    jpeg_started = time.perf_counter()
    if output_root_for_batch is not None:
        for image, map_name, file_frame in zip(images, map_names, file_frames):
            output_path = output_root_for_batch / task / "gen_imgs" / map_name / f"{file_frame}.jpg"
            _save_unilip_style(image, output_path)
    stage["jpeg_s"] = time.perf_counter() - jpeg_started
    stage["total_s"] = time.perf_counter() - batch_started

    allocated_peak = None
    reserved_peak = None
    if torch.cuda.is_available():
        allocated_peak = int(torch.cuda.max_memory_allocated(device))
        reserved_peak = int(torch.cuda.max_memory_reserved(device))
    return {
        "event": "batch",
        "task": task,
        "batch_index": batch_index,
        "sample_ids": sample_ids,
        "map_names": map_names,
        "file_frames": file_frames,
        "sample_count": len(sample_ids),
        "timing_s": stage,
        "throughput_samples_per_s": len(sample_ids) / max(stage["total_s"], 1e-12),
        "torch_allocated_peak_bytes": allocated_peak,
        "torch_reserved_peak_bytes": reserved_peak,
        "gpu_snapshot": _gpu_snapshot(),
        "vq_decode_strategy": "per_image_sequential_fp32",
        "write_images": bool(write_images),
        "seed_policy": "batch RNG seeded from the first sample ID hash; batch/compile may produce different pixels",
    }


def _iter_batches(loader: DataLoader) -> Iterator[tuple[dict[str, Any], float]]:
    iterator = iter(loader)
    while True:
        fetch_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            return
        yield batch, time.perf_counter() - fetch_started


def _prepare_compiled_step(mode: str) -> Any | None:
    if mode != "compiled":
        return None
    # This is intentionally the same fullgraph/reduce-overhead contract as
    # fast_inference.fast_engine's _decode_step_e2e, kept local so importing
    # the benchmark never pulls in T5 or HED.
    return torch.compile(_decode_step_e2e, mode="reduce-overhead", fullgraph=True)


def _build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return _resolve_path(args.output_dir)
    return (
        ROOT
        / "outputs"
        / "csgo_inference_microbench"
        / f"{args.mode}_bs{args.batch_size}_n{args.samples_per_task}_seed{args.seed}"
    ).resolve()


def _validate_args_and_paths(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[Path, Path, Path, Path]:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.samples_per_task < 1:
        raise ValueError("samples-per-task must be positive")
    if args.warmup_batches < 0:
        raise ValueError("warmup-batches cannot be negative")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if int(config.get("image_size", IMAGE_SIZE)) != IMAGE_SIZE:
        raise ValueError("CSGO microbenchmark requires image_size=448")
    if int(config.get("downsample_size", DOWNSAMPLE_SIZE)) != DOWNSAMPLE_SIZE:
        raise ValueError("CSGO microbenchmark requires downsample_size=16")
    if int(config.get("token_count", TOKEN_COUNT)) != TOKEN_COUNT:
        raise ValueError("CSGO microbenchmark requires token_count=120")
    if float(config.get("cfg_scale", CFG_SCALE)) != CFG_SCALE:
        raise ValueError("CSGO microbenchmark requires cfg_scale=4.0")
    if float(config.get("temperature", TEMPERATURE)) != TEMPERATURE:
        raise ValueError("CSGO microbenchmark requires temperature=1.0")
    if int(config.get("top_k", TOP_K)) != TOP_K:
        raise ValueError("CSGO microbenchmark requires top_k=2000")
    if float(config.get("top_p", TOP_P)) != TOP_P:
        raise ValueError("CSGO microbenchmark requires top_p=1.0")

    data_root = _resolve_path(args.data_root)
    vq_checkpoint = _resolve_path(args.vq_checkpoint)
    if args.checkpoint is None:
        output_base = _resolve_path(config.get("output_base"))
        checkpoint = output_base / f"seed_{args.seed}" / "checkpoints" / "best.pt"
    else:
        checkpoint = _resolve_path(args.checkpoint)
    output_dir = _build_output_dir(args)
    if not data_root.is_dir():
        raise FileNotFoundError(f"CSGO data root not found: {data_root}")
    if not vq_checkpoint.is_file():
        raise FileNotFoundError(f"VQ checkpoint not found: {vq_checkpoint}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected best checkpoint not found: {checkpoint}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_existing:
        raise FileExistsError(
            f"Output directory is non-empty: {output_dir}; use a new --output-dir or --allow-existing"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return data_root, vq_checkpoint, checkpoint, output_dir


def run_benchmark(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    data_root, vq_checkpoint, checkpoint_path, output_dir = _validate_args_and_paths(args, config)
    progress_path = output_dir / "progress.jsonl"
    final_path = output_dir / "benchmark_results.json"
    checkpoint_sha = _checkpoint_fingerprint(checkpoint_path)
    vq_checkpoint_sha = _checkpoint_fingerprint(vq_checkpoint)
    result: dict[str, Any] = {
        "status": "running",
        "started_at_unix": time.time(),
        "benchmark": "csgo_seen10_inference_microbenchmark",
        "mode": args.mode,
        "batch_size": args.batch_size,
        "samples_per_task": args.samples_per_task,
        "warmup_batches": args.warmup_batches,
        "seed": args.seed,
        "data_root": str(data_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "vq_checkpoint": str(vq_checkpoint),
        "vq_checkpoint_sha256": vq_checkpoint_sha,
        "output_dir": str(output_dir),
        "image_size": IMAGE_SIZE,
        "downsample_size": DOWNSAMPLE_SIZE,
        "code_length": CODE_LENGTH,
        "token_count": TOKEN_COUNT,
        "sampling": {
            "cfg_scale": CFG_SCALE,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "sample_logits": True,
        },
        "precision": {"gpt": "bf16", "vq": "fp32"},
        "environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "TORCHINDUCTOR_COMPILE_THREADS": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
            "pid": os.getpid(),
        },
        "timed_stages": [
            "cpu_collate_s",
            "h2d_s",
            "pose_s",
            "generation_s",
            "vq_decode_s",
            "jpeg_s",
            "total_s",
        ],
        "vq_decode_strategy": "per_image_sequential_fp32",
        "progress_jsonl": str(progress_path),
        "selected_samples": {},
        "warmup_records": [],
        "task_summaries": {},
    }
    _write_json(final_path, result)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CSGO inference benchmark requires CUDA; CPU mode is only for syntax checks")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        result["runtime"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "device_index": device.index,
        }

        started_loading = time.perf_counter()
        checkpoint = load_checkpoint(checkpoint_path)
        gpt_model_name = str(config.get("gpt_model", "GPT-XL"))
        caption_dim = int(config.get("caption_dim", 2048))
        adapter_size = str(config.get("adapter_size", "small"))
        condition_type = str(config.get("condition_type", "radar"))
        _validate_model_config(
            checkpoint,
            gpt_model=gpt_model_name,
            caption_dim=caption_dim,
            adapter_size=adapter_size,
            condition_type=condition_type,
        )
        gpt = build_gpt(
            model_name=gpt_model_name,
            image_size=IMAGE_SIZE,
            downsample_size=DOWNSAMPLE_SIZE,
            token_count=TOKEN_COUNT,
            adapter_size=adapter_size,
            condition_type=condition_type,
            dropout=0.0,
            token_dropout=0.0,
        ).to(device=device, dtype=torch.bfloat16)
        model = Seen10GenerationModel(gpt, caption_dim=caption_dim, token_count=TOKEN_COUNT).to(
            device=device, dtype=torch.bfloat16
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        del checkpoint
        vq_model = _load_vq_model(vq_checkpoint, device)
        loading_s = time.perf_counter() - started_loading
        result["model_loading_s_excluded_from_timing"] = loading_s

        selected_rows: dict[str, list[dict[str, Any]]] = {}
        datasets: dict[str, Seen10GenerationDataset] = {}
        for task, split in TASK_TO_SPLIT.items():
            rows = read_benchmark_rows(data_root, split, require_images=False)
            selected = _select_uniform_map_balanced(rows, args.samples_per_task)
            selected_rows[task] = selected
            result["selected_samples"][task] = [
                {
                    "sample_id": row["sample_id"],
                    "map_name": row["map_name"],
                    "file_frame": row["file_frame"],
                    "clip_id": row.get("clip_id"),
                    "frame_index": row.get("frame_index"),
                }
                for row in selected
            ]
            datasets[task] = Seen10GenerationDataset(
                selected, image_size=IMAGE_SIZE, include_target=False
            )
        _write_json(output_dir / "selected_samples.json", result["selected_samples"])

        compiled_step = _prepare_compiled_step(args.mode)
        compiled_cache = _CompiledCachePool() if args.mode == "compiled" else None
        with progress_path.open("a", encoding="utf-8") as progress:
            # A single global warmup batch is enough to separate compile/setup
            # costs from the two equally sized measured tasks.
            if args.warmup_batches:
                print(
                    json.dumps(
                        {
                            "mode": args.mode,
                            "batch_size": args.batch_size,
                            "phase": "warmup_start",
                            "time": time.time(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                warmup_loader = DataLoader(
                    datasets["discrete"],
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=False,
                    persistent_workers=args.num_workers > 0,
                )
                for warmup_index, (warmup_batch, cpu_collate_s) in enumerate(_iter_batches(warmup_loader)):
                    if warmup_index >= args.warmup_batches:
                        break
                    warmup_record = _run_one_batch(
                        batch=warmup_batch,
                        model=model,
                        vq_model=vq_model,
                        device=device,
                        mode=args.mode,
                        compiled_step=compiled_step,
                        compiled_cache=compiled_cache,
                        output_root=None,
                        task="warmup",
                        batch_index=warmup_index,
                        seed=args.seed,
                        write_images=False,
                        cpu_collate_s=cpu_collate_s,
                    )
                    warmup_record["event"] = "warmup"
                    result["warmup_records"].append(warmup_record)
                    _append_jsonl(progress, warmup_record)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

            for task in ("discrete", "continuous"):
                dataset = datasets[task]
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=False,
                    persistent_workers=args.num_workers > 0,
                )
                task_records: list[dict[str, Any]] = []
                for batch_index, (batch, cpu_collate_s) in enumerate(_iter_batches(loader)):
                    # Make each batch RNG deterministic while allowing
                    # documented eager/compiled or batch-size pixel differences.
                    # The first sample ID drives the batch RNG.
                    batch_seed = _sample_seed(args.seed, str(batch["sample_id"][0]))
                    torch.manual_seed(batch_seed)
                    torch.cuda.manual_seed_all(batch_seed)
                    record = _run_one_batch(
                        batch=batch,
                        model=model,
                        vq_model=vq_model,
                        device=device,
                        mode=args.mode,
                        compiled_step=compiled_step,
                        compiled_cache=compiled_cache,
                        output_root=output_dir,
                        task=task,
                        batch_index=batch_index,
                        seed=args.seed,
                        write_images=True,
                        cpu_collate_s=cpu_collate_s,
                    )
                    record["batch_seed"] = batch_seed
                    task_records.append(record)
                    _append_jsonl(progress, record)
                if len(task_records) != (args.samples_per_task + args.batch_size - 1) // args.batch_size:
                    raise RuntimeError(f"Unexpected number of {task} batches: {len(task_records)}")
                stage_totals = {
                    stage_name: sum(record["timing_s"].get(stage_name, 0.0) for record in task_records)
                    for stage_name in (
                        "cpu_collate_s",
                        "h2d_s",
                        "pose_s",
                        "generation_s",
                        "vq_decode_s",
                        "jpeg_s",
                        "total_s",
                    )
                }
                sample_count = sum(record["sample_count"] for record in task_records)
                task_summary = {
                    "sample_count": sample_count,
                    "batch_count": len(task_records),
                    "stage_totals_s": stage_totals,
                    "samples_per_s_total": sample_count / max(stage_totals["total_s"], 1e-12),
                    "max_torch_allocated_peak_bytes": max(
                        record["torch_allocated_peak_bytes"] or 0 for record in task_records
                    ),
                    "max_torch_reserved_peak_bytes": max(
                        record["torch_reserved_peak_bytes"] or 0 for record in task_records
                    ),
                    "sample_ids": [sample_id for record in task_records for sample_id in record["sample_ids"]],
                }
                result["task_summaries"][task] = task_summary
                _write_json(final_path, result)

        if compiled_cache is not None:
            result["compiled_cache"] = {
                "reuse_count": compiled_cache.reuse_count,
                "allocation_count": compiled_cache.allocate_count,
                "policy": "reuse same-shape KV storage; zero K/V before every call",
            }
        result["status"] = "complete"
        result["completed_at_unix"] = time.time()
        _write_json(final_path, result)
        return result
    except Exception as exc:
        error_payload = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "oom": isinstance(exc, torch.cuda.OutOfMemoryError),
        }
        result["status"] = "error"
        result["error"] = error_payload
        _write_json(final_path, result)
        try:
            with progress_path.open("a", encoding="utf-8") as progress:
                _append_jsonl(progress, {"event": "error", **error_payload})
        except OSError:
            pass
        raise


def main() -> None:
    args, config, _ = _config_from_cli()
    run_benchmark(args, config)


if __name__ == "__main__":
    main()
