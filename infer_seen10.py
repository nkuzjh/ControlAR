from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from autoregressive.models.generate import generate
from csgo_seen10.data import Seen10GenerationDataset, read_benchmark_rows
from csgo_seen10.model import Seen10GenerationModel, build_gpt, load_checkpoint
from tokenizer.tokenizer_image.vq_model import VQ_models


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "csgo_seen10.json"
TASK_TO_SPLIT = {"discrete": "seen_discrete_test", "continuous": "seen_continuous"}
LEGACY_EAGER_SEED_POLICY = (
    "legacy-v1: UTF-8 bytes of str(base_seed) + NUL + sample_id; "
    "seed is the first 8 SHA256 digest bytes as big-endian, masked to 63 bits"
)
COMPILED_BATCH_SEED_POLICY = (
    "sha256-json-v1: compact UTF-8 JSON with sorted keys for integer base_seed, "
    "task string, zero-based batch_index, and real_sample_ids in manifest order; "
    "seed is the first 8 SHA256 digest bytes as big-endian, masked to 63 bits"
)


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_CONFIG))
    config_args, _ = pre.parse_known_args()
    config = json.loads(Path(config_args.config).read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(description="Generate ControlAR Seen-10 predictions.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    for name, arg_type in (
        ("data-root", str),
        ("vq-checkpoint", str),
        ("output-base", str),
        ("output-root", str),
        ("checkpoint", str),
        ("gpt-model", str),
        ("image-size", int),
        ("downsample-size", int),
        ("token-count", int),
        ("caption-dim", int),
        ("adapter-size", str),
        ("condition-type", str),
        ("precision", str),
        ("batch-size", int),
        ("seed", int),
        ("cfg-scale", float),
        ("temperature", float),
        ("top-k", int),
        ("top-p", float),
        ("max-samples", int),
        ("task", str),
    ):
        key = name.replace("-", "_")
        default = config.get(key, 1 if key == "batch_size" else None)
        if key in ("output_root", "checkpoint", "seed", "max_samples"):
            default = None
        parser.add_argument(f"--{name}", dest=key, type=arg_type, default=default)
    parser.add_argument(
        "--inference-engine",
        choices=("eager", "compiled"),
        default="eager",
        help="Inference implementation; compiled uses the configured fixed batch size",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def default_output_root(output_base: str, seed: int, smoke: bool) -> str:
    if smoke:
        return os.environ.get("CSGO_SMOKE_ROOT") or str(
            ROOT / "outputs" / "csgo_benchmark_v2_smoke" / "ControlAR" / f"seed_{seed}"
        )
    return str(ROOT / output_base / f"seed_{seed}")


def load_vq_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    vq_model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8).to(device)
    checkpoint = load_checkpoint(checkpoint_path)
    state = checkpoint.get("model", checkpoint)
    vq_model.load_state_dict(state, strict=True)
    vq_model.eval()
    vq_model.requires_grad_(False)
    return vq_model


def checkpoint_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_seed(seed: int, sample_id: str) -> int:
    payload = f"{seed}\0{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def validate_existing_prediction(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.size != (448, 448) or image.mode != "RGB":
                raise ValueError(f"Existing prediction must be 448x448 RGB, got {image.size} {image.mode}: {path}")
    except Exception as exc:
        raise RuntimeError(f"Existing prediction is not a valid benchmark JPEG; refusing to overwrite: {path}") from exc


def save_unilip_style(image_tensor: torch.Tensor, path: Path) -> None:
    # Keep UniLIP's conversion path and Pillow's default JPEG encoder settings.
    pixels = ((image_tensor.detach().float().clamp(-1, 1) + 1.0) / 2.0 * 255.0).round().clamp(0, 255).to(torch.uint8)
    array = pixels.permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(array, mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not give the temporary file a .jpg suffix: an uncatchable SIGKILL or
    # host failure may leave it behind, and benchmark evaluators discover
    # predictions by that suffix.
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        # Keep Pillow's default JPEG options while avoiding partially written
        # benchmark outputs if the process is interrupted during encoding.
        image.save(temp_path, format="JPEG")
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def ensure_output_manifest(
    output_root: Path,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    vq_checkpoint_path: Path,
    vq_checkpoint_sha256: str,
    data_root: Path,
    seed: int,
    image_size: int,
    precision: str,
    cfg_scale: float,
    temperature: float,
    top_k: int,
    top_p: float,
    max_samples: int | None,
    smoke: bool,
    task: str,
    inference: dict[str, Any],
) -> None:
    manifest_path = output_root / "inference_manifest.json"
    expected = {
        "benchmark_id": "csgo_benchmark_v2",
        "model_name": "ControlAR",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "vq_checkpoint_path": str(vq_checkpoint_path),
        "vq_checkpoint_sha256": vq_checkpoint_sha256,
        "data_root": str(data_root),
        "seed": seed,
        "image_size": image_size,
        "precision": precision,
        "sampling": {
            "cfg_scale": cfg_scale,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        },
        "smoke_only": bool(smoke),
        "max_samples": max_samples,
        "inference": inference,
        "tasks": [],
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "benchmark_id",
            "model_name",
            "checkpoint_sha256",
            "vq_checkpoint_path",
            "vq_checkpoint_sha256",
            "data_root",
            "seed",
            "image_size",
            "precision",
            "sampling",
            "smoke_only",
            "max_samples",
        ):
            if previous.get(key) != expected[key]:
                raise RuntimeError(
                    f"Existing output root was created with a different {key}; refusing to mix or overwrite predictions"
                )
        if "inference" in previous:
            if previous["inference"] != inference:
                raise RuntimeError(
                    "Existing output root was created with a different inference configuration; "
                    "refusing to mix or overwrite predictions"
                )
        elif inference != {
            "engine": "eager",
            "batch_size": 1,
            "compile_mode": None,
            "batching": "per_sample",
            "seed_policy": LEGACY_EAGER_SEED_POLICY,
        }:
            raise RuntimeError(
                "Existing output root has a legacy manifest (eager batch_size=1); "
                "only eager batch_size=1 may resume it"
            )
        tasks = set(previous.get("tasks", []))
        tasks.add(task)
        previous["tasks"] = sorted(tasks)
        atomic_json_write(manifest_path, previous)
        return

    # A previous run without a manifest cannot be safely associated with this
    # checkpoint, so do not silently combine its images with current outputs.
    if output_root.exists() and any(output_root.rglob("*.jpg")):
        raise RuntimeError(f"Predictions already exist without an inference manifest under {output_root}")
    expected["tasks"] = [task]
    atomic_json_write(manifest_path, expected)


def run_compiled_task(
    *,
    args: argparse.Namespace,
    task: str,
    dataset: Seen10GenerationDataset,
    model: Seen10GenerationModel,
    vq_model: torch.nn.Module,
    device: torch.device,
    precision_dtype: torch.dtype,
    output_root: Path,
    runtime: dict[str, Any],
) -> tuple[int, int]:
    """Generate one task in fixed manifest blocks, saving only missing files."""

    from csgo_seen10.compiled_inference import (
        generate_compiled,
        pad_batch_to_fixed_size,
        prepare_compiled_step,
        stable_batch_seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    generated = 0
    existing = 0
    started = time.time()
    for batch_index, batch in enumerate(loader):
        sample_ids = [str(value) for value in batch["sample_id"]]
        map_names = [str(value) for value in batch["map_name"]]
        file_frames = [str(value) for value in batch["file_frame"]]
        real_count = len(sample_ids)
        output_paths = [
            output_root / task / "gen_imgs" / map_name / f"{file_frame}.jpg"
            for map_name, file_frame in zip(map_names, file_frames)
        ]

        existing_indices: list[int] = []
        for index, output_path in enumerate(output_paths):
            if output_path.exists():
                validate_existing_prediction(output_path)
                existing_indices.append(index)
        existing_index_set = set(existing_indices)
        missing_indices = [
            index for index in range(real_count) if index not in existing_index_set
        ]
        existing += len(existing_indices)

        if missing_indices:
            batch_seed = stable_batch_seed(
                args.seed, task, batch_index, sample_ids
            )
            torch.manual_seed(batch_seed)
            torch.cuda.manual_seed_all(batch_seed)
            padded_batch = pad_batch_to_fixed_size(batch, args.batch_size)
            radar = padded_batch["radar"].to(
                device=device, dtype=precision_dtype, non_blocking=True
            )
            pose = padded_batch["pose"].to(
                device=device, dtype=precision_dtype, non_blocking=True
            )
            map_id = padded_batch["map_id"].to(device=device, non_blocking=True)
            with torch.inference_mode():
                caption = model.pose_map_embedder(pose, map_id)
                if runtime["compiled_step"] is None:
                    runtime["compiled_step"] = prepare_compiled_step()
                codes = generate_compiled(
                    model.gpt,
                    caption,
                    max_new_tokens=(args.image_size // args.downsample_size) ** 2,
                    condition=radar,
                    cfg_scale=args.cfg_scale,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    e2e_step=runtime["compiled_step"],
                    cache_pool=runtime["cache_pool"],
                )

                # Decode each missing image independently in FP32 to bound VQ
                # memory, with the benchmark's per-code [1, 8, H, W] shape.
                shape = [
                    1,
                    8,
                    args.image_size // 16,
                    args.image_size // 16,
                ]
                for index in missing_indices:
                    image = vq_model.decode_code(codes[index : index + 1], shape)[0]
                    save_unilip_style(image, output_paths[index])
                    generated += 1

        if batch_index == 0 or (batch_index + 1) % 100 == 0 or batch_index + 1 == len(loader):
            elapsed = time.time() - started
            print(
                f"{task}: engine=compiled batch={args.batch_size} "
                f"batch_index={batch_index + 1}/{len(loader)} real={real_count} "
                f"generated={generated} existing={existing} elapsed={elapsed:.1f}s"
            )

    return generated, existing


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    if args.seed is None:
        args.seed = 0
    if args.task is None:
        args.task = "all"
    if args.task not in ("all", "discrete", "continuous"):
        raise ValueError("task must be all, discrete, or continuous")
    if args.batch_size is None:
        args.batch_size = 1
    if args.inference_engine == "compiled":
        if args.batch_size != 16:
            raise ValueError(
                "Compiled inference is the validated fixed-shape batch=16 path; "
                "pass --batch-size 16"
            )
        if args.top_p != 1.0:
            raise ValueError("Compiled inference requires top_p=1.0")
        if args.cfg_scale <= 1.0:
            raise ValueError("Compiled inference requires cfg_scale > 1")
        inference_config = {
            "engine": "compiled",
            "batch_size": args.batch_size,
            "compile_mode": "reduce-overhead",
            "batching": "fixed_manifest_blocks",
            "seed_policy": COMPILED_BATCH_SEED_POLICY,
        }
        from csgo_seen10.compiled_inference import CompiledCachePool

        compiled_runtime: dict[str, Any] | None = {
            "compiled_step": None,
            "cache_pool": CompiledCachePool(),
        }
    else:
        # Eager keeps the original per-sample generation and RNG contract;
        # batch-size remains a compiled-engine setting.
        inference_config = {
            "engine": "eager",
            "batch_size": 1,
            "compile_mode": None,
            "batching": "per_sample",
            "seed_policy": LEGACY_EAGER_SEED_POLICY,
        }
        compiled_runtime = None
    if args.smoke:
        args.max_samples = 1 if args.max_samples is None else args.max_samples
        if args.max_samples < 1:
            raise ValueError("Smoke inference needs max-samples >= 1")
    elif args.max_samples is not None:
        raise ValueError("Partial inference is only allowed with --smoke; formal inference must cover the full split")

    args.data_root = Path(args.data_root).expanduser().resolve()
    args.vq_checkpoint = Path(args.vq_checkpoint).expanduser().resolve()
    if args.output_root is None:
        args.output_root = default_output_root(args.output_base, args.seed, args.smoke)
    output_root = Path(args.output_root).expanduser().resolve()
    if args.checkpoint is None:
        args.checkpoint = str(output_root / "checkpoints" / "best.pt")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected validation checkpoint not found: {checkpoint_path}")
    if not args.vq_checkpoint.is_file():
        raise FileNotFoundError(f"VQ-16 checkpoint not found: {args.vq_checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("ControlAR Seen-10 inference requires CUDA")

    precision_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "none": torch.float32,
    }.get(args.precision)
    if precision_dtype is None:
        raise ValueError(f"Unsupported precision {args.precision!r}")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch_checkpoint = load_checkpoint(checkpoint_path)
    model_config = torch_checkpoint.get("model_config", {})
    for key, value in {
        "gpt_model": args.gpt_model,
        "image_size": args.image_size,
        "downsample_size": args.downsample_size,
        "token_count": args.token_count,
        "caption_dim": args.caption_dim,
        "adapter_size": args.adapter_size,
        "condition_type": args.condition_type,
    }.items():
        if model_config and model_config.get(key) != value:
            raise ValueError(f"Checkpoint {key}={model_config.get(key)!r}, requested {value!r}")

    gpt = build_gpt(
        model_name=args.gpt_model,
        image_size=args.image_size,
        downsample_size=args.downsample_size,
        token_count=args.token_count,
        adapter_size=args.adapter_size,
        condition_type=args.condition_type,
        dropout=0.0,
        token_dropout=0.0,
    ).to(device=device, dtype=precision_dtype)
    model = Seen10GenerationModel(gpt, caption_dim=args.caption_dim, token_count=args.token_count).to(
        device=device, dtype=precision_dtype
    )
    model.load_state_dict(torch_checkpoint["model"], strict=True)
    model.eval()
    del torch_checkpoint

    vq_model = load_vq_model(str(args.vq_checkpoint), device)
    vq_model.eval()
    checkpoint_sha256 = checkpoint_fingerprint(checkpoint_path)
    vq_checkpoint_sha256 = checkpoint_fingerprint(args.vq_checkpoint)
    selected_tasks = ("discrete", "continuous") if args.task == "all" else (args.task,)
    output_root.mkdir(parents=True, exist_ok=True)

    for task in selected_tasks:
        ensure_output_manifest(
            output_root,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            vq_checkpoint_path=args.vq_checkpoint,
            vq_checkpoint_sha256=vq_checkpoint_sha256,
            data_root=args.data_root,
            seed=args.seed,
            image_size=args.image_size,
            precision=args.precision,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_samples=args.max_samples,
            smoke=args.smoke,
            task=task,
            inference=inference_config,
        )
        rows = read_benchmark_rows(
            args.data_root,
            TASK_TO_SPLIT[task],
            max_samples=args.max_samples,
            require_images=False,
        )
        dataset = Seen10GenerationDataset(rows, image_size=args.image_size, include_target=False)
        if args.inference_engine == "compiled":
            assert compiled_runtime is not None
            generated, skipped = run_compiled_task(
                args=args,
                task=task,
                dataset=dataset,
                model=model,
                vq_model=vq_model,
                device=device,
                precision_dtype=precision_dtype,
                output_root=output_root,
                runtime=compiled_runtime,
            )
        else:
            # Keep the established eager batch=1 path and sample-specific RNG
            # untouched for existing launch scripts and legacy output roots.
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
            generated = 0
            skipped = 0
            started = time.time()
            for batch_index, batch in enumerate(loader):
                map_name = batch["map_name"][0]
                file_frame = batch["file_frame"][0]
                sample_id = batch["sample_id"][0]
                output_path = output_root / task / "gen_imgs" / map_name / f"{file_frame}.jpg"
                if output_path.exists():
                    validate_existing_prediction(output_path)
                    skipped += 1
                    if batch_index == 0 or (batch_index + 1) % 100 == 0:
                        elapsed = time.time() - started
                        print(
                            f"{task}: engine=eager batch=1 real=1 generated={generated} "
                            f"existing={skipped} rows={batch_index + 1}/{len(dataset)} "
                            f"elapsed={elapsed:.1f}s last={sample_id}"
                        )
                    continue

                pose = batch["pose"].to(device=device, dtype=precision_dtype, non_blocking=True)
                map_id = batch["map_id"].to(device=device, non_blocking=True)
                radar = batch["radar"].to(device=device, dtype=precision_dtype, non_blocking=True)
                seed = sample_seed(args.seed, sample_id)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                with torch.inference_mode():
                    caption = model.pose_map_embedder(pose, map_id)
                    codes = generate(
                        model.gpt,
                        caption,
                        max_new_tokens=(args.image_size // args.downsample_size) ** 2,
                        condition=radar,
                        cfg_scale=args.cfg_scale,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        sample_logits=True,
                    )
                    shape = [1, 8, args.image_size // 16, args.image_size // 16]
                    image = vq_model.decode_code(codes, shape)[0]
                save_unilip_style(image, output_path)
                generated += 1
                if batch_index == 0 or (batch_index + 1) % 100 == 0:
                    elapsed = time.time() - started
                    print(
                        f"{task}: engine=eager batch=1 real=1 generated={generated} "
                        f"existing={skipped} rows={batch_index + 1}/{len(dataset)} "
                        f"elapsed={elapsed:.1f}s last={sample_id}"
                    )

        missing = [
            row["sample_id"]
            for row in rows
            if not (output_root / task / "gen_imgs" / row["map_name"] / f"{row['file_frame']}.jpg").is_file()
        ]
        if missing:
            raise RuntimeError(f"Inference left {len(missing)} samples missing; first: {missing[:5]}")
        print(
            f"Completed {task}: engine={args.inference_engine} "
            f"batch={inference_config['batch_size']} real={len(dataset)} "
            f"generated={generated} existing={skipped} output={output_root / task / 'gen_imgs'}"
        )


if __name__ == "__main__":
    main()
