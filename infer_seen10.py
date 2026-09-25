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
from csgo_seen10.artifact_contract import (
    SCHEMA_VERSION as ARTIFACT_SCHEMA_VERSION,
    ArtifactContractError,
    audit_task_outputs,
    benchmark_data_contract,
    rows_contract,
    sha256_file,
    write_completion,
)
from csgo_seen10.data import Seen10GenerationDataset, read_benchmark_rows
from csgo_seen10.model import Seen10GenerationModel, build_gpt, load_checkpoint
from csgo_seen10.paths import data_root as resolve_data_root, project_path
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
ALIGNED_EXPERIMENT = "csgo_seen10_exp32gen_aligned"
ALIGNED_CONFIG = ROOT / "configs" / f"{ALIGNED_EXPERIMENT}.json"
PEFT_EXPERIMENT = "csgo_seen10_exp32gen_aligned_peft"
PEFT_CONFIG = ROOT / "configs" / f"{PEFT_EXPERIMENT}.json"
ALIGNED_COMPILED_SEED_POLICY = (
    "stateless-sample-v1: SHA256(UTF-8 bytes of str(inference_seed) + NUL + "
    "sample_id + NUL + decimal token index), first 64 bits mapped to (0, 1)"
)


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--experiment", default=None)
    config_args, _ = pre.parse_known_args()
    if config_args.config is None:
        config_path = (
            PEFT_CONFIG if config_args.experiment == PEFT_EXPERIMENT else
            ALIGNED_CONFIG if config_args.experiment == ALIGNED_EXPERIMENT else DEFAULT_CONFIG
        )
    else:
        config_path = Path(config_args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(description="Generate ControlAR Seen-10 predictions.")
    parser.add_argument("--config", default=str(config_path))
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
        ("experiment", str),
        ("checkpoint-role", str),
        ("inference-seed", int),
    ):
        key = name.replace("-", "_")
        default = config.get(key, 1 if key == "batch_size" else None)
        if key == "batch_size" and config.get("experiment") in (ALIGNED_EXPERIMENT, PEFT_EXPERIMENT):
            default = config.get("inference_batch_size", default)
        if key in ("output_root", "checkpoint", "seed", "max_samples", "data_root"):
            default = None
        parser.add_argument(f"--{name}", dest=key, type=arg_type, default=default)
    parser.add_argument(
        "--inference-engine",
        choices=("eager", "compiled"),
        default=None,
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


def aligned_default_output_root(
    output_base: str, seed: int, inference_seed: int, role: str, smoke: bool
) -> str:
    if smoke:
        smoke_root = os.environ.get("CSGO_ALIGNED_SMOKE_ROOT")
        if smoke_root:
            return str(Path(smoke_root).expanduser().resolve())
    base = Path(output_base).expanduser()
    if not base.is_absolute():
        base = ROOT / base
    root = base / f"seed_{seed}"
    if smoke:
        root = root / "smoke"
    return str(root / "predictions" / role / f"inference_seed_{inference_seed}")


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


def validate_aligned_checkpoint(
    checkpoint_path: Path,
    payload: dict[str, Any],
    *,
    checkpoint_role: str,
    experiment: str,
    config_sha256: str,
    data_root: Path,
    data_contract: dict[str, Any],
    train_seed: int,
    official_gpt_sha256: str,
    vq_sha256: str,
    smoke: bool,
) -> str:
    """Validate the training contract before loading an aligned model."""

    if checkpoint_path.name != f"{checkpoint_role}.pt":
        raise ValueError(
            f"Aligned checkpoint filename must be {checkpoint_role}.pt, got {checkpoint_path.name}"
        )
    if payload.get("format") != "csgo_seen10_exp32gen_aligned_v1":
        raise ValueError("Checkpoint format is not csgo_seen10_exp32gen_aligned_v1")
    if checkpoint_role not in ("late", "best"):
        raise ValueError("Aligned checkpoint role must be late or best")
    args_payload = payload.get("args")
    if not isinstance(args_payload, dict):
        raise ValueError("Aligned checkpoint has no training args metadata")
    if args_payload.get("experiment") != experiment:
        raise ValueError("Aligned checkpoint experiment identity mismatch")
    if int(args_payload.get("seed", -1)) != int(train_seed):
        raise ValueError("Aligned checkpoint training seed mismatch")
    sampler_state = payload.get("sampler_state")
    if not isinstance(sampler_state, dict) or int(sampler_state.get("seed", -1)) != int(train_seed):
        raise ValueError("Aligned checkpoint sampler seed mismatch")
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Aligned checkpoint has no identity metadata")
    identity_files = identity.get("files")
    if not isinstance(identity_files, dict):
        raise ValueError("Aligned checkpoint identity has no file hashes")
    if identity_files.get("config") != config_sha256:
        raise ValueError("Aligned checkpoint config identity hash mismatch")
    if identity.get("data_root") != str(data_root):
        raise ValueError("Aligned checkpoint data root identity mismatch")
    if identity_files.get("manifest") != data_contract.get("benchmark_manifest_sha256"):
        raise ValueError("Aligned checkpoint benchmark manifest hash mismatch")
    if identity_files.get("report") != data_contract.get("minimal_dataset_report_sha256"):
        raise ValueError("Aligned checkpoint minimal report hash mismatch")
    if identity_files.get("official_gpt") != official_gpt_sha256:
        raise ValueError("Aligned checkpoint official GPT identity hash mismatch")
    if identity_files.get("vq") != vq_sha256:
        raise ValueError("Aligned checkpoint VQ identity hash mismatch")
    if identity.get("benchmark_data_contract") != data_contract:
        raise ValueError(
            "Aligned checkpoint benchmark split/calibration contract does not match inference data"
        )
    step = int(payload.get("steps", -1))
    if int(payload.get("global_optimizer_step", -1)) != step:
        raise ValueError("Aligned checkpoint step/global_optimizer_step mismatch")
    training_config = payload.get("training_config")
    if not isinstance(training_config, dict):
        raise ValueError("Aligned checkpoint has no training_config metadata")
    effective_batch = int(training_config.get("effective_batch_size", -1))
    expected_effective_batch = 1 if smoke else 128
    if effective_batch != expected_effective_batch:
        raise ValueError(
            "Aligned checkpoint effective batch mismatch: "
            f"{effective_batch} != {expected_effective_batch}"
        )
    if int(payload.get("consumed_samples", -1)) != step * effective_batch:
        raise ValueError(
            "Aligned checkpoint consumed_samples must equal "
            f"step*{effective_batch}"
        )
    if checkpoint_role == "late":
        expected_late_step = 1 if smoke else 19_500
        if step != expected_late_step:
            raise ValueError(
                f"late.pt must contain optimizer step {expected_late_step}"
            )
    else:
        best_step = payload.get("best_step")
        if best_step is None or int(best_step) != step:
            raise ValueError("best.pt must contain the checkpoint's best_step")

    index_path = checkpoint_path.parent / "checkpoint_index.json"
    if not index_path.is_file():
        raise ValueError(f"Aligned checkpoint index is missing: {index_path}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot read aligned checkpoint index: {index_path}") from exc
    if not isinstance(index, dict) or not isinstance(index.get("checkpoints"), list):
        raise ValueError("Aligned checkpoint index has no checkpoint records")
    indexed_steps = tuple(
        sorted(int(item.get("step", -1)) for item in index["checkpoints"])
    )
    expected_steps = (1,) if smoke else (3_900, 7_800, 11_700, 15_600, 19_500)
    if indexed_steps != expected_steps:
        raise ValueError(
            f"Aligned checkpoint index steps are {indexed_steps}, expected {expected_steps}"
        )
    if sum(bool(item.get("is_best")) for item in index["checkpoints"]) != 1:
        raise ValueError("Aligned checkpoint index must identify exactly one best checkpoint")
    if checkpoint_role == "late":
        expected_late_step = 1 if smoke else 19_500
        if int(index.get("late_step", -1)) != expected_late_step:
            raise ValueError(
                f"Aligned checkpoint index late_step is not {expected_late_step}"
            )
        record = next(
            (item for item in index["checkpoints"] if int(item.get("step", -1)) == expected_late_step),
            None,
        )
    else:
        if int(index.get("best_step", -1)) != step:
            raise ValueError("Aligned best.pt does not match checkpoint_index best_step")
        record = next((item for item in index["checkpoints"] if int(item.get("step", -1)) == step), None)
    if not isinstance(record, dict):
        raise ValueError(f"Aligned checkpoint index has no {checkpoint_role} record")
    indexed_path = checkpoint_path.parent / str(record.get("path", ""))
    if not indexed_path.is_file() or not os.path.samefile(checkpoint_path, indexed_path):
        raise ValueError(f"{checkpoint_role}.pt is not the indexed checkpoint alias")
    actual_sha256 = checkpoint_fingerprint(checkpoint_path)
    if record.get("sha256") != actual_sha256:
        raise ValueError(f"Checkpoint index SHA256 does not match {checkpoint_role}.pt")
    return actual_sha256


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


def ensure_aligned_output_manifest(
    output_root: Path,
    *,
    experiment: str,
    checkpoint_role: str,
    config_path: Path,
    config_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    vq_checkpoint_path: Path,
    vq_checkpoint_sha256: str,
    data_root: Path,
    data_contract: dict[str, Any],
    seed: int,
    inference_seed: int,
    image_size: int,
    precision: str,
    cfg_scale: float,
    temperature: float,
    top_k: int,
    top_p: float,
    max_samples: int | None,
    smoke: bool,
    task: str,
    rows: list[dict[str, Any]],
    inference: dict[str, Any],
) -> None:
    """Create or validate the versioned aligned inference identity record."""

    if experiment != ALIGNED_EXPERIMENT:
        raise ValueError(f"Unsupported aligned experiment: {experiment!r}")
    if checkpoint_role not in ("late", "best"):
        raise ValueError("Aligned checkpoint role must be late or best")
    if image_size != 448:
        raise ValueError("Aligned inference requires image_size=448")
    manifest_path = output_root / "inference_manifest.json"
    expected = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "manifest_version": 2,
        "benchmark_id": "csgo_benchmark_v2",
        "model_name": "ControlAR",
        "experiment": experiment,
        "checkpoint_role": checkpoint_role,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "vq_checkpoint_path": str(vq_checkpoint_path),
        "vq_checkpoint_sha256": vq_checkpoint_sha256,
        "data_root": str(data_root),
        "data_contract": data_contract,
        "data_contract_sha256": data_contract["sha256"],
        "benchmark_manifest_sha256": data_contract["benchmark_manifest_sha256"],
        "minimal_dataset_report_sha256": data_contract["minimal_dataset_report_sha256"],
        "seed": int(seed),
        "inference_seed": int(inference_seed),
        "image_size": int(image_size),
        "output_size": [448, 448],
        "precision": precision,
        "sampling": {
            "cfg_scale": cfg_scale,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        },
        "encoding": {
            "format": "JPEG",
            "mode": "RGB",
            "size": [448, 448],
            "pillow": "default JPEG encoder options",
        },
        "smoke_only": bool(smoke),
        "max_samples": max_samples,
        "inference": inference,
        "target_loading": "disabled",
        "tasks": [],
        "task_rows": {},
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(previous, dict):
            raise RuntimeError(f"Existing aligned manifest is not an object: {manifest_path}")
        # Tasks and row contracts are accumulated as discrete and continuous
        # are run separately.  Every other field is immutable for this root.
        for key, value in expected.items():
            if key in ("tasks", "task_rows"):
                continue
            if previous.get(key) != value:
                raise RuntimeError(
                    f"Existing aligned output root has a different {key}; refusing to mix artifacts"
                )
        tasks = set(previous.get("tasks", []))
        tasks.add(task)
        previous["tasks"] = sorted(tasks)
        task_rows = dict(previous.get("task_rows", {}))
        current_rows = rows_contract(rows)
        if task in task_rows and task_rows[task] != current_rows:
            raise RuntimeError(
                f"Existing aligned manifest has a different ordered sample identity for {task}"
            )
        task_rows[task] = current_rows
        previous["task_rows"] = task_rows
        atomic_json_write(manifest_path, previous)
        return

    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise RuntimeError(
            f"Aligned predictions already exist without a manifest under {output_root}"
        )
    expected["tasks"] = [task]
    expected["task_rows"] = {task: rows_contract(rows)}
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
    aligned: bool = False,
    peft: bool = False,
) -> tuple[int, int]:
    """Generate one task in fixed manifest blocks, saving only missing files."""

    from csgo_seen10.compiled_inference import (
        generate_compiled,
        pad_batch_to_fixed_size,
        prepare_compiled_step,
        prepare_compiled_stateless_step,
        stable_batch_seed,
        stateless_uniforms_for_batch,
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
            padded_batch = pad_batch_to_fixed_size(batch, args.batch_size)
            if aligned:
                padded_sample_ids = [
                    str(value) for value in padded_batch["sample_id"]
                ]
                uniforms = stateless_uniforms_for_batch(
                    args.inference_seed,
                    padded_sample_ids,
                    (args.image_size // args.downsample_size) ** 2,
                    device=device,
                )
                if peft and runtime["stateless_logits_step"] is None:
                    from csgo_seen10.peft_compiled_inference import prepare_peft_compiled_logits_step

                    runtime["stateless_logits_step"] = prepare_peft_compiled_logits_step()
                elif not peft and runtime["stateless_step"] is None:
                    runtime["stateless_step"] = prepare_compiled_stateless_step()
            else:
                # Preserve the legacy batch-level RNG contract exactly.
                batch_seed = stable_batch_seed(
                    args.seed, task, batch_index, sample_ids
                )
                torch.manual_seed(batch_seed)
                torch.cuda.manual_seed_all(batch_seed)
                uniforms = None
            radar = padded_batch["radar"].to(
                device=device, dtype=precision_dtype, non_blocking=True
            )
            pose = padded_batch["pose"].to(
                device=device, dtype=precision_dtype, non_blocking=True
            )
            map_id = padded_batch["map_id"].to(device=device, non_blocking=True)
            with torch.inference_mode():
                caption = model.pose_map_embedder(pose, map_id)
                if not aligned and runtime["compiled_step"] is None:
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
                    e2e_step=runtime["compiled_step"] if not aligned else None,
                    cache_pool=runtime["cache_pool"],
                    stateless_uniforms=uniforms,
                    stateless_step=runtime["stateless_step"] if aligned and not peft else None,
                    stateless_logits_step=runtime["stateless_logits_step"] if peft else None,
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
    config_seed = config.get("seed")
    config_inference_seed = config.get("inference_seed", config_seed)
    config_inference_engine = config.get("inference_engine")
    config_inference_batch = config.get("inference_batch_size", config.get("batch_size"))
    for key, value in config.items():
        if key == "data_root":
            continue  # Keep CLI unset so the runtime resolver can inspect environment overrides.
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    if args.inference_engine is None:
        args.inference_engine = "eager"
    peft = args.experiment == PEFT_EXPERIMENT
    aligned = args.experiment in (ALIGNED_EXPERIMENT, PEFT_EXPERIMENT)
    if args.experiment not in (None, ALIGNED_EXPERIMENT, PEFT_EXPERIMENT):
        raise ValueError(
            f"Unsupported inference experiment {args.experiment!r}; "
            f"expected {ALIGNED_EXPERIMENT!r}, {PEFT_EXPERIMENT!r}, or no experiment"
        )
    if args.seed is None:
        if aligned and config_seed is None:
            raise ValueError("Aligned inference config must declare seed")
        args.seed = config_seed if aligned else 0
    if args.inference_seed is None:
        if aligned and config_inference_seed is None:
            raise ValueError("Aligned inference config must declare inference_seed")
        args.inference_seed = config_inference_seed if aligned else args.seed
    if aligned:
        canonical_config = PEFT_CONFIG if peft else ALIGNED_CONFIG
        if config_path != canonical_config.resolve():
            raise ValueError(
                f"Aligned inference requires the canonical config: {canonical_config.resolve()}"
            )
        if config_seed is None or int(args.seed) != int(config_seed):
            raise ValueError("Formal aligned inference requires seed equal to the config seed")
        if config_inference_seed is None or int(args.inference_seed) != int(config_inference_seed):
            raise ValueError("Formal aligned inference requires inference_seed equal to the config inference_seed")
        fixed_values = {
            "gpt_model": (args.gpt_model, config.get("gpt_model")),
            "image_size": (int(args.image_size), int(config.get("image_size", -1))),
            "downsample_size": (
                int(args.downsample_size), int(config.get("downsample_size", -1))
            ),
            "token_count": (int(args.token_count), int(config.get("token_count", -1))),
            "caption_dim": (int(args.caption_dim), int(config.get("caption_dim", -1))),
            "adapter_size": (args.adapter_size, config.get("adapter_size")),
            "condition_type": (args.condition_type, config.get("condition_type")),
            "precision": (args.precision, config.get("precision")),
            "cfg_scale": (float(args.cfg_scale), float(config.get("cfg_scale", float("nan")))),
            "temperature": (
                float(args.temperature), float(config.get("temperature", float("nan")))
            ),
            "top_k": (int(args.top_k), int(config.get("top_k", -1))),
            "top_p": (float(args.top_p), float(config.get("top_p", float("nan")))),
            "inference_engine": (args.inference_engine, config_inference_engine),
            "inference_batch_size": (int(args.batch_size), int(config_inference_batch or -1)),
        }
        mismatches = [
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in fixed_values.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError(
                "Aligned inference configuration was overridden: " + "; ".join(mismatches)
            )
        if config_inference_engine != "compiled" or args.inference_engine != "compiled":
            raise ValueError("Aligned inference requires inference_engine=compiled")
        if int(config_inference_batch or -1) != 16 or args.batch_size != 16:
            raise ValueError("Aligned inference requires inference batch_size=16")
    if aligned:
        args.checkpoint_role = args.checkpoint_role or "late"
        if args.checkpoint_role not in ("late", "best"):
            raise ValueError("Aligned checkpoint role must be late or best")
    elif args.checkpoint_role is not None:
        raise ValueError("--checkpoint-role is only valid with the aligned experiment")
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
            "seed_policy": ALIGNED_COMPILED_SEED_POLICY if aligned else COMPILED_BATCH_SEED_POLICY,
        }
        from csgo_seen10.compiled_inference import CompiledCachePool

        compiled_runtime: dict[str, Any] | None = {
            "compiled_step": None,
            "stateless_step": None,
            "stateless_logits_step": None,
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
            "seed_policy": (
                "aligned-eager-v1: SHA256(UTF-8 bytes of str(inference_seed) + NUL + sample_id), "
                "first 8 bytes as big-endian, masked to 63 bits"
                if aligned
                else LEGACY_EAGER_SEED_POLICY
            ),
        }
        compiled_runtime = None
    if aligned:
        inference_config["target_loading"] = "disabled"
    if peft:
        inference_config["sampling_backend"] = "compiled_logits+cuda_aten_fp32_inverse_cdf"
    if args.smoke:
        args.max_samples = 1 if args.max_samples is None else args.max_samples
        if args.max_samples < 1:
            raise ValueError("Smoke inference needs max-samples >= 1")
    elif args.max_samples is not None:
        raise ValueError("Partial inference is only allowed with --smoke; formal inference must cover the full split")

    args.data_root = resolve_data_root(config, args.data_root, root=ROOT).resolve()
    args.vq_checkpoint = project_path(args.vq_checkpoint, ROOT).resolve()
    if aligned:
        expected_vq = Path(config["vq_checkpoint"]).expanduser()
        if not expected_vq.is_absolute():
            expected_vq = ROOT / expected_vq
        expected_vq = expected_vq.resolve()
        expected_output_base = Path(config["output_base"]).expanduser()
        if not expected_output_base.is_absolute():
            expected_output_base = ROOT / expected_output_base
        expected_output_base = expected_output_base.resolve()
        actual_output_base = Path(args.output_base).expanduser()
        if not actual_output_base.is_absolute():
            actual_output_base = ROOT / actual_output_base
        actual_output_base = actual_output_base.resolve()
        if args.vq_checkpoint != expected_vq:
            raise ValueError(
                f"Aligned inference VQ checkpoint must be {expected_vq}, got {args.vq_checkpoint}"
            )
        if actual_output_base != expected_output_base:
            raise ValueError(
                f"Aligned inference output_base must be {expected_output_base}, got {actual_output_base}"
            )
    if args.output_root is None:
        if aligned:
            args.output_root = aligned_default_output_root(
                args.output_base,
                args.seed,
                args.inference_seed,
                args.checkpoint_role,
                args.smoke,
            )
        else:
            args.output_root = default_output_root(args.output_base, args.seed, args.smoke)
    output_root = Path(args.output_root).expanduser().resolve()
    if args.checkpoint is None:
        if aligned:
            base = Path(args.output_base).expanduser()
            if not base.is_absolute():
                base = ROOT / base
            args.checkpoint = str(
                base
                / f"seed_{args.seed}"
                / "checkpoints"
                / f"{args.checkpoint_role}.pt"
            )
        else:
            args.checkpoint = str(output_root / "checkpoints" / "best.pt")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if aligned and not args.smoke:
        expected_run_root = expected_output_base / f"seed_{int(args.seed)}"
        expected_checkpoint = (
            expected_run_root / "checkpoints" / f"{args.checkpoint_role}.pt"
        ).resolve()
        expected_output_root = (
            expected_run_root
            / "predictions"
            / str(args.checkpoint_role)
            / f"inference_seed_{int(args.inference_seed)}"
        ).resolve()
        if checkpoint_path != expected_checkpoint:
            raise ValueError(
                f"Formal aligned inference checkpoint must be {expected_checkpoint}, got {checkpoint_path}"
            )
        if output_root != expected_output_root:
            raise ValueError(
                f"Formal aligned inference output_root must be {expected_output_root}, got {output_root}"
            )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected validation checkpoint not found: {checkpoint_path}")
    if not args.vq_checkpoint.is_file():
        raise FileNotFoundError(f"VQ-16 checkpoint not found: {args.vq_checkpoint}")
    data_contract = benchmark_data_contract(args.data_root) if aligned else None
    config_sha256 = sha256_file(config_path) if aligned else None
    vq_checkpoint_sha256 = checkpoint_fingerprint(args.vq_checkpoint)
    if aligned and vq_checkpoint_sha256 != str(config.get("vq_sha256", "")):
        raise ValueError("Aligned inference VQ checkpoint SHA256 does not match the config")
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
    aligned_checkpoint_sha256: str | None = None
    if peft:
        from csgo_seen10.peft_artifact_contract import validate_peft_checkpoint

        assert data_contract is not None
        aligned_checkpoint_sha256 = validate_peft_checkpoint(
            checkpoint_path,
            torch_checkpoint,
            checkpoint_role=args.checkpoint_role,
            config_path=config_path,
            config=config,
            data_root=args.data_root,
            data_contract=data_contract,
            smoke=args.smoke,
        )
    elif aligned:
        assert data_contract is not None
        assert config_sha256 is not None
        aligned_checkpoint_sha256 = validate_aligned_checkpoint(
            checkpoint_path,
            torch_checkpoint,
            checkpoint_role=args.checkpoint_role,
            experiment=args.experiment,
            config_sha256=config_sha256,
            data_root=args.data_root,
            data_contract=data_contract,
            train_seed=args.seed,
            official_gpt_sha256=str(config.get("official_gpt_sha256", "")),
            vq_sha256=str(config.get("vq_sha256", "")),
            smoke=args.smoke,
        )
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

    construction_dtype = torch.float32 if peft else precision_dtype
    gpt = build_gpt(
        model_name=args.gpt_model,
        image_size=args.image_size,
        downsample_size=args.downsample_size,
        token_count=args.token_count,
        adapter_size=args.adapter_size,
        condition_type=args.condition_type,
        dropout=0.0,
        token_dropout=0.0,
    ).to(device=device, dtype=construction_dtype)
    model = Seen10GenerationModel(gpt, caption_dim=args.caption_dim, token_count=args.token_count).to(
        device=device, dtype=construction_dtype
    )
    if peft:
        from csgo_seen10.peft import inject_lora

        inject_lora(
            model.gpt,
            rank=int(config["lora_rank"]),
            alpha=int(config["lora_alpha"]),
            dropout=float(config["lora_dropout"]),
        )
    model.load_state_dict(torch_checkpoint["model"], strict=True)
    model.eval()
    if peft:
        from csgo_seen10.peft import merge_lora_

        merge_lora_(model.gpt)
        model.to(dtype=precision_dtype)
    del torch_checkpoint

    vq_model = load_vq_model(str(args.vq_checkpoint), device)
    vq_model.eval()
    checkpoint_sha256 = (
        aligned_checkpoint_sha256
        if aligned_checkpoint_sha256 is not None
        else checkpoint_fingerprint(checkpoint_path)
    )
    selected_tasks = ("discrete", "continuous") if args.task == "all" else (args.task,)
    output_root.mkdir(parents=True, exist_ok=True)

    for task in selected_tasks:
        if not aligned:
            # Keep the legacy manifest/resume ordering unchanged.
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
        if dataset.include_target:
            raise RuntimeError("Inference dataset unexpectedly enables target FPV loading")
        if peft:
            from csgo_seen10.peft_artifact_contract import ensure_peft_output_manifest

            assert data_contract is not None
            assert config_sha256 is not None
            ensure_peft_output_manifest(
                output_root,
                task=task,
                rows=rows,
                experiment=args.experiment,
                checkpoint_role=args.checkpoint_role,
                config_path=str(config_path),
                config_sha256=config_sha256,
                checkpoint_path=str(checkpoint_path),
                checkpoint_sha256=checkpoint_sha256,
                vq_checkpoint_path=str(args.vq_checkpoint),
                vq_checkpoint_sha256=vq_checkpoint_sha256,
                data_root=str(args.data_root),
                data_contract=data_contract,
                data_contract_sha256=data_contract["sha256"],
                benchmark_manifest_sha256=data_contract["benchmark_manifest_sha256"],
                minimal_dataset_report_sha256=data_contract["minimal_dataset_report_sha256"],
                seed=int(args.seed),
                inference_seed=int(args.inference_seed),
                image_size=int(args.image_size),
                precision=args.precision,
                sampling={
                    "cfg_scale": args.cfg_scale,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                },
                smoke_only=bool(args.smoke),
                max_samples=args.max_samples,
                inference={**inference_config, "peft_merge": "temporary_inference_model"},
            )
        elif aligned:
            assert data_contract is not None
            assert config_sha256 is not None
            ensure_aligned_output_manifest(
                output_root,
                experiment=args.experiment,
                checkpoint_role=args.checkpoint_role,
                config_path=config_path,
                config_sha256=config_sha256,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                vq_checkpoint_path=args.vq_checkpoint,
                vq_checkpoint_sha256=vq_checkpoint_sha256,
                data_root=args.data_root,
                data_contract=data_contract,
                seed=args.seed,
                inference_seed=args.inference_seed,
                image_size=args.image_size,
                precision=args.precision,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_samples=args.max_samples,
                smoke=args.smoke,
                task=task,
                rows=rows,
                inference=inference_config,
            )
        if aligned:
            # Validate existing images and reject extras before any generation.
            audit_task_outputs(
                output_root,
                task,
                rows,
                image_size=(448, 448),
                require_complete=False,
            )
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
                aligned=aligned,
                peft=peft,
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
                seed = sample_seed(args.inference_seed if aligned else args.seed, sample_id)
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

        if aligned:
            try:
                task_audit = audit_task_outputs(
                    output_root,
                    task,
                    rows,
                    image_size=(448, 448),
                    require_complete=not args.smoke,
                )
            except ArtifactContractError:
                raise
            assert config_sha256 is not None
            assert data_contract is not None
            write_completion(
                output_root,
                experiment=args.experiment,
                config_sha256=config_sha256,
                checkpoint_sha256=checkpoint_sha256,
                checkpoint_role=args.checkpoint_role,
                vq_checkpoint_sha256=vq_checkpoint_sha256,
                data_contract_sha256=data_contract["sha256"],
                seed=args.seed,
                inference_seed=args.inference_seed,
                selected_tasks=selected_tasks,
                task_audit=task_audit,
                formal=not args.smoke,
            )
        else:
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
