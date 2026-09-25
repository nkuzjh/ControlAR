from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from autoregressive.train.train_c2i import creat_optimizer
from csgo_seen10.artifact_contract import benchmark_data_contract
from csgo_seen10.data import MAP_ORDER, Seen10GenerationDataset, read_benchmark_rows
from csgo_seen10.model import (
    Seen10GenerationModel,
    build_gpt,
    load_checkpoint,
    load_official_gpt_weights,
)
from tokenizer.tokenizer_image.vq_model import VQ_models
from csgo_seen10.paths import data_root as resolve_data_root, project_path
from csgo_seen10.source_compat import check_resume_identity


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "csgo_seen10.json"
ALIGNED_EXPERIMENT = "csgo_seen10_exp32gen_aligned"
ALIGNED_CONFIG = ROOT / "configs" / f"{ALIGNED_EXPERIMENT}.json"
ALIGNED_CHECKPOINT_STEPS = (3900, 7800, 11700, 15600, 19500)


class StridedDistributedSampler(Sampler[int]):
    """Exact non-padding partition for validation; ranks may have unequal lengths."""

    def __init__(self, dataset_size: int, rank: int, world_size: int):
        self.dataset_size = dataset_size
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.dataset_size, self.world_size))

    def __len__(self) -> int:
        return max(0, (self.dataset_size - self.rank + self.world_size - 1) // self.world_size)


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--experiment", default=None)
    config_args, _ = pre.parse_known_args()
    if config_args.config is None:
        config_path = ALIGNED_CONFIG if config_args.experiment == ALIGNED_EXPERIMENT else DEFAULT_CONFIG
    else:
        config_path = Path(config_args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(description="Train ControlAR on CSGO Benchmark v2 Seen-10.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--experiment", default=config.get("experiment"))
    for name, arg_type in (
        ("data-root", str),
        ("official-gpt-checkpoint", str),
        ("vq-checkpoint", str),
        ("output-base", str),
        ("run-dir", str),
        ("gpt-model", str),
        ("image-size", int),
        ("downsample-size", int),
        ("token-count", int),
        ("caption-dim", int),
        ("adapter-size", str),
        ("condition-type", str),
        ("epochs", int),
        ("batch-size", int),
        ("num-workers", int),
        ("learning-rate", float),
        ("weight-decay", float),
        ("beta1", float),
        ("beta2", float),
        ("max-grad-norm", float),
        ("precision", str),
        ("checkpoint-every", int),
        ("validate-every", int),
        ("log-every", int),
        ("dropout", float),
        ("token-dropout", float),
        ("seed", int),
        ("max-steps", int),
        ("max-train-samples", int),
        ("max-val-samples", int),
        ("resume", str),
        ("gradient-accumulation-steps", int),
        ("effective-batch-size", int),
        ("max-optimizer-steps", int),
    ):
        key = name.replace("-", "_")
        default = config.get(key)
        if name in ("seed", "max_steps", "max_train_samples", "max_val_samples", "resume", "run_dir", "data-root"):
            default = None
        parser.add_argument(f"--{name}", dest=key, type=arg_type, default=default)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-legacy-source-resume", action="store_true")
    return parser.parse_args()


def default_run_dir(output_base: str, seed: int, smoke: bool) -> str:
    if smoke:
        return os.environ.get("CSGO_SMOKE_ROOT") or str(
            ROOT / "outputs" / "csgo_benchmark_v2_smoke" / "ControlAR" / f"seed_{seed}"
        )
    return str(ROOT / output_base / f"seed_{seed}")


def initialize_distributed() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
        return rank, world_size, local_rank, device
    torch.cuda.set_device(0)
    return 0, 1, 0, torch.device("cuda", 0)


def setup_logger(run_dir: Path, rank: int) -> logging.Logger:
    logger = logging.getLogger("csgo_seen10.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        file_handler = logging.FileHandler(run_dir / "train.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def seed_process(seed: int, rank: int) -> None:
    process_seed = seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed % (2**32))
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device)


def collect_rng_states(device: torch.device, rank: int, world_size: int) -> list[dict[str, Any]] | None:
    local = capture_rng_state(device)
    if world_size == 1:
        return [local]
    gathered: list[dict[str, Any] | None] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    return gathered  # type: ignore[return-value]


def autocast_context(precision: str):
    dtype_by_name = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, "none": torch.float32}
    if precision not in dtype_by_name:
        raise ValueError(f"Unsupported precision {precision!r}; choose bf16, fp16, or fp32")
    enabled = precision in ("bf16", "fp16")
    return torch.autocast(device_type="cuda", dtype=dtype_by_name[precision], enabled=enabled)


def get_scaler(precision: str):
    try:
        return torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=precision == "fp16")


def load_vq_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    vq_model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8).to(device)
    checkpoint = load_checkpoint(checkpoint_path)
    state = checkpoint.get("model", checkpoint)
    vq_model.load_state_dict(state, strict=True)
    vq_model.eval()
    vq_model.requires_grad_(False)
    return vq_model


def latent_mask(batch_size: int, token_count: int, code_length: int, device: torch.device) -> torch.Tensor:
    sequence_length = token_count + code_length - 1
    causal = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device))
    return causal[None, None].expand(batch_size, 1, sequence_length, sequence_length)


def batch_to_device(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "radar": batch["radar"].to(device=device, dtype=dtype, non_blocking=True),
        "target": batch["target"].to(device=device, non_blocking=True),
        "pose": batch["pose"].to(device=device, non_blocking=True),
        "map_id": batch["map_id"].to(device=device, non_blocking=True),
    }


@torch.no_grad()
def evaluate(
    model: Seen10GenerationModel,
    vq_model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    token_count: int,
    rank: int,
    world_size: int,
) -> float:
    model.eval()
    code_length = (model.gpt.block_size)
    local_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    local_sample_count = torch.zeros((), dtype=torch.float64, device=device)
    condition_dtype = torch.bfloat16 if precision == "bf16" else (
        torch.float16 if precision == "fp16" else torch.float32
    )

    for batch in loader:
        values = batch_to_device(batch, device, condition_dtype)
        # Keep VQ quantization in FP32 for train/eval target identity; AMP only
        # applies to the autoregressive model forward.
        _, _, info = vq_model.encode(values["target"])
        indices = info[2]
        targets = indices.reshape(values["target"].shape[0], -1).long()
        with autocast_context(precision):
            logits_in = targets[:, :-1]
            mask = latent_mask(targets.shape[0], token_count, code_length, device)
            _, loss = model(
                pose=values["pose"],
                map_id=values["map_id"],
                idx=logits_in,
                targets=targets,
                mask=mask,
                condition=values["radar"],
            )
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite validation loss: {loss}")
        local_loss_sum += loss.detach().double() * targets.shape[0]
        local_sample_count += targets.shape[0]

    totals = torch.stack((local_loss_sum, local_sample_count))
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if totals[1].item() == 0:
        raise RuntimeError("Validation split has no samples")
    return (totals[0] / totals[1]).item()


def grad_norm(parameters: Any) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().float().pow(2).sum().item()
    return math.sqrt(squared)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_checkpoint_alias(source: Path, destination: Path) -> None:
    """Atomically point a stable checkpoint name at an immutable step file."""
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        try:
            os.link(source, temporary)
        except OSError as exc:
            raise RuntimeError(
                "Aligned checkpoint aliases require hard-link support so aliases do not "
                f"duplicate multi-GB checkpoint storage: {source.parent}"
            ) from exc
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _row_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id"),
        "map_name": row.get("map_name"),
        "file_frame": row.get("file_frame"),
        "pose": row.get("pose"),
        "clip_id": row.get("clip_id"),
        "frame_index": row.get("frame_index"),
    }


def _aligned_identity(
    *,
    data_root: Path,
    config_path: Path,
    official_gpt_checkpoint: Path,
    vq_checkpoint: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = data_root / "benchmark_manifest.json"
    report_path = data_root / "minimal_dataset_report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration_rel = manifest.get("calibration", {}).get("file", "calibration/z_calibration.json")
    calibration_path = (data_root / calibration_rel).resolve()
    data_contract = benchmark_data_contract(data_root)
    files = {
        "manifest": _sha256_file(manifest_path),
        "report": _sha256_file(report_path),
        "calibration": _sha256_file(calibration_path),
        "official_gpt": _sha256_file(official_gpt_checkpoint),
        "vq": _sha256_file(vq_checkpoint),
        "config": _sha256_file(config_path),
    }
    code_files = {
        "train_seen10.py": ROOT / "train_seen10.py",
        "csgo_seen10/data.py": ROOT / "csgo_seen10" / "data.py",
        "csgo_seen10/model.py": ROOT / "csgo_seen10" / "model.py",
        "csgo_seen10/artifact_contract.py": ROOT / "csgo_seen10" / "artifact_contract.py",
        "csgo_seen10/paths.py": ROOT / "csgo_seen10" / "paths.py",
        "csgo_seen10/source_compat.py": ROOT / "csgo_seen10" / "source_compat.py",
        "autoregressive/models/gpt_t2i.py": ROOT / "autoregressive" / "models" / "gpt_t2i.py",
        "autoregressive/train/train_c2i.py": ROOT / "autoregressive" / "train" / "train_c2i.py",
    }
    files["code"] = {name: _sha256_file(path) for name, path in code_files.items()}
    identity = {
        "benchmark_id": manifest.get("benchmark_id"),
        "data_root": str(data_root),
        "seen_maps": list(MAP_ORDER),
        "files": files,
        "train_count": len(train_rows),
        "validation_count": len(val_rows),
        "train_rows_sha256": _sha256_json([_row_identity(row) for row in train_rows]),
        "validation_rows_sha256": _sha256_json([_row_identity(row) for row in val_rows]),
        "benchmark_data_contract": data_contract,
    }
    identity["identity_sha256"] = _sha256_json(identity)
    return identity


def _aligned_parameter_audit(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    run_dir: Path,
    *,
    vq_model: torch.nn.Module,
    logger: logging.Logger,
) -> dict[str, Any]:
    group_by_parameter: dict[int, tuple[int, float, float]] = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter in group["params"]:
            group_by_parameter[id(parameter)] = (
                group_index,
                float(group.get("lr", 0.0)),
                float(group.get("weight_decay", 0.0)),
            )
    rows: list[dict[str, Any]] = []
    trainable_numel = 0
    frozen_numel = 0
    optimizer_numel = 0
    for name, parameter in model.named_parameters():
        group_info = group_by_parameter.get(id(parameter))
        entry = {
            "name": name,
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
            "optimizer_group": None if group_info is None else group_info[0],
            "learning_rate": None if group_info is None else group_info[1],
            "weight_decay": None if group_info is None else group_info[2],
        }
        rows.append(entry)
        if parameter.requires_grad:
            trainable_numel += parameter.numel()
            if group_info is None:
                raise AssertionError(f"Trainable parameter is absent from optimizer: {name}")
        else:
            frozen_numel += parameter.numel()
            if group_info is not None:
                raise AssertionError(f"Frozen parameter is present in optimizer: {name}")
        if group_info is not None:
            optimizer_numel += parameter.numel()

    if trainable_numel != optimizer_numel:
        raise AssertionError(f"Optimizer/trainable mismatch: {optimizer_numel} != {trainable_numel}")
    if frozen_numel != 20_971_520 or trainable_numel != 817_343_360:
        raise AssertionError(
            "Aligned ControlAR parameter layout changed: "
            f"trainable={trainable_numel}, frozen={frozen_numel}; "
            "expected trainable=817343360 frozen=20971520"
        )
    total_numel = trainable_numel + frozen_numel
    if total_numel != 838_314_880:
        raise AssertionError(f"Aligned ControlAR total parameter count changed: {total_numel}")
    if any(parameter.requires_grad for parameter in vq_model.parameters()):
        raise AssertionError("VQ model must be fully frozen in aligned training")

    audit = {
        "model_total_numel": total_numel,
        "trainable_numel": trainable_numel,
        "frozen_numel": frozen_numel,
        "optimizer_numel": optimizer_numel,
        "optimizer_group_count": len(optimizer.param_groups),
        "optimizer_class": type(optimizer).__name__,
        "optimizer_defaults": {
            "betas": list(optimizer.defaults.get("betas", ())),
            "eps": optimizer.defaults.get("eps"),
            "weight_decay": optimizer.defaults.get("weight_decay"),
        },
        "vq_frozen": True,
        "parameters": rows,
    }
    if tuple(optimizer.defaults.get("betas", ())) != (0.9, 0.95):
        raise AssertionError(f"Aligned AdamW betas changed: {optimizer.defaults.get('betas')}")
    if float(optimizer.defaults.get("eps", float("nan"))) != 1e-8:
        raise AssertionError(f"Aligned AdamW epsilon changed: {optimizer.defaults.get('eps')}")
    _atomic_json_save(audit, run_dir / "audits" / "trainable_parameters.json")
    logger.info(
        "Aligned parameter audit: total=%s trainable=%s frozen=%s optimizer=%s",
        f"{total_numel:,}", f"{trainable_numel:,}", f"{frozen_numel:,}", f"{optimizer_numel:,}",
    )
    return audit


def _freeze_aligned_inactive_parameters(model: torch.nn.Module) -> list[str]:
    frozen_names: list[str] = []
    for name, parameter in model.named_parameters():
        if name.startswith("gpt.condition_embeddings."):
            parameter.requires_grad_(False)
            frozen_names.append(name)
        else:
            parameter.requires_grad_(True)
    if not frozen_names:
        raise AssertionError("Expected gpt.condition_embeddings.* parameters in ControlAR GPT")
    return frozen_names


def _reconcile_aligned_checkpoint_index(
    checkpoint_dir: Path,
    resume_path: Path,
    resume_checkpoint: dict[str, Any],
    *,
    checkpoint_steps: tuple[int, ...],
    max_optimizer_steps: int,
) -> tuple[int, float]:
    """Repair the JSON/alias side of an interrupted atomic step save.

    A step checkpoint is the authoritative recovery artifact.  The process can
    be killed after that file is atomically installed but before its small JSON
    index and stable aliases are updated.  Exact resume accepts that one narrow
    crash window and deterministically reconciles the missing record.
    """

    step = int(resume_checkpoint["global_optimizer_step"])
    if step not in checkpoint_steps:
        raise ValueError(
            f"Aligned resume step {step} is not a configured checkpoint milestone"
        )
    canonical_path = checkpoint_dir / f"step_{step:06d}.pt"
    if not canonical_path.is_file() or not os.path.samefile(resume_path, canonical_path):
        raise ValueError(
            "Aligned resume must resolve to its immutable milestone checkpoint: "
            f"{canonical_path}"
        )
    index_path = checkpoint_dir / "checkpoint_index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Aligned checkpoint index is corrupt: {index_path}") from exc
        if not isinstance(index, dict) or not isinstance(index.get("checkpoints"), list):
            raise ValueError(f"Aligned checkpoint index has no checkpoints list: {index_path}")
    else:
        index = {"checkpoints": []}

    records_by_step: dict[int, dict[str, Any]] = {}
    for raw_record in index["checkpoints"]:
        if not isinstance(raw_record, dict):
            raise ValueError("Aligned checkpoint index contains a non-object record")
        record_step = int(raw_record.get("step", -1))
        if record_step in records_by_step:
            raise ValueError(f"Aligned checkpoint index repeats step {record_step}")
        if record_step not in checkpoint_steps or record_step > step:
            raise ValueError(
                f"Aligned checkpoint index contains an invalid/future step {record_step}"
            )
        expected_name = f"step_{record_step:06d}.pt"
        if raw_record.get("path") != expected_name:
            raise ValueError(
                f"Aligned checkpoint index path mismatch at step {record_step}: "
                f"{raw_record.get('path')!r}"
            )
        if not (checkpoint_dir / expected_name).is_file():
            raise FileNotFoundError(
                f"Indexed aligned checkpoint is missing: {checkpoint_dir / expected_name}"
            )
        if not raw_record.get("sha256"):
            raise ValueError(f"Indexed aligned checkpoint has no SHA256 at step {record_step}")
        records_by_step[record_step] = dict(raw_record)

    # A missing current record is the expected crash window.  Missing earlier
    # records would indicate unrelated index loss and cannot be reconstructed
    # cheaply or safely without loading every multi-GB checkpoint.
    expected_prior = tuple(value for value in checkpoint_steps if value < step)
    missing_prior = [value for value in expected_prior if value not in records_by_step]
    if missing_prior:
        raise ValueError(
            "Aligned checkpoint index is missing earlier milestones and cannot be "
            f"reconciled from this resume point: {missing_prior}"
        )

    checkpoint_sha256 = _sha256_file(canonical_path)
    existing = records_by_step.get(step)
    if existing is not None and existing.get("sha256") != checkpoint_sha256:
        raise ValueError(f"Aligned checkpoint index SHA256 mismatch at step {step}")
    val_loss = float(resume_checkpoint["val_loss"])
    if not math.isfinite(val_loss):
        raise ValueError(f"Aligned resume checkpoint has non-finite validation loss: {val_loss}")
    records_by_step[step] = {
        "step": step,
        "path": canonical_path.name,
        "val_loss": val_loss,
        "is_best": False,
        "sha256": checkpoint_sha256,
    }
    records = [records_by_step[value] for value in sorted(records_by_step)]
    best_record = min(
        records, key=lambda record: (float(record["val_loss"]), int(record["step"]))
    )
    for record in records:
        record["is_best"] = int(record["step"]) == int(best_record["step"])
    index["checkpoints"] = records
    index["best_step"] = int(best_record["step"])
    if step == max_optimizer_steps:
        index["late_step"] = int(max_optimizer_steps)
    elif index.get("late_step") is not None:
        raise ValueError("Aligned checkpoint index declares late_step before training is complete")

    atomic_checkpoint_alias(
        checkpoint_dir / str(best_record["path"]), checkpoint_dir / "best.pt"
    )
    if step == max_optimizer_steps:
        atomic_checkpoint_alias(canonical_path, checkpoint_dir / "late.pt")
    _atomic_json_save(index, index_path)
    return int(best_record["step"]), float(best_record["val_loss"])


def main_aligned(args: argparse.Namespace, config: dict[str, Any], config_path: Path) -> None:
    """Run the explicitly versioned exp32_gen-aligned training contract.

    This is deliberately a separate path from the historical trainer below.
    The latter keeps its original microstep/checkpoint semantics when invoked
    without ``--experiment csgo_seen10_exp32gen_aligned``.
    """
    if config_path.resolve() != ALIGNED_CONFIG.resolve() or config.get("experiment") != ALIGNED_EXPERIMENT:
        raise ValueError(
            "Aligned training requires the canonical aligned config: "
            f"{ALIGNED_CONFIG.resolve()}"
        )
    if args.seed is None:
        args.seed = int(config.get("seed", 42))
    smoke = bool(args.smoke)
    formal_world_size = int(config.get("world_size", 1))
    formal_micro_batch = int(config.get("batch_size", 1))
    formal_accumulation = int(config.get("gradient_accumulation_steps", 128))
    formal_effective_batch = int(config.get("effective_batch_size", 128))
    formal_max_steps = int(config.get("max_optimizer_steps", config.get("max_steps", 19500)))
    formal_epochs = int(config.get("epochs", 50))
    formal_checkpoint_steps = tuple(
        int(step) for step in config.get("checkpoint_steps", ALIGNED_CHECKPOINT_STEPS)
    )
    if not smoke:
        if formal_world_size != 1 or formal_micro_batch != 1 or formal_accumulation != 128:
            raise ValueError("Aligned formal contract requires world_size=1, batch_size=1, accumulation=128")
        if formal_effective_batch != formal_world_size * formal_micro_batch * formal_accumulation:
            raise ValueError("Aligned effective batch does not equal world_size*micro_batch*accumulation")
        if formal_effective_batch != 128 or formal_max_steps != 19500 or formal_epochs != 50:
            raise ValueError("Aligned formal contract requires effective batch 128, 19500 steps and 50 epochs")
        if formal_checkpoint_steps != ALIGNED_CHECKPOINT_STEPS:
            raise ValueError(f"Aligned checkpoint schedule must be {ALIGNED_CHECKPOINT_STEPS}")

    args.data_root = resolve_data_root(config, args.data_root, root=ROOT).resolve()
    args.official_gpt_checkpoint = project_path(args.official_gpt_checkpoint, ROOT).resolve()
    args.vq_checkpoint = project_path(args.vq_checkpoint, ROOT).resolve()
    if args.run_dir is None:
        args.run_dir = default_run_dir(args.output_base, args.seed, smoke)
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"
    if not smoke:
        expected_output_base = Path(config["output_base"]).expanduser()
        if not expected_output_base.is_absolute():
            expected_output_base = ROOT / expected_output_base
        expected_run_dir = (expected_output_base / f"seed_{int(config['seed'])}").resolve()
        expected_gpt = (ROOT / config["official_gpt_checkpoint"]).resolve()
        expected_vq = (ROOT / config["vq_checkpoint"]).resolve()
        strict_values = {
            "seed": (int(args.seed), 42),
            "gpt_model": (args.gpt_model, "GPT-XL"),
            "image_size": (int(args.image_size), 448),
            "downsample_size": (int(args.downsample_size), 16),
            "token_count": (int(args.token_count), 120),
            "caption_dim": (int(args.caption_dim), 2048),
            "adapter_size": (args.adapter_size, "small"),
            "condition_type": (args.condition_type, "radar"),
            "epochs": (int(args.epochs), formal_epochs),
            "batch_size": (int(args.batch_size), formal_micro_batch),
            "learning_rate": (float(args.learning_rate), 5e-5),
            "weight_decay": (float(args.weight_decay), 0.05),
            "beta1": (float(args.beta1), 0.9),
            "beta2": (float(args.beta2), 0.95),
            "max_grad_norm": (float(args.max_grad_norm), 1.0),
            "precision": (args.precision, "bf16"),
            "dropout": (float(args.dropout), 0.1),
            "token_dropout": (float(args.token_dropout), 0.1),
            "scheduler_type": (config.get("scheduler_type"), "constant"),
            "adam_epsilon": (float(config.get("adam_epsilon", -1.0)), 1e-8),
            "random_image_augmentation": (config.get("random_image_augmentation"), False),
            "checkpoint_every": (int(args.checkpoint_every), 0),
            "validate_every": (int(args.validate_every), 0),
            "gradient_accumulation_steps": (
                int(args.gradient_accumulation_steps), formal_accumulation
            ),
            "effective_batch_size": (
                int(args.effective_batch_size), formal_effective_batch
            ),
            "max_optimizer_steps": (
                int(args.max_optimizer_steps), formal_max_steps
            ),
        }
        mismatches = [
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in strict_values.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError("Aligned formal configuration was overridden: " + "; ".join(mismatches))
        if run_dir != expected_run_dir:
            raise ValueError(f"Aligned run directory must be {expected_run_dir}, got {run_dir}")
        if args.official_gpt_checkpoint != expected_gpt or args.vq_checkpoint != expected_vq:
            raise ValueError("Aligned formal run must use the canonical official GPT and VQ checkpoints")
        if args.max_train_samples is not None or args.max_val_samples is not None:
            raise ValueError("Aligned formal training cannot use max-train-samples or max-val-samples")
        if args.max_steps is not None:
            raise ValueError(
                "Aligned formal training uses max_optimizer_steps; --max-steps is a legacy-only option"
            )
    if not torch.cuda.is_available():
        raise RuntimeError("ControlAR Seen-10 training requires CUDA")
    rank, world_size, local_rank, device = initialize_distributed()
    if world_size != formal_world_size:
        raise ValueError(f"Aligned checkpoint contract requires world_size={formal_world_size}, got {world_size}")
    if smoke:
        micro_batch = 1
        accumulation_steps = 1
        max_optimizer_steps = 1
        epochs = 1
        max_train_samples = 1
        max_val_samples = 1
        num_workers = 0
        checkpoint_steps = (1,)
    else:
        micro_batch = formal_micro_batch
        accumulation_steps = formal_accumulation
        max_optimizer_steps = formal_max_steps
        epochs = formal_epochs
        max_train_samples = args.max_train_samples
        max_val_samples = args.max_val_samples
        num_workers = int(args.num_workers)
        checkpoint_steps = formal_checkpoint_steps
    if micro_batch < 1 or accumulation_steps < 1 or epochs < 1:
        raise ValueError("Aligned batch, accumulation and epochs must be positive")
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Output directory already has files: {run_dir}; choose a new run-dir or resume")
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        if resume_path.parent.resolve() != checkpoint_dir.resolve():
            raise ValueError(
                "Aligned exact resume requires the checkpoint to be inside the current run checkpoint directory: "
                f"checkpoint={resume_path.parent.resolve()} current={checkpoint_dir.resolve()}"
            )
        args.resume = str(resume_path)
    if not args.official_gpt_checkpoint.is_file() and not args.resume:
        raise FileNotFoundError(f"Official ControlAR checkpoint not found: {args.official_gpt_checkpoint}")
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    logger = setup_logger(run_dir, rank)
    seed_process(args.seed, rank)
    logger.info("Starting aligned ControlAR Seen-10 training: %s", vars(args))
    logger.info("rank=%s world_size=%s local_rank=%s device=%s", rank, world_size, local_rank, device)

    resume_checkpoint: dict[str, Any] | None = None
    if args.resume:
        resume_checkpoint = load_checkpoint(args.resume)
        required = {
            "model", "optimizer", "scheduler", "scaler", "steps", "global_optimizer_step",
            "consumed_samples", "epoch", "batch_in_epoch", "micro_step_in_accum", "rng_states",
            "dataloader_generator_state", "sampler_state", "identity", "training_config",
        }
        missing = required - resume_checkpoint.keys()
        if missing:
            raise ValueError(f"Aligned exact resume checkpoint is missing fields: {sorted(missing)}")
        if int(resume_checkpoint["world_size"]) != world_size:
            raise ValueError("Cannot exactly resume aligned run with a different world size")
        if resume_checkpoint.get("format") != "csgo_seen10_exp32gen_aligned_v1":
            raise ValueError("Aligned resume checkpoint has the wrong format")
        resume_step_preview = int(resume_checkpoint.get("global_optimizer_step", -1))
        future_steps = sorted(
            int(path.stem.rsplit("_", 1)[-1])
            for path in checkpoint_dir.glob("step_*.pt")
            if int(path.stem.rsplit("_", 1)[-1]) > resume_step_preview
        )
        if future_steps:
            raise ValueError(
                "Refusing to rewind an aligned run over newer checkpoints: "
                f"resume_step={resume_step_preview} newer={future_steps}"
            )

    gpt = build_gpt(
        model_name=args.gpt_model,
        image_size=args.image_size,
        downsample_size=args.downsample_size,
        token_count=args.token_count,
        adapter_size=args.adapter_size,
        condition_type=args.condition_type,
        dropout=args.dropout,
        token_dropout=args.token_dropout,
    ).to(device)
    if resume_checkpoint is None:
        load_official_gpt_weights(gpt, args.official_gpt_checkpoint)
    model = Seen10GenerationModel(gpt, caption_dim=args.caption_dim, token_count=args.token_count).to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"], strict=True)
    _freeze_aligned_inactive_parameters(model)
    vq_model = load_vq_model(str(args.vq_checkpoint), device)

    expected_train_count = int(config.get("train_samples", 50_000))
    expected_val_count = int(config.get("validation_samples", 5_000))
    train_rows = read_benchmark_rows(
        args.data_root, "seen_train", max_samples=max_train_samples, require_images=True
    )
    val_rows = read_benchmark_rows(
        args.data_root, "seen_validation", max_samples=max_val_samples, require_images=True
    )
    if not smoke and (len(train_rows) != expected_train_count or len(val_rows) != expected_val_count):
        raise ValueError(
            f"Aligned split counts differ: train={len(train_rows)} validation={len(val_rows)}; "
            f"expected {expected_train_count}/{expected_val_count}"
        )
    identity = _aligned_identity(
        data_root=args.data_root,
        config_path=config_path,
        official_gpt_checkpoint=args.official_gpt_checkpoint,
        vq_checkpoint=args.vq_checkpoint,
        train_rows=train_rows,
        val_rows=val_rows,
    )
    expected_gpt_sha256 = str(config.get("official_gpt_sha256", ""))
    expected_vq_sha256 = str(config.get("vq_sha256", ""))
    if identity["files"]["official_gpt"] != expected_gpt_sha256:
        raise ValueError("Official ControlAR checkpoint SHA256 does not match the aligned config")
    if identity["files"]["vq"] != expected_vq_sha256:
        raise ValueError("VQ checkpoint SHA256 does not match the aligned config")
    source_transition = None
    if resume_checkpoint is not None:
        source_transition = check_resume_identity(
            resume_checkpoint["identity"], identity, profile="aligned",
            allow_legacy=args.allow_legacy_source_resume,
        )
    if rank == 0:
        _atomic_json_save(identity, run_dir / "audits" / "identity.json")
        if source_transition is not None:
            _atomic_json_save(source_transition, run_dir / "audits" / "legacy_source_resume.json")

    train_data = Seen10GenerationDataset(train_rows, image_size=args.image_size, include_target=True)
    val_data = Seen10GenerationDataset(val_rows, image_size=args.image_size, include_target=True)
    train_sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=False
    )
    val_sampler = StridedDistributedSampler(len(val_data), rank, world_size)
    generator = torch.Generator()
    generator.manual_seed(args.seed + 1729)
    train_loader = DataLoader(
        train_data,
        batch_size=micro_batch,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=micro_batch,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
    if len(train_loader) < accumulation_steps:
        raise RuntimeError("Aligned training loader is shorter than one accumulation window")
    updates_per_epoch = (len(train_data) // (world_size * micro_batch)) // accumulation_steps
    epoch_micro_batches = updates_per_epoch * accumulation_steps
    if smoke:
        updates_per_epoch = 1
        epoch_micro_batches = 1
    elif updates_per_epoch != 390 or epoch_micro_batches != 49_920:
        raise AssertionError(
            f"Aligned epoch contract changed: updates={updates_per_epoch}, micro_batches={epoch_micro_batches}"
        )
    effective_batch = world_size * micro_batch * accumulation_steps
    if not smoke and effective_batch != 128:
        raise AssertionError(f"Aligned effective batch is {effective_batch}, expected 128")
    logger.info(
        "Loaded manifest splits: train=%s validation=%s loader_batches=%s epoch_micro_batches=%s updates_per_epoch=%s",
        len(train_data), len(val_data), len(train_loader), epoch_micro_batches, updates_per_epoch,
    )

    optimizer = creat_optimizer(model, args.weight_decay, args.learning_rate, (args.beta1, args.beta2), logger)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = get_scaler(args.precision)
    _aligned_parameter_audit(model, optimizer, run_dir, vq_model=vq_model, logger=logger)
    if world_size > 1:
        train_model: torch.nn.Module = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
        )
    else:
        train_model = model

    model_config = {
        "gpt_model": args.gpt_model,
        "image_size": args.image_size,
        "downsample_size": args.downsample_size,
        "token_count": args.token_count,
        "caption_dim": args.caption_dim,
        "adapter_size": args.adapter_size,
        "condition_type": args.condition_type,
        "dropout": float(args.dropout),
        "token_dropout": float(args.token_dropout),
    }
    scheduler_name = "constant_lambda_lr"
    aligned_training_config = {
        "seed": int(args.seed),
        "batch_size": int(micro_batch),
        "gradient_accumulation_steps": int(accumulation_steps),
        "effective_batch_size": int(effective_batch),
        "epochs": int(epochs),
        "learning_rate": float(args.learning_rate),
        "betas": [float(args.beta1), float(args.beta2)],
        "weight_decay": float(args.weight_decay),
        "adam_epsilon": 1e-8,
        "scheduler": scheduler_name,
        "precision": str(args.precision),
        "max_grad_norm": None if args.max_grad_norm is None else float(args.max_grad_norm),
        "dropout": float(args.dropout),
        "token_dropout": float(args.token_dropout),
        "max_optimizer_steps": int(max_optimizer_steps),
        "checkpoint_steps": [int(step) for step in checkpoint_steps],
        "updates_per_epoch": int(updates_per_epoch),
        "epoch_micro_batches": int(epoch_micro_batches),
    }
    code_length = (args.image_size // args.downsample_size) ** 2
    train_steps = 0
    consumed_samples = 0
    start_epoch = 0
    resume_batch_offset = 0
    best_val_loss = float("inf")
    best_step: int | None = None
    last_validated_step = -1
    if resume_checkpoint is not None:
        saved_training = resume_checkpoint["training_config"]
        for key, expected in aligned_training_config.items():
            if saved_training.get(key) != expected:
                raise ValueError(f"Aligned resume {key} mismatch: {saved_training.get(key)!r} != {expected!r}")
        if resume_checkpoint.get("model_config") != model_config:
            raise ValueError("Aligned resume model configuration mismatch")
        if int(resume_checkpoint.get("batches_per_epoch")) != epoch_micro_batches:
            raise ValueError("Aligned resume epoch micro batch count differs")
        if int(resume_checkpoint.get("micro_step_in_accum", -1)) != 0:
            raise ValueError("Aligned checkpoints must be saved at accumulation boundaries")
        train_steps = int(resume_checkpoint["global_optimizer_step"])
        if int(resume_checkpoint["steps"]) != train_steps:
            raise ValueError("Aligned steps/global_optimizer_step disagree")
        consumed_samples = int(resume_checkpoint["consumed_samples"])
        start_epoch = int(resume_checkpoint["epoch"])
        if consumed_samples != train_steps * effective_batch:
            raise ValueError(
                "Aligned consumed sample count is inconsistent: "
                f"{consumed_samples} != {train_steps}*{effective_batch}"
            )
        if train_steps != start_epoch * updates_per_epoch:
            raise ValueError(
                "Aligned optimizer step/epoch state is inconsistent: "
                f"{train_steps} != {start_epoch}*{updates_per_epoch}"
            )
        resume_batch_offset = int(resume_checkpoint["batch_in_epoch"])
        if resume_batch_offset != 0:
            raise ValueError("Aligned checkpoints must resume at an epoch boundary")
        best_val_loss = float(resume_checkpoint.get("best_val_loss", float("inf")))
        best_step = resume_checkpoint.get("best_step")
        last_validated_step = int(resume_checkpoint.get("last_validated_step", -1))
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler_state = resume_checkpoint["scheduler"]
        if int(scheduler_state.get("last_epoch", -1)) != train_steps:
            raise ValueError(
                "Aligned scheduler last_epoch is inconsistent: "
                f"{scheduler_state.get('last_epoch')} != {train_steps}"
            )
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        scaler.load_state_dict(resume_checkpoint["scaler"])
        generator.set_state(resume_checkpoint["dataloader_generator_state"])
        rng_states = resume_checkpoint["rng_states"]
        if not isinstance(rng_states, list) or rank >= len(rng_states):
            raise ValueError("Aligned resume checkpoint has no per-rank RNG state")
        restore_rng_state(rng_states[rank], device)
        if resume_checkpoint.get("sampler_state", {}).get("epoch") != start_epoch:
            raise ValueError("Aligned sampler epoch state is inconsistent with checkpoint epoch")
        if rank == 0:
            best_step, best_val_loss = _reconcile_aligned_checkpoint_index(
                checkpoint_dir,
                Path(args.resume).resolve(),
                resume_checkpoint,
                checkpoint_steps=checkpoint_steps,
                max_optimizer_steps=max_optimizer_steps,
            )
        if world_size > 1:
            state = [best_step, best_val_loss] if rank == 0 else [None, None]
            dist.broadcast_object_list(state, src=0)
            best_step = int(state[0])
            best_val_loss = float(state[1])
        logger.info("Restored aligned checkpoint at epoch=%s step=%s", start_epoch, train_steps)
        # The model/optimizer tensors have been copied into their live objects;
        # release the multi-GB CPU checkpoint payload before training resumes.
        del resume_checkpoint

    if rank == 0:
        _atomic_json_save(
            {
                "args": vars(args),
                "config_path": str(config_path),
                "identity": identity,
                "model_config": model_config,
                "world_size": world_size,
                "micro_batch_per_device": micro_batch,
                "gradient_accumulation_steps": accumulation_steps,
                "effective_generation_batch": effective_batch,
                "updates_per_epoch": updates_per_epoch,
                "epoch_micro_batches": epoch_micro_batches,
                "max_optimizer_steps": max_optimizer_steps,
                "checkpoint_steps": list(checkpoint_steps),
                "scheduler": scheduler_name,
                "training_config": aligned_training_config,
            },
            run_dir / "run_config.json",
        )

    def save_aligned_checkpoint(*, step: int, epoch: int, val_loss: float) -> None:
        nonlocal best_val_loss, best_step
        if consumed_samples != step * effective_batch:
            raise AssertionError(
                f"Aligned checkpoint exposure mismatch: {consumed_samples} != {step}*{effective_batch}"
            )
        if step != epoch * updates_per_epoch:
            raise AssertionError(
                f"Aligned checkpoint epoch mismatch: step={step} epoch={epoch} updates_per_epoch={updates_per_epoch}"
            )
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_step = step
        rng_states = collect_rng_states(device, rank, world_size)
        if rank == 0:
            payload = {
                "format": "csgo_seen10_exp32gen_aligned_v1",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "steps": step,
                "global_optimizer_step": step,
                "consumed_samples": consumed_samples,
                "args": vars(args),
                "epoch": epoch,
                "batch_in_epoch": 0,
                "micro_step_in_accum": 0,
                "best_val_loss": best_val_loss,
                "best_step": best_step,
                "val_loss": val_loss,
                "last_validated_step": step,
                "world_size": world_size,
                "train_dataset_size": len(train_data),
                "validation_dataset_size": len(val_data),
                "batches_per_epoch": epoch_micro_batches,
                "updates_per_epoch": updates_per_epoch,
                "rng_states": rng_states,
                "dataloader_generator_state": generator.get_state(),
                "sampler_state": {
                    "epoch": epoch,
                    "seed": args.seed,
                    "num_replicas": world_size,
                    "rank": rank,
                    "micro_batch_per_device": micro_batch,
                    "gradient_accumulation_steps": accumulation_steps,
                },
                "identity": identity,
                "model_config": model_config,
                "training_config": aligned_training_config,
            }
            path = checkpoint_dir / f"step_{step:06d}.pt"
            atomic_torch_save(payload, path)
            checkpoint_sha256 = _sha256_file(path)
            logger.info(
                "Saved aligned checkpoint: %s step=%s epoch=%s val_loss=%.6f best_step=%s",
                path, step, epoch, val_loss, best_step,
            )
            index_path = checkpoint_dir / "checkpoint_index.json"
            try:
                index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {"checkpoints": []}
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Aligned checkpoint index is corrupt; refusing to discard its history: {index_path}"
                ) from exc
            index["checkpoints"] = [record for record in index.get("checkpoints", []) if record.get("step") != step]
            index["checkpoints"].append({
                "step": step,
                "path": path.name,
                "val_loss": val_loss,
                "is_best": False,
                "sha256": checkpoint_sha256,
            })
            index["checkpoints"].sort(key=lambda record: record["step"])
            if not index["checkpoints"]:
                raise AssertionError("Aligned checkpoint index unexpectedly has no checkpoints")
            best_record = min(
                index["checkpoints"],
                key=lambda record: (float(record["val_loss"]), int(record["step"])),
            )
            for record in index["checkpoints"]:
                record["is_best"] = int(record["step"]) == int(best_record["step"])
            best_step = int(best_record["step"])
            best_val_loss = float(best_record["val_loss"])
            atomic_checkpoint_alias(checkpoint_dir / best_record["path"], checkpoint_dir / "best.pt")
            index["best_step"] = best_step
            index["late_step"] = max_optimizer_steps if step == max_optimizer_steps else index.get("late_step")
            if step == max_optimizer_steps:
                atomic_checkpoint_alias(path, checkpoint_dir / "late.pt")
            _atomic_json_save(index, index_path)
        if world_size > 1:
            dist.barrier()

    run_start = time.time()
    stopped = train_steps >= max_optimizer_steps
    for epoch in range(start_epoch, epochs):
        if stopped:
            break
        train_sampler.set_epoch(epoch)
        iterator = iter(train_loader)
        offset = resume_batch_offset if epoch == start_epoch else 0
        if offset:
            for _ in range(offset):
                next(iterator)
        optimizer.zero_grad(set_to_none=True)
        micro_loss_sum = 0.0
        micro_loss_count = 0
        model.train()
        for batch_index in range(offset, epoch_micro_batches):
            batch = next(iterator)
            values = batch_to_device(batch, device, {
                "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, "none": torch.float32,
            }[args.precision])
            with torch.no_grad():
                _, _, info = vq_model.encode(values["target"])
                indices = info[2]
                targets = indices.reshape(values["target"].shape[0], -1).long()
            input_tokens = targets[:, :-1]
            mask = latent_mask(targets.shape[0], args.token_count, code_length, device)
            boundary = ((batch_index - offset + 1) % accumulation_steps) == 0
            sync_context = (
                train_model.no_sync() if world_size > 1 and not boundary else contextlib.nullcontext()
            )
            with sync_context:
                with autocast_context(args.precision):
                    _, loss = train_model(
                        pose=values["pose"], map_id=values["map_id"], idx=input_tokens,
                        targets=targets, mask=mask, condition=values["radar"],
                    )
                if loss is None or not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite aligned training loss at microstep {batch_index}: {loss}")
                micro_loss_sum += float(loss.detach().float().item())
                micro_loss_count += 1
                scaler.scale(loss / accumulation_steps).backward()
            if not boundary:
                continue
            if args.max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_window = torch.tensor([micro_loss_sum, float(micro_loss_count)], dtype=torch.float64, device=device)
            if world_size > 1:
                dist.all_reduce(loss_window, op=dist.ReduceOp.SUM)
            micro_loss_average = (loss_window[0] / loss_window[1]).item()
            optimizer.zero_grad(set_to_none=True)
            train_steps += 1
            consumed_samples += effective_batch
            micro_loss_sum = 0.0
            micro_loss_count = 0
            if train_steps % int(args.log_every) == 0 or smoke:
                elapsed = max(time.time() - run_start, 1e-6)
                logger.info(
                    "aligned step=%s epoch=%s micro=%s/%s micro_loss_avg_unscaled=%.6f samples=%s elapsed=%.1fs",
                    train_steps, epoch + 1, batch_index + 1, epoch_micro_batches,
                    micro_loss_average, consumed_samples, elapsed,
                )
            if train_steps in checkpoint_steps:
                val_loss = evaluate(
                    model, vq_model, val_loader, device=device, precision=args.precision,
                    token_count=args.token_count, rank=rank, world_size=world_size,
                )
                last_validated_step = train_steps
                if rank == 0:
                    logger.info("Aligned validation: step=%s epoch=%s loss=%.6f", train_steps, epoch + 1, val_loss)
                save_aligned_checkpoint(step=train_steps, epoch=epoch + 1, val_loss=val_loss)
            if train_steps >= max_optimizer_steps:
                stopped = True
                break
        resume_batch_offset = 0
        if stopped:
            break

    if not smoke and train_steps != max_optimizer_steps:
        raise RuntimeError(f"Aligned training ended at step={train_steps}, expected {max_optimizer_steps}")
    if not smoke:
        if tuple(
            int(path.stem.split("_")[-1])
            for path in sorted(checkpoint_dir.glob("step_*.pt"))
        ) != ALIGNED_CHECKPOINT_STEPS:
            raise AssertionError("Aligned run did not produce exactly the five required step checkpoints")
        expected_samples = max_optimizer_steps * effective_batch
        if consumed_samples != expected_samples or expected_samples != 2_496_000:
            raise AssertionError(
                f"Aligned exposure total mismatch: {consumed_samples} != {expected_samples} (expected 2496000)"
            )
        index_path = checkpoint_dir / "checkpoint_index.json"
        if not index_path.is_file():
            raise AssertionError("Aligned checkpoint index is missing")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        records = index.get("checkpoints", [])
        if len(records) != len(ALIGNED_CHECKPOINT_STEPS):
            raise AssertionError("Aligned checkpoint index does not contain exactly five records")
        if sum(bool(record.get("is_best")) for record in records) != 1:
            raise AssertionError("Aligned checkpoint index must contain exactly one is_best record")
        for record in records:
            if not record.get("sha256"):
                raise AssertionError(f"Checkpoint SHA256 missing from index for step={record.get('step')}")
        best_step_in_index = int(index["best_step"])
        best_record = next(record for record in records if int(record["step"]) == best_step_in_index)
        best_path = checkpoint_dir / "best.pt"
        late_path = checkpoint_dir / "late.pt"
        late_step_path = checkpoint_dir / f"step_{max_optimizer_steps:06d}.pt"
        if not best_path.is_file() or not late_path.is_file():
            raise AssertionError("Aligned best.pt/late.pt aliases are missing")
        if not os.path.samefile(best_path, checkpoint_dir / best_record["path"]):
            raise AssertionError("Aligned best.pt does not point to the indexed best checkpoint")
        if not os.path.samefile(late_path, late_step_path):
            raise AssertionError("Aligned late.pt does not point to step_019500.pt")
    if rank == 0:
        logger.info("Aligned training finished: steps=%s samples=%s best_val_loss=%.6f", train_steps, consumed_samples, best_val_loss)
    if smoke:
        checkpoint_path = checkpoint_dir / "step_000001.pt"
        if checkpoint_path.is_file():
            roundtrip = load_checkpoint(checkpoint_path)
            model.load_state_dict(roundtrip["model"], strict=True)
            optimizer.load_state_dict(roundtrip["optimizer"])
            scheduler.load_state_dict(roundtrip["scheduler"])
            logger.info("Aligned smoke checkpoint reload passed: %s", checkpoint_path)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, value in config.items():
        if key == "data_root":
            continue  # Keep CLI unset so the runtime resolver can inspect environment overrides.
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    if getattr(args, "experiment", None) not in (None, ALIGNED_EXPERIMENT):
        raise ValueError(f"Unknown experiment: {args.experiment!r}")
    if getattr(args, "experiment", None) == ALIGNED_EXPERIMENT or config_path.name == ALIGNED_CONFIG.name:
        main_aligned(args, config, config_path)
        return
    if args.seed is None:
        args.seed = 0
    if args.smoke:
        args.max_steps = 1
        args.max_train_samples = 1
        args.max_val_samples = 1
        args.batch_size = 1
        args.num_workers = 0
        args.checkpoint_every = 1
        args.validate_every = 1

    args.data_root = resolve_data_root(config, args.data_root, root=ROOT).resolve()
    args.official_gpt_checkpoint = project_path(args.official_gpt_checkpoint, ROOT).resolve()
    args.vq_checkpoint = project_path(args.vq_checkpoint, ROOT).resolve()
    if args.run_dir is None:
        args.run_dir = default_run_dir(args.output_base, args.seed, args.smoke)
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"

    if not torch.cuda.is_available():
        raise RuntimeError("ControlAR Seen-10 training requires CUDA")
    rank, world_size, local_rank, device = initialize_distributed()
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("batch_size and epochs must be positive")
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already has files: {run_dir}; choose a new run-dir or resume a checkpoint"
        )
    if args.resume and not Path(args.resume).is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    logger = setup_logger(run_dir, rank)
    seed_process(args.seed, rank)
    logger.info("Starting ControlAR Seen-10 training: %s", vars(args))
    logger.info("rank=%s world_size=%s local_rank=%s device=%s", rank, world_size, local_rank, device)

    if args.resume:
        resume_checkpoint = load_checkpoint(args.resume)
        required_resume_fields = {"model", "optimizer", "scaler", "steps", "epoch", "rng_states"}
        missing_resume_fields = required_resume_fields - resume_checkpoint.keys()
        if missing_resume_fields:
            raise ValueError(
                f"Exact resume needs a last.pt checkpoint with optimizer/RNG/epoch/step state; "
                f"missing {sorted(missing_resume_fields)}"
            )
        saved_config = resume_checkpoint.get("model_config", {})
        for key, value in {
            "gpt_model": args.gpt_model,
            "image_size": args.image_size,
            "downsample_size": args.downsample_size,
            "token_count": args.token_count,
            "adapter_size": args.adapter_size,
            "condition_type": args.condition_type,
        }.items():
            if saved_config and saved_config.get(key) != value:
                raise ValueError(f"Resume {key} mismatch: checkpoint={saved_config.get(key)!r}, requested={value!r}")
        logger.info("Resuming from %s", args.resume)
        gpt = build_gpt(
            model_name=args.gpt_model,
            image_size=args.image_size,
            downsample_size=args.downsample_size,
            token_count=args.token_count,
            adapter_size=args.adapter_size,
            condition_type=args.condition_type,
            dropout=args.dropout,
            token_dropout=args.token_dropout,
        ).to(device)
        model = Seen10GenerationModel(gpt, caption_dim=args.caption_dim, token_count=args.token_count).to(device)
        model.load_state_dict(resume_checkpoint["model"], strict=True)
    else:
        if not args.official_gpt_checkpoint.is_file():
            raise FileNotFoundError(f"Official ControlAR checkpoint not found: {args.official_gpt_checkpoint}")
        gpt = build_gpt(
            model_name=args.gpt_model,
            image_size=args.image_size,
            downsample_size=args.downsample_size,
            token_count=args.token_count,
            adapter_size=args.adapter_size,
            condition_type=args.condition_type,
            dropout=args.dropout,
            token_dropout=args.token_dropout,
        ).to(device)
        load_official_gpt_weights(gpt, args.official_gpt_checkpoint)
        model = Seen10GenerationModel(gpt, caption_dim=args.caption_dim, token_count=args.token_count).to(device)

    vq_model = load_vq_model(str(args.vq_checkpoint), device)

    train_rows = read_benchmark_rows(
        args.data_root, "seen_train", max_samples=args.max_train_samples, require_images=True
    )
    val_rows = read_benchmark_rows(
        args.data_root, "seen_validation", max_samples=args.max_val_samples, require_images=True
    )
    train_data = Seen10GenerationDataset(train_rows, image_size=args.image_size, include_target=True)
    val_data = Seen10GenerationDataset(val_rows, image_size=args.image_size, include_target=True)
    train_sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=False
    )
    val_sampler = StridedDistributedSampler(len(val_data), rank, world_size)
    generator = torch.Generator()
    generator.manual_seed(args.seed + 1729)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty; reduce batch-size or increase train rows")
    batches_per_epoch = len(train_loader)
    logger.info(
        "Loaded manifest splits: train=%s validation=%s batches_per_epoch=%s",
        len(train_data), len(val_data), batches_per_epoch,
    )

    optimizer = creat_optimizer(model, args.weight_decay, args.learning_rate, (args.beta1, args.beta2), logger)
    scaler = get_scaler(args.precision)
    if world_size > 1:
        train_model: torch.nn.Module = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    else:
        train_model = model

    model_config = {
        "gpt_model": args.gpt_model,
        "image_size": args.image_size,
        "downsample_size": args.downsample_size,
        "token_count": args.token_count,
        "caption_dim": args.caption_dim,
        "adapter_size": args.adapter_size,
        "condition_type": args.condition_type,
    }
    train_steps = 0
    start_epoch = 0
    resume_batch_offset = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = resume_checkpoint
        if int(checkpoint.get("world_size", world_size)) != world_size:
            raise ValueError(
                f"Cannot exactly resume with world_size={world_size}; checkpoint used {checkpoint.get('world_size')}"
            )
        if int(checkpoint.get("train_dataset_size", len(train_data))) != len(train_data):
            raise ValueError("Training row count differs from checkpoint; exact sampler resume is not possible")
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        train_steps = int(checkpoint["steps"])
        start_epoch = int(checkpoint["epoch"])
        resume_batch_offset = int(checkpoint.get("batch_in_epoch", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        saved_training = checkpoint.get("training_config", {})
        for key, value in {"seed": args.seed, "batch_size": args.batch_size}.items():
            if saved_training and saved_training.get(key) != value:
                raise ValueError(f"Resume {key} mismatch: checkpoint={saved_training.get(key)!r}, requested={value!r}")
        if int(checkpoint.get("batches_per_epoch", batches_per_epoch)) != batches_per_epoch:
            raise ValueError("Batches per epoch differ from checkpoint; exact resume is not possible")
        rng_states = checkpoint.get("rng_states")
        if not isinstance(rng_states, list) or rank >= len(rng_states):
            raise ValueError("Resume checkpoint is missing the per-rank RNG state")
        restore_rng_state(rng_states[rank], device)
        if resume_batch_offset > batches_per_epoch:
            raise ValueError("Resume batch offset exceeds current epoch length")
        logger.info(
            "Restored optimizer/scaler/RNG at epoch=%s batch=%s step=%s",
            start_epoch, resume_batch_offset, train_steps,
        )

    precision_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "none": torch.float32,
    }[args.precision]
    code_length = (args.image_size // args.downsample_size) ** 2
    last_validated_step = (
        int(resume_checkpoint.get("last_validated_step", -1)) if args.resume else -1
    )
    last_checkpointed_step = train_steps if args.resume else -1
    run_start = time.time()

    def save_checkpoint(
        path: Path,
        *,
        epoch: int,
        batch_in_epoch: int,
        val_loss: float,
        update_last_alias: bool = False,
    ) -> None:
        nonlocal last_checkpointed_step
        rng_states = collect_rng_states(device, rank, world_size)
        if rank == 0:
            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "steps": train_steps,
                "args": vars(args),
                "epoch": epoch,
                "batch_in_epoch": batch_in_epoch,
                "best_val_loss": best_val_loss,
                "val_loss": val_loss,
                "last_validated_step": last_validated_step,
                "world_size": world_size,
                "train_dataset_size": len(train_data),
                "batches_per_epoch": batches_per_epoch,
                "rng_states": rng_states,
                "model_config": model_config,
                "training_config": vars(args),
            }
            atomic_torch_save(payload, path)
            logger.info("Saved checkpoint: %s (step=%s epoch=%s batch=%s)", path, train_steps, epoch, batch_in_epoch)
            if update_last_alias:
                if path != checkpoint_dir / "last.pt":
                    atomic_checkpoint_alias(path, checkpoint_dir / "last.pt")
                logger.info("Updated last.pt to %s", path.name)
        last_checkpointed_step = train_steps
        if world_size > 1:
            dist.barrier()

    def save_best_checkpoint(*, epoch: int, batch_in_epoch: int, val_loss: float) -> None:
        if rank == 0:
            payload = {
                "model": model.state_dict(),
                "steps": train_steps,
                "args": vars(args),
                "epoch": epoch,
                "batch_in_epoch": batch_in_epoch,
                "best_val_loss": best_val_loss,
                "val_loss": val_loss,
                "model_config": model_config,
                "training_config": vars(args),
            }
            atomic_torch_save(payload, checkpoint_dir / "best.pt")
            logger.info(
                "Saved validation-selected weights: %s (step=%s val_loss=%.6f)",
                checkpoint_dir / "best.pt", train_steps, val_loss,
            )
        if world_size > 1:
            dist.barrier()

    def run_validation(epoch: int, batch_in_epoch: int, *, repeat_check: bool = False) -> float:
        nonlocal best_val_loss, last_validated_step
        val_loss = evaluate(
            model,
            vq_model,
            val_loader,
            device=device,
            precision=args.precision,
            token_count=args.token_count,
            rank=rank,
            world_size=world_size,
        )
        if repeat_check:
            repeat_loss = evaluate(
                model,
                vq_model,
                val_loader,
                device=device,
                precision=args.precision,
                token_count=args.token_count,
                rank=rank,
                world_size=world_size,
            )
            if not math.isclose(val_loss, repeat_loss, rel_tol=1e-6, abs_tol=1e-6):
                raise AssertionError(f"eval-mode teacher-forcing is stochastic: {val_loss} != {repeat_loss}")
            if rank == 0:
                logger.info("Repeated eval validation loss matches: %.8f vs %.8f", val_loss, repeat_loss)
        if rank == 0:
            logger.info("Validation: step=%s epoch=%s loss=%.6f", train_steps, epoch, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_checkpoint(epoch=epoch, batch_in_epoch=batch_in_epoch, val_loss=val_loss)
        model.train()
        last_validated_step = train_steps
        return val_loss

    if rank == 0 and not (run_dir / "run_config.json").exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_config.json").write_text(
            json.dumps(vars(args), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )

    stopped = args.max_steps is not None and train_steps >= args.max_steps
    initially_stopped = stopped
    if stopped and rank == 0:
        logger.info("Resume checkpoint already reached max_steps=%s; no optimizer step was run", args.max_steps)
    for epoch in range(start_epoch, args.epochs):
        if stopped:
            break
        train_sampler.set_epoch(epoch)
        iterator = iter(train_loader)
        offset = resume_batch_offset if epoch == start_epoch else 0
        if offset:
            for _ in range(offset):
                next(iterator)
        model.train()
        for batch_index in range(offset, batches_per_epoch):
            batch = next(iterator)
            values = batch_to_device(batch, device, precision_dtype)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                _, _, info = vq_model.encode(values["target"])
                indices = info[2]
                targets = indices.reshape(values["target"].shape[0], -1).long()
            input_tokens = targets[:, :-1]
            mask = latent_mask(targets.shape[0], args.token_count, code_length, device)
            with autocast_context(args.precision):
                _, loss = train_model(
                    pose=values["pose"],
                    map_id=values["map_id"],
                    idx=input_tokens,
                    targets=targets,
                    mask=mask,
                    condition=values["radar"],
                )
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {train_steps}: {loss}")
            scaler.scale(loss).backward()
            if args.smoke and batch_index == offset:
                pose_norm = grad_norm(model.pose_map_embedder.parameters())
                radar_norm = grad_norm(model.gpt.adapter.parameters())
                if rank == 0:
                    logger.info("Smoke gradient norms: numeric_pose_map=%.6f radar_adapter=%.6f", pose_norm, radar_norm)
                if pose_norm <= 0 or radar_norm <= 0:
                    raise AssertionError(f"Expected nonzero pose/radar gradients, got {pose_norm}, {radar_norm}")
            if args.max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            train_steps += 1
            batch_in_epoch = batch_index + 1

            if train_steps % args.log_every == 0 or args.smoke:
                values_loss = torch.tensor([loss.detach().float().item(), 1.0], device=device)
                if world_size > 1:
                    dist.all_reduce(values_loss, op=dist.ReduceOp.SUM)
                if rank == 0:
                    avg_loss = (values_loss[0] / values_loss[1]).item()
                    elapsed = max(time.time() - run_start, 1e-6)
                    logger.info("step=%s train_loss=%.6f elapsed=%.1fs", train_steps, avg_loss, elapsed)

            should_validate = args.validate_every > 0 and train_steps % args.validate_every == 0
            if should_validate:
                run_validation(epoch, batch_in_epoch, repeat_check=args.smoke)
            should_stop = args.max_steps is not None and train_steps >= args.max_steps
            if should_stop and last_validated_step != train_steps:
                run_validation(epoch, batch_in_epoch, repeat_check=args.smoke)

            should_checkpoint = args.checkpoint_every > 0 and train_steps % args.checkpoint_every == 0
            checkpoint_epoch = epoch + 1 if batch_in_epoch == batches_per_epoch else epoch
            checkpoint_batch = 0 if batch_in_epoch == batches_per_epoch else batch_in_epoch
            if should_checkpoint:
                save_checkpoint(
                    checkpoint_dir / f"step_{train_steps:06d}.pt",
                    epoch=checkpoint_epoch,
                    batch_in_epoch=checkpoint_batch,
                    val_loss=best_val_loss,
                    update_last_alias=True,
                )
            if should_stop:
                if not should_checkpoint:
                    save_checkpoint(
                        checkpoint_dir / "last.pt",
                        epoch=checkpoint_epoch,
                        batch_in_epoch=checkpoint_batch,
                        val_loss=best_val_loss,
                        update_last_alias=True,
                    )
                stopped = True
                break

        resume_batch_offset = 0
        if stopped:
            break

    # The default 300k-step run ends on a 60k milestone. Keep this fallback for
    # shortened or otherwise non-divisible runs without adding per-epoch saves.
    if train_steps > 0 and not initially_stopped:
        if last_validated_step != train_steps:
            run_validation(args.epochs, 0, repeat_check=args.smoke)
        if last_checkpointed_step != train_steps:
            save_checkpoint(
                checkpoint_dir / "last.pt",
                epoch=args.epochs,
                batch_in_epoch=0,
                val_loss=best_val_loss,
                update_last_alias=True,
            )

    if rank == 0:
        logger.info("Training finished: steps=%s best_val_loss=%.6f", train_steps, best_val_loss)

    if args.smoke:
        roundtrip = load_checkpoint(checkpoint_dir / "last.pt")
        model.load_state_dict(roundtrip["model"], strict=True)
        optimizer.load_state_dict(roundtrip["optimizer"])
        if rank == 0:
            logger.info("Smoke checkpoint reload passed: %s", checkpoint_dir / "last.pt")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
