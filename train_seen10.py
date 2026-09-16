from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
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
from csgo_seen10.data import MAP_ORDER, Seen10GenerationDataset, read_benchmark_rows
from csgo_seen10.model import (
    Seen10GenerationModel,
    build_gpt,
    load_checkpoint,
    load_official_gpt_weights,
)
from tokenizer.tokenizer_image.vq_model import VQ_models


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "csgo_seen10.json"


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
    pre.add_argument("--config", default=str(DEFAULT_CONFIG))
    config_args, _ = pre.parse_known_args()
    config = json.loads(Path(config_args.config).read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(description="Train ControlAR on CSGO Benchmark v2 Seen-10.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
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
    ):
        key = name.replace("-", "_")
        default = config.get(key)
        if name in ("seed", "max_steps", "max_train_samples", "max_val_samples", "resume", "run_dir"):
            default = None
        parser.add_argument(f"--{name}", dest=key, type=arg_type, default=default)
    parser.add_argument("--smoke", action="store_true")
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
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
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

    args.data_root = Path(args.data_root).expanduser().resolve()
    args.official_gpt_checkpoint = Path(args.official_gpt_checkpoint).expanduser().resolve()
    args.vq_checkpoint = Path(args.vq_checkpoint).expanduser().resolve()
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
