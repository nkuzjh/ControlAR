"""Independent Seen-10 PEFT trainer; the running aligned trainer is untouched."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import train_seen10 as common
from csgo_seen10.data import Seen10GenerationDataset, read_benchmark_rows
from csgo_seen10.model import Seen10GenerationModel, build_gpt, load_checkpoint, load_official_gpt_weights
from csgo_seen10.peft import audit_parameters, build_optimizer, configure_trainable, inject_lora


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "csgo_seen10_exp32gen_aligned_peft.json"
EXPERIMENT = "csgo_seen10_exp32gen_aligned_peft"
FORMAT = EXPERIMENT + "_v1"
MILESTONES = (3900, 7800, 11700, 15600, 19500)


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(CONFIG_PATH))
    initial, _ = pre.parse_known_args()
    config_path = Path(initial.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description="Independent aligned PEFT Seen-10 trainer")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--experiment", default=config["experiment"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=1)
    parser.add_argument("--smoke-micro-batch", type=int, default=1)
    parser.add_argument("--smoke-accumulation-steps", type=int, default=1)
    parser.add_argument("--smoke-stop-after-step", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--run-dir")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    for key in ("data_root", "official_gpt_checkpoint", "vq_checkpoint", "output_base",
                "gpt_model", "adapter_size", "condition_type", "precision"):
        parser.add_argument("--" + key.replace("_", "-"), default=config[key])
    for key in ("image_size", "downsample_size", "token_count", "caption_dim", "epochs",
                "batch_size", "gradient_accumulation_steps", "effective_batch_size",
                "max_optimizer_steps", "num_workers", "seed", "log_every"):
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=config[key])
    return parser.parse_args()


def validate_batch(world_size: int, micro_batch: int, accumulation_steps: int,
                   *, formal: bool) -> tuple[int, int, int]:
    if any(type(value) is not int or value < 1 for value in (world_size, micro_batch, accumulation_steps)):
        raise ValueError("world_size, micro batch and accumulation must be positive integers")
    effective = world_size * micro_batch * accumulation_steps
    if formal and effective != 128:
        raise ValueError(f"PEFT effective batch must be 128, got {effective}")
    updates_per_epoch = 49_920 // effective if formal else 0
    microsteps_per_epoch = updates_per_epoch * accumulation_steps
    return effective, updates_per_epoch, microsteps_per_epoch


def lr_factor(step: int, *, total_steps: int = 19_500, warmup_steps: int = 195,
              min_ratio: float = 0.1) -> float:
    """LR for the next optimizer update, indexed from one."""
    if step < 1 or total_steps < warmup_steps or not 0 < min_ratio <= 1:
        raise ValueError("Invalid scheduler steps or minimum LR ratio")
    if step <= warmup_steps:
        return step / warmup_steps
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    return min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * progress)) / 2


def apply_lr(optimizer: torch.optim.Optimizer, next_step: int) -> None:
    factor = lr_factor(next_step)
    for group in optimizer.param_groups:
        group["lr"] = group["peak_lr"] * factor


def build_identity(args: argparse.Namespace, config_path: Path,
                   train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    identity = common._aligned_identity(
        data_root=args.data_root, config_path=config_path,
        official_gpt_checkpoint=args.official_gpt_checkpoint, vq_checkpoint=args.vq_checkpoint,
        train_rows=train_rows, val_rows=val_rows,
    )
    code = identity["files"]["code"]
    code["train_seen10_peft.py"] = common._sha256_file(__file__)
    code["csgo_seen10/peft.py"] = common._sha256_file(ROOT / "csgo_seen10" / "peft.py")
    identity["experiment"] = EXPERIMENT
    identity["identity_sha256"] = common._sha256_json({
        key: value for key, value in identity.items() if key != "identity_sha256"
    })
    return identity


def _read_index(checkpoint_dir: Path) -> dict[str, Any]:
    path = checkpoint_dir / "checkpoint_index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"checkpoints": []}


def _update_index(checkpoint_dir: Path, checkpoint_path: Path, step: int,
                  val_loss: float, *, final_step: int) -> tuple[int, float]:
    index = _read_index(checkpoint_dir)
    records = index["checkpoints"]
    if any(int(record["step"]) == step for record in records):
        raise FileExistsError(f"Checkpoint index already contains step {step}")
    records.append({"step": step, "path": checkpoint_path.name, "val_loss": val_loss,
                    "is_best": False, "sha256": common._sha256_file(checkpoint_path)})
    records.sort(key=lambda record: int(record["step"]))
    best = min(records, key=lambda record: (float(record["val_loss"]), int(record["step"])))
    for record in records:
        record["is_best"] = record is best
    best_step = int(best["step"])
    index["best_step"] = best_step
    index["late_step"] = final_step if step == final_step else index.get("late_step")
    common.atomic_checkpoint_alias(checkpoint_dir / best["path"], checkpoint_dir / "best.pt")
    if step == final_step:
        common.atomic_checkpoint_alias(checkpoint_path, checkpoint_dir / "late.pt")
    common._atomic_json_save(index, checkpoint_dir / "checkpoint_index.json")
    return best_step, float(best["val_loss"])


def train(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    if config_path != CONFIG_PATH.resolve():
        raise ValueError(f"PEFT training requires canonical config {CONFIG_PATH}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.experiment != EXPERIMENT or config.get("experiment") != EXPERIMENT:
        raise ValueError("PEFT experiment mismatch")
    formal = not args.smoke
    if tuple(config["checkpoint_steps"]) != MILESTONES or config["max_optimizer_steps"] != 19_500:
        raise ValueError("PEFT budget or milestones changed")
    if config["epochs"] != 50 or config["effective_batch_size"] != 128:
        raise ValueError("PEFT formal epoch/effective-batch contract changed")
    validate_batch(int(config["world_size"]), int(config["batch_size"]),
                   int(config["gradient_accumulation_steps"]), formal=True)
    if any(config[key] != expected for key, expected in {
        "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05,
        "lora_learning_rate": 1e-4, "pose_learning_rate": 1e-4,
        "learning_rate": 5e-5, "weight_decay": 0.05,
        "beta1": 0.9, "beta2": 0.95, "adam_epsilon": 1e-8,
        "warmup_steps": 195, "min_lr_ratio": 0.1, "scheduler_type": "warmup_cosine",
        "precision": "bf16", "dropout": 0.1, "token_dropout": 0.1,
        "random_image_augmentation": False,
    }.items()):
        raise ValueError("PEFT optimizer/model recipe differs from canonical config")
    for key in ("seed", "gpt_model", "image_size", "downsample_size", "token_count",
                "caption_dim", "adapter_size", "condition_type", "precision", "epochs",
                "max_optimizer_steps", "effective_batch_size"):
        if getattr(args, key) != config[key]:
            raise ValueError(f"PEFT formal model/budget setting changed: {key}")
    if formal and (args.max_train_samples is not None or args.max_val_samples is not None):
        raise ValueError("Formal PEFT training requires full train and validation splits")
    if formal and (args.smoke_stop_after_step is not None):
        raise ValueError("stop-after-step is smoke-only")
    if formal and args.num_workers != config["num_workers"]:
        raise ValueError("Formal num_workers must match canonical config")
    if args.smoke and (args.smoke_steps < 1 or (args.smoke_stop_after_step is not None and
                         not 1 <= args.smoke_stop_after_step <= args.smoke_steps)):
        raise ValueError("Invalid smoke step budget or stop point")
    args.data_root = Path(args.data_root).expanduser().resolve()
    args.official_gpt_checkpoint = (ROOT / args.official_gpt_checkpoint).resolve()
    args.vq_checkpoint = (ROOT / args.vq_checkpoint).resolve()
    if formal and args.data_root != Path(config["data_root"]).expanduser().resolve():
        raise ValueError("Formal PEFT data root differs from canonical benchmark")
    if formal and (args.official_gpt_checkpoint != (ROOT / config["official_gpt_checkpoint"]).resolve() or
                   args.vq_checkpoint != (ROOT / config["vq_checkpoint"]).resolve()):
        raise ValueError("Formal PEFT GPT/VQ paths differ from canonical checkpoints")
    if not torch.cuda.is_available():
        raise RuntimeError("PEFT training requires CUDA")
    rank, world_size, local_rank, device = common.initialize_distributed()
    micro_batch = args.batch_size if formal else args.smoke_micro_batch
    accumulation = args.gradient_accumulation_steps if formal else args.smoke_accumulation_steps
    effective_batch, updates_per_epoch, epoch_micro_batches = validate_batch(
        world_size, micro_batch, accumulation, formal=formal
    )
    if args.smoke:
        if args.max_train_samples is None:
            args.max_train_samples = effective_batch * args.smoke_steps
        if args.max_val_samples is None:
            args.max_val_samples = 8
    if formal and args.effective_batch_size != effective_batch:
        raise ValueError("Configured effective batch does not match the actual batch product")
    if formal and updates_per_epoch != 390:
        raise AssertionError("PEFT epoch must have 390 optimizer updates")
    if args.run_dir is None:
        if args.smoke:
            args.run_dir = os.environ.get("CSGO_SMOKE_ROOT") or str(
                ROOT / "outputs" / "csgo_benchmark_v2_smoke_peft" / "ControlAR" / f"seed_{args.seed}"
            )
        else:
            args.run_dir = str((ROOT / args.output_base / f"seed_{args.seed}").resolve())
    run_dir = Path(args.run_dir).expanduser().resolve()
    if formal and run_dir != (ROOT / config["output_base"] / f"seed_{config['seed']}").resolve():
        raise ValueError("Formal PEFT run directory differs from canonical run")
    checkpoint_dir = run_dir / "checkpoints"
    if args.resume:
        args.resume = str(Path(args.resume).expanduser().resolve())
        if Path(args.resume).parent != checkpoint_dir or not Path(args.resume).is_file():
            raise ValueError("Exact PEFT resume needs an existing checkpoint within this run")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"PEFT run directory is not empty: {run_dir}")
    if not args.resume and not args.official_gpt_checkpoint.is_file():
        raise FileNotFoundError(args.official_gpt_checkpoint)
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    logger = common.setup_logger(run_dir, rank)
    common.seed_process(args.seed, rank)
    logger.info("Starting %s: world=%s micro=%s accumulation=%s effective=%s",
                EXPERIMENT, world_size, micro_batch, accumulation, effective_batch)
    resume = load_checkpoint(args.resume) if args.resume else None
    if resume is not None and resume.get("format") != FORMAT:
        raise ValueError("PEFT resume checkpoint format mismatch")

    gpt = build_gpt(model_name=args.gpt_model, image_size=args.image_size,
                    downsample_size=args.downsample_size, token_count=args.token_count,
                    adapter_size=args.adapter_size, condition_type=args.condition_type,
                    dropout=config["dropout"], token_dropout=config["token_dropout"]).to(device)
    if resume is None:
        load_official_gpt_weights(gpt, args.official_gpt_checkpoint)
    inject_lora(gpt, config["lora_rank"], config["lora_alpha"], config["lora_dropout"])
    model = Seen10GenerationModel(gpt, caption_dim=args.caption_dim, token_count=args.token_count).to(device)
    if resume is not None:
        model.load_state_dict(resume["model"], strict=True)
    configure_trainable(model)
    vq_model = common.load_vq_model(str(args.vq_checkpoint), device)

    train_rows = read_benchmark_rows(args.data_root, "seen_train",
                                     max_samples=args.max_train_samples if args.smoke else None,
                                     require_images=True)
    val_rows = read_benchmark_rows(args.data_root, "seen_validation",
                                   max_samples=args.max_val_samples if args.smoke else None,
                                   require_images=True)
    if formal and (len(train_rows), len(val_rows)) != (50_000, 5_000):
        raise ValueError("PEFT split counts differ from canonical 50000/5000")
    identity = build_identity(args, config_path, train_rows, val_rows)
    if identity["files"]["official_gpt"] != config["official_gpt_sha256"]:
        raise ValueError("Official GPT SHA256 mismatch")
    if identity["files"]["vq"] != config["vq_sha256"]:
        raise ValueError("VQ SHA256 mismatch")
    if resume is not None and resume.get("identity") != identity:
        raise ValueError("PEFT exact resume identity mismatch")
    if rank == 0:
        common._atomic_json_save(identity, run_dir / "audits" / "identity.json")
    train_data = Seen10GenerationDataset(train_rows, image_size=args.image_size, include_target=True)
    val_data = Seen10GenerationDataset(val_rows, image_size=args.image_size, include_target=True)
    sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=args.seed, drop_last=True)
    val_sampler = common.StridedDistributedSampler(len(val_data), rank, world_size)
    generator = torch.Generator().manual_seed(args.seed + 1729)
    val_generator = torch.Generator().manual_seed(args.seed + 2718)
    workers = args.num_workers if formal else 0
    train_loader = DataLoader(train_data, batch_size=micro_batch, sampler=sampler,
                              num_workers=workers, pin_memory=True, drop_last=True,
                              persistent_workers=False, generator=generator)
    val_loader = DataLoader(val_data, batch_size=micro_batch, sampler=val_sampler,
                            num_workers=workers, pin_memory=True, drop_last=False,
                            persistent_workers=False, generator=val_generator)
    if args.smoke:
        updates_per_epoch = len(train_loader) // accumulation
        epoch_micro_batches = updates_per_epoch * accumulation
        if updates_per_epoch < 1 or args.smoke_steps > updates_per_epoch * config["epochs"]:
            raise ValueError("Smoke split is too short for the requested accumulation/steps")
    elif len(train_loader) < epoch_micro_batches:
        raise RuntimeError("Formal train loader is shorter than expected epoch")
    logger.info("train=%s val=%s epoch_microsteps=%s updates_per_epoch=%s",
                len(train_data), len(val_data), epoch_micro_batches, updates_per_epoch)

    optimizer = build_optimizer(model, lora_lr=config["lora_learning_rate"],
                                pose_lr=config["pose_learning_rate"],
                                pretrained_lr=config["learning_rate"],
                                weight_decay=config["weight_decay"],
                                betas=(config["beta1"], config["beta2"]), eps=config["adam_epsilon"])
    scaler = common.get_scaler(args.precision)
    audit = audit_parameters(model, optimizer, vq_model=vq_model)
    if audit["total_numel"] != 866_921_344:
        raise AssertionError("Unexpected total PEFT parameter count")
    if rank == 0:
        common._atomic_json_save(audit, run_dir / "audits" / "trainable_parameters.json")
    train_model = (DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                           find_unused_parameters=True) if world_size > 1 else model)
    model_config = {"gpt_model": args.gpt_model, "image_size": args.image_size,
                    "downsample_size": args.downsample_size, "token_count": args.token_count,
                    "caption_dim": args.caption_dim, "adapter_size": args.adapter_size,
                    "condition_type": args.condition_type, "dropout": config["dropout"],
                    "token_dropout": config["token_dropout"],
                    "peft": {"rank": 32, "alpha": 64, "dropout": 0.05, "qkv_independent": True}}
    training_config = {
        "seed": args.seed, "world_size": world_size, "batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation, "effective_batch_size": effective_batch,
        "epochs": config["epochs"], "max_optimizer_steps": config["max_optimizer_steps"],
        "checkpoint_steps": list(MILESTONES), "updates_per_epoch": updates_per_epoch,
        "epoch_micro_batches": epoch_micro_batches, "lora_rank": 32, "lora_alpha": 64,
        "lora_dropout": 0.05, "lora_learning_rate": config["lora_learning_rate"],
        "pose_learning_rate": config["pose_learning_rate"],
        "pretrained_learning_rate": config["learning_rate"], "weight_decay": config["weight_decay"],
        "betas": [0.9, 0.95], "adam_epsilon": 1e-8,
        "scheduler": "warmup_cosine", "scheduler_horizon": 19_500,
        "warmup_steps": 195, "min_lr_ratio": 0.1,
        "precision": args.precision, "max_grad_norm": config["max_grad_norm"],
        "dropout": config["dropout"], "token_dropout": config["token_dropout"],
        "smoke": args.smoke, "smoke_steps": args.smoke_steps if args.smoke else None,
    }
    if rank == 0:
        common._atomic_json_save({"args": vars(args), "config_path": str(config_path),
                                  "identity": identity, "model_config": model_config,
                                  "training_config": training_config,
                                  "checkpoint_steps": list(range(1, args.smoke_steps + 1)) if args.smoke else list(MILESTONES)},
                                 run_dir / "run_config.json")
    checkpoint_steps = tuple(range(1, args.smoke_steps + 1)) if args.smoke else MILESTONES
    target_steps = args.smoke_steps if args.smoke else 19_500
    stop_after = args.smoke_stop_after_step if args.smoke_stop_after_step is not None else target_steps
    train_steps = consumed_samples = start_epoch = resume_offset = 0
    best_step: int | None = None
    best_val_loss = float("inf")
    loss_trace: list[dict[str, Any]] = []
    restored_generator_state: torch.Tensor | None = None
    if resume is not None:
        required = {"model", "optimizer", "scheduler", "scaler", "global_optimizer_step",
                    "consumed_samples", "epoch", "batch_in_epoch", "micro_step_in_accum",
                    "rng_states", "dataloader_generator_state", "validation_dataloader_generator_state",
                    "sampler_state", "identity",
                    "training_config", "model_config"}
        if missing := required - resume.keys():
            raise ValueError(f"PEFT resume fields missing: {sorted(missing)}")
        if resume["training_config"] != training_config or resume["model_config"] != model_config:
            raise ValueError("PEFT resume recipe mismatch")
        if int(resume["world_size"]) != world_size or int(resume["batches_per_epoch"]) != epoch_micro_batches:
            raise ValueError("PEFT resume world/epoch layout mismatch")
        train_steps = int(resume["global_optimizer_step"])
        consumed_samples = int(resume["consumed_samples"])
        start_epoch = int(resume["epoch"])
        resume_offset = int(resume["batch_in_epoch"])
        if (int(resume["steps"]) != train_steps or consumed_samples != train_steps * effective_batch or
            int(resume["micro_step_in_accum"]) != 0 or
            train_steps != start_epoch * updates_per_epoch + resume_offset // accumulation or
            resume_offset % accumulation != 0):
            raise ValueError("PEFT checkpoint is not at a consistent optimizer boundary")
        if resume_offset < 0 or resume_offset >= epoch_micro_batches:
            raise ValueError("PEFT resume batch offset outside epoch")
        if resume["sampler_state"] != {"epoch": start_epoch, "seed": args.seed,
                                       "num_replicas": world_size, "micro_batch_per_device": micro_batch,
                                       "gradient_accumulation_steps": accumulation}:
            raise ValueError("PEFT sampler state mismatch")
        indexed = _read_index(checkpoint_dir)
        indexed_record = next((record for record in indexed["checkpoints"]
                               if int(record["step"]) == train_steps), None)
        canonical = checkpoint_dir / f"step_{train_steps:06d}.pt"
        if not canonical.is_file() or not Path(args.resume).samefile(canonical):
            raise ValueError("PEFT resume checkpoint is not its canonical step file")
        disk_steps = {int(path.stem.rsplit("_", 1)[-1])
                      for path in checkpoint_dir.glob("step_*.pt")}
        if any(step > train_steps for step in disk_steps) or any(
            int(record["step"]) > train_steps for record in indexed["checkpoints"]
        ):
            raise ValueError("Refusing to rewind PEFT run over newer checkpoints")
        prior_steps = {step for step in checkpoint_steps if step < train_steps}
        indexed_prior = {int(record["step"]) for record in indexed["checkpoints"]
                         if int(record["step"]) < train_steps}
        if prior_steps != indexed_prior:
            raise ValueError("PEFT resume is missing an earlier indexed milestone")
        for record in indexed["checkpoints"]:
            record_path = checkpoint_dir / record["path"]
            if not record_path.is_file() or common._sha256_file(record_path) != record["sha256"]:
                raise ValueError("PEFT indexed checkpoint file is missing or changed")
        if indexed_record is None:
            if train_steps not in checkpoint_steps:
                raise ValueError("PEFT resume step lacks a canonical indexed milestone")
            if rank == 0:
                _update_index(checkpoint_dir, canonical, train_steps, float(resume["val_loss"]),
                              final_step=target_steps)
            if world_size > 1:
                dist.barrier()
            indexed = _read_index(checkpoint_dir)
            indexed_record = next(record for record in indexed["checkpoints"]
                                  if int(record["step"]) == train_steps)
        if indexed_record["sha256"] != common._sha256_file(canonical):
            raise ValueError("PEFT indexed checkpoint SHA256 mismatch")
        optimizer.load_state_dict(resume["optimizer"])
        expected_scheduler = {"type": "warmup_cosine", "step": train_steps,
                              "horizon": 19_500, "warmup_steps": 195, "min_lr_ratio": 0.1}
        if resume["scheduler"] != expected_scheduler:
            raise ValueError("PEFT scheduler state mismatch")
        expected_factor = lr_factor(train_steps + 1)
        if any(not math.isclose(group["lr"], group["peak_lr"] * expected_factor,
                                rel_tol=1e-12, abs_tol=1e-15)
               for group in optimizer.param_groups):
            raise ValueError("PEFT optimizer group LR differs from resumed scheduler")
        scaler.load_state_dict(resume["scaler"])
        restored_generator_state = resume["dataloader_generator_state"]
        generator.set_state(restored_generator_state)
        val_generator.set_state(resume["validation_dataloader_generator_state"])
        states = resume["rng_states"]
        if len(states) != world_size:
            raise ValueError("PEFT per-rank RNG count mismatch")
        common.restore_rng_state(states[rank], device)
        best_step = int(indexed["best_step"])
        best_val_loss = float(next(record["val_loss"] for record in indexed["checkpoints"]
                                   if int(record["step"]) == best_step))
        trace_path = run_dir / "step_trace.json"
        if trace_path.is_file():
            loss_trace = json.loads(trace_path.read_text(encoding="utf-8"))
        del resume
        logger.info("Restored PEFT step=%s epoch=%s batch_offset=%s", train_steps, start_epoch, resume_offset)
    apply_lr(optimizer, train_steps + 1)
    code_length = (args.image_size // args.downsample_size) ** 2
    run_start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, config["epochs"]):
        if train_steps >= stop_after:
            break
        sampler.set_epoch(epoch)
        generator.manual_seed(args.seed + 1729 + epoch)
        iterator = iter(train_loader)
        offset = resume_offset if epoch == start_epoch else 0
        for _ in range(offset):
            next(iterator)
        if offset and restored_generator_state is not None and not torch.equal(
            generator.get_state(), restored_generator_state
        ):
            raise ValueError("PEFT restored dataloader generator state differs from replayed iterator")
        optimizer.zero_grad(set_to_none=True)
        model.train()
        micro_loss_sum = 0.0
        for batch_index in range(offset, epoch_micro_batches):
            batch = next(iterator)
            values = common.batch_to_device(batch, device, {
                "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
                "none": torch.float32}[args.precision])
            with torch.no_grad():
                _, _, info = vq_model.encode(values["target"])
                targets = info[2].reshape(values["target"].shape[0], -1).long()
            mask = common.latent_mask(targets.shape[0], args.token_count, code_length, device)
            boundary = (batch_index + 1) % accumulation == 0
            sync = train_model.no_sync() if world_size > 1 and not boundary else contextlib.nullcontext()
            with sync:
                with common.autocast_context(args.precision):
                    _, loss = train_model(pose=values["pose"], map_id=values["map_id"],
                                          idx=targets[:, :-1], targets=targets, mask=mask,
                                          condition=values["radar"])
                if loss is None or not torch.isfinite(loss):
                    raise FloatingPointError(f"PEFT non-finite loss at epoch={epoch} micro={batch_index}")
                micro_loss_sum += float(loss.detach().float())
                scaler.scale(loss / accumulation).backward()
            if not boundary:
                continue
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"PEFT non-finite gradient at step {train_steps + 1}")
            step_lrs = {group["role"]: float(group["lr"]) for group in optimizer.param_groups}
            scaler.step(optimizer)
            scaler.update()
            train_steps += 1
            consumed_samples += effective_batch
            optimizer.zero_grad(set_to_none=True)
            loss_average = micro_loss_sum / accumulation
            micro_loss_sum = 0.0
            trace = {"step": train_steps, "loss": loss_average, "grad_norm": float(grad_norm),
                     "learning_rates": step_lrs, "consumed_samples": consumed_samples,
                     "epoch": epoch, "batch_in_epoch": batch_index + 1}
            loss_trace.append(trace)
            if rank == 0 and (args.smoke or train_steps % args.log_every == 0):
                logger.info("PEFT step=%s loss=%.6f grad=%.4f lrs=%s elapsed=%.1fs",
                            train_steps, loss_average, float(grad_norm), step_lrs,
                            time.time() - run_start)
                common._atomic_json_save(loss_trace, run_dir / "step_trace.json")
            apply_lr(optimizer, train_steps + 1)
            if train_steps in checkpoint_steps:
                val_loss = common.evaluate(model, vq_model, val_loader, device=device,
                                           precision=args.precision, token_count=args.token_count,
                                           rank=rank, world_size=world_size)
                model.train()
                logger.info("PEFT validation step=%s loss=%.6f", train_steps, val_loss)
                next_epoch = epoch + 1 if batch_index + 1 == epoch_micro_batches else epoch
                next_offset = 0 if next_epoch != epoch else batch_index + 1
                rng_states = common.collect_rng_states(device, rank, world_size)
                if rank == 0:
                    is_best = val_loss < best_val_loss
                    payload = {
                        "format": FORMAT, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": {"type": "warmup_cosine", "step": train_steps,
                                      "horizon": 19_500, "warmup_steps": 195, "min_lr_ratio": 0.1},
                        "scaler": scaler.state_dict(), "steps": train_steps,
                        "global_optimizer_step": train_steps, "consumed_samples": consumed_samples,
                        "args": vars(args), "epoch": next_epoch, "batch_in_epoch": next_offset,
                        "micro_step_in_accum": 0, "best_val_loss": min(best_val_loss, val_loss),
                        "best_step": train_steps if is_best else best_step,
                        "val_loss": val_loss, "last_validated_step": train_steps,
                        "world_size": world_size, "train_dataset_size": len(train_data),
                        "validation_dataset_size": len(val_data),
                        "batches_per_epoch": epoch_micro_batches,
                        "updates_per_epoch": updates_per_epoch, "rng_states": rng_states,
                        "dataloader_generator_state": generator.get_state(),
                        "validation_dataloader_generator_state": val_generator.get_state(),
                        "sampler_state": {"epoch": next_epoch, "seed": args.seed,
                                          "num_replicas": world_size,
                                          "micro_batch_per_device": micro_batch,
                                          "gradient_accumulation_steps": accumulation},
                        "identity": identity, "model_config": model_config,
                        "training_config": training_config,
                    }
                    path = checkpoint_dir / f"step_{train_steps:06d}.pt"
                    if path.exists():
                        raise FileExistsError(path)
                    common.atomic_torch_save(payload, path)
                    best_step, best_val_loss = _update_index(
                        checkpoint_dir, path, train_steps, val_loss, final_step=target_steps)
                    logger.info("Saved PEFT checkpoint %s", path)
                if world_size > 1:
                    dist.barrier()
            if train_steps >= stop_after:
                break
        resume_offset = 0
    if formal and train_steps != 19_500:
        raise RuntimeError(f"PEFT formal run stopped at {train_steps}/19500")
    if formal:
        if consumed_samples != 2_496_000 or tuple(
            int(path.stem.rsplit("_", 1)[-1]) for path in sorted(checkpoint_dir.glob("step_*.pt"))
        ) != MILESTONES:
            raise AssertionError("PEFT formal exposure/checkpoint budget mismatch")
    if rank == 0:
        common._atomic_json_save(loss_trace, run_dir / "step_trace.json")
        common._atomic_json_save({"step": train_steps, "target_steps": target_steps,
                                  "stopped_for_resume": train_steps < target_steps,
                                  "consumed_samples": consumed_samples,
                                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                                  "elapsed_seconds": time.time() - run_start,
                                  "last_loss": loss_trace[-1]["loss"] if loss_trace else None,
                                  "learning_rates_next_step": {group["role"]: group["lr"]
                                                               for group in optimizer.param_groups}},
                                 run_dir / "train_summary.json")
        logger.info("PEFT finished step=%s target=%s", train_steps, target_steps)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    train(parse_args())
