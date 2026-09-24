#!/usr/bin/env python3
"""Read-only PEFT config/data/artifact checks; never construct a model or use CUDA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_csgo_seen10_aligned import (
    ContractError, DEFAULT_DATA_ROOT, DEFAULT_EVAL_CONFIG, OFFICIAL_GPT_SHA256,
    VQ_SHA256, CHECKPOINT_STEPS, check_data_contract, check_checkpoint_contract,
    read_json, require, sha256_file,
)

EXPERIMENT = "csgo_seen10_exp32gen_aligned_peft"
DEFAULT_CONFIG = ROOT / "configs" / f"{EXPERIMENT}.json"


def check_batch(world_size: int, micro_batch: int, accumulation: int) -> dict:
    values = (world_size, micro_batch, accumulation)
    require(all(type(value) is int and value > 0 for value in values),
            "World size, micro batch and accumulation must be positive integers")
    effective = world_size * micro_batch * accumulation
    require(effective == 128, f"Effective generation batch must be 128, got {effective}")
    microsteps = (50_000 // (world_size * micro_batch) // accumulation) * accumulation
    require(microsteps * world_size * micro_batch == 49_920, "Epoch source exposure mismatch")
    return {"world_size": world_size, "micro_batch": micro_batch, "accumulation": accumulation,
            "effective_batch": effective, "epoch_microsteps": microsteps,
            "updates_per_epoch": microsteps // accumulation, "epoch_samples": 49_920}


def load_config(path: Path, *, verify_weights: bool = True) -> dict:
    config = read_json(path)
    require(isinstance(config, dict), "PEFT config must be an object")
    # Preserve the existing aligned data/model/sampling contract while explicitly
    # allowing the PEFT batch factorization and declared optimizer differences.
    baseline = read_json(ROOT / "configs/csgo_seen10_exp32gen_aligned.json")
    mutable = {"experiment", "output_base", "world_size", "batch_size",
               "gradient_accumulation_steps", "scheduler_type"}
    for key, value in baseline.items():
        if key not in mutable:
            require(config.get(key) == value, f"PEFT {key} must retain aligned value {value!r}")
    expected = {
        "experiment": EXPERIMENT, "output_base": f"outputs/{EXPERIMENT}/ControlAR",
        "scheduler_type": "warmup_cosine", "warmup_steps": 195, "min_lr_ratio": 0.1,
        "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05,
        "lora_learning_rate": 1e-4, "pose_learning_rate": 1e-4,
        "effective_batch_size": 128, "max_optimizer_steps": 19500,
    }
    for key, value in expected.items():
        require(config.get(key) == value, f"PEFT {key}={config.get(key)!r}, expected {value!r}")
    check_batch(config.get("world_size"), config.get("batch_size"), config.get("gradient_accumulation_steps"))
    require(tuple(config.get("checkpoint_steps", ())) == CHECKPOINT_STEPS, "PEFT checkpoint milestones changed")
    if verify_weights:
        for key, expected_sha in (("official_gpt_checkpoint", OFFICIAL_GPT_SHA256), ("vq_checkpoint", VQ_SHA256)):
            weight = ROOT / config[key]
            require(weight.is_file(), f"Missing official weight: {weight}")
            require(sha256_file(weight) == expected_sha, f"Weight SHA mismatch: {weight}")
    return config


def check_audit(run_root: Path) -> dict:
    audit = read_json(run_root / "audits/trainable_parameters.json")
    require(audit.get("trainable_numel") == 69_316_608, "PEFT trainable count mismatch")
    require(audit.get("optimizer_numel") == 69_316_608, "PEFT optimizer count mismatch")
    require(audit.get("vq_frozen") is True, "VQ freeze not audited")
    parameters = audit.get("parameters")
    require(isinstance(parameters, list) and bool(parameters), "Missing full parameter audit")
    for row in parameters:
        require(bool(row["requires_grad"]) == (row.get("optimizer_group") is not None),
                f"Optimizer membership mismatch: {row['name']}")
    return {"trainable_numel": audit["trainable_numel"], "optimizer_numel": audit["optimizer_numel"],
            "vq_frozen": True, "audited_parameters": len(parameters)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--checkpoint-role", choices=("best", "late"), default="late")
    parser.add_argument("--inference-seed", type=int, default=42)
    parser.add_argument("--task", choices=("discrete", "continuous", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-artifacts", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--verify-all-checkpoint-sha256", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    data_root = (args.data_root or Path(config["data_root"])).resolve()
    require(data_root == Path(config["data_root"]).resolve(), "Data root differs from config")
    require(args.eval_config.resolve().is_file(), "Shared evaluator config is missing")
    require(args.inference_seed == 42, "PEFT inference seed is fixed at 42")
    batch = check_batch(
        args.world_size if args.world_size is not None else config["world_size"],
        args.batch_size if args.batch_size is not None else config["batch_size"],
        args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else config["gradient_accumulation_steps"],
    )
    result = {"experiment": EXPERIMENT, "config": str(config_path), "config_sha256": sha256_file(config_path),
              "batch": batch, "optimizer_steps": 19500, "generation_exposure": 2_496_000,
              "data": check_data_contract(data_root)}
    if args.run_root is not None:
        run_root = args.run_root.resolve()
        result["parameter_audit"] = check_audit(run_root)
        if not args.smoke:
            result["checkpoint"] = check_checkpoint_contract(
                run_root, args.checkpoint_role, verify_all_sha256=args.verify_all_checkpoint_sha256)
        if args.check_artifacts:
            from csgo_seen10.peft_artifact_contract import preflight_peft_inference
            pred_root = run_root / "predictions" / args.checkpoint_role / f"inference_seed_{args.inference_seed}"
            checkpoint = run_root / "checkpoints" / f"{args.checkpoint_role}.pt"
            tasks = ("discrete", "continuous") if args.task == "all" else (args.task,)
            result["artifacts"] = {
                task: preflight_peft_inference(
                    pred_root, config_path=config_path, checkpoint_path=checkpoint,
                    checkpoint_role=args.checkpoint_role, data_root=data_root,
                    task=task, max_samples=args.max_samples, smoke=args.smoke)
                for task in tasks
            }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ContractError, ValueError, KeyError) as exc:
        print(f"PEFT contract check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
