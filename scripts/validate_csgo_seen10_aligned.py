#!/usr/bin/env python3
"""Static and artifact-contract checks for the aligned CSGO Seen-10 run.

This checker deliberately does not import ControlAR model code, torch, or a
checkpoint loader.  It validates the versioned configuration, benchmark
identity/count contract, source-level guardrails, and (when requested) the
small JSON/file contract surrounding a completed aligned run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from csgo_seen10.paths import data_root as resolve_data_root, evaluator_root, project_path
ALIGNED_EXPERIMENT = "csgo_seen10_exp32gen_aligned"
DEFAULT_CONFIG = ROOT / "configs" / f"{ALIGNED_EXPERIMENT}.json"
DEFAULT_DATA_ROOT = Path("/home/jiahao/task/UniLIP/data/csgo_benchmark_v2")
DEFAULT_EVAL_CONFIG = Path("/home/jiahao/task/csgo_benchmark_v2_eval_general/benchmark_v2.yaml")
OFFICIAL_GPT_SHA256 = "ef59b3c51e582e4742406480fb81160044b902bd46b2f00d923734800258545e"
VQ_SHA256 = "0e21fc1318e2e9ee641a07bdad0e20675e9ec35e6e3eb911d58b5d7a2cd8d4cb"
SEEN_MAPS = (
    "cs_agency",
    "cs_italy",
    "de_ancient",
    "de_anubis",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
    "de_overpass",
    "de_train",
)
EXPECTED_COUNTS = {
    "train_per_map": 5_000,
    "validation_per_map": 500,
    "discrete_per_map": 2_000,
    "continuous_clips_per_map": 20,
    "continuous_frames_per_map": 1_280,
    "train": 50_000,
    "validation": 5_000,
    "discrete": 20_000,
    "continuous": 12_800,
}
CHECKPOINT_STEPS = (3_900, 7_800, 11_700, 15_600, 19_500)


class ContractError(RuntimeError):
    """Raised when the aligned contract is not satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - the path is in the error
        raise ContractError(f"Cannot read JSON: {path}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load_config(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Aligned config not found: {path}")
    config = read_json(path)
    require(isinstance(config, dict), f"Aligned config must be a JSON object: {path}")
    require(config.get("experiment") == ALIGNED_EXPERIMENT, "Config experiment does not identify the aligned profile")
    expected = {
        "data_root": str(DEFAULT_DATA_ROOT),
        "official_gpt_checkpoint": "checkpoints/t2i/canny_MR.safetensors",
        "vq_checkpoint": "checkpoints/vq/vq_ds16_t2i.pt",
        "output_base": "outputs/csgo_seen10_exp32gen_aligned/ControlAR",
        "gpt_model": "GPT-XL",
        "image_size": 448,
        "downsample_size": 16,
        "token_count": 120,
        "caption_dim": 2_048,
        "adapter_size": "small",
        "condition_type": "radar",
        "world_size": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 128,
        "effective_batch_size": 128,
        "max_optimizer_steps": 19_500,
        "epochs": 50,
        "train_samples": 50_000,
        "validation_samples": 5_000,
        "learning_rate": 5e-5,
        "weight_decay": 0.05,
        "beta1": 0.9,
        "beta2": 0.95,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "precision": "bf16",
        "scheduler_type": "constant",
        "dropout": 0.1,
        "token_dropout": 0.1,
        "random_image_augmentation": False,
        "cfg_scale": 4.0,
        "temperature": 1.0,
        "top_k": 2_000,
        "top_p": 1.0,
        "seed": 42,
        "inference_seed": 42,
        "inference_engine": "compiled",
        "inference_batch_size": 16,
        "official_gpt_sha256": OFFICIAL_GPT_SHA256,
        "vq_sha256": VQ_SHA256,
    }
    for key, value in expected.items():
        require(config.get(key) == value, f"Config {key}={config.get(key)!r}, expected {value!r}")
    for config_key, expected_hash in (
        ("official_gpt_checkpoint", OFFICIAL_GPT_SHA256),
        ("vq_checkpoint", VQ_SHA256),
    ):
        checkpoint = Path(str(config[config_key])).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        checkpoint = checkpoint.resolve()
        require(checkpoint.is_file(), f"Configured checkpoint is missing: {checkpoint}")
        require(sha256_file(checkpoint) == expected_hash, f"Configured checkpoint SHA256 mismatch: {checkpoint}")
    require(tuple(config.get("checkpoint_steps", ())) == CHECKPOINT_STEPS, "Checkpoint schedule is not the five aligned milestones")
    require(config.get("checkpoint_every") == 0, "Aligned config must disable interval checkpointing")
    require(config.get("validate_every") == 0, "Aligned config must disable interval validation")
    require(config.get("data_root"), "Aligned config has no data_root")
    require(config.get("official_gpt_checkpoint"), "Aligned config has no official_gpt_checkpoint")
    require(config.get("vq_checkpoint"), "Aligned config has no vq_checkpoint")
    return config


def _split_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    require(isinstance(payload, list), f"Expected a JSON array: {path}")
    return [row for row in payload if isinstance(row, dict)]


def check_data_contract(data_root: Path) -> dict[str, Any]:
    data_root = data_root.expanduser().resolve()
    manifest_path = data_root / "benchmark_manifest.json"
    report_path = data_root / "minimal_dataset_report.json"
    calibration_path = data_root / "calibration" / "z_calibration.json"
    for path in (manifest_path, report_path, calibration_path):
        require(path.is_file(), f"Benchmark contract file not found: {path}")
    manifest = read_json(manifest_path)
    report = read_json(report_path)
    require(manifest.get("benchmark_id") == "csgo_benchmark_v2", "Wrong benchmark manifest id")
    require(report.get("benchmark_id") == "csgo_benchmark_v2", "Wrong dataset report id")
    require(report.get("status") == "verified", "Dataset report is not verified")
    require(tuple(manifest.get("protocol", {}).get("seen_maps", ())) == SEEN_MAPS, "Seen-10 map order differs")

    counts = {"train": 0, "validation": 0, "discrete": 0, "continuous": 0}
    file_hashes: list[dict[str, str]] = []
    for path in (manifest_path, report_path, calibration_path):
        file_hashes.append({"path": str(path.relative_to(data_root)), "sha256": sha256_file(path)})
    for map_name in SEEN_MAPS:
        base = data_root / "splits" / "seen" / map_name
        train = _split_rows(base / "train.json")
        validation = _split_rows(base / "validation.json")
        discrete = _split_rows(base / "discrete_test.json")
        require(len(train) == EXPECTED_COUNTS["train_per_map"], f"{map_name} train count mismatch")
        require(len(validation) == EXPECTED_COUNTS["validation_per_map"], f"{map_name} validation count mismatch")
        require(len(discrete) == EXPECTED_COUNTS["discrete_per_map"], f"{map_name} discrete count mismatch")
        counts["train"] += len(train)
        counts["validation"] += len(validation)
        counts["discrete"] += len(discrete)
        continuous_payload = read_json(base / "continuous_clips.json")
        clips = continuous_payload.get("clips") if isinstance(continuous_payload, dict) else None
        require(isinstance(clips, list), f"{map_name} continuous split has no clips list")
        require(len(clips) == EXPECTED_COUNTS["continuous_clips_per_map"], f"{map_name} continuous clip count mismatch")
        frame_count = 0
        for clip in clips:
            require(isinstance(clip, dict) and isinstance(clip.get("frames"), list), f"Invalid continuous clip in {map_name}")
            require(len(clip["frames"]) == 64, f"{map_name} continuous clip length is not 64")
            frame_count += len(clip["frames"])
        require(frame_count == EXPECTED_COUNTS["continuous_frames_per_map"], f"{map_name} continuous frame count mismatch")
        counts["continuous"] += frame_count
        file_hashes.append({"path": str((base / "train.json").relative_to(data_root)), "sha256": sha256_file(base / "train.json")})
        file_hashes.append({"path": str((base / "validation.json").relative_to(data_root)), "sha256": sha256_file(base / "validation.json")})
        file_hashes.append({"path": str((base / "discrete_test.json").relative_to(data_root)), "sha256": sha256_file(base / "discrete_test.json")})
        file_hashes.append({"path": str((base / "continuous_clips.json").relative_to(data_root)), "sha256": sha256_file(base / "continuous_clips.json")})
    for key in ("train", "validation", "discrete", "continuous"):
        require(counts[key] == EXPECTED_COUNTS[key], f"Total {key} count={counts[key]}, expected {EXPECTED_COUNTS[key]}")
    return {
        "data_root": str(data_root),
        "counts": counts,
        "seen_maps": list(SEEN_MAPS),
        "benchmark_manifest_sha256": next(item["sha256"] for item in file_hashes if item["path"] == "benchmark_manifest.json"),
        "minimal_dataset_report_sha256": next(item["sha256"] for item in file_hashes if item["path"] == "minimal_dataset_report.json"),
        "calibration_sha256": next(item["sha256"] for item in file_hashes if item["path"] == "calibration/z_calibration.json"),
        "files": file_hashes,
    }


def check_source_contract() -> list[str]:
    checks = {
        ROOT / "train_seen10.py": (
            "ALIGNED_EXPERIMENT =",
            "gradient_accumulation_steps",
            "scaler.scale(loss / accumulation_steps).backward()",
            "scheduler.step()",
            "micro_step_in_accum",
            "gpt.condition_embeddings.",
        ),
        ROOT / "infer_seen10.py": (
            "ALIGNED_EXPERIMENT =",
            "include_target=False",
            "require_images=False",
            "ALIGNED_COMPILED_SEED_POLICY",
            "checkpoint_role",
        ),
        ROOT / "csgo_seen10" / "data.py": (
            "read_benchmark_rows",
            "include_target: bool = True",
            "Image.Resampling.BICUBIC",
        ),
        ROOT / "scripts" / "run_csgo_seen10.sh": (
            "--experiment",
            "--checkpoint-role",
            "--inference-seed",
            "benchmark_v2.yaml",
            "csgo_seen10_exp32gen_aligned",
        ),
    }
    checked: list[str] = []
    for path, needles in checks.items():
        require(path.is_file(), f"Source file not found: {path}")
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            require(needle in text, f"Source contract missing {needle!r} in {path}")
        checked.append(str(path))
    return checked


def check_checkpoint_contract(
    run_root: Path, role: str, *, verify_all_sha256: bool = False,
    checkpoint_steps: tuple[int, ...] = CHECKPOINT_STEPS,
) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    checkpoint_dir = run_root / "checkpoints"
    require(checkpoint_dir.is_dir(), f"Checkpoint directory not found: {checkpoint_dir}")
    index_path = checkpoint_dir / "checkpoint_index.json"
    require(index_path.is_file(), f"Checkpoint index not found: {index_path}")
    index = read_json(index_path)
    records = index.get("checkpoints") if isinstance(index, dict) else None
    require(isinstance(records, list), "Checkpoint index has no checkpoints list")
    steps = tuple(sorted(int(record["step"]) for record in records))
    require(steps == checkpoint_steps, f"Checkpoint steps={steps}, expected {checkpoint_steps}")
    require(int(index.get("late_step", -1)) == checkpoint_steps[-1],
            f"Checkpoint index late_step is not {checkpoint_steps[-1]}")
    best_step = int(index.get("best_step", -1))
    require(best_step in checkpoint_steps, "Checkpoint index best_step is not one of the five milestones")
    for record in records:
        path = checkpoint_dir / str(record.get("path", ""))
        require(path.is_file(), f"Indexed checkpoint is missing: {path}")
        require(record.get("sha256"), f"Indexed checkpoint has no SHA256: {path}")
        if verify_all_sha256:
            require(
                sha256_file(path) == record["sha256"],
                f"Indexed checkpoint SHA256 mismatch: {path}",
            )
    role_path = checkpoint_dir / f"{role}.pt"
    require(role_path.is_file(), f"Requested checkpoint role is missing: {role_path}")
    late_path = checkpoint_dir / "late.pt"
    require(late_path.is_file(), "late.pt is missing")
    final_path = checkpoint_dir / f"step_{checkpoint_steps[-1]:06d}.pt"
    require(late_path.samefile(final_path), f"late.pt does not reference {final_path.name}")
    best_record = next(record for record in records if int(record["step"]) == best_step)
    require((checkpoint_dir / "best.pt").samefile(checkpoint_dir / str(best_record["path"])), "best.pt does not reference indexed best checkpoint")
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "role": role,
        "role_path": str(role_path),
        "steps": list(steps),
        "best_step": best_step,
        "late_step": checkpoint_steps[-1],
        "all_step_sha256_verified": bool(verify_all_sha256),
    }


def check_artifact_contract(
    run_root: Path,
    role: str,
    inference_seed: int,
    config_path: Path,
    data_contract: dict[str, Any],
    task: str,
    smoke: bool,
    verify_all_sha256: bool,
) -> dict[str, Any]:
    if smoke:
        checkpoint_dir = run_root.expanduser().resolve() / "checkpoints"
        require(checkpoint_dir.is_dir(), f"Checkpoint directory not found: {checkpoint_dir}")
        checkpoint = {"checkpoint_dir": str(checkpoint_dir), "role": role}
    else:
        checkpoint = check_checkpoint_contract(
            run_root, role, verify_all_sha256=verify_all_sha256
        )
    checkpoint_path: Path | None = None
    config_sha = sha256_file(config_path)
    prediction_root = run_root / "predictions" / role / f"inference_seed_{inference_seed}"
    manifest_path = prediction_root / "inference_manifest.json"
    completion_path = prediction_root / "completion.json"
    require(manifest_path.is_file(), f"Inference manifest not found: {manifest_path}")
    require(completion_path.is_file(), f"Inference completion record not found: {completion_path}")
    manifest = read_json(manifest_path)
    completion = read_json(completion_path)
    require(manifest.get("experiment") == ALIGNED_EXPERIMENT, "Inference manifest experiment mismatch")
    require(manifest.get("checkpoint_role") == role, "Inference manifest checkpoint role mismatch")
    manifest_checkpoint = Path(manifest.get("checkpoint_path", "")).expanduser().resolve()
    require(manifest_checkpoint.is_file(), f"Inference manifest checkpoint is missing: {manifest_checkpoint}")
    checkpoint_path = manifest_checkpoint
    if not smoke:
        require(checkpoint_path == Path(checkpoint["role_path"]).resolve(), "Inference manifest checkpoint path mismatch")
    else:
        require(checkpoint_path.parent == Path(checkpoint["checkpoint_dir"]).resolve(), "Smoke checkpoint is outside the run checkpoint directory")
    require(manifest.get("checkpoint_sha256") == sha256_file(checkpoint_path), "Inference manifest checkpoint SHA256 mismatch")
    require(manifest.get("config_sha256") == config_sha, "Inference manifest config SHA256 mismatch")
    require(manifest.get("benchmark_manifest_sha256") == data_contract["benchmark_manifest_sha256"], "Inference manifest benchmark identity mismatch")
    require(manifest.get("minimal_dataset_report_sha256") == data_contract["minimal_dataset_report_sha256"], "Inference manifest dataset report identity mismatch")
    require(manifest.get("target_loading") == "disabled", "Inference manifest does not record target loading disabled")
    require(int(manifest.get("inference_seed")) == int(inference_seed), "Inference seed mismatch")
    require(completion.get("experiment") == ALIGNED_EXPERIMENT, "Completion experiment mismatch")
    require(completion.get("checkpoint_sha256") == manifest.get("checkpoint_sha256"), "Completion checkpoint identity mismatch")
    require(completion.get("checkpoint_role") == role, "Completion checkpoint role mismatch")
    require(completion.get("config_sha256") == config_sha, "Completion config identity mismatch")
    require(completion.get("data_contract_sha256") == manifest.get("data_contract", {}).get("sha256"), "Completion data contract identity mismatch")
    selected = ("discrete", "continuous") if task == "all" else (task,)
    tasks = completion.get("tasks", {})
    require(isinstance(tasks, dict), "Completion record has no tasks object")
    for task_name in selected:
        record = tasks.get(task_name)
        require(isinstance(record, dict), f"Completion record has no {task_name} audit")
        if not smoke:
            require(bool(record.get("complete")), f"Completion record marks {task_name} incomplete")
    return {
        "prediction_root": str(prediction_root),
        "manifest": str(manifest_path),
        "completion": str(completion_path),
        "tasks": list(selected),
        "formal_complete": bool(completion.get("complete")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the aligned ControlAR CSGO contract without loading a model")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--eval-config", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--checkpoint-role", choices=("late", "best"), default="late")
    parser.add_argument("--inference-seed", type=int, default=42)
    parser.add_argument("--task", choices=("all", "discrete", "continuous"), default="all")
    parser.add_argument("--check-artifacts", action="store_true")
    parser.add_argument(
        "--verify-all-checkpoint-sha256",
        action="store_true",
        help="Re-hash all five multi-GB step checkpoints; intentionally optional because it is I/O heavy",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    data_root = resolve_data_root(config, args.data_root, root=ROOT).resolve()
    data_contract = check_data_contract(data_root)
    checked_source = check_source_contract()
    eval_config = project_path(args.eval_config, ROOT) if args.eval_config is not None else evaluator_root(root=ROOT) / "benchmark_v2.yaml"
    require(eval_config.is_file(), f"Shared evaluator config not found: {eval_config}")

    result: dict[str, Any] = {
        "experiment": ALIGNED_EXPERIMENT,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "data": data_contract,
        "source_files": checked_source,
        "effective_batch": int(config["world_size"]) * int(config["batch_size"]) * int(config["gradient_accumulation_steps"]),
        "optimizer_steps": int(config["max_optimizer_steps"]),
        "generation_exposure": int(config["max_optimizer_steps"]) * int(config["effective_batch_size"]),
    }
    require(result["effective_batch"] == 128, "Effective generation batch is not 128")
    require(result["generation_exposure"] == 2_496_000, "Generation exposure is not 2,496,000")
    if args.run_root is not None:
        if args.check_artifacts:
            result["artifacts"] = check_artifact_contract(
                args.run_root,
                args.checkpoint_role,
                args.inference_seed,
                config_path,
                data_contract,
                args.task,
                args.smoke,
                args.verify_all_checkpoint_sha256,
            )
        else:
            result["checkpoint"] = check_checkpoint_contract(
                args.run_root,
                args.checkpoint_role,
                verify_all_sha256=args.verify_all_checkpoint_sha256,
            )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"aligned contract check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
