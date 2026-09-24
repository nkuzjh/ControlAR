"""Checkpoint and prediction identity for the independent Seen-10 PEFT profile."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Any

from csgo_seen10.artifact_contract import (
    SCHEMA_VERSION,
    benchmark_data_contract,
    rows_contract,
    sha256_file,
)


EXPERIMENT = "csgo_seen10_exp32gen_aligned_peft"
FORMAT = f"{EXPERIMENT}_v1"


def validate_peft_checkpoint(
    checkpoint_path: Path,
    payload: dict[str, Any],
    *,
    checkpoint_role: str,
    config_path: Path,
    config: dict[str, Any],
    data_root: Path,
    data_contract: dict[str, Any],
    smoke: bool,
) -> str:
    """Check the saved training identity and indexed best/late alias."""

    checkpoint_path = checkpoint_path.resolve()
    if checkpoint_role not in ("late", "best") or checkpoint_path.name != f"{checkpoint_role}.pt":
        raise ValueError("PEFT inference requires the selected best.pt or late.pt alias")
    if payload.get("format") != FORMAT:
        raise ValueError("Checkpoint is not the PEFT training format")
    args = payload.get("args")
    identity = payload.get("identity")
    training = payload.get("training_config")
    if not all(isinstance(value, dict) for value in (args, identity, training)):
        raise ValueError("PEFT checkpoint lacks args, identity, or training configuration")
    if args.get("experiment") != EXPERIMENT or config.get("experiment") != EXPERIMENT:
        raise ValueError("PEFT checkpoint experiment identity mismatch")
    if int(args.get("seed", -1)) != int(config["seed"]):
        raise ValueError("PEFT checkpoint training seed mismatch")
    if bool(args.get("smoke", False)) != smoke:
        raise ValueError("PEFT checkpoint smoke/formal identity mismatch")
    sampler = payload.get("sampler_state")
    if not isinstance(sampler, dict) or int(sampler.get("seed", -1)) != int(config["seed"]):
        raise ValueError("PEFT checkpoint sampler seed mismatch")
    files = identity.get("files")
    if not isinstance(files, dict):
        raise ValueError("PEFT checkpoint lacks identity file hashes")
    expected_files = {
        "config": sha256_file(config_path),
        "manifest": data_contract["benchmark_manifest_sha256"],
        "report": data_contract["minimal_dataset_report_sha256"],
        "official_gpt": str(config["official_gpt_sha256"]),
        "vq": str(config["vq_sha256"]),
    }
    for name, expected in expected_files.items():
        if files.get(name) != expected:
            raise ValueError(f"PEFT checkpoint {name} identity hash mismatch")
    if identity.get("data_root") != str(data_root):
        raise ValueError("PEFT checkpoint data root mismatch")
    if identity.get("experiment") != EXPERIMENT:
        raise ValueError("PEFT checkpoint identity experiment mismatch")
    if identity.get("benchmark_data_contract") != data_contract:
        raise ValueError("PEFT checkpoint benchmark data contract mismatch")
    identity_copy = dict(identity)
    saved_identity_sha256 = identity_copy.pop("identity_sha256", None)
    # Training identity uses compact JSON without a trailing newline; the
    # artifact helper hashes newline-terminated JSON for a different contract.
    identity_bytes = json.dumps(identity_copy, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    if saved_identity_sha256 != hashlib.sha256(identity_bytes).hexdigest():
        raise ValueError("PEFT checkpoint identity digest mismatch")
    code = files.get("code")
    if not isinstance(code, dict) or "csgo_seen10/peft.py" not in code:
        raise ValueError("PEFT checkpoint lacks LoRA implementation identity")
    peft_source = Path(__file__).resolve().parent / "peft.py"
    if code["csgo_seen10/peft.py"] != sha256_file(peft_source):
        raise ValueError("PEFT checkpoint LoRA implementation identity mismatch")

    step = int(payload.get("steps", -1))
    if step < 1 or int(payload.get("global_optimizer_step", -1)) != step:
        raise ValueError("PEFT checkpoint optimizer step mismatch")
    effective = int(training.get("effective_batch_size", -1))
    if effective != 128 and not smoke:
        raise ValueError("Formal PEFT checkpoint effective batch must be 128")
    if effective < 1 or int(payload.get("consumed_samples", -1)) != step * effective:
        raise ValueError("PEFT checkpoint consumed samples mismatch")
    micro_batch = int(training.get("batch_size", -1))
    accumulation = int(training.get("gradient_accumulation_steps", -1))
    world_size = int(training.get("world_size", -1))
    if min(micro_batch, accumulation, world_size) < 1 or world_size * micro_batch * accumulation != effective:
        raise ValueError("PEFT checkpoint effective batch decomposition mismatch")
    if any(int(sampler.get(key, -1)) != expected for key, expected in {
        "num_replicas": world_size,
        "micro_batch_per_device": micro_batch,
        "gradient_accumulation_steps": accumulation,
    }.items()):
        raise ValueError("PEFT checkpoint sampler batch decomposition mismatch")
    if int(training.get("seed", -1)) != int(config["seed"]):
        raise ValueError("PEFT checkpoint training seed configuration mismatch")
    saved_steps = (
        tuple(range(1, int(training.get("smoke_steps", 0)) + 1))
        if smoke else tuple(int(value) for value in training.get("checkpoint_steps", ()))
    )
    formal_steps = tuple(int(value) for value in config["checkpoint_steps"])
    if not saved_steps or len(set(saved_steps)) != len(saved_steps) or tuple(sorted(saved_steps)) != saved_steps:
        raise ValueError("PEFT checkpoint milestone schedule is invalid")
    if not smoke and saved_steps != formal_steps:
        raise ValueError("Formal PEFT checkpoint milestone schedule mismatch")
    maximum = saved_steps[-1]
    if int(training.get("max_optimizer_steps", -1)) != int(config["max_optimizer_steps"]):
        raise ValueError("PEFT checkpoint training budget mismatch")
    if step not in saved_steps:
        raise ValueError("PEFT checkpoint step is not a milestone")
    if checkpoint_role == "late":
        if step != maximum:
            raise ValueError("PEFT late.pt is not the final optimizer step")
    elif int(payload.get("best_step", -1)) != step:
        raise ValueError("PEFT best.pt does not contain its best step")
    model_config = payload.get("model_config")
    peft = model_config.get("peft") if isinstance(model_config, dict) else None
    expected_peft = {
        "rank": int(config["lora_rank"]),
        "alpha": int(config["lora_alpha"]),
        "dropout": float(config["lora_dropout"]),
        "qkv_independent": True,
    }
    if peft != expected_peft:
        raise ValueError(f"PEFT checkpoint LoRA model configuration mismatch: {peft!r}")

    index_path = checkpoint_path.parent / "checkpoint_index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot read PEFT checkpoint index: {index_path}") from exc
    records = index.get("checkpoints") if isinstance(index, dict) else None
    if not isinstance(records, list) or len(records) != len(saved_steps):
        raise ValueError("PEFT checkpoint index has an incomplete milestone record")
    indexed_steps = tuple(sorted(int(record.get("step", -1)) for record in records))
    if indexed_steps != saved_steps or sum(bool(record.get("is_best")) for record in records) != 1:
        raise ValueError("PEFT checkpoint index milestone/best record mismatch")
    if int(index.get("late_step", -1)) != maximum:
        raise ValueError("PEFT checkpoint index final step mismatch")
    best_record = next(record for record in records if record.get("is_best"))
    if int(index.get("best_step", -1)) != int(best_record["step"]):
        raise ValueError("PEFT checkpoint index best step mismatch")
    selected = next(record for record in records if int(record["step"]) == step)
    if checkpoint_role == "best" and selected is not best_record:
        raise ValueError("PEFT best.pt does not match the indexed best record")
    if selected.get("path") != f"step_{step:06d}.pt":
        raise ValueError("PEFT checkpoint index has a noncanonical milestone path")
    milestone_path = checkpoint_path.parent / selected["path"]
    if not milestone_path.is_file() or not os.path.samefile(checkpoint_path, milestone_path):
        raise ValueError("PEFT checkpoint alias differs from its indexed milestone")
    digest = sha256_file(checkpoint_path)
    if selected.get("sha256") != digest:
        raise ValueError("PEFT checkpoint index SHA256 mismatch")
    return digest


def ensure_peft_output_manifest(output_root: Path, *, task: str, rows: list[dict[str, Any]], **identity: Any) -> None:
    """Only accumulate task rows when every other prediction identity matches."""

    if identity.get("experiment") != EXPERIMENT:
        raise ValueError("PEFT prediction manifest experiment mismatch")
    if identity.get("checkpoint_role") not in ("best", "late"):
        raise ValueError("PEFT prediction manifest role mismatch")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "manifest_version": 1,
        "benchmark_id": "csgo_benchmark_v2",
        "model_name": "ControlAR",
        "target_loading": "disabled",
        "output_size": [448, 448],
        "encoding": {"format": "JPEG", "mode": "RGB", "size": [448, 448], "pillow": "default JPEG encoder options"},
        **identity,
        "tasks": [],
        "task_rows": {},
    }
    manifest_path = output_root / "inference_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(previous, dict):
            raise ValueError("Existing PEFT inference manifest is invalid")
        for key, value in expected.items():
            if key not in ("tasks", "task_rows") and previous.get(key) != value:
                raise ValueError(f"Existing PEFT prediction identity mismatch: {key}")
        prior_rows = previous.get("task_rows")
        if not isinstance(prior_rows, dict):
            raise ValueError("Existing PEFT inference manifest lacks task rows")
        row_contract = rows_contract(rows)
        if task in prior_rows and prior_rows[task] != row_contract:
            raise ValueError("Existing PEFT prediction row identity mismatch")
        previous["tasks"] = sorted(set(previous.get("tasks", [])) | {task})
        previous["task_rows"] = {**prior_rows, task: row_contract}
        expected = previous
    else:
        if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
            raise ValueError("PEFT output root already contains files without an inference manifest")
        expected["tasks"] = [task]
        expected["task_rows"] = {task: rows_contract(rows)}
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(manifest_path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)


def preflight_peft_inference(
    output_root: str | Path,
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_role: str,
    data_root: str | Path,
    task: str,
    max_samples: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Read-only, task-specific prediction audit for a PEFT run."""

    import torch
    from csgo_seen10.artifact_contract import audit_task_outputs
    from csgo_seen10.data import read_benchmark_rows

    config_path = Path(config_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment") != EXPERIMENT:
        raise ValueError("PEFT preflight requires the PEFT profile")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    digest = validate_peft_checkpoint(
        checkpoint_path, payload, checkpoint_role=checkpoint_role,
        config_path=config_path, config=config, data_root=data_root,
        data_contract=benchmark_data_contract(data_root), smoke=smoke,
    )
    rows = read_benchmark_rows(data_root, {"discrete": "seen_discrete_test", "continuous": "seen_continuous"}[task], max_samples=max_samples, require_images=False)
    audit = audit_task_outputs(output_root, task, rows, require_complete=not smoke)
    manifest_path = Path(output_root).expanduser().resolve() / "inference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "experiment": EXPERIMENT,
        "checkpoint_role": checkpoint_role,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": digest,
        "data_root": str(data_root),
        "data_contract": benchmark_data_contract(data_root),
        "seed": int(config["seed"]),
        "inference_seed": int(config["inference_seed"]),
        "smoke_only": bool(smoke),
        "max_samples": max_samples,
        "sampling": {key: config[key] for key in ("cfg_scale", "temperature", "top_k", "top_p")},
        "inference": {
            "engine": "compiled", "batch_size": 16, "compile_mode": "reduce-overhead",
            "batching": "fixed_manifest_blocks",
            "seed_policy": "stateless-sample-v1: SHA256(UTF-8 bytes of str(inference_seed) + NUL + sample_id + NUL + decimal token index), first 64 bits mapped to (0, 1)",
            "target_loading": "disabled", "peft_merge": "temporary_inference_model",
            "sampling_backend": "compiled_logits+cuda_aten_fp32_inverse_cdf",
        },
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise ValueError(f"PEFT inference manifest identity mismatch: {key}")
    if manifest.get("task_rows", {}).get(task) != rows_contract(rows):
        raise ValueError("PEFT inference manifest task rows mismatch")
    return {"experiment": EXPERIMENT, "checkpoint_role": checkpoint_role, "checkpoint_sha256": digest, "task_audit": audit}


def preflight_evaluation(
    output_root: str | Path,
    task: str,
    checkpoint_path: str | Path,
    expected_role: str,
    expected_config_sha256: str,
    data_root: str | Path,
) -> dict[str, Any]:
    """Audit a complete formal PEFT prediction root before shared evaluation."""

    from csgo_seen10.artifact_contract import audit_task_outputs
    from csgo_seen10.data import read_benchmark_rows

    if task not in ("discrete", "continuous") or expected_role not in ("late", "best"):
        raise ValueError("Invalid PEFT evaluation task or checkpoint role")
    root = Path(output_root).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    manifest = json.loads((root / "inference_manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    config_path = Path(__file__).resolve().parents[1] / "configs" / f"{EXPERIMENT}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256_file(config_path) != expected_config_sha256:
        raise ValueError("PEFT evaluation canonical config SHA256 mismatch")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("manifest_version") != 1:
        raise ValueError("PEFT inference manifest schema mismatch")
    if manifest.get("experiment") != EXPERIMENT or completion.get("experiment") != EXPERIMENT:
        raise ValueError("Evaluation root does not belong to the PEFT experiment")
    if manifest.get("smoke_only") is not False or completion.get("formal") is not True:
        raise ValueError("PEFT evaluation requires a complete formal prediction root")
    if manifest.get("max_samples") is not None:
        raise ValueError("PEFT formal evaluation cannot use max_samples")
    if checkpoint.name != f"{expected_role}.pt" or not checkpoint.is_file():
        raise ValueError("Selected PEFT evaluation checkpoint alias is invalid")
    checkpoint_sha256 = sha256_file(checkpoint)
    current_contract = benchmark_data_contract(data_root)
    expected = {
        "checkpoint_role": expected_role,
        "config_sha256": expected_config_sha256,
        "config_path": str(config_path.resolve()),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "data_root": str(data_root),
        "data_contract": current_contract,
        "data_contract_sha256": current_contract["sha256"],
        "target_loading": "disabled",
        "image_size": 448,
        "output_size": [448, 448],
        "seed": int(config["seed"]),
        "inference_seed": int(config["inference_seed"]),
        "sampling": {key: config[key] for key in ("cfg_scale", "temperature", "top_k", "top_p")},
        "inference": {
            "engine": "compiled", "batch_size": 16, "compile_mode": "reduce-overhead",
            "batching": "fixed_manifest_blocks",
            "seed_policy": "stateless-sample-v1: SHA256(UTF-8 bytes of str(inference_seed) + NUL + sample_id + NUL + decimal token index), first 64 bits mapped to (0, 1)",
            "target_loading": "disabled", "peft_merge": "temporary_inference_model",
            "sampling_backend": "compiled_logits+cuda_aten_fp32_inverse_cdf",
        },
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"PEFT evaluation manifest identity mismatch: {key}")
    for key in ("checkpoint_role", "config_sha256", "checkpoint_sha256", "data_contract_sha256", "vq_checkpoint_sha256", "seed", "inference_seed"):
        if completion.get(key) != manifest.get(key):
            raise ValueError(f"PEFT evaluation completion identity mismatch: {key}")
    if task not in manifest.get("tasks", ()) or task not in completion.get("selected_tasks", ()):
        raise ValueError("PEFT evaluation task is absent from prediction records")
    if completion.get("complete") is not True:
        raise ValueError("PEFT evaluation selected tasks are incomplete")
    split = "seen_discrete_test" if task == "discrete" else "seen_continuous"
    rows = read_benchmark_rows(data_root, split, max_samples=None, require_images=False)
    expected_count = 20_000 if task == "discrete" else 12_800
    if len(rows) != expected_count:
        raise ValueError("PEFT evaluation benchmark split count mismatch")
    audit = audit_task_outputs(root, task, rows, require_complete=True)
    if manifest.get("task_rows", {}).get(task) != audit["rows"]:
        raise ValueError("PEFT evaluation manifest row identity mismatch")
    recorded = completion.get("tasks", {}).get(task)
    if not isinstance(recorded, dict) or any(recorded.get(key) != audit[key] for key in ("rows", "expected_count", "actual_count", "complete")):
        raise ValueError("PEFT evaluation completion audit mismatch")
    index = json.loads((checkpoint.parent / "checkpoint_index.json").read_text(encoding="utf-8"))
    records = index.get("checkpoints", ())
    if tuple(sorted(int(record.get("step", -1)) for record in records)) != (3900, 7800, 11700, 15600, 19500):
        raise ValueError("PEFT evaluation checkpoint index milestones mismatch")
    selected_step = 19500 if expected_role == "late" else int(index.get("best_step", -1))
    if int(index.get("late_step", -1)) != 19500:
        raise ValueError("PEFT evaluation index has no final checkpoint")
    selected = next((record for record in records if int(record.get("step", -1)) == selected_step), None)
    if not isinstance(selected, dict) or selected.get("path") != f"step_{selected_step:06d}.pt":
        raise ValueError("PEFT evaluation checkpoint index alias mismatch")
    milestone = checkpoint.parent / selected["path"]
    if not milestone.is_file() or not os.path.samefile(checkpoint, milestone) or selected.get("sha256") != checkpoint_sha256:
        raise ValueError("PEFT evaluation checkpoint alias/hash mismatch")
    return {"manifest": manifest, "completion": completion, "checkpoint_sha256": checkpoint_sha256, "task_audit": audit}
