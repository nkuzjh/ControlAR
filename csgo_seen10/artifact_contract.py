"""Artifact and identity checks for aligned CSGO Seen-10 inference.

The legacy inference path predates this module and keeps its original manifest
and resume behaviour.  The functions here are used only by the aligned
profile.  They intentionally depend on the benchmark metadata and Pillow,
but never open target FPV images.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


SCHEMA_VERSION = 2
EXPECTED_IMAGE_SIZE = (448, 448)
EXPECTED_IMAGE_FORMAT = "JPEG"
EXPECTED_IMAGE_MODE = "RGB"


class ArtifactContractError(RuntimeError):
    """Raised when inference artifacts cannot be associated safely."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes((canonical_json(value) + "\n").encode("utf-8"))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - error text is the useful part
        raise ArtifactContractError(f"Cannot read benchmark metadata: {path}") from exc


def _required_data_files(data_root: Path) -> list[Path]:
    """Return metadata files whose contents define the benchmark protocol."""

    manifest_path = data_root / "benchmark_manifest.json"
    report_path = data_root / "minimal_dataset_report.json"
    manifest = _read_json(manifest_path)
    report = _read_json(report_path)
    maps = tuple(manifest.get("protocol", {}).get("seen_maps", ()))
    if not maps:
        raise ArtifactContractError("Benchmark manifest has no Seen-10 map order")

    calibration_rel = manifest.get("calibration", {}).get(
        "file", "calibration/z_calibration.json"
    )
    calibration_path = (data_root / str(calibration_rel)).resolve()
    if data_root.resolve() not in calibration_path.parents:
        raise ArtifactContractError("Benchmark calibration path escapes data root")

    paths = [manifest_path, report_path, calibration_path]
    for map_name in maps:
        for split_name in ("train.json", "validation.json", "discrete_test.json", "continuous_clips.json"):
            paths.append(data_root / "splits" / "seen" / map_name / split_name)
    # Preserve order while rejecting accidental duplicate paths.
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.is_file():
            raise FileNotFoundError(f"Benchmark contract file not found: {resolved}")
        seen.add(resolved)
        result.append(resolved)
    # Keep the report read above as an explicit validation of JSON validity.
    if not isinstance(report, Mapping):
        raise ArtifactContractError(f"Benchmark report is not a JSON object: {report_path}")
    return result


def benchmark_data_contract(data_root: str | Path) -> dict[str, Any]:
    """Hash the manifest, calibration, split identities and protocol metadata."""

    root = Path(data_root).expanduser().resolve()
    paths = _required_data_files(root)
    files = [
        {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    by_name = {item["path"]: item["sha256"] for item in files}
    manifest_rel = "benchmark_manifest.json"
    report_rel = "minimal_dataset_report.json"
    return {
        "root": str(root),
        "files": files,
        "benchmark_manifest_sha256": by_name[manifest_rel],
        "minimal_dataset_report_sha256": by_name[report_rel],
        "sha256": sha256_json(files),
    }


def row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only manifest identity fields; image bytes are never read."""

    identity: dict[str, Any] = {
        "sample_id": str(row["sample_id"]),
        "map_name": str(row["map_name"]),
        "file_frame": str(row["file_frame"]),
    }
    for key in ("clip_id", "frame_index"):
        if key in row:
            identity[key] = row[key]
    return identity


def rows_contract(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    identities = [row_identity(row) for row in rows]
    return {
        "count": len(identities),
        "sample_ids_sha256": sha256_json([item["sample_id"] for item in identities]),
        "rows_sha256": sha256_json(identities),
    }


def expected_prediction_relpaths(
    task: str, rows: Sequence[Mapping[str, Any]]
) -> set[str]:
    expected: set[str] = set()
    for row in rows:
        relative = f"{task}/gen_imgs/{row['map_name']}/{row['file_frame']}.jpg"
        if relative in expected:
            raise ArtifactContractError(f"Duplicate output identity: {relative}")
        expected.add(relative)
    return expected


def validate_prediction_image(
    path: str | Path,
    *,
    image_size: tuple[int, int] = EXPECTED_IMAGE_SIZE,
) -> None:
    """Verify encoded JPEG bytes and the benchmark RGB/size contract."""

    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            if image.format != EXPECTED_IMAGE_FORMAT:
                raise ValueError(f"format={image.format!r}")
            if image.mode != EXPECTED_IMAGE_MODE:
                raise ValueError(f"mode={image.mode!r}")
            if image.size != image_size:
                raise ValueError(f"size={image.size!r}")
            image.load()
    except Exception as exc:
        raise ArtifactContractError(
            f"Invalid aligned prediction (expected JPEG RGB {image_size[0]}x{image_size[1]}): {image_path}"
        ) from exc


def audit_task_outputs(
    output_root: str | Path,
    task: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    image_size: tuple[int, int] = EXPECTED_IMAGE_SIZE,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Audit missing/extra/corrupt files without opening any target image."""

    root = Path(output_root).expanduser().resolve()
    expected = expected_prediction_relpaths(task, rows)
    gen_root = root / task / "gen_imgs"
    actual: set[str] = set()
    if gen_root.exists():
        if not gen_root.is_dir():
            raise ArtifactContractError(f"Prediction root is not a directory: {gen_root}")
        for path in gen_root.rglob("*"):
            if path.is_symlink():
                raise ArtifactContractError(
                    f"Aligned prediction tree must not contain symlinks: {path}"
                )
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if extra:
        raise ArtifactContractError(
            f"Aligned inference has unexpected files for task={task}: {extra[:6]}"
        )
    for relative in sorted(actual):
        validate_prediction_image(root / relative, image_size=image_size)
    if require_complete and missing:
        raise ArtifactContractError(
            f"Aligned inference is incomplete for task={task}: missing={len(missing)} {missing[:6]}"
        )
    return {
        "task": task,
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing": missing,
        "extra": extra,
        "complete": not missing and not extra,
        "rows": rows_contract(rows),
    }


def write_completion(
    output_root: str | Path,
    *,
    experiment: str,
    config_sha256: str,
    checkpoint_sha256: str,
    checkpoint_role: str,
    vq_checkpoint_sha256: str,
    data_contract_sha256: str,
    seed: int,
    inference_seed: int,
    selected_tasks: Sequence[str],
    task_audit: Mapping[str, Any],
    formal: bool,
) -> Path:
    """Atomically update the root completion record after a task audit."""

    root = Path(output_root).expanduser().resolve()
    path = root / "completion.json"
    previous: dict[str, Any] = {}
    if path.exists():
        previous = _read_json(path)
        if not isinstance(previous, dict):
            raise ArtifactContractError(f"Existing completion record is not an object: {path}")
        for key, value in {
            "schema_version": SCHEMA_VERSION,
            "experiment": experiment,
            "config_sha256": config_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_role": checkpoint_role,
            "vq_checkpoint_sha256": vq_checkpoint_sha256,
            "data_contract_sha256": data_contract_sha256,
            "seed": int(seed),
            "inference_seed": int(inference_seed),
        }.items():
            if previous.get(key) != value:
                raise ArtifactContractError(
                    f"Existing completion record has a different {key}; refusing to mix artifacts"
                )
    previous.update(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment": experiment,
            "config_sha256": config_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_role": checkpoint_role,
            "vq_checkpoint_sha256": vq_checkpoint_sha256,
            "data_contract_sha256": data_contract_sha256,
            "seed": int(seed),
            "inference_seed": int(inference_seed),
            "formal": bool(formal),
        }
    )
    tasks = dict(previous.get("tasks", {}))
    tasks[str(task_audit["task"])] = dict(task_audit)
    previous["tasks"] = tasks
    previous_selected = set(str(task) for task in previous.get("selected_tasks", []))
    previous_tasks = set(str(task) for task in tasks)
    requested_tasks = set(str(task) for task in selected_tasks)
    previous["selected_tasks"] = sorted(previous_selected | previous_tasks | requested_tasks)
    previous["complete"] = all(
        bool(tasks.get(task, {}).get("complete", False))
        for task in previous["selected_tasks"]
    )
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(previous, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)
    return path


def _load_object(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ArtifactContractError(f"Expected a JSON object: {path}")
    return value


def _expected_task_count(task: str) -> int:
    if task == "discrete":
        return 20_000
    if task == "continuous":
        return 12_800
    raise ArtifactContractError(f"Unsupported evaluation task: {task}")


def preflight_evaluation(
    output_root: str | Path,
    task: str,
    checkpoint_path: str | Path,
    expected_role: str,
    expected_config_sha256: str,
    data_root: str | Path,
) -> dict[str, Any]:
    """Check an aligned prediction root before invoking the shared evaluator.

    The check never opens benchmark target FPV images.  It re-hashes the current
    manifest/split/calibration files and re-audits every prediction JPEG so a
    stale completion record cannot authorize evaluation of changed artifacts.
    """

    if expected_role not in ("late", "best"):
        raise ArtifactContractError("expected_role must be late or best")
    if task not in ("discrete", "continuous"):
        raise ArtifactContractError(f"Unsupported evaluation task: {task}")
    root = Path(output_root).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    current_data_root = Path(data_root).expanduser().resolve()
    current_data_contract = benchmark_data_contract(current_data_root)
    manifest = _load_object(root / "inference_manifest.json")
    completion = _load_object(root / "completion.json")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("manifest_version") != 2:
        raise ArtifactContractError("Evaluation requires aligned inference manifest schema v2")
    if manifest.get("experiment") != "csgo_seen10_exp32gen_aligned":
        raise ArtifactContractError("Evaluation root is not the aligned ControlAR experiment")
    if manifest.get("checkpoint_role") != expected_role:
        raise ArtifactContractError("Manifest checkpoint role does not match evaluation role")
    if manifest.get("config_sha256") != expected_config_sha256:
        raise ArtifactContractError("Manifest config identity does not match the requested config")
    if manifest.get("checkpoint_path") != str(checkpoint):
        raise ArtifactContractError("Manifest checkpoint path does not match the requested checkpoint")
    if checkpoint.name != f"{expected_role}.pt":
        raise ArtifactContractError(
            f"Aligned evaluation checkpoint must be named {expected_role}.pt, got {checkpoint.name}"
        )
    if not checkpoint.is_file():
        raise ArtifactContractError(f"Checkpoint file does not exist: {checkpoint}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if manifest.get("checkpoint_sha256") != checkpoint_sha256:
        raise ArtifactContractError("Manifest checkpoint SHA256 does not match the checkpoint file")
    data_contract = manifest.get("data_contract")
    if not isinstance(data_contract, Mapping) or not data_contract.get("sha256"):
        raise ArtifactContractError("Manifest has no data contract SHA256")
    if manifest.get("data_root") != str(current_data_root):
        raise ArtifactContractError(
            "Manifest data root does not match the data root passed to the evaluator"
        )
    if dict(data_contract) != current_data_contract:
        raise ArtifactContractError(
            "Current benchmark manifest/split/calibration contract differs from inference"
        )
    data_contract_sha256 = str(data_contract["sha256"])
    if manifest.get("data_contract_sha256") != data_contract_sha256:
        raise ArtifactContractError("Manifest data contract SHA256 field is inconsistent")
    if task not in set(str(value) for value in manifest.get("tasks", [])):
        raise ArtifactContractError(f"Task {task} is absent from inference_manifest.json")

    if completion.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactContractError("Completion record is not schema v2")
    if completion.get("formal") is not True:
        raise ArtifactContractError("Evaluation requires formal completion")
    for key, expected in {
        "experiment": manifest.get("experiment"),
        "config_sha256": expected_config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_role": expected_role,
        "vq_checkpoint_sha256": manifest.get("vq_checkpoint_sha256"),
        "data_contract_sha256": data_contract_sha256,
        "seed": manifest.get("seed"),
        "inference_seed": manifest.get("inference_seed"),
    }.items():
        if completion.get(key) != expected:
            raise ArtifactContractError(
                f"Completion {key} does not match the inference manifest/checkpoint"
            )
    selected_tasks = set(str(value) for value in completion.get("selected_tasks", []))
    if task not in selected_tasks:
        raise ArtifactContractError(f"Task {task} is absent from completion selected_tasks")
    if completion.get("complete") is not True:
        raise ArtifactContractError("Completion record does not cover all selected tasks")
    completion_tasks = completion.get("tasks", {})
    if not isinstance(completion_tasks, Mapping):
        raise ArtifactContractError("Completion record has no task records")
    task_record = completion_tasks.get(task)
    if not isinstance(task_record, Mapping) or task_record.get("complete") is not True:
        raise ArtifactContractError(f"Task {task} is not complete in completion.json")

    expected_count = _expected_task_count(task)
    manifest_task = manifest.get("task_rows", {}).get(task)
    if not isinstance(manifest_task, Mapping) or int(manifest_task.get("count", -1)) != expected_count:
        raise ArtifactContractError(f"Manifest task_rows count is wrong for {task}")
    completion_rows = task_record.get("rows")
    if not isinstance(completion_rows, Mapping) or int(completion_rows.get("count", -1)) != expected_count:
        raise ArtifactContractError(f"Completion task row count is wrong for {task}")
    if int(task_record.get("expected_count", -1)) != expected_count:
        raise ArtifactContractError(f"Completion artifact count is wrong for {task}")
    if int(task_record.get("actual_count", -1)) != expected_count:
        raise ArtifactContractError(f"Completion actual artifact count is wrong for {task}")

    try:
        from csgo_benchmark_v2_eval.protocol import BenchmarkData
    except ImportError as exc:
        raise ArtifactContractError(
            "Shared benchmark protocol is unavailable for the live evaluation preflight"
        ) from exc
    benchmark_manifest = _load_object(current_data_root / "benchmark_manifest.json")
    maps = benchmark_manifest.get("protocol", {}).get("seen_maps")
    if not isinstance(maps, list) or not maps:
        raise ArtifactContractError("Current benchmark manifest has no Seen-10 map order")
    split = "seen_discrete_test" if task == "discrete" else "seen_continuous"
    rows = BenchmarkData(str(current_data_root)).rows(split, maps=maps)
    live_audit = audit_task_outputs(
        root,
        task,
        rows,
        image_size=EXPECTED_IMAGE_SIZE,
        require_complete=True,
    )
    if live_audit["rows"] != dict(manifest_task):
        raise ArtifactContractError(
            f"Live ordered sample identity differs from inference manifest for {task}"
        )
    if live_audit["rows"] != dict(completion_rows):
        raise ArtifactContractError(
            f"Live ordered sample identity differs from completion record for {task}"
        )
    for key in ("expected_count", "actual_count", "complete"):
        if live_audit[key] != task_record.get(key):
            raise ArtifactContractError(
                f"Live prediction audit {key} differs from completion record for {task}"
            )

    index = _load_object(checkpoint.parent / "checkpoint_index.json")
    records = index.get("checkpoints")
    if not isinstance(records, list):
        raise ArtifactContractError("Checkpoint index has no checkpoint records")
    if expected_role == "late":
        if int(index.get("late_step", -1)) != 19_500:
            raise ArtifactContractError("Checkpoint index late_step is not 19500")
        expected_record = next((r for r in records if int(r.get("step", -1)) == 19_500), None)
    else:
        best_step = int(index.get("best_step", -1))
        expected_record = next((r for r in records if int(r.get("step", -1)) == best_step), None)
    if not isinstance(expected_record, Mapping):
        raise ArtifactContractError(f"Checkpoint index has no {expected_role} alias record")
    indexed_path = checkpoint.parent / str(expected_record.get("path", ""))
    if not indexed_path.is_file() or not os.path.samefile(checkpoint, indexed_path):
        raise ArtifactContractError(f"{expected_role}.pt is not the indexed checkpoint alias")
    if expected_record.get("sha256") != checkpoint_sha256:
        raise ArtifactContractError(f"Checkpoint index SHA256 does not match {expected_role}.pt")
    return {
        "manifest": manifest,
        "completion": completion,
        "checkpoint_sha256": checkpoint_sha256,
        "data_contract_sha256": data_contract_sha256,
        "live_audit": live_audit,
        "task": task,
        "expected_count": expected_count,
    }
