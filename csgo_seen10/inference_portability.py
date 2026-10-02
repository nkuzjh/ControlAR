"""Read-only identity checks for PEFT checkpoints used on another server."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def validate_inference_data_identity(
    identity: Mapping[str, Any], data_root: Path, data_contract: Mapping[str, Any]
) -> dict[str, str]:
    """Verify the saved identity and allow only a data-root relocation."""

    saved_identity_sha256 = identity.get("identity_sha256")
    identity_bytes = json.dumps(
        {key: value for key, value in identity.items() if key != "identity_sha256"},
        sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    if saved_identity_sha256 != hashlib.sha256(identity_bytes).hexdigest():
        raise ValueError("PEFT checkpoint identity digest mismatch")

    saved_root = identity.get("data_root")
    saved_contract = identity.get("benchmark_data_contract")
    if not isinstance(saved_root, str) or not saved_root or not isinstance(saved_contract, dict):
        raise ValueError("PEFT checkpoint lacks saved data identity")
    if saved_contract.get("root") != saved_root:
        raise ValueError("PEFT checkpoint saved data root/contract mismatch")
    if data_contract.get("root") != str(data_root):
        raise ValueError("PEFT current data root/contract mismatch")
    if saved_contract.keys() != data_contract.keys() or any(
        saved_contract[key] != data_contract[key]
        for key in saved_contract if key != "root"
    ):
        raise ValueError("PEFT checkpoint benchmark data contract mismatch")
    return {
        "training_data_root": saved_root,
        "training_identity_sha256": saved_identity_sha256,
    }


def validate_manifest_checkpoint_origin(
    manifest: Mapping[str, Any], checkpoint_origin: Mapping[str, str], data_root: Path
) -> None:
    """Allow legacy same-root manifests; require provenance for relocated ones."""

    recorded = manifest.get("checkpoint_origin")
    if "checkpoint_origin" not in manifest and checkpoint_origin["training_data_root"] == str(data_root):
        return
    if recorded != dict(checkpoint_origin):
        raise ValueError("PEFT prediction checkpoint origin mismatch")
