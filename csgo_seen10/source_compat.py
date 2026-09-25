"""Audited, explicit source transition for same-path legacy checkpoint resume."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

LEDGER = Path(__file__).with_name("legacy_source_resume.json")


def _digest(identity: Mapping[str, Any]) -> str:
    content = {key: value for key, value in identity.items() if key != "identity_sha256"}
    payload = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def check_resume_identity(
    saved: Mapping[str, Any], current: Mapping[str, Any], *,
    profile: str, allow_legacy: bool, ledger_path: Path = LEDGER,
) -> dict[str, Any] | None:
    """Require equality, or one pinned old-to-new source transition.

    Return an audit record only for an explicitly allowed transition. No
    checkpoint identity or recorded source hash is modified.
    """
    if saved == current:
        return None
    if not allow_legacy:
        raise ValueError("Resume identity mismatch; --allow-legacy-source-resume is required for an audited source transition")
    if saved.get("identity_sha256") != _digest(saved):
        raise ValueError("Saved resume identity digest mismatch")
    if current.get("identity_sha256") != _digest(current):
        raise ValueError("Current resume identity digest mismatch")
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        approved = ledger["profiles"][profile]
        old_code = saved["files"]["code"]
        new_code = current["files"]["code"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing approved source transition for {profile}") from exc
    if not isinstance(old_code, dict) or old_code != approved.get("source"):
        raise ValueError("Saved source SHA tuple is not the approved legacy source")
    if not isinstance(new_code, dict) or new_code != approved.get("target"):
        raise ValueError("Current source SHA tuple is not the approved migrated source")

    def without_code(identity: Mapping[str, Any]) -> dict[str, Any]:
        result = {key: value for key, value in identity.items() if key not in ("identity_sha256", "files")}
        result["files"] = {key: value for key, value in identity["files"].items() if key != "code"}
        return result

    if without_code(saved) != without_code(current):
        raise ValueError("Resume data, path, config, weights, or split identity differs")
    return {
        "allow_legacy_source_resume": True,
        "profile": profile,
        "saved_identity_sha256": saved["identity_sha256"],
        "current_identity_sha256": current["identity_sha256"],
        "saved_source_sha256": old_code,
        "current_source_sha256": new_code,
        "approved_ledger": str(ledger_path),
    }
