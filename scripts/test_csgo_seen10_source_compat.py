"""CPU-only checks for audited source transitions on exact resume."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from csgo_seen10.source_compat import _digest, check_resume_identity


def identity(code: dict[str, str], *, data_root: str = "/old/data", recipe: str = "canonical") -> dict:
    value = {"data_root": data_root, "files": {"code": code, "config": recipe, "manifest": "fixed"}, "train_count": 50_000}
    value["identity_sha256"] = _digest(value)
    return value


class SourceCompatTests(unittest.TestCase):
    def test_strict_and_pinned_transition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "transition.json"
            old_code = {"train_seen10.py": "old", "model.py": "unchanged"}
            new_code = {"train_seen10.py": "new", "model.py": "unchanged", "paths.py": "path-source"}
            ledger.write_text(json.dumps({"profiles": {"aligned": {"source": old_code, "target": new_code}}}))
            saved = identity(old_code)
            current = identity(new_code)
            self.assertIsNone(check_resume_identity(current, current, profile="aligned", allow_legacy=False, ledger_path=ledger))
            with self.assertRaisesRegex(ValueError, "--allow-legacy-source-resume"):
                check_resume_identity(saved, current, profile="aligned", allow_legacy=False, ledger_path=ledger)
            audit = check_resume_identity(saved, current, profile="aligned", allow_legacy=True, ledger_path=ledger)
            self.assertEqual(audit["saved_source_sha256"], old_code)
            self.assertEqual(audit["current_source_sha256"], new_code)

            for changed in (
                identity(new_code, data_root="/new/data"),
                identity(new_code, recipe="changed"),
                identity({**new_code, "model.py": "unreviewed"}),
            ):
                with self.assertRaises(ValueError):
                    check_resume_identity(saved, changed, profile="aligned", allow_legacy=True, ledger_path=ledger)
            tampered = copy.deepcopy(saved)
            tampered["files"]["manifest"] = "tampered"
            with self.assertRaisesRegex(ValueError, "digest"):
                check_resume_identity(tampered, current, profile="aligned", allow_legacy=True, ledger_path=ledger)


if __name__ == "__main__":
    unittest.main()
