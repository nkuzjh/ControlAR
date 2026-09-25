"""Portable path precedence and legacy fallback checks; no model imports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from csgo_seen10 import paths


class Seen10PathsTests(unittest.TestCase):
    def test_data_precedence_and_relative_checkout_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ControlAR"
            config = {"data_root": "data/from-config"}
            env = {
                "CSGO_DATA_ROOT": "data/primary",
                "CSGO_BENCHMARK_V2_DATA": "data/compat",
                "DATA_ROOT": "data/generic",
            }
            self.assertEqual(paths.data_root(config, "data/cli", root=root, env=env), root / "data/cli")
            self.assertEqual(paths.data_root(config, root=root, env=env), root / "data/primary")
            del env["CSGO_DATA_ROOT"]
            self.assertEqual(paths.data_root(config, root=root, env=env), root / "data/compat")
            del env["CSGO_BENCHMARK_V2_DATA"]
            self.assertEqual(paths.data_root(config, root=root, env=env), root / "data/generic")
            self.assertEqual(paths.data_root(config, root=root, env={}), root / "data/from-config")

    def test_only_missing_original_data_default_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ControlAR"
            old = Path(directory) / "old" / "csgo_benchmark_v2"
            sibling = root.parent / "UniLIP/data/csgo_benchmark_v2"
            with patch.object(paths, "LEGACY_DATA_ROOT", str(old)):
                self.assertEqual(paths.data_root({"data_root": str(old)}, root=root, env={}), sibling)
                self.assertEqual(paths.data_root({"data_root": str(old) + "-custom"}, root=root, env={}), Path(str(old) + "-custom"))
                self.assertEqual(paths.data_root({"data_root": str(old)}, str(old), root=root, env={}), old)
                old.mkdir(parents=True)
                self.assertEqual(paths.data_root({"data_root": str(old)}, root=root, env={}), old)

    def test_evaluator_root_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ControlAR"
            old = Path(directory) / "legacy-eval"
            with patch.object(paths, "LEGACY_EVAL_ROOT", str(old)):
                self.assertEqual(paths.evaluator_root({"shared_eval_dir": str(old)}, root=root, env={}), root.parent / "csgo_benchmark_v2_eval_general")
                old.mkdir()
                self.assertEqual(paths.evaluator_root({}, root=root, env={}), root.parent / "csgo_benchmark_v2_eval_general")
                self.assertEqual(paths.evaluator_root({"shared_eval_dir": str(old)}, root=root, env={}), old)
                self.assertEqual(paths.evaluator_root({}, root=root, env={"CSGO_EVAL_ROOT": "secondary"}), root / "secondary")
                self.assertEqual(paths.evaluator_root({}, root=root, env={"SHARED_EVAL_DIR": "primary", "CSGO_EVAL_ROOT": "secondary"}), root / "primary")
                self.assertEqual(paths.evaluator_root({}, "explicit", root=root, env={"SHARED_EVAL_DIR": "primary"}), root / "explicit")

    def test_evaluator_python_precedence_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ControlAR"
            old = Path(directory) / "old-python"
            venv = root / ".venv" / "bin" / "python"
            project_eval = root / ".venv-eval" / "bin" / "python"
            venv.parent.mkdir(parents=True)
            venv.symlink_to("/usr/bin/python3")
            with patch.object(paths, "LEGACY_EVAL_PYTHON", str(old)):
                self.assertEqual(paths.evaluator_python(root=root, env={}), venv)
                old.touch()
                self.assertEqual(paths.evaluator_python(root=root, env={}), old)
                project_eval.parent.mkdir(parents=True)
                project_eval.symlink_to("/usr/bin/python3")
                self.assertEqual(paths.evaluator_python(root=root, env={}), project_eval)
                self.assertEqual(paths.evaluator_python({"unilip_python": "custom/python"}, root=root, env={}), root / "custom/python")
                self.assertEqual(paths.evaluator_python({}, root=root, env={"UNILIP_PYTHON": "env/python"}), root / "env/python")
                self.assertEqual(paths.evaluator_python({}, root=root, env={"EVAL_PYTHON": "preferred", "UNILIP_PYTHON": "secondary"}), root / "preferred")
                self.assertEqual(paths.evaluator_python({}, "explicit", root=root, env={"EVAL_PYTHON": "preferred"}), root / "explicit")


if __name__ == "__main__":
    unittest.main()
