"""Exercise CLI parsing through runtime path resolution before GPU setup."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import infer_seen10
import train_seen10
import train_seen10_peft


class ResolvedPath(Exception):
    pass


class EntryPathTests(unittest.TestCase):
    def _check(self, module, argv: list[str], expected: Path, *, explicit: str | None) -> None:
        actual_resolver = module.resolve_data_root

        def capture(config, value, *, root):
            self.assertEqual(value, explicit)
            raise ResolvedPath(actual_resolver(config, value, root=root).resolve())

        with patch.object(module, "resolve_data_root", side_effect=capture), patch.object(sys, "argv", argv):
            with self.assertRaises(ResolvedPath) as caught:
                if module is train_seen10_peft:
                    module.train(module.parse_args())
                else:
                    module.main()
        self.assertEqual(caught.exception.args[0], expected)

    def test_train_infer_and_peft_env_reaches_resolver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory) / "benchmark"
            with patch.dict(os.environ, {"CSGO_DATA_ROOT": str(selected)}):
                self._check(train_seen10, ["train_seen10.py", "--experiment", train_seen10.ALIGNED_EXPERIMENT], selected, explicit=None)
                self._check(infer_seen10, ["infer_seen10.py", "--experiment", infer_seen10.ALIGNED_EXPERIMENT], selected, explicit=None)
                self._check(train_seen10_peft, ["train_seen10_peft.py"], selected, explicit=None)
                cli = Path(directory) / "cli"
                self._check(train_seen10, ["train_seen10.py", "--experiment", train_seen10.ALIGNED_EXPERIMENT, "--data-root", str(cli)], cli, explicit=str(cli))


if __name__ == "__main__":
    unittest.main()
