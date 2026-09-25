"""Read-only setup and asset checks using only temporary files."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "csgo_assets", ROOT / "scripts/download_csgo_seen10_assets.py"
)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


class AssetChecks(unittest.TestCase):
    def test_local_hashes_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            payload = b"test asset"
            model = temp / "model.bin"
            model.write_bytes(payload)
            metadata_dir = temp / "autoregressive/models/dinov2-small"
            metadata_dir.mkdir(parents=True)
            metadata = metadata_dir / "config.json"
            metadata.write_bytes(b"{}")
            spec = [{
                "target": model,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }]
            with patch.object(assets, "ROOT", temp), \
                 patch.object(assets, "ASSETS", spec), \
                 patch.object(assets, "DINO_SMALL_FILES", {"config.json": assets.git_blob_id(b"{}")}), \
                 patch.object(assets, "get_session", side_effect=AssertionError("network used")):
                self.assertTrue(assets.check_assets())
                model.write_bytes(b"wrong data")
                self.assertFalse(assets.check_assets())
                model.unlink()
                self.assertFalse(assets.check_assets())

    def test_endpoint_override(self):
        asset = {"repo": "owner/model", "revision": "abcdef", "filename": "model.bin"}
        with patch.dict(os.environ, {"HF_ENDPOINT": "https://mirror.example/"}):
            self.assertEqual(
                assets.asset_url(asset),
                "https://mirror.example/owner/model/resolve/abcdef/model.bin?download=true",
            )


class SetupChecks(unittest.TestCase):
    def copy_setup(self, temp):
        (temp / "scripts").mkdir()
        shutil.copy2(ROOT / "scripts/setup_csgo_seen10.sh", temp / "scripts/setup_csgo_seen10.sh")
        shutil.copy2(ROOT / "requirements-csgo-seen10.txt", temp / "requirements-csgo-seen10.txt")
        shutil.copy2(ROOT / "requirements-csgo-eval.txt", temp / "requirements-csgo-eval.txt")
        return temp / "scripts/setup_csgo_seen10.sh"

    def mock_python(self, temp):
        script = temp / "mock-python"
        script.write_text("""#!/usr/bin/env bash
if [[ "$1" == -c ]]; then
    if [[ "$3" =~ ^[0-9]+$ && "${MOCK_VERSION:-3.11.0}" == 3.10.* && "$3" -gt 10 ]]; then exit 0; fi
    echo "${MOCK_VERSION:-3.11.0}"
    exit 0
fi
if [[ "$1" == -m && "$2" == venv ]]; then
    mkdir -p "$3/bin"
    cp "$0" "$3/bin/python"
    exit 0
fi
if [[ "$1" == -m && "$2" == pip ]]; then
    if [[ "$3" == --version ]]; then echo 'pip 25.0'; exit 0; fi
    printf '%s\\n' "$*" >> "$MOCK_LOG"
    exit 0
fi
exit 0
""")
        script.chmod(0o755)
        return script

    def test_help_and_missing_environment_check_are_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            script = self.copy_setup(temp)
            for args, expected in [(["--help"], 0), (["--env-only", "--check"], 2)]:
                result = subprocess.run(["bash", str(script), *args], cwd=temp, capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stderr)
            self.assertEqual(sorted(p.name for p in temp.iterdir()), ["requirements-csgo-eval.txt", "requirements-csgo-seen10.txt", "scripts"])

    def test_existing_environment_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            script = self.copy_setup(temp)
            mock = self.mock_python(temp)
            env_python = temp / ".venv/bin/python"
            env_python.parent.mkdir(parents=True)
            shutil.copy2(mock, env_python)
            sentinel = temp / ".venv/keep.txt"
            sentinel.write_text("preserve")
            log = temp / "pip.log"
            env = {**os.environ, "MOCK_LOG": str(log)}
            for args in (["--env-only", "--check"], ["--env-only"]):
                result = subprocess.run(["bash", str(script), *args], cwd=temp, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sentinel.read_text(), "preserve")
            self.assertFalse(log.exists(), "existing environment triggered pip install")

    def test_invalid_explicit_interpreter_and_eval_python_version(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            script = self.copy_setup(temp)
            mock = self.mock_python(temp)
            for args, value in [(["--env-only"], str(temp / "missing-python")), (["--eval-only"], str(mock))]:
                env = {**os.environ, "CONTROLAR_BOOTSTRAP_PYTHON": value, "MOCK_VERSION": "3.10.13"}
                result = subprocess.run(["bash", str(script), *args], cwd=temp, env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("bootstrap Python", result.stderr)
            self.assertFalse((temp / ".venv").exists())
            self.assertFalse((temp / ".venv-eval").exists())

    def test_mocked_cpu_and_cu128_install_selection(self):
        for backend in ("cpu", "cu128"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                temp = Path(directory)
                script = self.copy_setup(temp)
                mock = self.mock_python(temp)
                log = temp / "pip.log"
                env = {**os.environ, "CONTROLAR_BOOTSTRAP_PYTHON": str(mock),
                       "CONTROLAR_TORCH_BACKEND": backend, "MOCK_LOG": str(log)}
                result = subprocess.run(["bash", str(script), "--env-only"], cwd=temp, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"https://download.pytorch.org/whl/{backend}", log.read_text())
                self.assertIn("torch==2.7.1 torchvision==0.22.1", log.read_text())


if __name__ == "__main__":
    unittest.main()
