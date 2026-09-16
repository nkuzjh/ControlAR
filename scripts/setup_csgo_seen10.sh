#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Prefer a byte-for-byte independent conda copy of the already validated cu128
# runtime on this host. Nothing is installed into the source environment.
CONDA_EXE="${CONTROLAR_CONDA_EXE:-/home/jiahao/miniconda3/bin/conda}"
CLONE_FROM="${CONTROLAR_CLONE_FROM:-/home/jiahao/miniconda3/envs/UniLIP}"
BOOTSTRAP_PYTHON="${CONTROLAR_BOOTSTRAP_PYTHON:-/home/jiahao/miniconda3/envs/UniLIP/bin/python3.11}"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    if [[ -x "$CONDA_EXE" && -d "$CLONE_FROM/conda-meta" ]]; then
        "$CONDA_EXE" create --copy --prefix "$PROJECT_ROOT/.venv" --clone "$CLONE_FROM" -y
    else
        if [[ ! -x "$BOOTSTRAP_PYTHON" ]]; then
            echo "Python 3.11 bootstrap interpreter not found: $BOOTSTRAP_PYTHON" >&2
            echo "Set CONTROLAR_BOOTSTRAP_PYTHON to an existing Python 3.10-3.12 executable." >&2
            exit 2
        fi
        "$BOOTSTRAP_PYTHON" -m venv "$PROJECT_ROOT/.venv"
    fi
fi

# A cloned UniLIP environment carries an editable-install .pth file. Remove
# that link only from the project copy, so imports cannot fall back to UniLIP.
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import site

for directory in site.getsitepackages():
    for pth in Path(directory).glob("__editable__*.pth"):
        contents = pth.read_text(errors="replace").lower()
        if "unilip" in pth.name.lower() or "/home/jiahao/task/unilip" in contents:
            pth.unlink()
            print(f"Removed cloned editable path from project copy: {pth}")
PY

"$VENV_PYTHON" - <<'PY'
import sys
from pathlib import Path
import site

assert (3, 10) <= sys.version_info[:2] <= (3, 12), sys.version
prefix = Path(sys.prefix).resolve()
expected = Path.cwd().joinpath(".venv").resolve()
assert prefix == expected, (prefix, expected)
site_dirs = [Path(path).resolve() for path in site.getsitepackages()]
assert all(path.is_relative_to(prefix) for path in site_dirs), site_dirs
assert not any("/home/jiahao/task/unilip" in path.lower() for path in sys.path), sys.path
if (prefix / "conda-meta").is_dir():
    env_kind = "copied conda prefix"
else:
    cfg = (prefix / "pyvenv.cfg").read_text().lower()
    assert "include-system-site-packages = false" in cfg, cfg
    env_kind = "venv without system site packages"
print(f"Using isolated {env_kind}: Python {sys.version.split()[0]} at {prefix}")
PY

if ! "$VENV_PYTHON" - <<'PY' >/dev/null 2>&1
import torch

assert torch.version.cuda is not None
assert tuple(map(int, torch.version.cuda.split(".")[:2])) >= (12, 8)
assert "sm_120" in torch.cuda.get_arch_list()
PY
then
    "$VENV_PYTHON" -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --upgrade 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128'
fi
"$VENV_PYTHON" -m pip install -r requirements-csgo-seen10.txt

"$VENV_PYTHON" - <<'PY'
import torch

assert torch.version.cuda is not None, "A CUDA PyTorch build is required"
cuda_version = tuple(map(int, torch.version.cuda.split(".")[:2]))
assert cuda_version >= (12, 8), torch.version.cuda
arch_list = torch.cuda.get_arch_list()
assert "sm_120" in arch_list, arch_list
assert torch.cuda.is_available(), "CUDA device is unavailable"
print(f"Torch {torch.__version__}; CUDA {torch.version.cuda}; compiled archs={arch_list}")
PY

mkdir -p checkpoints/t2i checkpoints/vq autoregressive/models/dinov2-small
export HF_HUB_DISABLE_XET=1
"$VENV_PYTHON" scripts/download_csgo_seen10_assets.py

echo "Environment and pinned official model assets are ready."
