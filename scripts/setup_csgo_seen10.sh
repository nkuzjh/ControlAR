#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=setup
TARGET=training
DOWNLOAD=1
ENV_ONLY=0
EVAL_ONLY=0

usage() {
    cat <<'HELP'
Usage: bash scripts/setup_csgo_seen10.sh [--env-only | --eval-only] [--check | --check-cuda] [--help]

Default: prepare the project-local .venv and verify/download pinned official model assets.
--env-only   Prepare only the training/inference environment; skip model assets.
--eval-only  Prepare the shared evaluator in a separate .venv-eval; skip model assets.
             Uses Python 3.11-3.12 for the evaluator's scipy==1.17.0.
--check      Inspect the selected environment and, by default, hash all local assets.
             CPU imports only; no installation, downloads, writes, or CUDA initialization.
--check-cuda Check the selected environment and run a small CUDA matrix operation.
             This initializes CUDA and requires a GPU; it does not download assets.
--help       Print this help without changing anything.

Optional configuration:
  CONTROLAR_BOOTSTRAP_PYTHON  Python 3.10-3.12 executable for a new environment.
  CONTROLAR_CONDA_EXE        Conda executable (fallback when Python is unavailable).
  CONTROLAR_CLONE_FROM       Explicit Conda source prefix to clone into .venv.
  CONTROLAR_TORCH_BACKEND    cu128 (default) or cpu for a new environment.
  CONTROLAR_TORCH_INDEX_URL  Matching official PyTorch wheel index or mirror.
  HF_ENDPOINT                Hugging Face Hub endpoint for model downloads.
HELP
}

die() { echo "setup_csgo_seen10: $*" >&2; exit 2; }

for arg in "$@"; do
    case "$arg" in
        --help|-h) usage; exit 0 ;;
        --check) MODE=check ;;
        --check-cuda) MODE=check-cuda; DOWNLOAD=0 ;;
        --env-only) ENV_ONLY=1; DOWNLOAD=0 ;;
        --eval-only) EVAL_ONLY=1; TARGET=eval; DOWNLOAD=0 ;;
        *) die "unknown argument: $arg (use --help)" ;;
    esac
done
(( ! (ENV_ONLY && EVAL_ONLY) )) || die "--env-only conflicts with --eval-only"

if [[ "$TARGET" == eval ]]; then
    ENV_DIR="$PROJECT_ROOT/.venv-eval"
    REQUIREMENTS="$PROJECT_ROOT/requirements-csgo-eval.txt"
    MIN_PYTHON=11
else
    ENV_DIR="$PROJECT_ROOT/.venv"
    REQUIREMENTS="$PROJECT_ROOT/requirements-csgo-seen10.txt"
    MIN_PYTHON=10
fi
[[ -f "$REQUIREMENTS" ]] || die "missing requirements: $REQUIREMENTS"
PYTHON="$ENV_DIR/bin/python"

python_version() {
    "$1" -c 'import sys; print(sys.version.split()[0] if (3, int(sys.argv[1])) <= sys.version_info[:2] <= (3, 12) else "")' "$MIN_PYTHON" 2>/dev/null
}

check_environment() {
    [[ -x "$PYTHON" ]] || die "missing environment Python: $PYTHON"
    "$PYTHON" - "$ENV_DIR" "$REQUIREMENTS" "$MIN_PYTHON" "$TARGET" <<'PY'
import importlib
import importlib.metadata as metadata
import sys
from pathlib import Path

prefix = Path(sys.argv[1]).resolve()
requirements = Path(sys.argv[2])
if Path(sys.prefix).resolve() != prefix:
    raise SystemExit(f"Environment points to {sys.prefix}, expected {prefix}; recreate the moved environment")
if not (3, int(sys.argv[3])) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit(f"Python {sys.version.split()[0]} is unsupported for this environment")
venv_cfg = prefix / "pyvenv.cfg"
if venv_cfg.exists():
    if sys.prefix == sys.base_prefix or "include-system-site-packages = false" not in venv_cfg.read_text().lower():
        raise SystemExit("Environment is not an isolated venv")
elif not (prefix / "conda-meta").is_dir():
    raise SystemExit("Environment has neither isolated pyvenv.cfg nor conda-meta")
required = ["torch", "torchvision"]
for line in requirements.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        required.append(line.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0])
missing = []
for package in required:
    try:
        metadata.version(package)
    except metadata.PackageNotFoundError:
        missing.append(package)
if missing:
    raise SystemExit("Missing packages in " + str(prefix) + ": " + ", ".join(missing))
modules = ["torch", "torchvision", "numpy", "PIL", "cv2", "requests"]
if sys.argv[4] == "eval":
    modules += ["torchmetrics", "torch_fidelity", "scipy", "yaml"]
else:
    modules += ["einops", "transformers", "datasets", "accelerate", "safetensors", "huggingface_hub", "timm"]
failures = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
if failures:
    raise SystemExit("Package import failures:\n  " + "\n  ".join(failures))
import torch
if torch.cuda.is_initialized():
    raise SystemExit("CUDA was initialized during CPU-only environment check")
print(f"Python {sys.version.split()[0]} at {prefix}")
print(f"torch {metadata.version('torch')}; torchvision {metadata.version('torchvision')} (CPU imports passed; CUDA uninitialized)")
PY
}

asset_check() {
    # The asset command imports only the standard library for --check.
    "$PYTHON" "$PROJECT_ROOT/scripts/download_csgo_seen10_assets.py" --check
}

if [[ "$MODE" == check ]]; then
    check_environment
    if (( DOWNLOAD )); then asset_check; fi
    exit 0
fi

if [[ "$MODE" == check-cuda ]]; then
    check_environment
    "$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to PyTorch in this environment")
device = torch.device("cuda:0")
capability = torch.cuda.get_device_capability(device)
compiled = torch.cuda.get_arch_list()
if not torch.cuda.is_bf16_supported():
    raise SystemExit("GPU does not support BF16 required by ControlAR training/inference")
matrix = torch.eye(8, device=device, dtype=torch.bfloat16)
result = matrix @ matrix
torch.cuda.synchronize(device)
if not torch.allclose(result, matrix):
    raise SystemExit("CUDA matrix operation failed")
print(f"CUDA matrix check passed: {torch.cuda.get_device_name(device)}, sm_{capability[0]}{capability[1]}, torch {torch.__version__}, runtime {torch.version.cuda}, compiled {compiled}")
PY
    exit 0
fi

if [[ -e "$ENV_DIR" || -L "$ENV_DIR" ]]; then
    [[ -x "$PYTHON" ]] || die "$ENV_DIR exists without bin/python; preserve and repair it manually"
    "$PYTHON" - "$ENV_DIR" "$MIN_PYTHON" <<'PY'
import sys
from pathlib import Path
prefix = Path(sys.argv[1]).resolve()
if Path(sys.prefix).resolve() != prefix or not (3, int(sys.argv[2])) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit("Existing environment is moved or uses unsupported Python; preserved unchanged")
if not (prefix / "conda-meta").is_dir():
    cfg = prefix / "pyvenv.cfg"
    if not cfg.is_file() or sys.prefix == sys.base_prefix or "include-system-site-packages = false" not in cfg.read_text().lower():
        raise SystemExit("Existing environment is not isolated; preserved unchanged")
PY
    ENV_KIND=existing
else
    ENV_KIND=fresh
    if [[ -n "${CONTROLAR_CLONE_FROM:-}" ]]; then
        [[ "$TARGET" == training ]] || die "CONTROLAR_CLONE_FROM applies only to training .venv"
        [[ -x "$CONTROLAR_CLONE_FROM/bin/python" && -d "$CONTROLAR_CLONE_FROM/conda-meta" ]] || die "invalid Conda clone source: $CONTROLAR_CLONE_FROM"
        CONDA_BIN="${CONTROLAR_CONDA_EXE:-$(command -v conda || true)}"
        [[ -n "$CONDA_BIN" && -x "$CONDA_BIN" ]] || die "Conda executable required for CONTROLAR_CLONE_FROM"
        "$CONDA_BIN" create --copy --prefix "$ENV_DIR" --clone "$CONTROLAR_CLONE_FROM" -y
        ENV_KIND=clone
    else
        BOOTSTRAP="${CONTROLAR_BOOTSTRAP_PYTHON:-}"
        if [[ -n "$BOOTSTRAP" ]]; then
            command -v "$BOOTSTRAP" >/dev/null 2>&1 || die "bootstrap Python unavailable: $BOOTSTRAP"
            [[ -n "$(python_version "$BOOTSTRAP")" ]] || die "bootstrap Python version is unsupported for $TARGET: $BOOTSTRAP"
            "$BOOTSTRAP" -c 'import ensurepip, venv' >/dev/null 2>&1 || die "bootstrap Python lacks venv/ensurepip: $BOOTSTRAP; install its venv package or use Conda"
        else
            if [[ "$TARGET" == eval ]]; then
                CANDIDATES=(python3.12 python3.11 python3)
            else
                CANDIDATES=(python3.12 python3.11 python3.10 python3)
            fi
            for candidate in "${CANDIDATES[@]}"; do
                if command -v "$candidate" >/dev/null 2>&1 && [[ -n "$(python_version "$candidate")" ]] && \
                    "$candidate" -c 'import ensurepip, venv' >/dev/null 2>&1; then
                    BOOTSTRAP="$candidate"
                    break
                fi
            done
        fi
        if [[ -n "$BOOTSTRAP" ]]; then
            "$BOOTSTRAP" -m venv "$ENV_DIR" || die "venv creation failed; install python3-venv or use Conda"
        else
            CONDA_BIN="${CONTROLAR_CONDA_EXE:-$(command -v conda || true)}"
            [[ -n "$CONDA_BIN" && -x "$CONDA_BIN" ]] || die "Python 3.10-3.12 or Conda is required"
            "$CONDA_BIN" create --prefix "$ENV_DIR" python=3.11 pip -y
        fi
    fi
fi

[[ -x "$PYTHON" ]] || die "environment Python is unavailable: $PYTHON"
if [[ "$ENV_KIND" == clone ]]; then
    "$PYTHON" - "$ENV_DIR" <<'PY'
import site
import sys
from pathlib import Path

prefix = Path(sys.argv[1]).resolve()
for directory in site.getsitepackages():
    path = Path(directory).resolve()
    if not path.is_relative_to(prefix):
        continue
    for pth in path.glob("*.pth"):
        if "unilip" in pth.name.lower() or "unilip" in pth.read_text(errors="replace").lower():
            pth.unlink()
            print(f"Removed foreign UniLIP editable path from cloned copy: {pth}")
PY
fi
"$PYTHON" -m pip --version >/dev/null || die "pip is missing from $ENV_DIR"

install_torch_pair() {
    BACKEND="${CONTROLAR_TORCH_BACKEND:-cu128}"
    case "$BACKEND" in cu128|cpu) ;; *) die "unsupported CONTROLAR_TORCH_BACKEND=$BACKEND; choose cu128 or cpu" ;; esac
    INDEX="${CONTROLAR_TORCH_INDEX_URL:-https://download.pytorch.org/whl/$BACKEND}"
    "$PYTHON" -m pip install --disable-pip-version-check --no-input \
        --index-url "$INDEX" 'torch==2.7.1' 'torchvision==0.22.1'
}

if [[ "$ENV_KIND" == fresh ]]; then
    install_torch_pair
    "$PYTHON" -m pip install --disable-pip-version-check --no-input -r "$REQUIREMENTS"
else
    # Existing environments, including nightly stacks, retain every installed
    # distribution version. Only truly missing requirements are installed.
    mapfile -t MISSING < <("$PYTHON" - "$REQUIREMENTS" <<'PY'
import importlib.metadata as metadata
import sys
from pathlib import Path
lines = ["torch", "torchvision"] + [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
for line in lines:
    name = line.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0]
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        print(line)
PY
    )
    if (( ${#MISSING[@]} )); then
        # A valid but interrupted empty venv can safely get the stable pair.
        # One missing member of an existing pair requires manual repair.
        if [[ " ${MISSING[*]} " == *" torch "* && " ${MISSING[*]} " == *" torchvision "* ]]; then
            install_torch_pair
            MISSING=("${MISSING[@]:2}")
        elif [[ " ${MISSING[*]} " == *" torch "* || " ${MISSING[*]} " == *" torchvision "* ]]; then
            die "incomplete existing torch/torchvision pair; repair it explicitly in $ENV_DIR"
        fi
    fi
    if (( ${#MISSING[@]} )); then
        CONSTRAINTS="$(mktemp)"
        trap 'rm -f "$CONSTRAINTS"' EXIT
        "$PYTHON" - "$CONSTRAINTS" <<'PY'
import importlib.metadata as metadata
import sys
from pathlib import Path
pins = sorted({f"{dist.metadata['Name']}=={dist.version}" for dist in metadata.distributions() if dist.metadata.get('Name')})
Path(sys.argv[1]).write_text("\n".join(pins) + "\n")
PY
        "$PYTHON" -m pip install --disable-pip-version-check --no-input \
            --upgrade-strategy only-if-needed --constraint "$CONSTRAINTS" "${MISSING[@]}"
    fi
fi

check_environment
if (( DOWNLOAD )); then
    "$PYTHON" "$PROJECT_ROOT/scripts/download_csgo_seen10_assets.py"
fi
printf 'CSGO %s environment ready: %s\n' "$TARGET" "$ENV_DIR"
