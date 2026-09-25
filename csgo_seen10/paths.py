"""Resolve machine-local Seen-10 paths without changing canonical configs.

Explicit paths never fall back silently. Relative paths are rooted at this
checkout, regardless of the caller's working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DATA_ROOT = "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2"
LEGACY_EVAL_ROOT = "/home/jiahao/task/csgo_benchmark_v2_eval_general"
LEGACY_EVAL_PYTHON = "/home/jiahao/miniconda3/envs/UniLIP/bin/python"


def project_path(value: Any, root: Path = PROJECT_ROOT) -> Path:
    """Anchor a path to the checkout while retaining virtualenv symlinks."""
    candidate = Path(os.fspath(value)).expanduser()
    return candidate if candidate.is_absolute() else Path(root) / candidate


def _first_env(env: Mapping[str, str], names: tuple[str, ...]) -> str | None:
    return next((env[name] for name in names if env.get(name)), None)


def _configured(config: Mapping[str, Any], *keys: str) -> Any:
    return next((config[key] for key in keys if config.get(key)), None)


def data_root(
    config: Mapping[str, Any] = {}, explicit: Any = None, *,
    root: Path = PROJECT_ROOT, env: Mapping[str, str] | None = None,
) -> Path:
    """CLI > CSGO_DATA_ROOT > compatibility envs > config > sibling UniLIP."""
    environment = os.environ if env is None else env
    selected = explicit if explicit is not None else _first_env(
        environment, ("CSGO_DATA_ROOT", "CSGO_BENCHMARK_V2_DATA", "DATA_ROOT")
    )
    if selected is not None:
        return project_path(selected, root)
    configured = _configured(config, "data_root")
    if configured is not None:
        candidate = project_path(configured, root)
        if str(configured) != LEGACY_DATA_ROOT or candidate.exists():
            return candidate
    return Path(root).parent / "UniLIP" / "data" / "csgo_benchmark_v2"


def evaluator_root(
    config: Mapping[str, Any] = {}, explicit: Any = None, *,
    root: Path = PROJECT_ROOT, env: Mapping[str, str] | None = None,
) -> Path:
    """Locate the shared evaluator, preserving an existing legacy checkout."""
    environment = os.environ if env is None else env
    selected = explicit if explicit is not None else _first_env(
        environment, ("SHARED_EVAL_DIR", "CSGO_EVAL_ROOT")
    )
    if selected is not None:
        return project_path(selected, root)
    configured = _configured(config, "shared_eval_dir", "eval_root")
    if configured is not None:
        candidate = project_path(configured, root)
        if str(configured) != LEGACY_EVAL_ROOT or candidate.exists():
            return candidate
    return Path(root).parent / "csgo_benchmark_v2_eval_general"


def evaluator_python(
    config: Mapping[str, Any] = {}, explicit: Any = None, *,
    root: Path = PROJECT_ROOT, env: Mapping[str, str] | None = None,
) -> Path:
    """Prefer a project eval venv, then the legacy UniLIP env, then project venv."""
    environment = os.environ if env is None else env
    selected = explicit if explicit is not None else _first_env(
        environment, ("EVAL_PYTHON", "UNILIP_PYTHON")
    )
    if selected is not None:
        return project_path(selected, root)
    configured = _configured(config, "unilip_python", "eval_python")
    if configured is not None and str(configured) != LEGACY_EVAL_PYTHON:
        return project_path(configured, root)
    project_eval = Path(root) / ".venv-eval" / "bin" / "python"
    if project_eval.exists():
        return project_eval
    legacy = Path(LEGACY_EVAL_PYTHON)
    if legacy.exists():
        return legacy
    return Path(root) / ".venv" / "bin" / "python"
