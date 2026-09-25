"""Select the shared evaluator's Python without importing or running it."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

from csgo_seen10.paths import LEGACY_EVAL_PYTHON, PROJECT_ROOT, project_path


@dataclass(frozen=True)
class EvalPythonSelection:
    path: Path
    source: str
    ready: bool


def _usable_python(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def resolve_eval_python(
    eval_root: Path,
    explicit: Any = None,
    *,
    action: str,
    root: Path = PROJECT_ROOT,
    env: Mapping[str, str] | None = None,
) -> EvalPythonSelection:
    """Choose CLI > shared .venv > env > legacy UniLIP, with no other fallback.

    ``eval_root`` is the already selected shared evaluator directory. An absent
    default is inspectable for train/infer, while eval/smoke require it to run.
    For train/infer, an unusable candidate remains visible as not ready.
    """
    if action not in ('train', 'infer', 'eval', 'smoke'):
        raise ValueError(f'Unknown action: {action}')
    environment = os.environ if env is None else env
    if explicit is not None:
        path = project_path(explicit, root)
        source = 'CLI --eval-python/--unilip-python'
        ready = _usable_python(path)
        if not ready and action in ('eval', 'smoke'):
            raise ValueError(f'{source} is not an executable file: {path}')
        return EvalPythonSelection(path, source, ready)

    shared_python = Path(eval_root) / '.venv' / 'bin' / 'python'
    if _usable_python(shared_python):
        return EvalPythonSelection(shared_python, 'shared evaluator .venv', True)

    for name in ('EVAL_PYTHON', 'UNILIP_PYTHON'):
        if environment.get(name):
            path = project_path(environment[name], root)
            ready = _usable_python(path)
            if not ready and action in ('eval', 'smoke'):
                raise ValueError(f'{name} is not an executable file: {path}')
            return EvalPythonSelection(path, name, ready)

    path = Path(LEGACY_EVAL_PYTHON)
    ready = _usable_python(path)
    if not ready and action in ('eval', 'smoke'):
        raise ValueError(
            f'Evaluator Python is missing: {path}; prepare {shared_python} '
            'or set EVAL_PYTHON/--eval-python.'
        )
    return EvalPythonSelection(path, 'legacy UniLIP', ready)
