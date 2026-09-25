#!/usr/bin/env python3
"""Resolve launch paths using only the standard library; never start a job."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from csgo_seen10.eval_runtime import resolve_eval_python
from csgo_seen10.paths import data_root, evaluator_root, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='')
    parser.add_argument('--action', choices=('smoke', 'train', 'infer', 'eval'), default='train')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-root')
    parser.add_argument('--eval-root')
    parser.add_argument('--eval-python', '--unilip-python', dest='eval_python')
    parser.add_argument('--model-python')
    parser.add_argument('--run-root')
    parser.add_argument('--lines', action='store_true')
    args = parser.parse_args()
    profiles = {'': 'csgo_seen10', 'csgo_seen10_exp32gen_aligned': 'csgo_seen10_exp32gen_aligned',
                'csgo_seen10_exp32gen_aligned_peft': 'csgo_seen10_exp32gen_aligned_peft'}
    if args.experiment not in profiles:
        parser.error(f'Unknown experiment: {args.experiment}')
    config_path = ROOT / 'configs' / (profiles[args.experiment] + '.json')
    config = json.loads(config_path.read_text())
    selected_eval_root = evaluator_root(config, args.eval_root).resolve()
    try:
        selected_eval_python = resolve_eval_python(
            selected_eval_root, args.eval_python, action=args.action
        )
    except ValueError as exc:
        parser.error(str(exc))
    values = {
        'model_python': str(project_path(args.model_python or os.environ.get('CONTROLAR_PYTHON') or '.venv/bin/python')),
        'data_root': str(data_root(config, args.data_root).resolve()),
        'eval_root': str(selected_eval_root),
        'eval_python': str(selected_eval_python.path),
        'run_root': str(project_path(args.run_root or str(Path(config['output_base']) / f'seed_{args.seed}')).resolve()),
    }
    if any('\n' in value or '\r' in value for value in values.values()):
        parser.error('Paths may not contain newlines')
    if args.lines:
        print('\n'.join(values.values()))
    else:
        print(json.dumps({
            'project_root': str(ROOT), 'experiment': args.experiment or 'legacy',
            'config': str(config_path), **values,
            'evaluator': str(Path(values['eval_root']) / 'run_eval.py'),
            'eval_config': str(Path(values['eval_root']) / 'benchmark_v2.yaml'),
            'official_gpt_checkpoint': str(project_path(config['official_gpt_checkpoint'])),
            'vq_checkpoint': str(project_path(config['vq_checkpoint'])),
            'paths_exist': {name: Path(value).exists() for name, value in values.items()},
            'eval_python_source': selected_eval_python.source,
            'eval_python_ready': selected_eval_python.ready,
            'inspection_only': True,
        }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
