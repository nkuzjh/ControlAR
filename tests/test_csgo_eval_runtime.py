"""Selection rules for the shared evaluator interpreter; no interpreter runs."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from csgo_seen10.eval_runtime import resolve_eval_python


def executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\nexit 0\n')
    path.chmod(0o755)
    return path


class EvalRuntimeTest(unittest.TestCase):
    def test_cli_wins_and_keeps_symlink_with_spaces(self):
        with TemporaryDirectory(prefix='eval runtime ') as temp:
            root = Path(temp)
            shared = executable(root / 'shared eval/.venv/bin/python')
            target = executable(root / 'custom env/bin/python')
            alias = root / 'python link'
            alias.symlink_to(target)
            chosen = resolve_eval_python(root / 'shared eval', 'python link',
                                         action='eval', root=root,
                                         env={'EVAL_PYTHON': '/ignored/python'})
            self.assertEqual(chosen.path, alias)
            self.assertEqual(chosen.source, 'CLI --eval-python/--unilip-python')
            self.assertTrue(chosen.ready)
            self.assertNotEqual(chosen.path, shared)

    def test_shared_venv_wins_over_environment(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            shared = executable(root / 'shared/.venv/bin/python')
            chosen = resolve_eval_python(root / 'shared', action='eval', root=root,
                                         env={'EVAL_PYTHON': '/missing/python'})
            self.assertEqual(chosen.path, shared)
            self.assertEqual(chosen.source, 'shared evaluator .venv')

    def test_missing_or_nonexecutable_shared_venv_selects_env_in_alias_order(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            shared = root / 'shared/.venv/bin/python'
            shared.parent.mkdir(parents=True)
            shared.write_text('not executable')
            primary = executable(root / 'new env/python')
            secondary = executable(root / 'old env/python')
            chosen = resolve_eval_python(root / 'shared', action='smoke', root=root,
                                         env={'EVAL_PYTHON': str(primary),
                                              'UNILIP_PYTHON': str(secondary)})
            self.assertEqual(chosen.path, primary)
            chosen = resolve_eval_python(root / 'shared', action='smoke', root=root,
                                         env={'UNILIP_PYTHON': 'old env/python'})
            self.assertEqual(chosen.path, secondary)

    def test_broken_shared_symlink_selects_environment(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            shared_python = root / 'shared/.venv/bin/python'
            shared_python.parent.mkdir(parents=True)
            shared_python.symlink_to(root / 'missing-target')
            env_python = executable(root / 'environment/bin/python')
            chosen = resolve_eval_python(root / 'shared', action='eval', root=root,
                                         env={'EVAL_PYTHON': str(env_python)})
            self.assertEqual(chosen.path, env_python)
            self.assertEqual(chosen.source, 'EVAL_PYTHON')

    def test_bad_explicit_paths_fail_without_fallback(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            executable(root / 'shared/.venv/bin/python')
            with self.assertRaisesRegex(ValueError, r'CLI .*not an executable file'):
                resolve_eval_python(root / 'shared', 'absent/python',
                                    action='eval', root=root, env={})
            shared = root / 'missing-shared'
            with self.assertRaisesRegex(ValueError, r'EVAL_PYTHON .*not an executable file'):
                resolve_eval_python(shared, action='eval', root=root,
                                    env={'EVAL_PYTHON': str(root / 'absent/python')})
            with self.assertRaisesRegex(ValueError, r'UNILIP_PYTHON .*not an executable file'):
                resolve_eval_python(shared, action='eval', root=root,
                                    env={'UNILIP_PYTHON': str(root / 'absent/python')})

    def test_train_and_infer_report_invalid_explicit_candidates(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            legacy = root / 'missing-legacy/python'
            with patch('csgo_seen10.eval_runtime.LEGACY_EVAL_PYTHON', str(legacy)):
                for action in ('train', 'infer'):
                    for explicit, env, expected_source in (
                        ('bad CLI/python', {}, 'CLI --eval-python/--unilip-python'),
                        (None, {'EVAL_PYTHON': 'bad env/python'}, 'EVAL_PYTHON'),
                        (None, {'UNILIP_PYTHON': 'bad old env/python'}, 'UNILIP_PYTHON'),
                        (None, {}, 'legacy UniLIP'),
                    ):
                        with self.subTest(action=action, source=expected_source):
                            chosen = resolve_eval_python(root / 'missing', explicit,
                                                         action=action, root=root, env=env)
                            self.assertEqual(chosen.source, expected_source)
                            self.assertFalse(chosen.ready)

    def test_legacy_only_when_no_usable_shared_or_explicit_env(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            legacy = executable(root / 'legacy env/bin/python')
            with patch('csgo_seen10.eval_runtime.LEGACY_EVAL_PYTHON', str(legacy)):
                chosen = resolve_eval_python(root / 'missing', action='eval', root=root, env={})
            self.assertEqual(chosen.path, legacy)
            self.assertEqual(chosen.source, 'legacy UniLIP')

    def test_missing_default_is_inspectable_for_train_infer_but_required_for_eval_smoke(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            legacy = root / 'missing-legacy/python'
            with patch('csgo_seen10.eval_runtime.LEGACY_EVAL_PYTHON', str(legacy)):
                for action in ('train', 'infer'):
                    chosen = resolve_eval_python(root / 'missing', action=action,
                                                 root=root, env={})
                    self.assertEqual(chosen.path, legacy)
                    self.assertFalse(chosen.ready)
                for action in ('eval', 'smoke'):
                    with self.assertRaisesRegex(ValueError, 'Evaluator Python is missing'):
                        resolve_eval_python(root / 'missing', action=action,
                                            root=root, env={})

    def test_no_project_venv_fallback(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            executable(root / '.venv/bin/python')
            executable(root / '.venv-eval/bin/python')
            legacy = root / 'missing-legacy/python'
            with patch('csgo_seen10.eval_runtime.LEGACY_EVAL_PYTHON', str(legacy)):
                chosen = resolve_eval_python(root / 'missing-shared', action='train',
                                             root=root, env={})
                self.assertEqual(chosen.path, legacy)
                self.assertFalse(chosen.ready)
                with self.assertRaisesRegex(ValueError, 'Evaluator Python is missing'):
                    resolve_eval_python(root / 'missing-shared', action='eval',
                                        root=root, env={})


if __name__ == '__main__':
    unittest.main()
