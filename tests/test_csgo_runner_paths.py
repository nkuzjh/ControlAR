"""Launch-path regressions: run only stdlib inspection or a fake interpreter."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'scripts/run_csgo_seen10.sh'
PEFT = 'csgo_seen10_exp32gen_aligned_peft'


class RunnerPathsTest(unittest.TestCase):
    def environment(self, **updates):
        env = os.environ.copy()
        for key in ('CSGO_DATA_ROOT', 'CSGO_BENCHMARK_V2_DATA', 'DATA_ROOT', 'SHARED_EVAL_DIR',
                    'CSGO_EVAL_ROOT', 'EVAL_PYTHON', 'UNILIP_PYTHON', 'CONTROLAR_PYTHON',
                    'CONTROLAR_PATHS_PYTHON', 'NPROC_PER_NODE'):
            env.pop(key, None)
        env.update(updates)
        return env

    def inspect(self, action='train', *options, **env):
        result = subprocess.run(['bash', str(RUNNER), action, '--print-paths', *options],
                                cwd='/tmp', env=self.environment(**env), text=True,
                                capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_all_profiles_and_actions_before_model_environment_exists(self):
        for experiment, seed in (('', 0), ('csgo_seen10_exp32gen_aligned', 42), (PEFT, 42)):
            opts = ['--experiment', experiment] if experiment else []
            for action in ('train', 'infer', 'eval', 'smoke'):
                with self.subTest(experiment=experiment, action=action):
                    result = self.inspect(action, *opts, CONTROLAR_PYTHON='not prepared/python')
                    self.assertTrue(result['inspection_only'])
                    self.assertEqual(result['model_python'], str(ROOT / 'not prepared/python'))
                    self.assertTrue(result['run_root'].endswith(f'/seed_{seed}'))

    def test_cli_precedence_space_paths_and_relative_project_anchor(self):
        result = self.inspect('eval', '--experiment', PEFT, '--data-root', 'some data',
                              '--eval-root=shared eval', '--eval-python', 'eval env/bin/python',
                              CSGO_DATA_ROOT='/unused/data', EVAL_PYTHON='/unused/python')
        self.assertEqual(result['data_root'], str(ROOT / 'some data'))
        self.assertEqual(result['eval_root'], str(ROOT / 'shared eval'))
        self.assertEqual(result['eval_python'], str(ROOT / 'eval env/bin/python'))

    def test_environment_alias_priority(self):
        result = self.inspect(CSGO_DATA_ROOT='/selected', CSGO_BENCHMARK_V2_DATA='/old', DATA_ROOT='/older',
                              EVAL_PYTHON='/eval/new', UNILIP_PYTHON='/eval/old')
        self.assertEqual(result['data_root'], '/selected')
        self.assertEqual(result['eval_python'], '/eval/new')
        result = self.inspect(CSGO_BENCHMARK_V2_DATA='/old')
        self.assertEqual(result['data_root'], '/old')

    def test_explicit_missing_path_is_not_replaced(self):
        result = self.inspect('train', '--data-root', '/nonexistent/explicit/benchmark')
        self.assertEqual(result['data_root'], '/nonexistent/explicit/benchmark')
        self.assertFalse(result['paths_exist']['data_root'])

    def test_train_routing_with_fake_interpreter(self):
        with tempfile.TemporaryDirectory(prefix='controlar runner ') as temp:
            fake = Path(temp) / 'fake python'
            fake.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o755)
            result = subprocess.run(['bash', str(RUNNER), 'train', '--experiment', PEFT,
                                     '--data-root', 'new data', '--batch-size', '16',
                                     '--gradient-accumulation-steps', '8'], cwd='/tmp',
                                    env=self.environment(CONTROLAR_PYTHON=str(fake), NPROC_PER_NODE='1'),
                                    text=True, capture_output=True, check=True)
            args = json.loads(result.stdout)
            self.assertEqual(args[0], 'train_seen10_peft.py')
            for i, value in enumerate(args):
                if value == '--data-root': self.assertEqual(args[i + 1], str(ROOT / 'new data'))
            self.assertIn('--gradient-accumulation-steps', args)

    def test_unknown_action_rejected_even_when_inspecting(self):
        result = subprocess.run(['bash', str(RUNNER), 'typo', '--print-paths'], cwd='/tmp',
                                env=self.environment(), text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
