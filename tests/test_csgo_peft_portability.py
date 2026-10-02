"""CPU-only migration regressions using tiny checkpoints, never model tensors."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from csgo_seen10.artifact_contract import rows_contract, sha256_file, sha256_json
from csgo_seen10.peft_artifact_contract import (
    EXPERIMENT, FORMAT, PEFT_CHECKPOINT_STEPS, validate_peft_checkpoint,
    ensure_peft_output_manifest, preflight_peft_inference, preflight_evaluation,
)
from csgo_seen10.inference_portability import validate_inference_data_identity
from csgo_seen10.source_compat import check_resume_identity
from scripts.validate_csgo_seen10_aligned import ContractError, check_checkpoint_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs' / f'{EXPERIMENT}.json'


def sign_identity(identity):
    identity.pop('identity_sha256', None)
    identity['identity_sha256'] = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(',', ':'), default=str,
    ).encode()).hexdigest()


class PeftPortabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='peft migration ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data_root = self.root / 'current data'
        self.ckpt_dir = self.root / 'checkpoints'
        self.ckpt_dir.mkdir()
        self.config = json.loads(CONFIG.read_text())
        files = [{'path': name, 'sha256': hashlib.sha256(name.encode()).hexdigest()}
                 for name in ('benchmark_manifest.json', 'minimal_dataset_report.json',
                              'calibration/z_calibration.json', 'splits/seen/train.json')]
        self.contract = {
            'root': str(self.data_root), 'files': files,
            'benchmark_manifest_sha256': files[0]['sha256'],
            'minimal_dataset_report_sha256': files[1]['sha256'],
            'sha256': sha256_json(files),
        }
        self.identity = {
            'experiment': EXPERIMENT, 'data_root': str(self.data_root),
            'benchmark_data_contract': copy.deepcopy(self.contract),
            'files': {
                'config': sha256_file(CONFIG),
                'manifest': files[0]['sha256'], 'report': files[1]['sha256'],
                'calibration': files[2]['sha256'],
                'official_gpt': self.config['official_gpt_sha256'],
                'vq': self.config['vq_sha256'],
                'code': {'csgo_seen10/peft.py': sha256_file(ROOT / 'csgo_seen10/peft.py')},
            },
        }
        sign_identity(self.identity)
        # A two-GPU training origin is valid for one-GPU inference.
        self.payload = {
            'format': FORMAT,
            'args': {'experiment': EXPERIMENT, 'seed': 42, 'smoke': False,
                     'data_root': str(self.data_root)},
            'identity': self.identity,
            'training_config': {
                'seed': 42, 'world_size': 2, 'batch_size': 16,
                'gradient_accumulation_steps': 4, 'effective_batch_size': 128,
                'max_optimizer_steps': 19500, 'checkpoint_steps': list(PEFT_CHECKPOINT_STEPS),
            },
            'sampler_state': {'seed': 42, 'num_replicas': 2,
                              'micro_batch_per_device': 16, 'gradient_accumulation_steps': 4},
            'steps': 19500, 'global_optimizer_step': 19500,
            'consumed_samples': 2496000, 'best_step': 19500,
            'model_config': {'peft': {'rank': 32, 'alpha': 64, 'dropout': .05,
                                      'qkv_independent': True}},
        }
        self.index = {'best_step': 19500, 'late_step': 19500, 'checkpoints': []}
        for step in PEFT_CHECKPOINT_STEPS:
            p = self.ckpt_dir / f'step_{step:06d}.pt'
            p.write_bytes(f'fixture checkpoint {step}'.encode())
            self.index['checkpoints'].append({
                'step': step, 'path': p.name, 'sha256': sha256_file(p),
                'is_best': step == 19500,
            })
        for role in ('late', 'best'):
            shutil.copyfile(self.final, self.ckpt_dir / f'{role}.pt')
        self.save_index()

    @property
    def final(self):
        return self.ckpt_dir / 'step_019500.pt'

    def save_index(self):
        (self.ckpt_dir / 'checkpoint_index.json').write_text(json.dumps(self.index))

    def relocate(self):
        old_root = '/different/server/data/csgo_benchmark_v2'
        self.identity['data_root'] = old_root
        self.identity['benchmark_data_contract']['root'] = old_root
        self.payload['args']['data_root'] = old_root
        sign_identity(self.identity)

    def validate(self, role='late'):
        return validate_peft_checkpoint(
            self.ckpt_dir / f'{role}.pt', self.payload, checkpoint_role=role,
            config_path=CONFIG, config=self.config, data_root=self.data_root,
            data_contract=self.contract, smoke=False,
        )

    def test_relocated_copies_accept_both_roles_without_mutation(self):
        self.relocate()
        before = copy.deepcopy(self.payload)
        disk_before = {p.name: (sha256_file(p), p.stat().st_ino, p.stat().st_mtime_ns)
                       for p in self.ckpt_dir.iterdir()}
        for role in ('late', 'best'):
            self.assertFalse((self.ckpt_dir / f'{role}.pt').samefile(self.final))
            self.assertEqual(self.validate(role), sha256_file(self.final))
        self.assertEqual(self.payload, before)
        self.assertEqual(disk_before, {
            p.name: (sha256_file(p), p.stat().st_ino, p.stat().st_mtime_ns)
            for p in self.ckpt_dir.iterdir()})

    def test_same_root_hardlink_remains_accepted(self):
        alias = self.ckpt_dir / 'late.pt'
        alias.unlink()
        os.link(self.final, alias)
        self.assertEqual(self.validate(), sha256_file(self.final))

    def test_alias_wrong_content_is_rejected(self):
        (self.ckpt_dir / 'late.pt').write_bytes(b'wrong checkpoint')
        with self.assertRaises(ValueError):
            self.validate()

    def test_milestone_wrong_content_is_rejected(self):
        self.final.write_bytes(b'wrong checkpoint')
        with self.assertRaises(ValueError):
            self.validate()

    def test_index_wrong_hash_even_when_both_files_match_is_rejected(self):
        self.index['checkpoints'][-1]['sha256'] = '0' * 64
        self.save_index()
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_milestone_is_rejected(self):
        self.final.unlink()
        with self.assertRaises(ValueError):
            self.validate()

    def test_tampered_saved_identity_is_rejected_before_relocation(self):
        self.relocate()
        self.identity['data_root'] = '/tampered/root'
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.validate()

    def test_saved_root_inconsistent_with_saved_contract_is_rejected(self):
        self.relocate()
        self.identity['data_root'] = '/inconsistent/root'
        sign_identity(self.identity)
        with self.assertRaises(ValueError):
            self.validate()

    def test_current_root_inconsistent_with_contract_is_rejected(self):
        self.contract['root'] = '/inconsistent/current'
        with self.assertRaises(ValueError):
            self.validate()

    def test_relocation_rejects_changed_protocol_content(self):
        self.relocate()
        original = copy.deepcopy(self.contract)
        for name in ('benchmark_manifest_sha256', 'minimal_dataset_report_sha256',
                     'sha256', 'files', 'extra_protocol_field'):
            with self.subTest(field=name):
                self.contract = copy.deepcopy(original)
                self.contract[name] = [] if name == 'files' else 'changed'
                with self.assertRaises(ValueError):
                    self.validate()

    def test_relocation_rejects_changed_calibration_and_split(self):
        self.relocate()
        for entry in (2, 3):
            with self.subTest(entry=entry):
                changed = copy.deepcopy(self.contract)
                changed['files'][entry]['sha256'] = '0' * 64
                changed['sha256'] = sha256_json(changed['files'])
                previous = self.contract
                self.contract = changed
                with self.assertRaises(ValueError):
                    self.validate()
                self.contract = previous

    def test_recipe_and_base_weight_guards_remain_strict(self):
        self.relocate()
        original = copy.deepcopy(self.identity['files'])
        for field in ('config', 'official_gpt', 'vq', 'code'):
            with self.subTest(field=field):
                self.identity['files'] = copy.deepcopy(original)
                self.identity['files'][field] = (
                    {'csgo_seen10/peft.py': '0' * 64} if field == 'code' else '0' * 64)
                sign_identity(self.identity)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_budget_and_alias_role_guards_remain_strict(self):
        for section, field, value in (
            ('training_config', 'effective_batch_size', 256),
            ('training_config', 'max_optimizer_steps', 20000),
            ('training_config', 'checkpoint_steps', [3900, 7800, 11700, 15600, 19500]),
            ('sampler_state', 'num_replicas', 1),
            ('args', 'seed', 0),
        ):
            with self.subTest(field=field):
                original = self.payload[section][field]
                self.payload[section][field] = value
                with self.assertRaises(ValueError):
                    self.validate()
                self.payload[section][field] = original
        self.payload['best_step'] = 16000
        with self.assertRaises(ValueError):
            self.validate('best')

    def test_static_alias_check_opt_in_and_original_default(self):
        result = check_checkpoint_contract(
            self.root, 'late', checkpoint_steps=PEFT_CHECKPOINT_STEPS,
            allow_copied_aliases=True)
        self.assertEqual(result['late_step'], 19500)
        with self.assertRaisesRegex(ContractError, 'does not reference'):
            check_checkpoint_contract(self.root, 'late', checkpoint_steps=PEFT_CHECKPOINT_STEPS)
        (self.ckpt_dir / 'best.pt').write_bytes(b'wrong best')
        with self.assertRaises((ContractError, ValueError)):
            check_checkpoint_contract(
                self.root, 'late', checkpoint_steps=PEFT_CHECKPOINT_STEPS,
                allow_copied_aliases=True)

    def test_static_alias_check_rejects_noncanonical_index_path(self):
        replacement = self.ckpt_dir / 'other_checkpoint.pt'
        shutil.copyfile(self.final, replacement)
        self.index['checkpoints'][-1]['path'] = replacement.name
        self.save_index()
        with self.assertRaises(ContractError):
            check_checkpoint_contract(
                self.root, 'late', checkpoint_steps=PEFT_CHECKPOINT_STEPS,
                allow_copied_aliases=True)

    def test_static_alias_check_rejects_inconsistent_best_flags(self):
        self.index['checkpoints'][0]['is_best'] = True
        self.save_index()
        with self.assertRaises(ContractError):
            check_checkpoint_contract(
                self.root, 'late', checkpoint_steps=PEFT_CHECKPOINT_STEPS,
                allow_copied_aliases=True)

    def test_copied_best_before_final_uses_its_own_indexed_milestone(self):
        self.relocate()
        best_step = 16000
        self.index['best_step'] = best_step
        for record in self.index['checkpoints']:
            record['is_best'] = record['step'] == best_step
        self.save_index()
        selected = self.ckpt_dir / 'step_016000.pt'
        shutil.copyfile(selected, self.ckpt_dir / 'best.pt')
        self.payload.update(steps=best_step, global_optimizer_step=best_step,
                            consumed_samples=best_step * 128, best_step=best_step)
        self.assertEqual(self.validate('best'), sha256_file(selected))
        self.assertEqual(check_checkpoint_contract(
            self.root, 'best', checkpoint_steps=PEFT_CHECKPOINT_STEPS,
            allow_copied_aliases=True)['best_step'], best_step)

    def test_exact_training_resume_still_rejects_relocation(self):
        current = copy.deepcopy(self.identity)
        self.relocate()
        with self.assertRaises(ValueError):
            check_resume_identity(self.identity, current, profile='peft', allow_legacy=False)
        ledger = self.root / 'source_ledger.json'
        ledger.write_text(json.dumps({'profiles': {'peft': {
            'source': self.identity['files']['code'], 'target': current['files']['code'],
        }}}))
        with self.assertRaisesRegex(ValueError, 'data, path, config, weights, or split'):
            check_resume_identity(self.identity, current, profile='peft',
                                  allow_legacy=True, ledger_path=ledger)

    def make_predictions_metadata(self, *, include_origin=True, task='discrete'):
        """Only metadata is created; image completeness is mocked below."""
        count = 20000 if task == 'discrete' else 12800
        rows = [{'sample_id': f'de_dust2/frame_{i}', 'map_name': 'de_dust2',
                 'file_frame': f'frame_{i}'} for i in range(count)]
        out = self.root / 'predictions'
        inference = {
            'engine': 'compiled', 'batch_size': 16, 'compile_mode': 'reduce-overhead',
            'batching': 'fixed_manifest_blocks',
            'seed_policy': 'stateless-sample-v1: SHA256(UTF-8 bytes of str(inference_seed) + NUL + sample_id + NUL + decimal token index), first 64 bits mapped to (0, 1)',
            'target_loading': 'disabled', 'peft_merge': 'temporary_inference_model',
            'sampling_backend': 'compiled_logits+cuda_aten_fp32_inverse_cdf',
        }
        self.manifest_identity = {
            'experiment': EXPERIMENT, 'checkpoint_role': 'late',
            'config_path': str(CONFIG), 'config_sha256': sha256_file(CONFIG),
            'checkpoint_path': str(self.ckpt_dir / 'late.pt'),
            'checkpoint_sha256': sha256_file(self.final),
            'data_root': str(self.data_root), 'data_contract': self.contract,
            'data_contract_sha256': self.contract['sha256'],
            'vq_checkpoint_sha256': self.config['vq_sha256'],
            'seed': 42, 'inference_seed': 42, 'smoke_only': False, 'max_samples': None,
            'image_size': 448,
            'sampling': {k: self.config[k] for k in ('cfg_scale', 'temperature', 'top_k', 'top_p')},
            'inference': inference,
        }
        self.manifest_identity['checkpoint_origin'] = validate_inference_data_identity(
            self.identity, self.data_root, self.contract)
        ensure_peft_output_manifest(out, task=task, rows=rows, **self.manifest_identity)
        if not include_origin:
            # Simulate an artifact produced before provenance was introduced.
            self.manifest_identity.pop('checkpoint_origin')
            p = out / 'inference_manifest.json'
            legacy = json.loads(p.read_text())
            legacy.pop('checkpoint_origin')
            p.write_text(json.dumps(legacy))
        audit = {'rows': rows_contract(rows), 'expected_count': count,
                 'actual_count': count, 'complete': True}
        completion = {k: self.manifest_identity[k] for k in (
            'experiment', 'checkpoint_role', 'config_sha256', 'checkpoint_sha256',
            'data_contract_sha256', 'vq_checkpoint_sha256', 'seed', 'inference_seed')}
        completion.update(formal=True, selected_tasks=[task], complete=True, tasks={task: audit})
        (out / 'completion.json').write_text(json.dumps(completion))
        return out, rows, audit

    def preflight(self, out, rows, audit, *, task='discrete', evaluation=False):
        with patch('torch.load', return_value=self.payload), \
             patch('csgo_seen10.peft_artifact_contract.benchmark_data_contract', return_value=self.contract), \
             patch('csgo_seen10.data.read_benchmark_rows', return_value=rows) as read_rows, \
             patch('csgo_seen10.artifact_contract.audit_task_outputs', return_value=audit):
            if evaluation:
                result = preflight_evaluation(
                    out, task, self.ckpt_dir / 'late.pt', 'late', sha256_file(CONFIG), self.data_root)
            else:
                result = preflight_peft_inference(
                    out, task=task, config_path=CONFIG, checkpoint_path=self.ckpt_dir / 'late.pt',
                    checkpoint_role='late', data_root=self.data_root)
            self.assertFalse(read_rows.call_args.kwargs['require_images'])
            return result

    def test_migrated_prediction_preflights_for_both_tasks(self):
        self.relocate()
        for task in ('discrete', 'continuous'):
            with self.subTest(task=task):
                out, rows, audit = self.make_predictions_metadata(task=task)
                for evaluation in (False, True):
                    result = self.preflight(out, rows, audit, task=task, evaluation=evaluation)
                    self.assertEqual(result['checkpoint_sha256'], sha256_file(self.final))

    def test_legacy_same_root_manifest_stays_compatible_without_rewriting_origin(self):
        out, rows, audit = self.make_predictions_metadata(include_origin=False)
        origin = validate_inference_data_identity(self.identity, self.data_root, self.contract)
        ensure_peft_output_manifest(out, task='discrete', rows=rows,
                                    checkpoint_origin=origin, **self.manifest_identity)
        self.assertNotIn('checkpoint_origin', json.loads((out / 'inference_manifest.json').read_text()))
        self.preflight(out, rows, audit)
        self.preflight(out, rows, audit, evaluation=True)

    def test_migrated_manifest_requires_correct_origin(self):
        self.relocate()
        out, rows, audit = self.make_predictions_metadata()
        p = out / 'inference_manifest.json'
        original = json.loads(p.read_text())
        for change in ('missing', 'null', 'wrong'):
            manifest = copy.deepcopy(original)
            if change == 'missing':
                manifest.pop('checkpoint_origin')
            elif change == 'null':
                manifest['checkpoint_origin'] = None
            else:
                manifest['checkpoint_origin']['training_identity_sha256'] = '0' * 64
            p.write_text(json.dumps(manifest))
            for evaluation in (False, True):
                with self.subTest(change=change, evaluation=evaluation), self.assertRaisesRegex(ValueError, 'origin'):
                    self.preflight(out, rows, audit, evaluation=evaluation)
            with self.assertRaisesRegex(ValueError, 'origin'):
                ensure_peft_output_manifest(out, task='discrete', rows=rows, **self.manifest_identity)

    def test_old_prediction_paths_and_other_checkpoints_cannot_be_mixed(self):
        self.relocate()
        out, rows, audit = self.make_predictions_metadata()
        p = out / 'inference_manifest.json'
        original = json.loads(p.read_text())
        for field in ('data_root', 'config_path', 'checkpoint_path', 'checkpoint_sha256'):
            manifest = copy.deepcopy(original)
            manifest[field] = '/old/machine/path' if field.endswith(('root', 'path')) else '0' * 64
            p.write_text(json.dumps(manifest))
            for evaluation in (False, True):
                with self.subTest(field=field, evaluation=evaluation), self.assertRaises(ValueError):
                    self.preflight(out, rows, audit, evaluation=evaluation)
            with self.assertRaises(ValueError):
                ensure_peft_output_manifest(out, task='discrete', rows=rows, **self.manifest_identity)


if __name__ == '__main__':
    unittest.main()
