"""CPU-only reuse guard tests using the actual runtime config and patch files."""
import copy
import hashlib
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / 'scripts/reuse-firmus-kernel-selection.py'
spec = importlib.util.spec_from_file_location('selection_reuse', HELPER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SelectionReuse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.repo = self.base / 'repo'
        self.runtime = self.base / 'runtime'
        self.config = json.loads((ROOT / 'configs/firmus-framework.json').read_text())
        patches = self.config['sglang_overlay_patches'] + self.config['deep_gemm_overlay_patches']
        extra = [self.config['prefill']['expert_location']]
        for name in patches + extra:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, path)
        def record(name):
            return {'path': name, 'sha256': hashlib.sha256((self.repo / name).read_bytes()).hexdigest()}
        self.record = {'schema': 'firmus-kernel-selection/v1', 'source_run': '40313004834',
                       'source_commit': 'e45acd554737ad0b75196c9064fe677ba53f5016',
                       'normalized_config_sha256': module.normalized_config_sha256(self.config),
                       'image_sha256': self.config['images']['sglang']['sha256'],
                       'patches': [record(name) for name in patches], 'extra_files': [record(name) for name in extra],
                       'environment': dict(module.ENVIRONMENT), 'evidence_sha256': {'selection.json': 'a' * 64}}

    def accept(self):
        payload = module.reuse(self.record, self.config, self.repo, self.runtime)
        for subdir in ('source', 'results'):
            saved = json.loads((self.runtime / subdir / 'reused-kernel-selection.json').read_text())
            self.assertTrue(saved['reused'] and saved['no_gpu_selection_rerun'])
            self.assertEqual(saved, payload)
        merged = {}
        for name in ('prefill-kernels.env', 'decode-kernels.env'):
            for line in (self.runtime / 'source' / name).read_text().splitlines():
                key, value = shlex.split(line)[0].split('=', 1)
                merged[key] = value
        self.assertEqual(merged, module.ENVIRONMENT)
        return payload

    def reject(self):
        with self.assertRaises((ValueError, KeyError, TypeError, OSError)):
            module.reuse(self.record, self.config, self.repo, self.runtime)
        self.assertFalse(self.runtime.exists(), 'Rejected inputs created output files')

    def test_actual_config_accepts_without_mutation(self):
        original = copy.deepcopy(self.config)
        self.accept()
        self.assertEqual(self.config, original)

    def test_draft_precision_record_path_and_image_paths_may_change(self):
        self.config['speculation']['draft_quantization'] = 'bf16'
        self.config['reuse_kernel_selection'] = 'different/record.json'
        for image in self.config['images'].values():
            image['path'] = '/same-bytes/new/path.sqsh'
            image['provenance'] = '/new/sidecar.json'
        self.accept()

    def test_other_runtime_setting_rejects(self):
        self.config['speculation']['draft_tokens'] += 1
        self.reject()

    def test_patch_order_rejects(self):
        self.record['patches'][0], self.record['patches'][1] = self.record['patches'][1], self.record['patches'][0]
        self.reject()

    def test_patch_content_rejects(self):
        (self.repo / self.record['patches'][0]['path']).write_text('changed source')
        self.reject()

    def test_image_sha_rejects(self):
        self.record['image_sha256'] = '0' * 64
        self.reject()

    def test_image_size_config_change_rejects(self):
        self.config['images']['sglang']['size_bytes'] += 1
        self.reject()

    def test_extra_file_hash_rejects(self):
        self.record['extra_files'][0]['sha256'] = '0' * 64
        self.reject()

    def test_missing_expert_mapping_rejects(self):
        self.record['extra_files'] = []
        self.reject()

    def test_unsafe_or_unreviewed_environment_rejects(self):
        for change in ({'PATH': '/tmp/unsafe'}, {'SGLANG_PREFILL_FA3_VERIFY': 'n192\necho unsafe'}, {'SGLANG_DECODE_DENSE_GEMM_O': 'other'}):
            with self.subTest(change=change):
                original = dict(self.record['environment'])
                self.record['environment'].update(change)
                self.reject()
                self.record['environment'] = original

    def test_missing_evidence_rejects(self):
        self.record['evidence_sha256'] = {}
        self.reject()

    def test_cli(self):
        record, config = self.base / 'record.json', self.base / 'config.json'
        record.write_text(json.dumps(self.record))
        config.write_text(json.dumps(self.config))
        result = subprocess.run([sys.executable, str(HELPER), '--record', str(record), '--config', str(config), '--repo', str(self.repo), '--runtime', str(self.runtime)], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('no GPU selection rerun', result.stdout)


if __name__ == '__main__':
    unittest.main()
