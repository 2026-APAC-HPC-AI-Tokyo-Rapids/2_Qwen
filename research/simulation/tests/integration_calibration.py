"""Run explicitly with the patched AISimulate uv environment (CPU only)."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from aisimulate_core.sdk import RustForwardPassPerfModel

spec = importlib.util.spec_from_file_location('calibrate', Path(__file__).resolve().parents[1] / 'calibrate.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NativeCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        source = self.root / 'fixture.json'
        source.write_text(json.dumps({'timing_boundary': 'fixture', 'plan': {'model_revision': 'fixture'}, 'measurements': [{'tp': 4, 'ep': 1, 'batch_size': b, 'input_len': length, 'prefill_ms_samples': [100] * 3, 'decode_ms_samples': [10] * 3} for b in [1, 2, 4] for length in [512, 2048, 4000, 4200]]}))
        self.parquet = module.export(source, self.root / 'data')
        os.environ['AIC_ALLOW_UNLISTED_VERSIONS'] = '1'
        self.config = {'model': str(Path('research/simulation/model').resolve()), 'system': 'h100_sxm', 'backend': 'sglang', 'backend_version': module.VERSION, 'worker_type': 'prefill', 'tp': 4, 'pp': 1, 'attention_dp': 1, 'moe_tp_size': 4, 'moe_ep_size': 1, 'fmha_quant_mode': 'bfloat16', 'kvcache_quant_mode': 'bfloat16', 'estimation_mode': 'fpm_interpolation', 'fallback_policy': 'deny', 'estimator_config': {'fpm_interpolation': {'fpm_parquet_path': str(self.parquet)}}}

    def test_measured_policy_and_interpolation(self):
        model = RustForwardPassPerfModel.best_available(self.config)
        try:
            self.assertEqual(model.diagnostics()['readiness'], 'ready')
            result = model.estimate_forward_pass_time_ms({'scheduled_requests': {'num_prefill_requests': 2, 'sum_prefill_tokens': 8000, 'sum_prefill_kv_tokens': 0, 'num_decode_requests': 0, 'sum_decode_kv_tokens': 0}})
            self.assertAlmostEqual(result, 100)
        finally:
            model.close()

    def test_unknown_policy_still_rejected(self):
        metadata = self.parquet.with_suffix('.metadata.json')
        value = json.loads(metadata.read_text())
        value['measurement_policy'] = 'unknown_policy'
        metadata.write_text(json.dumps(value))
        with self.assertRaisesRegex(Exception, 'measurement_policy'):
            RustForwardPassPerfModel.best_available(self.config)

    def test_hash_mismatch_still_rejected(self):
        metadata = self.parquet.with_suffix('.metadata.json')
        value = json.loads(metadata.read_text())
        value['parquet_sha256'] = '0' * 64
        metadata.write_text(json.dumps(value))
        with self.assertRaisesRegex(Exception, 'digest mismatch'):
            RustForwardPassPerfModel.best_available(self.config)


if __name__ == '__main__':
    unittest.main()
