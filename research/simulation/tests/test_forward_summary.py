import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('forward_summary', Path(__file__).resolve().parents[1] / 'summarize_forward.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ForwardSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = {'workers': [{'name': 'prefill-0', 'tp': 4, 'ep': 1}], 'input_lengths': [4000], 'batch_sizes': [1], 'output_length': 32, 'repetitions': 3}
        (self.root / 'forward-plan.json').write_text(json.dumps(self.plan))
        self.rows = [{'run_name': 'prefill-0', 'batch_size': 1, 'input_len': 4000, 'output_len': 32, 'prefill_latency': .1, 'median_decode_latency': .01}] * 3
        self.write()

    def write(self):
        (self.root / 'prefill-0.jsonl').write_text('\n'.join(json.dumps(r) for r in self.rows))

    def test_units_and_boundary(self):
        result = module.summarize(self.root)
        self.assertEqual(result['measurements'][0]['prefill_ms'], 100)
        self.assertEqual(result['measurements'][0]['decode_ms'], 10)
        self.assertFalse(result['official_result'])

    def test_missing_repetition(self):
        self.rows.pop()
        self.write()
        with self.assertRaises(ValueError):
            module.summarize(self.root)

    def test_invalid_latency(self):
        self.rows[0]['prefill_latency'] = -1
        self.write()
        with self.assertRaises(ValueError):
            module.summarize(self.root)
