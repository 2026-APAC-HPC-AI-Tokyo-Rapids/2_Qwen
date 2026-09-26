import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class IncrementalForwardTests(unittest.TestCase):
    def test_warmup_excluded_and_completed_cells_survive_failure(self):
        bench = types.ModuleType('sglang.benchmark.one_batch')
        counter = 0

        def measure():
            nonlocal counter
            counter += 1
            if counter == 4:
                raise RuntimeError('next shape failed')
            return {'sample': counter}

        def latency_test(*args):
            for _ in range(4):
                bench.latency_test_run_once()

        bench.latency_test = latency_test
        bench.latency_test_run_once = measure
        benchmark = types.ModuleType('sglang.benchmark')
        benchmark.one_batch = bench
        with patch.dict(sys.modules, {'sglang': types.ModuleType('sglang'), 'sglang.benchmark': benchmark}):
            spec = importlib.util.spec_from_file_location('probe_fixture', Path(__file__).resolve().parents[1] / 'forward_probe.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.jsonl'
            args = types.SimpleNamespace(result_filename=str(path))
            with self.assertRaisesRegex(RuntimeError, 'next shape'):
                module.recording_latency_test(None, None, args, 0, 0)
            self.assertEqual([json.loads(line) for line in path.read_text().splitlines()], [{'sample': 2}, {'sample': 3}])
            self.assertIs(bench.latency_test_run_once, measure)
