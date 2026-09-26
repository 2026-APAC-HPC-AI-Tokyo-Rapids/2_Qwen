import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('calibrate', Path(__file__).resolve().parents[1] / 'calibrate.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CalibrationTests(unittest.TestCase):
    def test_quantization_coordinates_and_replica_pooling(self):
        point = {'tp': 4, 'ep': 1, 'batch_size': 2, 'input_len': 4000, 'prefill_ms_samples': [90, 100, 110], 'decode_ms_samples': [8, 10, 12]}
        replica = {**point, 'prefill_ms_samples': [110, 120, 130], 'decode_ms_samples': [10, 12, 14]}
        rows = module.rows_from_summary({'measurements': [point, replica]}, '/model')
        self.assertEqual(len(rows), 2)
        prefill, decode = rows
        self.assertEqual(prefill['latency_ms'], 110)
        self.assertEqual(prefill['total_prefill_tokens'], 8000)
        self.assertEqual(prefill['moe_quant_mode'], 'fp8_block')
        self.assertEqual(decode['latency_ms'], 11)
        self.assertEqual(decode['total_kv_read_tokens'], 8032)
        self.assertEqual(module.rows_from_summary({'measurements': [point]}, '/model', held_out_length=4000), [])

    def test_partial_repetition_rejected(self):
        point = {'tp': 8, 'ep': 8, 'batch_size': 1, 'input_len': 4000, 'prefill_ms_samples': [90], 'decode_ms_samples': [10]}
        with self.assertRaisesRegex(ValueError, 'three repetitions'):
            module.rows_from_summary({'measurements': [point]}, '/model')

    def test_real_prefix_coordinates_and_duplicate_observations(self):
        point = {'tp': 4, 'ep': 1, 'batch_size': 2, 'input_len': 4000, 'prefill_ms_samples': [90, 100, 110], 'decode_ms_samples': [8, 10, 12]}
        chunk = {'tp': 4, 'ep': 1, 'batch_size': 2, 'input_len': 512, 'prefix_len': 192, 'kind': 'prefill', 'latency_ms_samples': [20, 22, 24]}
        decode = {**chunk, 'kind': 'decode', 'input_len': 4000, 'prefix_len': 0, 'latency_ms_samples': [11, 13, 15]}
        rows = module.rows_from_summary({'measurements': [point], 'chunk_measurements': [chunk, decode]}, '/model')
        self.assertEqual(len(rows), 3)
        continuation = next(row for row in rows if row['workload_kind'] == 'prefill' and row['total_kv_read_tokens'])
        self.assertEqual(continuation['total_kv_read_tokens'], 384)
        self.assertEqual(continuation['total_prefill_tokens'], 1024)
        self.assertEqual(next(row for row in rows if row['workload_kind'] == 'decode')['latency_ms'], 11.5)
