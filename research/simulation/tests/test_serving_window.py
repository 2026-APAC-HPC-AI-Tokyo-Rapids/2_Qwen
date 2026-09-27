import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("serving_window", Path(__file__).parents[1] / "serving_window.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def request(arrival, end, ttft=100):
    return {"arrival_time_ms": arrival, "terminal_time_ms": end, "terminal_status": "completed", "e2e_latency_ms": end-arrival, "ttft_ms": ttft, "itl_ms": 10}


class ServingWindowTests(unittest.TestCase):
    def test_warmup_is_excluded_and_drain_is_counted(self):
        rows = [request(59000, 61000), request(60000, 62000), request(179000, 181000)]
        result = module.summarize(rows)
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["elapsed_seconds"], 121)
        self.assertAlmostEqual(result["good_tokens_per_second"], 400/121)

    def test_arrivals_during_drain_require_a_new_simulation(self):
        with self.assertRaisesRegex(ValueError, "rerun"):
            module.summarize([request(60000, 62000), request(180000, 182000)])

    def test_incomplete_measurement_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "before the measurement"):
            module.summarize([request(60000, 62000)])

    def test_sla_failure_is_not_silently_dropped(self):
        result = module.summarize([request(60000, 62000), request(179000, 183000, ttft=3100)])
        self.assertEqual(result["joint_sla_fraction"], .5)
        self.assertEqual(result["good_tokens_per_second"], 0)
