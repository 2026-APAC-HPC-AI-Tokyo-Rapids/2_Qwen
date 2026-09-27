import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("serving_profile", Path(__file__).parents[1] / "summarize_serving_profile.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ServingProfileTests(unittest.TestCase):
    def test_counter_difference_excludes_prior_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"metrics-before-prefill-30100.prom").write_text('sglang:latency_seconds_sum{stage="prefill"} 1.0\nsglang:latency_seconds_count{stage="prefill"} 2\n')
            (root/"metrics-after-prefill-30100.prom").write_text('sglang:latency_seconds_sum{stage="prefill"} 7.0\nsglang:latency_seconds_count{stage="prefill"} 5\n')
            row = module.summarize(root)["workers"]["prefill-30100"][0]
            self.assertEqual(row["count"], 3)
            self.assertEqual(row["mean"], 2)

    def test_counter_reset_rejects_misleading_negative_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"metrics-before-decode-30200.prom").write_text('sglang:x_sum 2\nsglang:x_count 2\n')
            (root/"metrics-after-decode-30200.prom").write_text('sglang:x_sum 1\nsglang:x_count 1\n')
            with self.assertRaisesRegex(ValueError, "counter reset"):
                module.summarize(root)

    def test_missing_capture_is_not_empty_success(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "no profile"):
            module.summarize(Path(directory))
