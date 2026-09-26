import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("summary", Path(__file__).resolve().parents[1] / "summarize_probe.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class ProbeSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.measurements = [{"operation": "device_copy", "latency_ms_samples": [2] * 5, "accounted_bytes_per_iteration": 536870912}, {"operation": "bf16_gemm", "latency_ms_samples": [1] * 5, "flops_per_iteration": 2 * 4096**3}]
        for op in ("nccl_intra_node_all_reduce", "nccl_inter_node_pair_all_reduce"):
            for size in (1048576, 16777216, 268435456):
                self.measurements.append({"operation": op, "payload_bytes": size, "latency_ms_samples": [2] * 5})
        for rank in range(16):
            (self.root / f"rank-{rank}.json").write_text(json.dumps({"rank": rank, "job_id": "fixture", "hostname": f"node-{rank // 8}", "cluster": "aspire2ap", "measurements": self.measurements, "limitations": ["fixture"]}))

    def test_units_and_no_nixl_inference(self):
        result = summary.summarize(self.root)
        copy = next(m for m in result["measurements"] if m["operation"] == "device_copy")
        self.assertAlmostEqual(copy["read_plus_write_GB_per_second"]["median"], 268.435456)
        self.assertIsNone(result["nixl_bandwidth_gb_per_second"])
        self.assertFalse(result["accuracy_improvement_demonstrated"])

    def test_missing_rank_is_not_success(self):
        (self.root / "rank-15.json").unlink()
        with self.assertRaises(ValueError):
            summary.summarize(self.root)

    def test_zero_latency_is_rejected(self):
        p = self.root / "rank-0.json"
        data = json.loads(p.read_text())
        data["measurements"][0]["latency_ms_samples"][0] = 0
        p.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            summary.summarize(self.root)

    def add_nixl(self):
        points = [{"source_tp": tp, "prompt_tokens": tokens, "integrity_verified": True, "latency_ms_samples": [2, 3, 4, 5, 6], "physical_bytes": 8 * 188 * tokens * 256, "unique_kv_bytes": 4 * 188 * tokens * 256} for tp in (4, 8) for tokens in (128, 1024, 4000, 4200)]
        for p in self.root.glob("rank-*.json"):
            data = json.loads(p.read_text())
            data["nixl_measurements"] = points
            p.write_text(json.dumps(data))

    def test_nixl_physical_and_unique_rates(self):
        self.add_nixl()
        points = summary.summarize(self.root)["nixl_segmented_writes"]
        self.assertEqual(len(points), 8)
        self.assertEqual(points[0]["median_latency_ms"], 4)
        self.assertAlmostEqual(points[0]["physical_GB_per_second"], 8 * 188 * 128 * 256 / 4e6)
        self.assertEqual(points[0]["physical_GB_per_second"], 2 * points[0]["effective_unique_KV_GB_per_second"])

    def test_nixl_partial_results_rejected(self):
        self.add_nixl()
        p = self.root / "rank-15.json"
        data = json.loads(p.read_text())
        data["nixl_measurements"].pop()
        p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "incomplete NIXL"):
            summary.summarize(self.root)

    def test_nixl_failed_integrity_rejected(self):
        self.add_nixl()
        p = self.root / "rank-0.json"
        data = json.loads(p.read_text())
        data["nixl_measurements"][0]["integrity_verified"] = False
        p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "invalid NIXL"):
            summary.summarize(self.root)
