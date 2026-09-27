import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "collect.py"
spec = importlib.util.spec_from_file_location("collect", MODULE)
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, value):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
        return p

    def fixture(self, cluster="aspire2ap", profile=False):
        self.write("source/source/provenance.json", {"schema_version": "qwen3-pd-provenance/v1", "cluster": cluster, "profile": profile, "container": {"sha256": "fixture"}})
        self.write("source/source/model.json", {"model": {"revision": "fixture"}})
        self.write("source/source/container-sidecar.json", {"source_commits": {"sglang": "fixture"}})
        self.write("score.json", {"concurrency": 16, "official_validation": {"valid": True}})
        self.write("source/source/aspire-serving.json", {"concurrency": 16})
        (self.root / "source/source/node-addresses.tsv").write_text("0\tnode-a\t10.0.0.1\n1\tnode-b\t10.0.0.2\n")
        (self.root / "run-success").touch()
        logs = self.root / "logs"
        logs.mkdir()
        (logs / "prefill-0.log").write_text("server_args=ServerArgs(tp_size=4, ep_size=1, speculative_algorithm=None)\n")
        (logs / "node-0-environment.log").write_text("GPU 0: NVIDIA H100 80GB HBM3\n")
        data = {k: {"unit": unit, "avg": 1.0} for k, unit in collect.METRICS.items()}
        data.update({"benchmark_id": "fixture", "was_cancelled": False, "error_summary": [], "input_config": {"phases": [{"kind": "profiling", "type": "concurrency", "concurrency": 16}]}})
        self.write("results/profile_export_aiperf.json", data)
        return data

    def test_nested_source_and_duplicate_export(self):
        data = self.fixture()
        self.write("results/artifacts/final/profile_export_aiperf.json", data)
        self.write("results/warmup/profile_export_aiperf.json", data)
        result = collect.collect(self.root, "aspire2ap")
        self.assertEqual(result["summary"]["target_exports"], 1)
        self.assertEqual(len(result["runs"][0]["duplicate_exports"]), 1)
        self.assertTrue(result["runs"][0]["candidate_for_matched_comparison"])

    def test_routing_policy_separates_comparison_groups(self):
        self.fixture()
        first = collect.collect(self.root, "aspire2ap")["runs"][0]["condition_key"]
        self.write("source/source/aspire-serving.json", {"concurrency": 16, "prefill_routing_policy": "power_of_two"})
        second = collect.collect(self.root, "aspire2ap")["runs"][0]["condition_key"]
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)

    def test_load_poll_period_separates_comparison_groups(self):
        self.fixture()
        self.write("source/source/aspire-serving.json", {"concurrency": 16, "prefill_routing_policy": "power_of_two"})
        first = collect.collect(self.root, "aspire2ap")["runs"][0]["condition_key"]
        self.write("source/source/aspire-serving.json", {"concurrency": 16, "prefill_routing_policy": "power_of_two", "router_load_poll_seconds": 1})
        second = collect.collect(self.root, "aspire2ap")["runs"][0]["condition_key"]
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)

    def test_physical_node_pair_separates_comparison_groups(self):
        self.fixture()
        first = collect.collect(self.root, "aspire2ap")["runs"][0]["condition_key"]
        (self.root / "source/source/node-addresses.tsv").write_text("0\tnode-c\t10.0.0.3\n1\tnode-d\t10.0.0.4\n")
        second = collect.collect(self.root, "aspire2ap")["runs"][0]["condition_key"]
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)

    def test_missing_physical_nodes_blocks_comparison(self):
        self.fixture()
        (self.root / "source/source/node-addresses.tsv").unlink()
        run = collect.collect(self.root, "aspire2ap")["runs"][0]
        self.assertIsNone(run["condition_key"])
        self.assertIn("missing_physical_node_identity", run["comparison_blockers"])

    def test_other_backend_requires_native_worker_evidence(self):
        self.fixture()
        self.write("source/source/aspire-serving.json", {"concurrency": 16, "backend": "vllm"})
        run = collect.collect(self.root, "aspire2ap")["runs"][0]
        self.assertIsNone(run["condition_key"])
        self.assertIn("unsupported_backend_worker_evidence", run["comparison_blockers"])

    def test_other_h100_cluster_is_not_aspire(self):
        self.fixture(cluster="genkai")
        result = collect.collect(self.root, "aspire2ap")
        self.assertEqual(result["summary"]["candidate_runs"], 0)
        self.assertIn("different_or_unknown_cluster", result["runs"][0]["comparison_blockers"])

    def test_profile_cannot_be_normal_benchmark(self):
        self.fixture(profile=True)
        result = collect.collect(self.root, "aspire2ap")
        self.assertEqual(result["summary"]["candidate_runs"], 0)

    def test_missing_validation_rejected(self):
        self.fixture()
        (self.root / "score.json").unlink()
        self.assertEqual(collect.collect(self.root, "aspire2ap")["summary"]["candidate_runs"], 0)

    def test_score_is_traceable_and_must_match_committed_concurrency(self):
        self.fixture()
        self.write("source/source/aspire-serving.json", {"concurrency": 40})
        run = collect.collect(self.root, "aspire2ap")["runs"][0]
        self.assertIn("configured_concurrency_mismatch", run["comparison_blockers"])
        self.assertEqual(run["local_scores"][0]["value"]["concurrency"], 16)
        self.assertEqual(run["local_scores"][0]["source"], collect.evidence(self.root / "score.json"))

    def test_score_must_match_actual_profiling_concurrency(self):
        data = self.fixture()
        data["input_config"]["phases"][0]["concurrency"] = 48
        self.write("results/profile_export_aiperf.json", data)
        run = collect.collect(self.root, "aspire2ap")["runs"][0]
        self.assertIn("benchmark_concurrency_mismatch", run["comparison_blockers"])

    def test_speculation_is_a_comparison_condition(self):
        self.fixture()
        def key():
            return collect.collect(self.root, "aspire2ap")["runs"][0]["condition_key"]
        baseline = key()
        log = self.root / "logs/prefill-0.log"
        log.write_text("server_args=ServerArgs(tp_size=4, ep_size=1, speculative_algorithm='NGRAM', speculative_num_draft_tokens=4)\n")
        ngram = key()
        self.assertIsNotNone(ngram)
        self.assertNotEqual(baseline, ngram)
        log.write_text(log.read_text().replace("tokens=4", "tokens=8"))
        self.assertNotEqual(ngram, key())

    def test_deterministic_mode_is_a_comparison_condition(self):
        self.fixture()
        first = collect.collect(self.root, "aspire2ap")["runs"][0]["condition_key"]
        log = self.root / "logs/prefill-0.log"
        log.write_text(log.read_text().replace("speculative_algorithm=None", "enable_deterministic_inference=True, speculative_algorithm=None"))
        self.assertNotEqual(first, collect.collect(self.root, "aspire2ap")["runs"][0]["condition_key"])

    def test_draft_weights_and_runtime_config_identify_eagle(self):
        self.fixture()
        (self.root / "logs/prefill-0.log").write_text("server_args=ServerArgs(tp_size=4, ep_size=1, speculative_algorithm='EAGLE3')\n")
        run = collect.collect(self.root, "aspire2ap")["runs"][0]
        self.assertIn("missing_draft_identity", run["comparison_blockers"])
        draft = {"repository": "fixture", "revision": "one", "weight_sha256": "weights1", "runtime_config_sha256": "config1"}
        self.write("source/source/draft-provenance.json", draft)
        first = collect.collect(self.root, "aspire2ap")["runs"][0]["condition_key"]
        self.assertIsNotNone(first)
        draft["weight_sha256"] = "weights2"
        self.write("source/source/draft-provenance.json", draft)
        self.assertNotEqual(first, collect.collect(self.root, "aspire2ap")["runs"][0]["condition_key"])

    def test_unit_mismatch_is_not_rescaled_silently(self):
        data = self.fixture()
        data["time_to_first_token"]["unit"] = "s"
        p = self.write("results/profile_export_aiperf.json", data)
        self.assertIn("missing_or_wrong_unit:time_to_first_token", collect.export_metrics(p)["issues"])

    def test_port_counter_four_octets_and_reset(self):
        p = self.root / "counters.csv"
        p.write_text("timestamp,device,port_xmit_data,port_rcv_data\n2026-09-26T00:00:00Z,mlx5_0,100,200\n2026-09-26T00:00:02Z,mlx5_0,150,300\n2026-09-26T00:00:03Z,mlx5_0,0,0\n")
        d = collect.counter_rates(p)["devices"]["mlx5_0"]
        self.assertEqual(d["tx_bytes_per_second"], 100)
        self.assertEqual(d["rx_bytes_per_second"], 200)
        self.assertEqual(d["discarded_intervals"], 1)


if __name__ == "__main__":
    unittest.main()
