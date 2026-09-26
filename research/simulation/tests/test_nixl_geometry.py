import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('nixl_probe', Path(__file__).resolve().parents[1] / 'nixl_probe.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GeometryTests(unittest.TestCase):
    def test_replica_mapping_preserves_qwen_kv_heads(self):
        for source_tp in (4, 8):
            for destination_tp in (4, 8):
                destinations = []
                for rank in range(8):
                    peers = module.destinations(source_tp, destination_tp, rank)
                    for peer in peers:
                        self.assertEqual(rank // (source_tp // 4), (peer - 8) // (destination_tp // 4))
                    destinations.extend(peers)
                self.assertEqual(sorted(destinations), list(range(8, 8 + destination_tp)))
