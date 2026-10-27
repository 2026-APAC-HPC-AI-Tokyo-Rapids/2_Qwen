"""CPU ownership and patch-scope fixture for the bounded QKV gate."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("prefill_memory", Path(__file__).with_name("prefill-memory.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class HeadOwnership(unittest.TestCase):
    def test_head4_boundary_coverage(self):
        for rows in (0, 1, 7, 8, 9, 31, 32, 33, 129, 3986, 4032, 4096):
            with self.subTest(rows=rows):
                probe.check_mapping(4, rows)

    def test_missing_source_anchor_fails_closed(self):
        with self.assertRaises((AssertionError, ValueError)):
            probe.tiled_header("unrelated source", 4)

    def test_patch_changes_only_prefill(self):
        patch = (Path(__file__).parents[2] / "patches/sglang/prefill-qkv-memory.patch").read_text()
        self.assertIn("constexpr int kHeadTile=4;", patch)
        self.assertNotIn("run_decode", patch)
        self.assertIn("num_qo_heads==64 && num_kv_heads==4", patch)


if __name__ == "__main__":
    unittest.main()
