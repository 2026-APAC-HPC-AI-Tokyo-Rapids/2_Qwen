"""Check WGMMA register-to-output ownership and the combine quad transpose."""
import importlib.util
import unittest
from pathlib import Path


def permute(col):
    return (col & ~31) | ((col & 6) << 2) | ((col & 24) >> 2) | (col & 1)


class LayoutTests(unittest.TestCase):
    def test_register_scatter_ownership(self):
        # SM90 non-swap WGMMA: four lanes hold pairs of columns in each
        # eight-column chunk, with two rows eight apart per thread.
        for width in (64, 128, 256):
            for valid in (1, 7, 8, 9, 63, 64):
                writes = {}
                for warp in range(4):
                    for lane in range(32):
                        for half in range(2):
                            row = warp * 16 + lane // 4 + 8 * half
                            if row >= valid:
                                continue
                            for group in range(width // 32):
                                for pair in range(4):
                                    for bit in range(2):
                                        original = group * 32 + pair * 8 + lane % 4 * 2 + bit
                                        packed = group * 32 + lane % 4 * 8 + pair * 2 + bit
                                        self.assertNotIn((row, packed), writes)
                                        writes[row, packed] = original
                self.assertEqual(len(writes), valid * width)
                for (row, packed), original in writes.items():
                    self.assertEqual(permute(original), packed)
                    self.assertEqual(permute(packed), original)

    def test_actual_shuffle_network(self):
        # Execute the shuffle network simultaneously across a quad, starting
        # from symbolic original BF16 pairs. This checks source-lane selection.
        v = [[4*j+c for j in range(4)] for c in range(4)]
        for bit, pairs in [(1, [(0, 1), (2, 3)]), (2, [(0, 2), (1, 3)])]:
            for a, b in pairs:
                old = [x[:] for x in v]
                for c in range(4):
                    other = c ^ bit
                    value = old[other][a if other & bit else b]
                    if c & bit:
                        v[c][a] = value
                    else:
                        v[c][b] = value
        self.assertEqual(v, [[4*c+j for j in range(4)] for c in range(4)])

    def test_all_hidden_channels_and_tile_boundaries(self):
        self.assertEqual(sorted(map(permute, range(4096))), list(range(4096)))
        for width in (64, 128, 256):
            for begin in range(0, 4096, width):
                self.assertEqual({permute(i) for i in range(begin, begin+width)}, set(range(begin, begin+width)))

    def test_probe_rejects_missing_sections(self):
        path = Path(__file__).with_name('build-fc2-output-probe.py')
        spec = importlib.util.spec_from_file_location('probe', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with self.assertRaises(IndexError):
            mod.extract('not a MegaMoE header')


if __name__ == '__main__':
    unittest.main()
