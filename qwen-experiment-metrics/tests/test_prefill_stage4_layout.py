"""Symbolic ownership check for the Prefill FC2 strip staging and scatter."""


def test_stage4_scatter_preserves_every_element_for_all_partial_tiles():
    # Distinct row/column values detect both wrong ownership and swizzle errors.
    expected = list(range(128 * 256))
    for valid_m in range(129):
        observed = [-1] * len(expected)
        writes = [0] * len(expected)
        for warp in range(8):
            row_base = warp * 16
            for strip in range(4):
                stage = [-1] * (16 * 64)
                for lane in range(32):
                    row, col_pair = divmod(lane, 4)
                    for chunk in range(8):
                        col = chunk * 8 + col_pair * 2
                        swizzled = ((col // 8) ^ row) * 8 + col % 8
                        for local_row in (row, row + 8):
                            if row_base + local_row < valid_m:
                                source = (row_base + local_row) * 256 + strip * 64 + col
                                start = local_row * 64 + swizzled
                                stage[start:start + 2] = expected[source:source + 2]
                for lane in range(32):
                    for j in range(4):
                        row = j * 4 + lane // 8
                        col = (lane % 8) * 8
                        if row_base + row < valid_m:
                            swizzled = ((col // 8) ^ (row % 8)) * 8
                            dest = (row_base + row) * 256 + strip * 64 + col
                            start = row * 64 + swizzled
                            observed[dest:dest + 8] = stage[start:start + 8]
                            for offset in range(8):
                                writes[dest + offset] += 1
        end = valid_m * 256
        assert observed[:end] == expected[:end]
        assert all(count == 1 for count in writes[:end])
        assert all(value == -1 for value in observed[end:])
        assert not any(writes[end:])
