"""CPU tile-prefix semantic oracle; never used by the runtime selector."""
from itertools import accumulate


def prepare_reference(ids, tile_rows=512, experts=128, topk=8, intermediate_size=384):
    if len(ids) % topk or tile_rows <= 0 or experts <= 0 or intermediate_size not in (384, 768):
        raise ValueError('invalid routing dimensions')
    tiles = [ids[i:i + tile_rows] for i in range(0, len(ids), tile_rows)]
    counts = [[tile.count(e) for e in range(experts)] for tile in tiles]
    totals = [sum(c[e] for c in counts) for e in range(experts)]
    offsets = [0, *accumulate(totals)]
    cursors = offsets[:-1].copy()
    tile_offsets = []
    for c in counts:
        tile_offsets.append(cursors.copy())
        cursors = [a + b for a, b in zip(cursors, c)]
    a_map = [-1] * len(ids)
    c_map = [-1] * len(ids)
    for tile, values in enumerate(tiles):
        local = [0] * experts
        for j, expert in enumerate(values):
            if 0 <= expert < experts:
                dst = tile_offsets[tile][expert] + local[expert]
                local[expert] += 1
                route = tile * tile_rows + j
                a_map[dst] = route // topk
                c_map[route] = dst
    return {'counts': counts, 'tile_offsets': tile_offsets, 'offsets': offsets, 'a_map': a_map, 'c_map': c_map,
            'p1': [[c, 2 * intermediate_size, 4096] for c in totals], 'p2': [[c, 4096, intermediate_size] for c in totals]}
