"""CPU mirror of L's integer 7/10, expert-order M-block assignment (no CUDA)."""
import json
import runpy
from collections import Counter
from pathlib import Path


def assign(rows):
    loads = [0] * 11
    groups = [[] for _ in loads]
    pool = 0
    for expert, count in enumerate(rows):
        for m in range((count + 127) // 128):
            winner = min(range(11), key=lambda g: (loads[g], g))
            valid = min(128, count - m * 128)
            groups[winner].append((expert, m, pool + m, valid))
            loads[winner] += 7 if valid <= 64 else 10
        pool += (count + 127) // 128
    return groups


def distribution(values):
    return {'min': min(values), 'max': max(values), 'mean': sum(values)/len(values), 'histogram': dict(sorted(Counter(round(v,6) for v in values).items()))}


def model(rows):
    tiles = [(e, m, min(128, count-m*128)) for e, count in enumerate(rows) for m in range((count+127)//128) for _ in range(12)]
    control = [tiles[c::132] for c in range(132)]
    groups = assign(rows)
    result = {'expert_rows': rows, 'blocks': len(tiles)//12, 'tiles': len(tiles), 'candidate': {}, 'control': {}}
    for label, weight in (('mma', .5), ('cost07', .7)):
        result['control'][label] = distribution([sum(weight if t[2]<=64 else 1 for t in c) for c in control])
        result['candidate'][label] = distribution([round(sum(weight if t[3]<=64 else 1 for t in group), 6) for group in groups for _ in range(12)])
    result['candidate']['tiles'] = distribution([len(group) for group in groups for _ in range(12)])
    return result


def fixtures():
    import numpy as np
    root = Path(__file__).resolve().parents[1]
    cases = runpy.run_path(str(root/'scripts/run-prefill-megamoe-diag.py'))['routing_cases'](np, root/'kernels/probes/megamoe-local-ep1-routing.npz')
    return {name: model(np.bincount(ids.ravel(), minlength=128).tolist()) for name, ids in cases.items()}


if __name__ == '__main__':
    print(json.dumps(fixtures(), indent=2))
