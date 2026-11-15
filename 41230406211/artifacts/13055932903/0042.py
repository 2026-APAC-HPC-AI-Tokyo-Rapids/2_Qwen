"""Exhaustive logical queue model; does not model CUDA memory visibility."""
import json
from pathlib import Path

records = []
for n in range(1, 23):
    initial = (0,) * 8
    seen, todo, terminal = {initial}, [initial], 0
    while todo:
        state = todo.pop()
        k, v, r0, r1, p0, p1, o0, o1 = state
        assert 0 <= k - v <= 1
        assert all(0 <= r - p <= 1 for r, p in ((r0, p0), (r1, p1)))
        assert k - min(r0, r1) <= 2 and v - min(o0, o1) <= 2
        assert p0 - o0 <= 2 and p1 - o1 <= 2
        if all(x == n for x in state):
            terminal += 1
            continue
        successors = []

        def advance(index, state=state, successors=successors):
            row = list(state)
            row[index] += 1
            successors.append(tuple(row))

        if k == v and k < n and k - min(r0, r1) < 2:
            advance(0)
        if v < k and v - min(o0, o1) < 2:
            advance(1)
        for i in (0, 1):
            r, p, o = state[2 + i], state[4 + i], state[6 + i]
            if r == p and r < k:
                advance(2 + i)
            if r == p + 1 and p - o < 2:
                advance(4 + i)
            if o < p and o < v:
                advance(6 + i)
        assert successors, ('deadlock', n, state)
        for row in successors:
            if row not in seen:
                seen.add(row)
                todo.append(row)
    assert terminal == 1
    records.append({'kv_blocks': n, 'reachable_states': len(seen), 'terminal_states': terminal, 'deadlocks': 0})

# Each SM subpartition initially owns five warps with 96 registers/lane.
# Producer 96->32 releases 64, covering the two QK 96->128 increases.
# One QK 128->24 release covers its corresponding PV 96->176 increase.
assert 96 - 32 == 2 * (128 - 96)
assert 128 - 24 > 176 - 96
assert 2 * 128 + 2 * 96 + 32 == 5 * 96 <= 512
assert 2 * 24 + 2 * 176 + 32 <= 5 * 96
record = {'gpu_executed': False, 'scope': 'Logical queues and per-SM-subpartition CTA pool only', 'cases': records, 'initial_cta_pool_required': 61440, 'active_role_registers': 61440, 'requires_native_initial_registers': 96, 'threads': 640}
Path(__file__).with_name('protocol.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
