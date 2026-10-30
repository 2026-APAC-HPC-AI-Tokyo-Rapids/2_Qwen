"""CPU coverage of the four-KV-heads-per-CTA stream-split Decode FA3 verify row and its PP4 gate."""

import ast
import json
import random
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "kernels/fa3-verify"
NAME = "p32stream-h4-fp8"
SLOTS, SMS, BLOCK_N = 6, 132, 32


def selector():
    return runpy.run_path(str(ROOT / "scripts/select-firmus-decode-fa3.py"))


def slot_counts():
    path = ROOT / "scripts/check-firmus-decode-kernels.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "stream_slot_counts")
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)  # noqa: S102 - tracked GPU-free helper
    return ns["stream_slot_counts"]


def segments(lengths):
    """VerifyStreamScheduler, CTA by CTA: (cta, request, lo, hi, slot) in kBlockN blocks."""
    blocks = [-(-x // BLOCK_N) for x in lengths]
    incl = [sum(blocks[:b + 1]) for b in range(len(blocks))]
    total = incl[-1]
    groups = min(SMS, total // -(-max(blocks) // (SLOTS - 1)))

    def cta_of(block):
        return ((block + 1) * groups - 1) // total

    out = []
    for cta in range(groups):
        end = (cta + 1) * total // groups
        block = cta * total // groups
        while block < end and any(x > block for x in incl):
            b = next(i for i, x in enumerate(incl) if x > block)
            start = 0 if b == 0 else incl[b - 1]
            out.append((cta, b, block - start, min(end, incl[b]) - start, cta - cta_of(start)))
            block = incl[b]
    return out, groups


def test_row_is_the_only_four_head_row_and_fits_shared_memory():
    rows = json.loads((KERNELS / "candidates.json").read_text())
    four = [r for r in rows if r.get("heads", 1) == 4]
    assert four == [{"name": NAME, "block_n": BLOCK_N, "stages": 3, "splits": SLOTS, "output": "fp8", "heads": 4}]
    # SharedStorageSize computed on the host from the patched templates (CUDA 13.4): 198656 B; build.py asserts it.
    assert 198656 <= runpy.run_path(str(KERNELS / "build.py"))["SM90_OPTIN_BYTES"]


def test_build_applies_the_stream_patch_last_and_generates_a_distinct_main(tmp_path):
    build = (KERNELS / "build.py").read_text()
    assert build.index('"prefill-pipe.patch"') < build.index('"decode-stream-h4.patch"') < build.index('cuda = Path("/usr/local/cuda")')
    ns = runpy.run_path(str(KERNELS / "build.py"))
    sources = {name: path.read_text() for path, name, _ in ns["generate_dispatch"](KERNELS, tmp_path)}
    assert "#define VERIFY_HEADS 4\n" in sources["verify_n32_s3_split1_merge0_h4"]
    assert "#define VERIFY_BLOCK_N 32\n" in sources["verify_n32_s3_split1_merge0_h4"]
    assert f'{{"{NAME}", 32, 3, 6, true, false, false, 1, 4, verify_n32_s3_split1_merge0_h4}},' in (tmp_path / "candidates.h").read_text()


def test_patch_encodes_and_decodes_the_same_segment_fields():
    patch = (KERNELS / "decode-stream-h4.patch").read_text()
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    assert "return {b, slot | ((block - start) << 3) | ((min(end, stop) - start) << 17), stop};" in added
    assert "n_block_min = (split_idx >> 3) & 0x3FFF;" in added and "n_block_max = std::min(n_block_max, (split_idx >> 17) & 0x3FFF);" in added
    assert added.count("split_idx &= 0x7;") == 2  # store and store_zero
    assert "static constexpr int kVerifyStreamSlots = 6;" in added
    assert "VerifyHeads == 4, flash::VerifyStreamScheduler<kBlockN, kVerifyStreamSlots>" in added
    # 640 threads: one cp.async producer warpgroup and four one-head consumer warpgroups fill the register file.
    assert "kHeads == 4 ? 64 :" in added and "kHeads == 4 ? 112 :" in added
    assert 128 * (64 + 4 * 112) == 65536
    # Whole 512-byte token rows per warp instruction.
    assert "static constexpr int kBlockKGmem = FullRow ? kHeadDimGCD :" in added and "Transpose_V || !IntraWGOverlap /*KV_Same_Iter*/, 1, kHeads == 4>;" in added


def test_wrapper_and_combine_agree_on_six_slots_from_the_main_kernel():
    wrapper = (KERNELS / "wrapper.cpp").read_text()
    assert "else if (candidate(id).heads != 4) { prepare_varlen_num_blocks(" in wrapper
    assert "candidate(id).heads != 4 || (p.num_splits == 6 && candidate(id).fp8" in wrapper
    combine = (KERNELS / "combine.cu").read_text()
    assert "if (S == 6) { active = dynamic[token / 4]; }" in combine
    assert "else if (p.num_splits == 6) { launch_combine<6, true>(p, stream); }" in combine


@pytest.mark.parametrize("batch", [44, 46, 48])
def test_partition_balances_the_selection_lengths(batch):
    lengths = [3900 + (37 * i + 13) % 301 for i in range(batch)]
    out, groups = segments(lengths)
    assert groups == SMS
    work = [0] * groups
    for cta, _, lo, hi, _ in out:
        work[cta] += hi - lo
    total = sum(-(-x // BLOCK_N) for x in lengths)
    assert max(work) - min(work) <= 1 and sum(work) == total
    # At most two segments (two FP32 partials) per CTA; about SMs + batch partials in all.
    assert max(sum(1 for s in out if s[0] == cta) for cta in range(groups)) == 2
    assert len(out) <= SMS + batch


def test_partition_covers_every_block_once_within_six_slots():
    rng = random.Random(7)
    counts_of = slot_counts()
    cases = [[3900 + (37 * i + 13) % 301 for i in range(b)] for b in (32, 40, 48)]
    cases += [([4, 63, 64, 65, 95, 96, 97, 127, 128, 129] * 5)[:b] for b in range(32, 49)]
    cases += [[rng.choice([1, 4, 64, 65, rng.randint(1, 40960)]) for _ in range(rng.randint(1, 64))] for _ in range(300)]
    for lengths in cases:
        out, _ = segments(lengths)
        seen = set()
        for _, b, lo, hi, slot in out:
            assert 0 <= lo < hi < 1 << 14 and 0 <= slot < SLOTS
            assert not seen & {(b, k) for k in range(lo, hi)}
            seen |= {(b, k) for k in range(lo, hi)}
        assert len(seen) == sum(-(-x // BLOCK_N) for x in lengths)
        counts = counts_of(lengths, BLOCK_N)
        for b in range(len(lengths)):
            assert sorted(s[4] for s in out if s[1] == b) == list(range(counts[b]))
        assert max(counts) <= SLOTS


def endurance_records(entry=None):
    ns = selector()
    good = {"replays": ns["ENDURANCE_REPLAYS"] + 1, "bytes_equal": True}
    ranks = []
    for rank in range(4):
        batches = {}
        for batch in ("44", "46", "48"):
            batches[batch] = {NAME: {"numeric_passed": True, "endurance": dict(good), "blocks": [{"stock": [74.0], "candidate": [68.0]} for _ in range(2)]}}
        ranks.append({"rank": rank, "batches": batches})
    if entry is not None:
        entry(ranks[2]["batches"]["46"][NAME])
    return ns, ranks


def test_screen_adopts_the_stream_row_only_after_its_endurance_gate():
    ns, ranks = endurance_records()
    assert ns["PINNED_SCREEN"] == (NAME,)
    result = ns["choose_pinned"](ranks, (44, 46, 48), "p128split2-s3-fp8", ns["PINNED_SCREEN"])
    assert result["winner"] == NAME and result["env"] == {"SGLANG_DECODE_FA3_VERIFY": NAME}
    assert result["rule"]["endurance_replays"] == ns["ENDURANCE_REPLAYS"] + 1 >= 300


@pytest.mark.parametrize("failure", ["missing", "mismatch", "short", "error"])
def test_a_failed_endurance_gate_on_one_gpu_keeps_the_pin(failure):
    def damage(entry):
        if failure == "missing":
            del entry["endurance"]
        elif failure == "mismatch":
            entry["endurance"]["bytes_equal"] = False
        elif failure == "short":
            entry["endurance"]["replays"] -= 1
        else:
            entry["endurance"] = {"error": "RuntimeError: x"}

    ns, ranks = endurance_records(damage)
    result = ns["choose_pinned"](ranks, (44, 46, 48), "p128split2-s3-fp8", ns["PINNED_SCREEN"])
    assert result["winner"] is None and result["env"] == {}
    assert result["choices"][NAME]["reason"] == "endurance-failed"


def test_endurance_replays_back_to_back_with_rotating_lengths_and_compares_the_first_output():
    ns = selector()

    class Lengths(list):
        def clone(self):
            return Lengths(self)

        def __sub__(self, k):
            return Lengths(x - k for x in self)

        def contiguous(self):
            return self

        def copy_(self, other):
            self[:] = other

    lengths = Lengths([4000, 4100, 4200])
    seen = []

    class Step:
        output = None

    class Graph:
        def replay(self):
            seen.append(tuple(lengths))
            Step.output = Lengths(lengths)

    class Cuda:
        synchronize = staticmethod(lambda: seen.append("sync"))

    class Torch:
        cuda = Cuda

    def equal(torch, actual, expected):
        expected = expected if isinstance(expected, tuple) else (expected,)
        return len(expected) == 1 and list(actual) == list(expected[0])

    ns["endurance"].__globals__["bytes_equal"] = equal
    result = ns["endurance"](Torch, Graph(), Step, lengths)
    replays = [x for x in seen if x != "sync"]
    assert result["replays"] == len(replays) - 1 == ns["ENDURANCE_REPLAYS"] + 1
    assert result["bytes_equal"] and lengths == [4000, 4100, 4200]
    # Only two synchronizations: after the reference replay and after the last one.
    assert seen.count("sync") == 2 and seen[1] == "sync" and seen[-1] == "sync"
    shifts = {r[0] - 4000 for r in replays}
    assert len(shifts) == ns["ENDURANCE_SHIFTS"] and max(shifts) <= 0 and min(shifts) > -64
