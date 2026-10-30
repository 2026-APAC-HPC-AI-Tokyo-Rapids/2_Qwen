"""CPU coverage of the two-CTA-per-SM Decode FA3 verify rows and their PP4 pinned selection."""

import json
import runpy
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "kernels/fa3-verify"


def selector():
    return runpy.run_path(str(ROOT / "scripts/select-firmus-decode-fa3.py"))


# Two rows exercise the pinned rule's choice between candidates (the PP4 screen now has one row).
NAMES = ("p128split2-h2-fp8", "p128split3-h2-fp8")


def pinned_records(names, control_us=71.0, candidate_us=60.0, batches=("44",)):
    endurance = {"replays": selector()["ENDURANCE_REPLAYS"] + 1, "bytes_equal": True}
    return [{"rank": rank, "batches": {batch: {name: {"numeric_passed": True, "endurance": dict(endurance), "blocks": [{"stock": [control_us, control_us + .1], "candidate": [candidate_us, candidate_us + .1]} for _ in range(2)]} for name in names} for batch in batches}} for rank in range(4)]


def test_two_cta_rows_change_only_occupancy_and_split_layout():
    rows = json.loads((KERNELS / "candidates.json").read_text())
    two = [r for r in rows if r.get("ctas", 1) == 2]
    assert [r["name"] for r in two] == ["p128split2-c2-fp8", "p128split3-c2-fp8", "p128-uneven-c2-fp8"]
    # Stage two: 2 x (109568 + 1024) bytes fit the 228 KiB of one SM; stage three does not.
    assert all(r["stages"] == 2 and r["block_n"] == 128 and r["output"] == "fp8" and not r.get("merge") for r in two)
    assert all(r.get("ctas", 1) in (1, 2) for r in rows)


def test_build_generates_a_distinct_two_cta_main_and_keeps_existing_symbols(tmp_path):
    ns = runpy.run_path(str(KERNELS / "build.py"))
    tasks = ns["generate_dispatch"](KERNELS, tmp_path)
    names = {name for _, name, _ in tasks}
    assert "verify_n128_s2_split1_merge0_c2" in names and "verify_n128_s3_split1_merge0" in names
    sources = {name: path.read_text() for path, name, _ in tasks}
    assert "#define VERIFY_CTAS 2\n" in sources["verify_n128_s2_split1_merge0_c2"]
    assert all("#define VERIFY_CTAS 1\n" in text for name, text in sources.items() if not name.endswith("_c2"))
    table = (tmp_path / "candidates.h").read_text()
    assert '{"p128split2-c2-fp8", 128, 2, 2, true, false, false, 2, 1, verify_n128_s2_split1_merge0_c2},' in table
    assert '{"p128split2-s3-fp8", 128, 3, 2, true, false, false, 1, 1, verify_n128_s3_split1_merge0},' in table
    build = (KERNELS / "build.py").read_text()
    # Applied after every other companion patch, as the template signature test assumes.
    assert build.index('"prefill-varlen.patch"') < build.index('"decode-two-cta.patch"') < build.index('cuda = Path("/usr/local/cuda")')
    assert 'row.get("ctas", 1) * (size + 1024) <= SM90_SM_BYTES' in build and "SM90_SM_BYTES = 228 * 1024" in build


def test_wrapper_launches_a_persistent_grid_of_ctas_per_sm():
    wrapper = (KERNELS / "wrapper.cpp").read_text()
    assert "bool fp8, merge, uneven; int ctas, heads; Launch launch; };" in wrapper
    assert "p.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount * candidate(id).ctas;" in wrapper
    patch = (KERNELS / "decode-two-cta.patch").read_text()
    assert "+    static constexpr uint32_t MinBlocksPerMultiprocessor = MinBlocks_;" in patch
    assert "MinBlocks_ == 2 ? 56 :" in patch and "MinBlocks_ == 2 ? 200 :" in patch
    assert "FlashAttnFwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler, VerifyCtas>" in patch
    # 2 CTAs x 128 threads x (56 + 200) registers fill the 64K register file exactly.
    assert 2 * 128 * (56 + 200) == 65536


def test_exact_reference_requires_equal_splits():
    ns = selector()
    assert ns["exact_reference"]("p128split2-s3-fp8", "p128split2-c2-fp8")
    assert not ns["exact_reference"]("p128split2-s3-fp8", "p128split3-c2-fp8")
    assert not ns["exact_reference"]("p128split2-s3-fp8", "p128-uneven-c2-fp8")
    assert not ns["exact_reference"]("stock", "p128split2-c2-fp8")
    assert ns["exact_reference"]("p128split2-s3-fp8", "p128split2-h2-fp8")
    assert not ns["exact_reference"]("p128split2-s3-fp8", "p128split3-h2-fp8")
    # The two-CTA rows lost in 40925558339 and the two-heads-per-CTA rows in 40991744289 and 41005872470;
    # the PP4 job screens the four-heads-per-CTA stream-split row instead (test_fa3_verify_stream.py).
    assert ns["PINNED_SCREEN"] == ("p32stream-h4-fp8",)
    assert not ns["exact_reference"]("p128split2-s3-fp8", "p32stream-h4-fp8")


def test_pinned_rule_picks_the_fastest_row_saving_a_microsecond_on_every_gpu():
    ns = selector()
    names = NAMES
    ranks = pinned_records(names)
    for rank in ranks:
        rank["batches"]["44"]["p128split3-h2-fp8"]["blocks"] = [{"stock": [71.0], "candidate": [58.0]}] * 2
    result = ns["choose_pinned"](ranks, (44,), "p128split2-s3-fp8", names)
    assert result["winner"] == "p128split3-h2-fp8" and result["env"] == {"SGLANG_DECODE_FA3_VERIFY": "p128split3-h2-fp8"}
    assert result["control"] == "p128split2-s3-fp8" and result["rule"]["minimum_saving_us_per_layer"] == 1.0


@pytest.mark.parametrize("failure", ["not-bit-exact", "numeric", "one-block-too-small", "missing-block"])
def test_pinned_rule_excludes_a_failing_row_and_keeps_the_others(failure):
    ns = selector()
    names = NAMES
    ranks = pinned_records(names)
    for rank in ranks:
        rank["batches"]["44"]["p128split2-h2-fp8"]["blocks"] = [{"stock": [71.0], "candidate": [55.0]} for _ in range(2)]
        rank["batches"]["44"]["p128split3-h2-fp8"]["blocks"] = [{"stock": [71.0], "candidate": [58.0]} for _ in range(2)]
    entry = ranks[3]["batches"]["44"]["p128split2-h2-fp8"]
    if failure == "not-bit-exact":
        entry["bytes_equal_to_control"] = False
    elif failure == "numeric":
        entry["numeric_passed"] = False
    elif failure == "one-block-too-small":
        entry["blocks"][1] = {"stock": [71.0], "candidate": [70.5]}
    else:
        entry["blocks"].pop()
    result = ns["choose_pinned"](ranks, (44,), "p128split2-s3-fp8", names)
    assert result["winner"] == "p128split3-h2-fp8"
    assert not result["choices"]["p128split2-h2-fp8"]["eligible"]


def test_pinned_rule_without_a_winner_writes_no_override():
    ns = selector()
    names = NAMES
    result = ns["choose_pinned"](pinned_records(names, 71.0, 70.5), (44,), "p128split2-s3-fp8", names)
    assert result["winner"] is None and result["env"] == {}


def test_pinned_rule_needs_the_saving_at_every_measured_batch():
    # Cap 48 per rank: the Decode runs at 44-48, so a row faster only at cap/4 keeps the pin.
    ns = selector()
    names = NAMES
    ranks = pinned_records(names, 71.0, 70.5, batches=("44", "46", "48"))
    for rank in ranks:
        rank["batches"]["48"]["p128split3-h2-fp8"]["blocks"] = [{"stock": [76.0], "candidate": [60.0]}] * 2
        for batch in ("44", "46", "48"):
            rank["batches"][batch]["p128split2-h2-fp8"]["blocks"] = [{"stock": [71.0], "candidate": [58.0]}] * 2
    result = ns["choose_pinned"](ranks, (44, 46, 48), "p128split2-s3-fp8", names)
    assert result["winner"] == "p128split2-h2-fp8" and result["rule"]["selection_batches"] == [44, 46, 48]
    assert not result["choices"]["p128split3-h2-fp8"]["eligible"]
    del ranks[1]["batches"]["46"]["p128split2-h2-fp8"]["blocks"][1]
    assert ns["choose_pinned"](ranks, (44, 46, 48), "p128split2-s3-fp8", names)["winner"] is None


def test_pp4_job_keeps_the_decode_fa3_pin_without_any_decode_screen(tmp_path):
    shell = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    pp4 = shell[shell.index('if [[ -s "${ROOT}/source/pp4-kernel-settings.json" ]]; then'):shell.index('elif [[ -s "${ROOT}/source/reused-kernel-selection.json" ]]; then')]
    assert "select-firmus-decode-kernels.sh" not in pp4
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    for directory in ("source", "results"):
        (tmp_path / directory).mkdir()
    prepare = runpy.run_path(str(ROOT / "scripts/prepare-firmus-pp4-settings.py"))
    prepare["prepare"](config, ROOT, tmp_path)
    env = tmp_path / "source/decode-kernels.env"
    result = subprocess.run(["bash", "-c", 'source "$1"; printf "%s" "$SGLANG_DECODE_FA3_VERIFY"', "bash", str(env)], capture_output=True, text=True, check=True)
    assert result.stdout == "p128split2-s3-fp8"
    worker = shell.split('cat >"${ROOT}/source/worker.sh" <<\'EOF\'\n', 1)[1].split("\nEOF", 1)[0]
    assert 'if [[ "$role" == decode ]]; then set -a; source "$(dirname "$0")/decode-kernels.env"; set +a; fi' in worker
