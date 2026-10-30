import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def probe():
    return runpy.run_path(str(ROOT / "scripts/check-firmus-pp4-copy-engine.py"))


def write_ranks(probe, directory):
    directory.mkdir(parents=True)
    stages, transfers = probe["STAGES"], probe["TRANSFERS"]
    for rank in range(stages):
        entry = {"rank": rank, "transfers_received": 0 if rank == 0 else transfers, "transfers_sent": 0 if rank == stages - 1 else transfers, "mismatches": None if rank == 0 else 0}
        (directory / f"rank-{rank}.json").write_text(json.dumps(entry))


def aggregate(probe, tmp_path):
    out, env = tmp_path / "selection.json", tmp_path / "prefill-kernels.env"
    env.write_text("SGLANG_PREFILL_FA3_VERIFY=n192\n")
    probe["aggregate"](tmp_path / "results", out, env)
    return json.loads(out.read_text()), env.read_text()


def test_exact_transfers_on_all_ranks_enable_the_copy_engine(probe, tmp_path):
    write_ranks(probe, tmp_path / "results")
    result, env = aggregate(probe, tmp_path)
    assert result["transport"] == "copy"
    assert env == "SGLANG_PREFILL_FA3_VERIFY=n192\nSGLANG_PP4_PROXY_TRANSPORT=copy\n"


@pytest.mark.parametrize("change", ["missing", "mismatch", "incomplete"])
def test_anything_else_keeps_pynccl(probe, tmp_path, change):
    write_ranks(probe, tmp_path / "results")
    path = tmp_path / "results/rank-2.json"
    entry = json.loads(path.read_text())
    if change == "missing":
        path.unlink()
    else:
        entry.update(mismatches=3) if change == "mismatch" else entry.update(transfers_sent=5)
        path.write_text(json.dumps(entry))
    result, env = aggregate(probe, tmp_path)
    assert result["transport"] == "nccl" and env == "SGLANG_PREFILL_FA3_VERIFY=n192\n"


def test_pp4_pins_copy_transport_and_keeps_the_probe_staged():
    shell = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    assert "check-firmus-pp4-copy-engine.py" not in shell
    prepare = runpy.run_path(str(ROOT / "scripts/prepare-firmus-pp4-settings.py"))
    assert prepare["PP4_SELECTED"]["SGLANG_PP4_PROXY_TRANSPORT"] == "copy"
    assert prepare["PP4_SELECTED_RUNS"]["SGLANG_PP4_PROXY_TRANSPORT"] == [41005872470, 41020974704, 41040348302]
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    assert 'install -m 0644 "${GITHUB_WORKSPACE}/scripts/check-firmus-pp4-copy-engine.py" "${runtime_dir}/scripts/"' in provision


def test_patch_defaults_to_pynccl_and_opens_events_on_the_own_device():
    patch = (ROOT / "patches/sglang/pp4-proxy-copy-engine.patch").read_text()
    assert '+_QWEN_PP_PROXY_TRANSPORT = os.environ.get("SGLANG_PP4_PROXY_TRANSPORT", "nccl")' in patch
    assert "+                ce.peer_events = [torch.cuda.Event.from_ipc_handle(device, h) for h in handle]" in patch
    assert "peer_device" not in patch
