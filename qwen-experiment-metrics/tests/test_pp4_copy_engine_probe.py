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


def test_pp4_job_probes_on_decode_gpus_before_the_prefill_workers():
    shell = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    pp4 = shell[shell.index('if [[ -s "${ROOT}/source/pp4-kernel-settings.json" ]]; then'):shell.index('elif [[ -s "${ROOT}/source/reused-kernel-selection.json" ]]; then')]
    launch = next(line for line in pp4.splitlines() if 'check-firmus-pp4-copy-engine.py" --output' in line)
    group = pp4[pp4.rindex("  {\n", 0, pp4.index(launch)):pp4.index("  } &\n") + 6]
    assert launch in group and '-w "${TAIL}" --gres=gpu:4' in launch and "timeout --kill-after=5 60" in launch and launch.rstrip().endswith("|| true")
    take = shell[shell.index('if [[ -n "${decode_gpu_selections:-}" ]]; then'):]
    take = take[:take.index("\nfi\n")]
    assert take.index('wait "${decode_gpu_selections}" || true') < take.index('check-firmus-pp4-copy-engine.py" --aggregate')
    assert shell.index(take) < shell.index('bash "${ROOT}/source/worker.sh" prefill-0')
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    assert 'install -m 0644 "${GITHUB_WORKSPACE}/scripts/check-firmus-pp4-copy-engine.py" "${runtime_dir}/scripts/"' in provision


def test_patch_defaults_to_pynccl_and_opens_events_on_the_own_device():
    patch = (ROOT / "patches/sglang/pp4-proxy-copy-engine.patch").read_text()
    assert '+_QWEN_PP_PROXY_TRANSPORT = os.environ.get("SGLANG_PP4_PROXY_TRANSPORT", "nccl")' in patch
    assert "+                ce.peer_events = [torch.cuda.Event.from_ipc_handle(device, h) for h in handle]" in patch
    assert "peer_device" not in patch
