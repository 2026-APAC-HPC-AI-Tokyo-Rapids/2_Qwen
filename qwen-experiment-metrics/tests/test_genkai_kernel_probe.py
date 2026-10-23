"""Check extraction boundaries and fail-fast execution without a GPU."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HEADER = """launch_geometry_unchanged
                auto postscale_l1_final = []() {};
                auto prescale_l2_final = []() {};
constexpr bool kQwenPrefillMainloop = true;
                    auto issue = []() {};
                    if (block_phase == sched::BlockPhase::Linear1) {
                        const uint32_t gate_row = 0;
                        original_loop();
                    } else {
                        // Per-128 activation scales:
                        fc2_unchanged();
"""


@pytest.mark.parametrize("outside", [False, True])
def test_probe_extraction_rejects_unmeasured_changes(tmp_path, outside):
    baseline, candidate, output = (tmp_path / name for name in ("base.cuh", "candidate.cuh", "probe.cu"))
    baseline.write_text(HEADER)
    text = HEADER.replace("original_loop()", "candidate_loop()")
    candidate.write_text(text.replace("fc2_unchanged", "changed_fc2") if outside else text)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/build-prefill-fc1-probe.py"), str(baseline), str(candidate), str(output)], capture_output=True, text=True, check=False)
    if outside:
        assert result.returncode != 0
        assert "specialized FC1 body only" in result.stderr
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        generated = output.read_text()
        assert generated.count("original_loop();") == 1
        assert generated.count("candidate_loop();") == 1
        assert "@LOOP@" not in generated
        assert "fc2_unchanged" not in generated


@pytest.mark.parametrize("probe_exit", [0, 7])
def test_pjm_stops_on_a_failed_probe_and_records_exit(tmp_path, probe_exit):
    for name in ("source", "results", "bin"):
        (tmp_path / name).mkdir()
    (tmp_path / "source/provenance.json").write_text("{}")
    executable = tmp_path / "source/probe"
    executable.write_text(f"#!/bin/bash\necho called >> calls\nexit {probe_exit}\n")
    executable.chmod(0o755)
    module = tmp_path / "bin/module"
    module.write_text("#!/bin/bash\nexit 0\n")
    module.chmod(0o755)
    # Tests use the same host timeout utility as the job (GNU coreutils on macOS).
    env = {**os.environ, "RUNTIME_DIR": str(tmp_path), "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}
    result = subprocess.run(["bash", str(ROOT / "platforms/genkai/kernel-probe/run.pjm")], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == probe_exit, result.stderr
    assert (tmp_path / "results/exit-code").read_text().strip() == str(probe_exit)
    assert len((tmp_path / "calls").read_text().splitlines()) == (1 if probe_exit else 3)
    assert (tmp_path / "run-success").exists() == (probe_exit == 0)
