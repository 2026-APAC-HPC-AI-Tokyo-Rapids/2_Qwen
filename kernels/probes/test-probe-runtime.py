"""CPU regression fixtures for probe metadata mounts and failure-log propagation."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "kernels/fa3-verify/candidates.json"


def executable(path, source):
    path.write_text(source)
    path.chmod(0o755)


def test_prepare_snapshots_candidate_table_for_both_wrappers(tmp_path):
    workspace, runtime, bin_dir = (tmp_path / name for name in ("workspace", "runtime", "bin"))
    for path in (workspace / "configs", workspace / "kernels/fa3-verify", workspace / "kernels/probes", runtime / "source", runtime / "logs", bin_dir):
        path.mkdir(parents=True)
    (workspace / "configs/firmus-framework.json").write_text('{"sglang_overlay_patches":[]}')
    shutil.copyfile(TABLE, workspace / "kernels/fa3-verify/candidates.json")
    for name in ("run-sglang.sh", "run-container.sh"):
        shutil.copyfile(ROOT / "kernels/probes" / name, workspace / "kernels/probes" / name)
    (runtime / "sglang.sqsh").touch()
    executable(bin_dir / "unsquashfs", '''#!/bin/bash
set -eu
while [[ "$1" != -d ]]; do shift; done
dest=$2
mkdir -p "$dest/opt/sglang/venv/lib/python3.12/site-packages/deep_gemm" "$dest/opt/sglang/source/python/sglang" "$dest/opt/apac"
printf '{}\n' >"$dest/opt/apac/sm90-megamoe-build.json"
''')
    env = {**os.environ, "RUNTIME_DIR": str(runtime), "GITHUB_WORKSPACE": str(workspace), "CANDIDATE_PATCH": "", "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    result = subprocess.run(["bash", str(ROOT / "kernels/probes/prepare-sglang.sh")], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    staged = runtime / "source/fa3-verify-candidates.json"
    assert staged.read_bytes() == TABLE.read_bytes()
    (workspace / "kernels/fa3-verify/candidates.json").write_text("[]")
    assert staged.read_bytes() == TABLE.read_bytes(), "queued runtime must retain its checkout snapshot"


@pytest.mark.parametrize("platform", ["aspire2ap", "genkai"])
@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_wrappers_bind_metadata_and_preserve_diagnostics(tmp_path, platform, profile, failed):
    for name in ("source", "results", "logs", "scripts", "home/.local", "bin", "tmp"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "home/.local/.setenv").write_text("")
    (tmp_path / "scripts/build-env.sh").write_text("")
    shutil.copyfile(ROOT / "kernels/probes/run-container.sh", tmp_path / "scripts/run-container.sh")
    (tmp_path / "source/provenance.json").write_text('{"calibration":true}')
    shutil.copyfile(TABLE, tmp_path / "source/fa3-verify-candidates.json")
    name = "run-fa3-prefill-profile.sh" if profile else "run-sglang.sh"
    shutil.copyfile(ROOT / "kernels/probes" / name, tmp_path / "source/probe")
    (tmp_path / "source/probe").chmod(0o755)
    executable(tmp_path / "bin/module", "#!/bin/bash\nexit 0\n")
    executable(tmp_path / "bin/enroot", '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
runtime = Path(os.environ['RUNTIME_DIR'])
(runtime/'enroot-argv.json').write_text(json.dumps(sys.argv[1:]))
profile = os.environ['FIXTURE_PROFILE'] == '1'
if os.environ['FIXTURE_FAILED'] == '1':
    if profile:
        print('==ERROR== ERR_NVGPUCTRPERM - permission denied', file=sys.stderr)
        print('PASS kernel=fa3_prefill_m2 control=n224_fused shape=4096 launches=4')
    else:
        print('FileNotFoundError: missing dependency fixture', file=sys.stderr)
    sys.exit(7)
if profile:
    (runtime/'results/fa3-prefill-m2.ncu-rep').write_text('fixture report')
    print('PASS kernel=fa3_prefill_m2 control=n224_fused shape=4096 launches=4')
else:
    print('PASS kernel=fixture')
''')
    executable(tmp_path / "bin/ncu", '''#!/bin/bash
set -eu
printf 'import-called\n' >>"$RUNTIME_DIR/ncu-calls"
printf 'metric,value\nfixture,1\n'
''')
    env = {**os.environ, "RUNTIME_DIR": str(tmp_path), "HOME": str(tmp_path / "home"), "TMPDIR": str(tmp_path / "tmp"), "PATH": f"{tmp_path}/bin:{os.environ['PATH']}", "FIXTURE_PROFILE": str(int(profile)), "FIXTURE_FAILED": str(int(failed))}
    runner = ROOT / "platforms" / platform / "kernel-probe" / ("run.pbs" if platform == "aspire2ap" else "run.pjm")
    result = subprocess.run(["bash", str(runner)], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == (7 if failed else 0), result.stdout + result.stderr
    trial = (tmp_path / "results/trial-1.log").read_text()
    if failed:
        assert ("ERR_NVGPUCTRPERM" if profile else "FileNotFoundError") in trial
        assert "PASS " not in trial
        assert not (tmp_path / "ncu-calls").exists()
    else:
        assert "PASS " in trial
        assert (tmp_path / "ncu-calls").exists() == profile
    assert (tmp_path / "results/exit-code").read_text().strip() == str(7 if failed else 0)
    assert (tmp_path / "run-success").exists() != failed
    args = json.loads((tmp_path / "enroot-argv.json").read_text())
    mounts = [args[i + 1] for i, value in enumerate(args) if value == "-m"]
    mount = next(value for value in mounts if ":/opt/apac/fa3-verify/candidates.json:" in value)
    source, destination, fs_type, options = mount.split(":")
    assert Path(source).read_bytes() == TABLE.read_bytes()
    assert destination == "/opt/apac/fa3-verify/candidates.json" and fs_type == "none"
    assert {"x-create=file", "bind", "ro"} <= set(options.split(","))
    assert not list((tmp_path / "tmp").iterdir()), "probe must clean its node-local directory"
