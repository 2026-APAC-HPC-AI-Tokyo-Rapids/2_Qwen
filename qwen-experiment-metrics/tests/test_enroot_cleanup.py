"""The node-local enroot cleanup removes only this account's earlier pyxis unpacks."""
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def cleanup_script():
    shell = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    line = next(l for l in shell.splitlines() if 'enroot-cleanup.log' in l)
    assert line.startswith("srun -N2 -n2 --ntasks-per-node=1") and "--container-image" not in line
    return re.search(r"bash -c '(.*)' >", line).group(1)


def test_cleanup_keeps_current_job_and_other_accounts(tmp_path):
    data = tmp_path / "data"
    mine = data / f"enroot/data/user-{os.getuid()}"
    other = data / "enroot/data/user-99999"
    for path in (mine / "pyxis_100.3", mine / "pyxis_101.0", mine / "pyxis_7.1", mine / "keep-me", other / "pyxis_100.3"):
        path.mkdir(parents=True)
        (path / "f").write_text("x")
    script = cleanup_script().replace('"/data/enroot', f'"{data}/enroot').replace("df -h /data", f"df -h {data}")
    result = subprocess.run(["bash", "-c", script], env=dict(os.environ, SLURM_JOB_ID="7"), capture_output=True, text=True, check=True)
    assert sorted(p.name for p in mine.iterdir()) == ["keep-me", "pyxis_7.1"]
    assert (other / "pyxis_100.3").exists()
    assert result.stdout.count(" before: ") == 1 and result.stdout.count(" after: ") == 1


def test_cleanup_runs_before_any_container_step():
    shell = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    assert shell.index("enroot-cleanup.log") < shell.index("--container-image")
