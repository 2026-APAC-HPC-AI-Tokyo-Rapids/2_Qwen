import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_every_script_the_job_runs_is_staged_into_the_run_directory():
    # 40964714831 lost its Decode FA3 and MoE-input selections to a ModuleNotFoundError and a missing
    # file: the job ran scripts that provision never copied into ${ROOT}/scripts.
    provision = (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    staged = set(re.findall(r'"\$\{GITHUB_WORKSPACE\}/scripts/([A-Za-z0-9_.-]+)"', provision))
    referenced = set(re.findall(r"\$\{ROOT\}/scripts/([A-Za-z0-9_.-]+)", (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()))
    for helper in ("select-firmus-decode-kernels.sh", "select-firmus-prefill-kernels.sh"):
        referenced |= set(re.findall(r"\$\{root\}/scripts/([A-Za-z0-9_.-]+)", (ROOT / "scripts" / helper).read_text()))
    # Sibling modules the staged Python scripts import from the run directory.
    modules = {path.stem: path.name for path in (ROOT / "scripts").glob("*.py") if path.stem.isidentifier()}
    for name in sorted(referenced | staged):
        if name.endswith(".py") and (ROOT / "scripts" / name).is_file():
            text = (ROOT / "scripts" / name).read_text()
            referenced |= {modules[m] for m in re.findall(r"^\s*(?:import|from) ([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE) if m in modules}
    # official-aiperf.sh is installed from platforms/firmus/framework-comparison.
    assert referenced - staged == {"official-aiperf.sh"}
