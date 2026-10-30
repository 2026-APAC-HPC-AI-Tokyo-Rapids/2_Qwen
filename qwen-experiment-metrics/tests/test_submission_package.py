"""CPU contracts for the fixed Firmus submission, independent of cluster access."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "submission"
CONFIG = json.loads((ROOT / "configs/firmus-framework.json").read_text())
# The frozen submission retains c240 and the measured overlay chain.
EXPERIMENT_ONLY_PATCHES = {"patches/sglang/prefill-start-align.patch", "patches/deep-gemm-bench/fc2-address/fc2-output-address.patch", "patches/sglang/decode-norm-quant-wide.patch", "patches/sglang/decode-dense-swapab.patch", "patches/sglang/verify-penalty-broadcast.patch"}
SUBMISSION_OVERLAYS = {key: [patch for patch in CONFIG[key] if patch not in EXPERIMENT_ONLY_PATCHES] for key in ("sglang_overlay_patches", "deep_gemm_overlay_patches")}
BATCH = PACKAGE / "run/submit.sbatch"
ORIGINAL = ROOT / "platforms/firmus/framework-comparison/sglang.sbatch"


@pytest.mark.parametrize("family,key,count", [("sglang", "sglang_overlay_patches", 14), ("deep-gemm", "deep_gemm_overlay_patches", 7)])
def test_numbered_overlay_order_and_bytes(family, key, count):
    copies = sorted((PACKAGE / "patches/overlay" / family).glob("*.patch"))
    assert len(copies) == len(SUBMISSION_OVERLAYS[key]) == count
    for index, (copy, original) in enumerate(zip(copies, SUBMISSION_OVERLAYS[key], strict=True), 1):
        assert copy.name == f"{index:02d}-{Path(original).name}"
        assert copy.read_bytes() == (ROOT / original).read_bytes()


@pytest.mark.parametrize("copy,source", [
    ("official-aiperf.sh", "platforms/firmus/framework-comparison/official-aiperf.sh"),
    ("run-aiperf.py", "scripts/run-aiperf.py"),
    ("parse-aiperf-result.py", "scripts/parse-aiperf-result.py"),
    ("prepare_eagle.py", "research/simulation/prepare_eagle.py"),
])
def test_client_and_draft_copies(copy, source):
    assert (PACKAGE / "run" / copy).read_bytes() == (ROOT / source).read_bytes()


def test_serving_configuration_matches_canonical_fields():
    # The package already pins these same kernels in its role env files; the
    # research runner's prior-run reuse record and profile-only routing capture
    # are not serving parameters.
    expected = {key: value for key, value in CONFIG.items() if key not in ("images", "sglang_overlay_patches", "deep_gemm_overlay_patches", "reuse_kernel_selection")}
    expected["concurrency"] = 240
    expected["prefill"] = {key: value for key, value in CONFIG["prefill"].items() if key != "routing_capture_samples"}
    expected["images"] = {"sglang": {key: CONFIG["images"]["sglang"][key] for key in ("sha256", "size_bytes")}}
    assert json.loads((PACKAGE / "config/serving.json").read_text()) == expected
    assert (PACKAGE / "config/prefill-expert-location.json").read_bytes() == (ROOT / CONFIG["prefill"]["expert_location"]).read_bytes()


def worker(text):
    return text.split('cat >"${ROOT}/source/worker.sh" <<\'EOF\'\n', 1)[1].split("\nEOF\n", 1)[0]


def test_worker_is_unchanged_except_dormant_nsight_block():
    expected = worker(ORIGINAL.read_text())
    dormant = re.compile(r"if \[\[ -x /opt/nsys/bin/nsys \]\]; then\n.*?\nfi\n", re.DOTALL)
    expected, count = dormant.subn("", expected)
    assert count == 1
    # The opt-in PP4 runtime block is absent from the frozen EP4 submission.
    pp4_start = expected.index("# The Decode peer also needs explicit global target/draft KV region IDs.\n")
    pp4_end = expected.index("# Keep inherited diagnostic variables", pp4_start)
    expected = expected[:pp4_start] + expected[pp4_end:]
    # Profile-only Prefill routing capture; the package never captures routing.
    capture = re.compile(r"runtime_dir=\$\(dirname \"\$\(dirname \"\$0\"\)\"\)\n# Keep inherited diagnostic variables.*?\nfi\n", re.DOTALL)
    expected, count = capture.subn("", expected)
    assert count == 1
    # The research runner clears the experimental tail switch and reports it; the frozen submission has no tail overlay.
    assert expected.count("unset DG_SM90_1D2D_PREFILL_TAIL_SKIP\n") == 1
    expected = expected.replace("unset DG_SM90_1D2D_PREFILL_TAIL_SKIP\n", "")
    assert expected.count("tail_skip=${DG_SM90_1D2D_PREFILL_TAIL_SKIP:-0} ") == 1
    expected = expected.replace("tail_skip=${DG_SM90_1D2D_PREFILL_TAIL_SKIP:-0} ", "")
    assert worker(BATCH.read_text()) == expected


def test_submission_has_no_selection_gate_telemetry_or_profile_execution():
    text = BATCH.read_text()
    active = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    for forbidden in ("PROFILE", "/opt/nsys", "profile_capture", "profiler_mount", "select-firmus-", "check-firmus-megamoe.py", "telemetry_pids", "gpu-telemetry", "--loop=", "collect-eagle-acceptance", "dense_selection", "fa3_selection", "--aggregate"):
        assert forbidden not in active
    # worker.sh retains the measured one-shot inventory query, not a telemetry loop.
    outside_worker = text.replace(worker(text), "")
    assert "nvidia-smi" not in outside_worker
    for directive in ("#SBATCH -p hpcai", "#SBATCH -N 2", "#SBATCH -t 00:17:00", "#SBATCH --gres=gpu:8"):
        assert directive in text
    assert 'ROOT=${SLURM_SUBMIT_DIR}' in text
    assert 'MODEL=$(jq -er .model.path "${ROOT}/source/model.json")' in text
    assert 'SGLANG=${ROOT}/sglang.sqsh' in text


def test_serving_launches_and_warmup_are_preserved_in_order():
    assert "tokenizer_backend" not in CONFIG
    assert "tokenizer_backend" not in json.loads((PACKAGE / "config/serving.json").read_text())
    def critical(text):
        text = text.split('test -s "${ROOT}/results/score.json"', 1)[0]
        # Normal EP4 still resolves to TP4; PP4 deliberately resolves to TP1.
        text = text.replace('"${MODEL}" "${prefill_tp}"', f'"${{MODEL}}" {CONFIG["prefill"]["tp"]}')
        # Both EP4 configs use HF, so the optional tokenizer array expands to no arguments.
        text = text.replace(' "${tokenizer_args[@]}"', "")
        return [line for line in text.splitlines() if line.startswith(("srun --exact", "srun --overlap -N1", "wait_http ", 'echo "READY', "  curl --fail", "curl --fail")) and " profile " not in line]
    assert critical(BATCH.read_text()) == critical(ORIGINAL.read_text())
    text = BATCH.read_text()
    sanity = text.index('>"${ROOT}/results/sanity-chat.json"')
    official = text.index('bash "${ROOT}/scripts/official-aiperf.sh"')
    assert text.index('>"${ROOT}/results/server-info-${worker}.json"') < sanity < official
    assert text.count('>"${ROOT}/results/sanity-chat.json"') == 1


def test_draft_precision_launch_and_runtime_check_match_canonical():
    package = BATCH.read_text()
    canonical = ORIGINAL.read_text()
    # The shared speculation arguments reach all four workers. The runtime
    # precision check must also precede the official client in both launchers.
    def arguments(text):
        return text.split("speculation=(", 1)[1].split("decode_graph=(", 1)[0].split("\ntokenizer_backend=", 1)[0].rstrip()

    assert arguments(package) == arguments(canonical)
    precision_check = next(line for line in canonical.splitlines() if ".speculative_draft_model_quantization ==" in line)
    assert precision_check in package
    assert package.index(precision_check) < package.index('bash "${ROOT}/scripts/official-aiperf.sh"')


def test_pinned_role_env_files(tmp_path):
    (tmp_path / "source").mkdir()
    lines = [line for line in BATCH.read_text().splitlines() if line.startswith("printf ") and '-kernels.env"' in line]
    assert len(lines) == 2
    subprocess.run(["bash", "-euc", "\n".join(lines)], env=os.environ | {"ROOT": str(tmp_path)}, check=True)
    assert (tmp_path / "source/prefill-kernels.env").read_text() == "SGLANG_PREFILL_FA3_VERIFY=n192\n"
    assert (tmp_path / "source/decode-kernels.env").read_text() == "SGLANG_DECODE_FA3_VERIFY=p128split2-s3-fp8\nDG_SM90_1D2D_DECODE_CONFIG=QKV:128,144,5,1,0\nSGLANG_DECODE_DENSE_GEMM_QKV=\nSGLANG_DECODE_DENSE_GEMM_O=\n"


@pytest.mark.parametrize("value,expected", [(None, []), ("", []), ("registry.example/qwen-cache", ["--cache-from", "registry.example/qwen-cache", "--cache-to", "registry.example/qwen-cache"])])
def test_optional_build_cache_keeps_actions_arguments(value, expected):
    build = (ROOT / "scripts/build-sglang-container.sh").read_text()
    code = build.split("cache_args=()\n", 1)[1].split("\nbuildah ", 1)[0]
    env = dict(os.environ)
    env.pop("BUILD_CACHE_REPOSITORY", None)
    if value is not None:
        env["BUILD_CACHE_REPOSITORY"] = value
    result = subprocess.run(["bash", "-euc", 'cache_args=()\n' + code + '\nfor arg in "${cache_args[@]}"; do printf "%s\\0" "$arg"; done'], env=env, check=True, capture_output=True)
    assert result.stdout.decode().split("\0")[:-1] == expected
    assert '--layers "${cache_args[@]}" --cache-ttl 336h' in build
    assert '${BUILD_CACHE_REPOSITORY:?' not in (PACKAGE / "build/setup.sh").read_text()


def test_package_shell_commands_stay_on_physical_lines():
    paths = [*PACKAGE.rglob("*.sh"), *PACKAGE.rglob("*.sbatch"), ROOT / "platforms/shared/qwen-pd.sbatch"]
    for path in paths:
        assert not any(line.rstrip().endswith("\\") for line in path.read_text().splitlines()), path
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_generic_launcher_remains_byte_identical_to_parent():
    assert hashlib.sha256((ROOT / "platforms/shared/qwen-pd.sbatch").read_bytes()).hexdigest() == "12a2fb09c70eb36f1afa8c7ec83934007be8629f47fb1c9de8d564f824ac32a6"
    provision = (ROOT / "scripts/provision-qwen-pd.sh").read_text()
    assert '"${GITHUB_WORKSPACE}/platforms/shared/qwen-pd.sbatch" "${runtime_dir}/scripts/submit.sbatch"' in provision


def test_curated_evidence_checksums_and_local_official_parser(tmp_path):
    evidence = PACKAGE / "results/40225743220"
    checked = set()
    for line in (evidence / "SHA256SUMS").read_text().splitlines():
        digest, path = line.split("  ", 1)
        target = evidence / path
        assert target.stat().st_size <= 20_000_000
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        checked.add(target)
    assert checked == {p for p in evidence.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
    output = tmp_path / "score.json"
    subprocess.run(["python3", str(PACKAGE / "run/parse-aiperf-result.py"), str(evidence / "profile_export_aiperf.json"), str(output), "--concurrency", "240", "--require-eligible", "--command", str(evidence / "aiperf-command.txt"), "--provenance", str(evidence / "aiperf-provenance.txt")], check=True)
    assert json.loads(output.read_text()) == json.loads((evidence / "score.json").read_text())


def test_recreated_boundary_commit_has_pinned_object_id():
    raw = (PACKAGE / "build/sglang-6b350686.commit").read_bytes()
    oid = subprocess.check_output(["git", "hash-object", "-t", "commit", "--stdin"], input=raw, cwd=ROOT).decode().strip()
    assert oid == "6b3506869d4840b18678de19d18d751a34f9b737"
    assert raw.splitlines()[0] == b"tree 9ea4c7300d38bd9e024330b45bf5fc9cbf70f97d"


def test_boundary_commit_copy_matches_available_source_mirror():
    mirror = Path("/home/USER/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src")
    if not os.access(mirror, os.R_OK | os.X_OK):
        pytest.skip("fixed source mirror unavailable")
    original = subprocess.check_output(["git", "-C", str(mirror), "cat-file", "commit", "6b350686"])
    assert (PACKAGE / "build/sglang-6b350686.commit").read_bytes() == original
