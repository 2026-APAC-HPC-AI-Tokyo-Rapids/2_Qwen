"""CPU contracts for the fixed Firmus PP4 round 25 submission, independent of cluster access."""
from __future__ import annotations

import hashlib
import json
import os
import re
import runpy
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "submission"
CONFIG_PATH = ROOT / "configs/firmus-prefill-pp4.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
BATCH = PACKAGE / "run/submit.sbatch"
ORIGINAL = ROOT / "platforms/firmus/framework-comparison/sglang.sbatch"
EVIDENCE = PACKAGE / "results/41573354672"
# Final env files the workers of Actions run 41573354672 sourced (source/source/*-kernels.env of its archive),
# after the Prefill FA3 pair, Prefill FA3 prepack (P3) and Decode greedy gates adopted their candidates.
PREFILL_ENV = "SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen-pipe+m192n96-pair-prepack-v3\nSGLANG_PREFILL_MOE_INPUT=onepass\nSGLANG_PREFILL_QKNORM=fp32\nDG_SM90_PREFILL_4STAGE=1\nDG_SM90_PREFILL_LEAN=1\nDG_SM90_PREFILL_SPLIT_ACCUM=1\nSGLANG_PP4_PROXY_TRANSPORT=copy\n"
DECODE_ENV = "SGLANG_DECODE_FA3_VERIFY=p128split2-s3-fp8\nDG_SM90_1D2D_DECODE_CONFIG='QKV:128,144,5,1,0;QKV>128:192,80,5,1,0'\nSGLANG_DECODE_DENSE_GEMM_QKV=''\nSGLANG_DECODE_DENSE_GEMM_O=''\nSGLANG_DECODE_NORM_QUANT=wide\nSGLANG_DECODE_MOE_INPUT=fused\nSGLANG_DECODE_GREEDY_VERIFY=1\n"
# Overlay SHA-256 values in application order, as recorded in that run's sglang-overlay.json and deep-gemm-overlay.json.
ARCHIVED_OVERLAYS = {
    "sglang_overlay_patches": [
        "0ea92882f7ba4d1804eeb54fe81d8d610796fa890c61c3039922bd49d28bf40c",
        "d58039ee0149e433e39f3ab13f8be7775cfb4861dd770fd8211304feff027e1e",
        "6f2b64cdc4f05ea31af7be60c313d5365646885afef73a3d2ce3f3a8e9846ea4",
        "bc4a3a2a6aa786e19dc57cb489c742f4e47e27d357dbe33b7cc6df83e89f829e",
        "5a86b227fea9225109999d48505efd510dae0f34b93d2bedb9fedbf8e6a8c652",
        "d779e4e864e65554ea527bfd0a2c200bcd154c83ae48aaf8427b1ff22f650574",
        "a5049b688615a8e89c1790ce4b3d190a396db33e1182bb4dcde38ebc1235111e",
        "e5623b0cd28cfa81e897a1847a602f418acd500112f4b24d858b95dceda4cfa8",
        "a9c26d49a24feb1b989759ce8d07949a6d3c874c999c83bb50d6006b8f98d8a9",
        "99e97664d83610c539eaa62e992a396f4d5bb1981760934ca3df3bb5c3de42ef",
        "11bb55ee3fc86db2c4b84254033105863d0db333919e4d771b26a04612d043a6",
        "e08c44f43ceded413c1652a02c9f65fcb41d9eaafc068c749cb1d5786dd0ad10",
        "71375901315fc74fe13551fc8de985e1617d56f438d5deb3b091c1be5817b03a",
        "3291c98dbb54cedc3b2e758c9e25caeea60c719ab4b9327d1caf9d83d484ee3b",
        "9e4044a9c295aadec36ff9b8ca22554afd010b942c66538726ebb201f1719cde",
        "773d876c62561b97be8d739d2f8b819c54b1afa2a76e26ee5e4f5e844682529f",
        "6ea1cb131b9e78e1ba54dd05ddf02a678c419a5b9946b26c620edde1159d5f61",
        "c0bf6da9e783ebf6dc5711591dd709e4d5bc81f29ec6c1ba5777351d7f460fba",
        "5d1a14e321aa708064e377274bbabea28248f96b4fb18edfd3f66c682a997258",
        "1279a16d1fb3692459159ea8232d3fa6ddee78ca583d4f29e9efc408b8f80d65",
        "f2fe604a83d48c49c05dd6df62f65915a920640244904d4107867f2f020f7404",
        "27127a371defbfc67136e4bfdcb501bb9186e3afdb2963a53673073b70f3c068",
        "b6938fcb022b7b06bd3f21de1950e51b6bf49cb657bd15a93f9818f730fe70bd",
        "6e3c149a422c54d3e633cdaa6d8a0ccbf75933fbd7e123adcfa071424b482dd5",
        "06461257621491a63254f5b7b216e3ad9ce0b3cb3da2998df0655b61243f5801",
        "e63f8d546fab7b43364884902bf58edadcba7fc43e71e00a0bc6dce6456028c5",
        "cfa174ba65f7fe996153007db12f52f4ef262e4d0f6af47fd4b270d6345a02e9",
        "c28c43e36621d7d582c7209bfeac64e67d50a88917866a62f93ebd489af4e539",
        "0b64f08a082df05a51f58bd188123ccf7ef45b0edfe4752b89f8cd2e5719d254",
        "2b7a7eef760fb862a438b2e666ecae1400b1b85938ba895d3f68858a224c4811",
        "9b0736da011b34b8e42cbc32e45b004be3ef0207d7bc1b44463d0ace9783cfa7",
        "0d9ea8b1ea34d34425f0e7bfa160c8ce040032c5bbdd8f9dc7a17939dde50189",
    ],
    "deep_gemm_overlay_patches": [
        "d7738280f64f25feaa53f8cbb8d4b28f25a1b952c5f590974de87dd5073a5667",
        "822fa80af98deebbadaecef46ab8119abb9c87760d1fba227a9def2eaf27b1b4",
        "4a91639165075a01d824bbb398d291037de4483160e0c15f487365b9f1fc1906",
        "ab5efd41b3089e5a3527cce98312fd97ef2342721b1fe0855ed6b82080be33ae",
        "0c337c9adeb6ce09bfe90c790d31686b187e59022b1ae754d83392d7ccbe177b",
        "2167001084fe5c9c50fd35aaef0ac3d0648af20b064c1355b7fe52519dd36803",
        "a27e42d3224305e4088b6c6cfbe17ff1457db8ef6866f4f32d12024cb8ffa230",
        "9b9543ec4f4d185980038a387e36217edafa42724976e5e86a335a24ab30a31d",
        "eaf177b91c868c7f95d9e885c870f6b309bd4fbe349d08aed30cc1724f61cd99",
        "075d70da25b38741ba8ba9d2fc1b2f69c7e3994ebfe1e329ed0e77f87bd1e871",
        "703f3d7ef44c2e5f4b4bf29de67a3878e5aa03ef3ad06a04ead484b1c1d0fba6",
        "225026eb287ae95aeb38294aed51f0b42f69e04e06dcfc61a045d0127d33cf91",
        "c08c6ba8b0805dca3e4e66b5be31c18c007a20267270b04ba82f479d8de627c5",
    ],
}
# Runtime overlays added after that run, appended in application order: the PP4 direct loop logprob relay.
ADDED_OVERLAYS = {"sglang_overlay_patches": ["b311fd9a488015ee886e74027b7042c940a0893aa943b20d4862c2508c078c04"], "deep_gemm_overlay_patches": []}
IMAGE = {"path": "/scratch/home/ACCOUNT/qwen/containers/41573354672.sqsh", "sha256": "a001f90142b571380894e9532e467c57b831397c9e2e9d613d9e04004c87def9", "size_bytes": 12298305536}
# Config keys that only fed removed gates, diagnostics, overlay staging from the repository, or the vLLM path.
DROPPED_KEYS = ("sglang_overlay_patches", "deep_gemm_overlay_patches", "kernel_settings_reference", "container_build_run_id", "container_sha256", "container_size_bytes", "container_source_commit", "profile_only", "decode_greedy_verify_gate")
DROPPED_PREFILL_KEYS = ("gpu_sm_clock", "fa3_pair_gate", "fa3_prepack_gate", "fa3_v3_gate")
# Lines of submit.sbatch that are not lines of main's sglang.sbatch: submission paths, the pinned-env comment and the greedy probe.
ADDED_LINES = [
    "#SBATCH -o logs/slurm-%j.log",
    "#SBATCH -e logs/slurm-%j.stderr.log",
    'MODEL=$(jq -er .model.path "${ROOT}/source/model.json")',
    "  local overlay_mount=",
    "  # Patched packages staged by submission/run/prepare.sh.",
    '  echo "/scratch:/scratch,${ROOT}/home/$1:${HOME},${ROOT}:${ROOT},${ROOT}/draft:/draft,/dev/infiniband:/dev/infiniband${overlay_mount}"',
    "# Kernel choices adopted by the in-job gates of Actions run 41573354672 (prefill-fa3-pair, prefill-fa3-prepack P3, decode-greedy);",
    "# submission/run/prepare.sh installed the final env files those workers sourced. Workers source their role's file.",
    'test -s "${ROOT}/source/prefill-kernels.env" && test -s "${ROOT}/source/decode-kernels.env"',
    "  # Team-side output check after the official benchmark (same gateway as the warm-up chat): eight fixed greedy prompts, 64 tokens each, with token logprobs.",
    '  srun --overlap -N1 -n1 -w "${HEAD}" --gres=gpu:0 --cpus-per-task=2 --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_DISABLE_REQUIRE,MELLANOX_VISIBLE_DEVICES,MELLANOX_MOUNT_DRIVER --container-image="${SGLANG}" --container-mounts="$(mounts greedy-probe)" /opt/sglang/venv/bin/python "${ROOT}/scripts/greedy-probe.py" "${ROOT}/results" "http://${HEAD_IP}:${R_PORT}/v1/chat/completions" qwen >"${ART}/greedy-probe.log" 2>&1',
]


@pytest.mark.parametrize("family,key,count", [("sglang", "sglang_overlay_patches", 33), ("deep-gemm", "deep_gemm_overlay_patches", 13)])
def test_numbered_overlay_order_bytes_and_archived_hashes(family, key, count):
    copies = sorted((PACKAGE / "patches/overlay" / family).glob("*.patch"))
    digests = ARCHIVED_OVERLAYS[key] + ADDED_OVERLAYS[key]
    assert len(copies) == len(CONFIG[key]) == len(digests) == count
    for index, (copy, original, digest) in enumerate(zip(copies, CONFIG[key], digests, strict=True), 1):
        assert copy.name == f"{index:02d}-{Path(original).name}"
        assert copy.read_bytes() == (ROOT / original).read_bytes()
        assert hashlib.sha256(copy.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("copy,source", [
    ("official-aiperf.sh", "platforms/firmus/framework-comparison/official-aiperf.sh"),
    ("run-aiperf.py", "scripts/run-aiperf.py"),
    ("parse-aiperf-result.py", "scripts/parse-aiperf-result.py"),
    ("prepare_eagle.py", "research/simulation/prepare_eagle.py"),
    ("order-firmus-pp4-gpus.py", "scripts/order-firmus-pp4-gpus.py"),
])
def test_helper_copies_are_verbatim(copy, source):
    assert (PACKAGE / "run" / copy).read_bytes() == (ROOT / source).read_bytes()


def test_greedy_probe_changes_only_gateway_and_model():
    source = (ROOT / "research/simulation/greedy_probe.py").read_text()
    for old, new in (("    root = Path(sys.argv[1])\n", "    root = Path(sys.argv[1])\n    url = sys.argv[2]\n    model = sys.argv[3]\n"), ("{'model': 'qwen', ", "{'model': model, "), ("urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions', ", "urllib.request.Request(url, ")):
        assert source.count(old) == 1
        source = source.replace(old, new)
    assert (PACKAGE / "run/greedy-probe.py").read_text() == source


def test_serving_configuration_matches_round25_config_and_image():
    expected = {key: value for key, value in CONFIG.items() if key not in DROPPED_KEYS}
    expected["prefill"] = {key: value for key, value in CONFIG["prefill"].items() if key not in DROPPED_PREFILL_KEYS}
    expected["images"] = {"sglang": {key: CONFIG["images"]["sglang"][key] for key in ("path", "sha256", "size_bytes")}}
    serving = json.loads((PACKAGE / "config/serving.json").read_text())
    assert serving == expected
    assert serving["images"]["sglang"] == IMAGE
    assert CONFIG["container_build_run_id"] == 41573354672 and CONFIG["container_sha256"] == IMAGE["sha256"]
    sidecar = json.loads((PACKAGE / "build/provenance.json").read_text())
    assert sidecar["container_artifact"]["sha256"] == IMAGE["sha256"] and sidecar["container_artifact"]["size_bytes"] == IMAGE["size_bytes"]
    assert sidecar["source_commits"] == {"qwen": CONFIG["container_source_commit"], "sglang": "fc275674a97d78553b22597ee72c8fd48f5420f4"}
    assert "sglang_revision=fc275674a97d78553b22597ee72c8fd48f5420f4\n" in (ROOT / "scripts/build-sglang-container.sh").read_text()


def test_static_inputs_match_repository_sources():
    assert (PACKAGE / "config/firmus-gpu-sm-clock.json").read_bytes() == (ROOT / CONFIG["prefill"]["gpu_sm_clock"]).read_bytes()
    # PP4 Prefill uses trivial expert placement, so no expert location file is packaged.
    assert CONFIG["prefill"]["expert_location"] == "trivial"
    assert sorted(path.name for path in (PACKAGE / "config").iterdir()) == ["decode-kernels.env", "firmus-gpu-sm-clock.json", "prefill-kernels.env", "serving.json"]


def test_pinned_role_env_files_equal_archived_values():
    assert (PACKAGE / "config/prefill-kernels.env").read_text() == PREFILL_ENV
    assert (PACKAGE / "config/decode-kernels.env").read_text() == DECODE_ENV
    printed = subprocess.check_output(["bash", "-euc", 'set -a; source "$1"; source "$2"; set +a; printf "%s\\n" "${DG_SM90_1D2D_DECODE_CONFIG}" "${SGLANG_DECODE_DENSE_GEMM_QKV:-empty}" "${SGLANG_PREFILL_FA3_VERIFY}"', "env", str(PACKAGE / "config/prefill-kernels.env"), str(PACKAGE / "config/decode-kernels.env")], text=True)
    assert printed.splitlines() == ["QKV:128,144,5,1,0;QKV>128:192,80,5,1,0", "empty", "n192-direct-o-varlen-pipe+m192n96-pair-prepack-v3"]


def test_pins_are_the_pp4_settings_plus_the_adopted_gate_outcomes(tmp_path):
    # Drift check: the provisioning settings script still produces the pre-gate env files, and the pins
    # differ from them only by the three adopted gate outcomes recorded in the evidence.
    (tmp_path / "source").mkdir()
    (tmp_path / "results").mkdir()
    runpy.run_path(str(ROOT / "scripts/prepare-firmus-pp4-settings.py"))["prepare"](CONFIG, ROOT, tmp_path)
    pair = json.loads((EVIDENCE / "prefill-fa3-pair/selection.json").read_text())
    prepack = json.loads((EVIDENCE / "prefill-fa3-prepack/selection.json").read_text())
    greedy = json.loads((EVIDENCE / "decode-greedy/selection.json").read_text())
    assert pair["adopted"] is True and prepack["adopted"] is True and prepack["selected"] == "P3" and greedy["adopted"] is True and greedy["vocab_size"] == 151936
    prefill = (tmp_path / "source/prefill-kernels.env").read_text()
    decode = (tmp_path / "source/decode-kernels.env").read_text()
    assert "SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen-pipe\n" in prefill
    assert pair["env"] == {"SGLANG_PREFILL_FA3_VERIFY": "n192-direct-o-varlen-pipe+m192n96-pair"}
    assert prefill.replace("SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen-pipe\n", "SGLANG_PREFILL_FA3_VERIFY=" + pair["env"]["SGLANG_PREFILL_FA3_VERIFY"] + "-prepack-v3\n") == PREFILL_ENV
    assert decode.replace("SGLANG_DECODE_GREEDY_VERIFY=0\n", "SGLANG_DECODE_GREEDY_VERIFY=1\n") == DECODE_ENV


def worker(text):
    return text.split('cat >"${ROOT}/source/worker.sh" <<\'EOF\'\n', 1)[1].split("\nEOF\n", 1)[0]


def test_worker_is_unchanged_except_nsight_and_prefill_routing_capture():
    expected = worker(ORIGINAL.read_text())
    dormant = re.compile(r"if \[\[ -x /opt/nsys/bin/nsys \]\]; then\n.*?\nfi\n", re.DOTALL)
    expected, count = dormant.subn("", expected)
    assert count == 1
    capture = re.compile(r"# Keep inherited diagnostic variables out of Decode and ordinary benchmark workers\.\n.*?\nfi\n", re.DOTALL)
    expected, count = capture.subn("", expected)
    assert count == 1
    assert worker(BATCH.read_text()) == expected


def test_diff_against_main_is_deletions_plus_listed_lines():
    package = BATCH.read_text().splitlines()
    original = ORIGINAL.read_text().splitlines()
    assert [line for line in package if line not in original] == ADDED_LINES
    # The remaining lines keep main's order: only deletions happened between them.
    position = 0
    for line in (line for line in package if line not in ADDED_LINES):
        position = original.index(line, position) + 1


def test_submission_has_no_selection_gate_telemetry_or_profile_execution():
    text = BATCH.read_text()
    active = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    for forbidden in ("PROFILE", "/opt/nsys", "profile_capture", "profiler_mount", "scripts/select-firmus-", "check-firmus-", "telemetry", "--loop", "collect-eagle-acceptance", "eagle-acceptance", "dense_selection", "fa3_selection", "--aggregate", "fa3_pair_gate", "fa3_prepack_gate", "decode_greedy_verify_gate", "profile_only", "routing_capture_samples", "SGLANG_PREFILL_ROUTING", "pp4-kernel-settings.json", "reused-kernel-selection", "timeout --", "torchrun", "uv run"):
        assert forbidden not in active, forbidden
    outside_worker = text.replace(worker(text), "")
    assert "nvidia-smi" not in outside_worker
    for directive in ("#SBATCH -p hpcai", "#SBATCH -N 2", "#SBATCH -t 00:17:00", "#SBATCH --gres=gpu:8"):
        assert directive in text
    assert 'ROOT=${SLURM_SUBMIT_DIR}' in text
    assert 'SGLANG=${ROOT}/sglang.sqsh' in text
    assert 'test -s "${ROOT}/source/prefill-kernels.env" && test -s "${ROOT}/source/decode-kernels.env"' in text


def test_serving_launches_warmup_benchmark_and_probe_order():
    def critical(text):
        text = text.split('test -s "${ROOT}/results/score.json"', 1)[0]
        return [line for line in text.splitlines() if line.startswith(("srun --exact", "srun --overlap -N1", "wait_http ", 'echo "READY', "  curl --fail", "curl --fail", "while IFS=$'\\t' read -r worker host port _; do curl"))]
    assert critical(BATCH.read_text()) == critical(ORIGINAL.read_text())
    text = BATCH.read_text()
    marks = ['>"${ROOT}/results/server-info-${worker}.json"', '>"${ROOT}/results/sanity-chat.json"', "/freeze_gc", 'bash "${ROOT}/scripts/official-aiperf.sh"', 'test -s "${ROOT}/results/score.json"', "results/decode-greedy-dispatch.txt", "if (( benchmark_status == 0 )); then", '"${ROOT}/scripts/greedy-probe.py"', 'echo "AIPERF ok"']
    positions = [text.index(mark) for mark in marks]
    assert positions == sorted(positions)
    assert text.count('"${ROOT}/scripts/greedy-probe.py"') == 1
    assert text.count('bash "${ROOT}/scripts/official-aiperf.sh"') == 1
    dispatch = next(line for line in ORIGINAL.read_text().splitlines() if "decode-greedy-dispatch.txt" in line)
    assert "(using fused kernels|fallback).*'" in dispatch and dispatch in text


def test_draft_precision_launch_and_runtime_check_match_canonical():
    package = BATCH.read_text()
    canonical = ORIGINAL.read_text()
    def arguments(text):
        return text.split("speculation=(", 1)[1].split("decode_graph=(", 1)[0]
    assert arguments(package) == arguments(canonical)
    precision_check = next(line for line in canonical.splitlines() if ".speculative_draft_model_quantization ==" in line)
    assert precision_check in package
    assert package.index(precision_check) < package.index('bash "${ROOT}/scripts/official-aiperf.sh"')
    assert json.loads((PACKAGE / "config/serving.json").read_text())["speculation"]["draft_quantization"] == "fp8"


def test_prepare_stages_the_keep_actions_only():
    text = (PACKAGE / "run/prepare.sh").read_text()
    for required in ('[[ $# -eq 4 ]]', 'mkdir "${runtime_dir}"', '.images.sglang.size_bytes', '.images.sglang.sha256', 'e156cb4efae43fbee1a1ab073f946a1377e6b969', '== 24 ]]', 'ac9c66cc9b46af7306746a9250f23d47083d689e', 'bf226a966868fe413a22e62e471aa1130bd94f17fd2c40d322b1c8b4def1ae86', 'stage_overlay sglang opt/sglang/source/python/sglang python/sglang', 'stage_overlay deep-gemm opt/sglang/venv/lib/python3.12/site-packages/deep_gemm deep_gemm', 'install -m 0644 "${package}/config/prefill-kernels.env" "${package}/config/decode-kernels.env" "${runtime_dir}/source/"', 'install -m 0644 "${package}/config/firmus-gpu-sm-clock.json" "${runtime_dir}/source/firmus-gpu-sm-clock.json"', '"${package}/run/order-firmus-pp4-gpus.py"', '"${package}/run/greedy-probe.py"', "sbatch scripts/submit.sbatch"):
        assert required in text, required
    for forbidden in ("check-firmus-pp4-cpu", "check-firmus-image", "prepare-firmus-pp4-settings", "prefill-expert-location", "GITHUB_"):
        assert forbidden not in text, forbidden


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


def test_generic_launcher_shell_remains_byte_identical_to_parent():
    # The local result parser now accepts the isolated Basic client path. Its
    # own fixtures check exact workload matching; freeze the serving shell here.
    launcher = (ROOT / "platforms/shared/qwen-pd.sbatch").read_text()
    prefix, embedded = launcher.split("<<'PY'\n", 1)
    parser, suffix = embedded.split("\nPY\n", 1)
    assert parser + "\n" == (ROOT / "scripts/parse-aiperf-result.py").read_text()
    shell = prefix + "<<'PY'\n<PARSER>\nPY\n" + suffix
    assert hashlib.sha256(shell.encode()).hexdigest() == "9fe400956c7e641ee6f33b45bbf44849c6db6ef6234b15e6601020b7d9cc1fd3"
    provision = (ROOT / "scripts/provision-qwen-pd.sh").read_text()
    assert '"${GITHUB_WORKSPACE}/platforms/shared/qwen-pd.sbatch" "${runtime_dir}/scripts/submit.sbatch"' in provision


def test_curated_evidence_checksums_and_local_official_parser(tmp_path):
    assert sorted(path.name for path in (PACKAGE / "results").iterdir()) == ["41573354672"]
    checked = set()
    for line in (EVIDENCE / "SHA256SUMS").read_text().splitlines():
        digest, path = line.split("  ", 1)
        target = EVIDENCE / path
        assert target.stat().st_size <= 20_000_000
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        checked.add(target)
    assert checked == {p for p in EVIDENCE.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
    output = tmp_path / "score.json"
    subprocess.run(["python3", str(PACKAGE / "run/parse-aiperf-result.py"), str(EVIDENCE / "profile_export_aiperf.json"), str(output), "--concurrency", str(CONFIG["concurrency"]), "--require-eligible", "--command", str(EVIDENCE / "aiperf-command.txt"), "--provenance", str(EVIDENCE / "aiperf-provenance.txt")], check=True)
    score = json.loads((EVIDENCE / "score.json").read_text())
    assert json.loads(output.read_text()) == score
    assert round(score["good_output_tokens_per_second"], 2) == 10021.60 and score["request_count"] == 6320 and score["error_request_count"] == 0
    assert (EVIDENCE / "logs/slurm-accounting.txt").read_text() == "28478|COMPLETED|0:0|00:13:22|2\n"
    dispatch = re.findall(r"Decode greedy verify using fused kernels greedy_tiles/greedy_finish: rows=4 vocab=151936 stride=151936", (EVIDENCE / "logs/decode-0.log").read_text(errors="replace"))
    assert len(dispatch) == 4


def test_patch_inventory_lists_every_applied_patch_once_with_its_hash():
    readme = (PACKAGE / "patches/README.md").read_text()
    rows = [line for line in readme.splitlines() if line.startswith("| [")]
    listed = {}
    for row in rows:
        cells = row.strip("|").split(" | ")
        match = re.search(r"\(source: \[([^\]]+)\]", cells[0]) or re.match(r" ?\[([^\]]+)\]", cells[0])
        path = match.group(1)
        assert path not in listed, path
        listed[path] = (cells[1].strip("` "), cells[4].strip())
    build = (ROOT / "scripts/build-sglang-container.sh").read_text()
    expected = re.findall(r'^[a-z0-9_]+_patch="(patches/sglang/[^"]+)"$', build, re.MULTILINE)
    assert len(expected) == 23
    expected += ["patches/deep-gemm/" + name for name in (ROOT / "patches/deep-gemm/series").read_text().split()]
    expected += ["patches/deep-gemm-host/" + name for name in (ROOT / "patches/deep-gemm-host/series").read_text().split()]
    expected += ["patches/sglang-router/prefill-load-lifetime.patch", "patches/sglang-router/no-delay-acceptor.patch", "patches/sglang-router/Cargo.lock", "kernels/probes/fa3-fp8-output.patch", "containers/install-accelerator-dependencies.sh"]
    expected += CONFIG["sglang_overlay_patches"] + CONFIG["deep_gemm_overlay_patches"]
    expected += [path.relative_to(ROOT).as_posix() for directory in ("kernels/dense-dg1", "kernels/fa3-verify", "kernels/prefill") for path in (ROOT / directory).iterdir() if path.is_file()]
    assert len(expected) == len(set(expected))
    for path in expected:
        assert listed[path][0] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), path
    for path in ("build/sglang-fc275674.bundle", "build/sglang-6b350686.commit"):
        assert listed[path][0] == hashlib.sha256((PACKAGE / path).read_bytes()).hexdigest()
    assert set(listed) == set(expected) | {"build/sglang-fc275674.bundle", "build/sglang-6b350686.commit"}
    classes = [klass for _, klass in listed.values()]
    assert set(classes) <= {"bit-identical", "FP-order", "no numeric effect", "to confirm"}
    assert f"**{len(rows)} rows; {classes.count('to confirm')} are marked `to confirm`**" in readme


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
