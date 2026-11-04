"""CPU contracts for the fixed Firmus PP4 round 34 submission, independent of cluster access."""
from __future__ import annotations

import hashlib
import json
import os
import re
import runpy
import subprocess
from pathlib import Path

import pytest
import submission_harness

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "submission"
CONFIG_PATH = ROOT / "configs/firmus-prefill-pp4.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
BATCH = PACKAGE / "run/submit.sbatch"
EVIDENCE = PACKAGE / "results/41767886175"
# Final env files the workers of Actions run 41767886175 sourced (source/source/*-kernels.env of its archive),
# after the Prefill FA3 pair, Prefill FA3 prepack (P3), Decode greedy and Prefill FC1 first-K gates adopted their candidates.
PREFILL_ENV = "SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen-pipe+m192n96-pair-prepack-v3\nSGLANG_PREFILL_MOE_INPUT=onepass\nSGLANG_PREFILL_QKNORM=fp32\nDG_SM90_PREFILL_4STAGE=1\nDG_SM90_PREFILL_LEAN=1\nDG_SM90_PREFILL_SPLIT_ACCUM=1\nSGLANG_PP4_PROXY_TRANSPORT=copy\nDG_SM90_PREFILL_FC1_FIRST_K=1\n"
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
        "d5cab287c80a2c3772598b617607e6696fff100da557aa4ac0c56f207fba9ff8",
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
        "4a64fab225bd8a671631cdc34691f87492432e1179193dcfbb9c4912af7c4383",
    ],
}
# Runtime overlays added after that run, appended in application order: the PP4 direct loop logprob relay.
ADDED_OVERLAYS = {"sglang_overlay_patches": ["b311fd9a488015ee886e74027b7042c940a0893aa943b20d4862c2508c078c04"], "deep_gemm_overlay_patches": []}
IMAGE = {"path": "/scratch/home/ACCOUNT/qwen/containers/41729881283.sqsh", "sha256": "3a47b757bb88f6bc14ad5f151e8405f645007f25f381f631d88046d9bd1b7b04", "size_bytes": 12296638464}
# Config keys that only fed removed gates, diagnostics, overlay staging from the repository, or the vLLM path.
DROPPED_KEYS = ("sglang_overlay_patches", "deep_gemm_overlay_patches", "kernel_settings_reference", "container_build_run_id", "container_sha256", "container_size_bytes", "container_source_commit", "profile_only", "decode_greedy_verify_gate")
DROPPED_PREFILL_KEYS = ("gpu_sm_clock", "fa3_pair_gate", "fa3_prepack_gate", "fa3_v3_gate", "megamoe_fc1_first_k_gate")


@pytest.mark.parametrize("family,key,count", [("sglang", "sglang_overlay_patches", 34), ("deep-gemm", "deep_gemm_overlay_patches", 14)])
def test_numbered_overlay_order_bytes_and_archived_hashes(family, key, count):
    copies = sorted((PACKAGE / "patches/overlay" / family).glob("*.patch"))
    digests = ARCHIVED_OVERLAYS[key] + ADDED_OVERLAYS[key]
    assert len(copies) == len(CONFIG[key]) == len(digests) == count
    for index, (copy, original, digest) in enumerate(zip(copies, CONFIG[key], digests, strict=True), 1):
        assert copy.name == f"{index:02d}-{Path(original).name}"
        assert copy.read_bytes() == (ROOT / original).read_bytes()
        assert hashlib.sha256(copy.read_bytes()).hexdigest() == digest


def test_serving_configuration_matches_round34_config_and_image():
    expected = {key: value for key, value in CONFIG.items() if key not in DROPPED_KEYS}
    expected["prefill"] = {key: value for key, value in CONFIG["prefill"].items() if key not in DROPPED_PREFILL_KEYS}
    expected["images"] = {"sglang": {key: CONFIG["images"]["sglang"][key] for key in ("path", "sha256", "size_bytes")}}
    serving = json.loads((PACKAGE / "config/serving.json").read_text())
    assert serving == expected
    assert serving["images"]["sglang"] == IMAGE
    assert CONFIG["container_build_run_id"] == 41729881283 and CONFIG["container_sha256"] == IMAGE["sha256"]
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
    # differ from them only by the four adopted gate outcomes recorded in the evidence.
    (tmp_path / "source").mkdir()
    (tmp_path / "results").mkdir()
    runpy.run_path(str(ROOT / "scripts/prepare-firmus-pp4-settings.py"))["prepare"](CONFIG, ROOT, tmp_path)
    pair = json.loads((EVIDENCE / "prefill-fa3-pair/selection.json").read_text())
    prepack = json.loads((EVIDENCE / "prefill-fa3-prepack/selection.json").read_text())
    greedy = json.loads((EVIDENCE / "decode-greedy/selection.json").read_text())
    fc1 = json.loads((EVIDENCE / "prefill-megamoe-fc1-first-k/selection.json").read_text())
    assert fc1["adopted"] is True
    assert pair["adopted"] is True and prepack["adopted"] is True and prepack["selected"] == "P3" and greedy["adopted"] is True and greedy["vocab_size"] == 151936
    prefill = (tmp_path / "source/prefill-kernels.env").read_text()
    decode = (tmp_path / "source/decode-kernels.env").read_text()
    assert "SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen-pipe\n" in prefill
    assert pair["env"] == {"SGLANG_PREFILL_FA3_VERIFY": "n192-direct-o-varlen-pipe+m192n96-pair"}
    assert prefill.replace("SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen-pipe\n", "SGLANG_PREFILL_FA3_VERIFY=" + pair["env"]["SGLANG_PREFILL_FA3_VERIFY"] + "-prepack-v3\n").replace("DG_SM90_PREFILL_FC1_FIRST_K=0\n", "DG_SM90_PREFILL_FC1_FIRST_K=1\n") == PREFILL_ENV
    assert decode.replace("SGLANG_DECODE_GREEDY_VERIFY=0\n", "SGLANG_DECODE_GREEDY_VERIFY=1\n") == DECODE_ENV


# The package scripts were rewritten after the measured runs: for readability, and down to the steps of a
# successful run (no argument, tool or input checks, traps, timeouts or fallbacks). The reference records are
# submission_harness.py recordings of the scripts of commit a688495 (the scripts of the round 25 standalone run
# 41694678252 and of the measured Basic runs 41078642203 and 41093781934), run with this package's current
# configuration and overlays. They are compared with exactly these documented differences:
#   - the Bonus worker launcher moved from a heredoc written to source/worker.sh to scripts/worker.sh;
#   - removed checks: `git apply --check` before each overlay, and the in-job Basic server-info validator
#     (check-basic-server-info.py, which the team workflow still runs on the collected files) and its copy;
#   - added for round 34: the Bonus job records each Prefill worker's FC1 first-K dispatch lines;
#   - the helpers were rewritten too: the AIPerf wrapper run-aiperf.py (it only stopped a failed client start) is
#     gone, the Basic launcher scripts/basic-worker.py became submission/basic/worker.py (same argv and environment
#     as each measured worker's launch record), and the Basic job no longer runs the diagnostic EAGLE collector.
REFERENCE = ROOT / "qwen-experiment-metrics/tests/fixtures/submission-scripts"
SCRIPT_NAMES = ("submit.sbatch", "worker.sh", "worker.py", "basic-worker.py", "prepare.sh", "greedy-probe.py", "official-aiperf.sh", "parse-aiperf-result.py", "prepare_eagle.py", "order-firmus-pp4-gpus.py")
SCRIPT_FILES = {f"{directory}/{name}" for directory in ("scripts", "source") for name in SCRIPT_NAMES}
REMOVED_EVENTS = {"bonus-prepare": re.compile(r"^git \S+ apply --check "), "basic-sbatch": re.compile(r"/check-basic-server-info\.py |/collect-eagle-acceptance\.py ")}
REMOVED_FILES = {"bonus-prepare": {"scripts/run-aiperf.py", "source/run-aiperf.py"}, "bonus-sbatch": {"scripts/run-aiperf.py"}, "basic-prepare": {"scripts/check-basic-server-info.py", "scripts/collect-eagle-acceptance.py", "scripts/run-aiperf.py"}, "basic-sbatch": {"scripts/check-basic-server-info.py", "scripts/collect-eagle-acceptance.py"}}
ADDED_FILES = {"bonus-sbatch": {f"results/prefill-r32-dispatch-{worker}.txt" for worker in range(3)}}


@pytest.mark.parametrize("kind", submission_harness.KINDS)
def test_scripts_do_what_the_measured_scripts_did(kind):
    expected = json.loads((REFERENCE / f"{kind}.json").read_text().replace("H/run/source/worker.sh", "H/run/scripts/worker.sh").replace("H/run/scripts/basic-worker.py", "H/run/scripts/worker.py"))
    actual = submission_harness.record(kind, ROOT)
    # Same srun steps with the same argv, and the same launch_server argv and environment for every worker.
    for key in ("srun", "cleanup_steps", "launch"):
        assert actual[key] == expected[key], key
    # The same ordered HTTP calls and helper, extraction, patch and download commands, less the removed checks.
    removed = REMOVED_EVENTS.get(kind)
    assert actual["events"] == [event for event in expected["events"] if not (removed and removed.search(event))]
    # The same files with the same content, apart from the scripts themselves and the worker logs' banner lines.
    def files(tree):
        return {path: None if path.startswith("logs/") and path.endswith(".log") else digest for path, digest in tree.items() if path not in SCRIPT_FILES | REMOVED_FILES.get(kind, set()) | ADDED_FILES.get(kind, set())}
    assert ADDED_FILES.get(kind, set()) <= actual["tree"].keys()
    assert not REMOVED_FILES.get(kind, set()) & actual["tree"].keys()
    assert files(actual["tree"]) == files(expected["tree"])


def test_bonus_job_has_no_selection_gate_telemetry_or_profile_step():
    text = BATCH.read_text() + (PACKAGE / "run/worker.sh").read_text()
    active = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    for forbidden in ("PROFILE", "/opt/nsys", "profile_capture", "scripts/select-firmus-", "check-firmus-", "telemetry", "--loop", "collect-eagle-acceptance", "fa3_pair_gate", "fa3_prepack_gate", "decode_greedy_verify_gate", "profile_only", "routing_capture_samples", "SGLANG_PREFILL_ROUTING", "SGLANG_DECODE_ROUTING", "pp4-kernel-settings.json", "timeout --", "torchrun", "uv run"):
        assert forbidden not in active, forbidden
    for directive in ("#SBATCH -p hpcai", "#SBATCH -N 2", "#SBATCH -t 00:17:00", "#SBATCH --gres=gpu:8"):
        assert directive in text
    assert json.loads((PACKAGE / "config/serving.json").read_text())["speculation"]["draft_quantization"] == "fp8"


def test_bonus_prepare_stages_no_tuning_step():
    text = (PACKAGE / "run/prepare.sh").read_text()
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
    assert sorted(path.name for path in (PACKAGE / "results").iterdir()) == ["41767886175"]
    checked = set()
    for line in (EVIDENCE / "SHA256SUMS").read_text().splitlines():
        digest, path = line.split("  ", 1)
        target = EVIDENCE / path
        assert target.stat().st_size <= 20_000_000
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        checked.add(target)
    assert checked == {p for p in EVIDENCE.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
    output = tmp_path / "score.json"
    subprocess.run(["python3", str(PACKAGE / "run/parse-aiperf-result.py"), str(EVIDENCE / "profile_export_aiperf.json"), str(output)], check=True)
    score = json.loads((EVIDENCE / "score.json").read_text())
    # The rewritten scorer reproduces every value it reports of the recorded score (which also held the removed checks).
    assert json.loads(output.read_text()).items() <= score.items()
    assert round(score["good_output_tokens_per_second"], 2) == 10054.49 and score["request_count"] == 6333 and score["error_request_count"] == 0
    assert (EVIDENCE / "logs/slurm-accounting.txt").read_text() == "28660|COMPLETED|0:0|00:14:52|2\n"
    dispatch = re.findall(r"Decode greedy verify using fused kernels greedy_tiles/greedy_finish: rows=4 vocab=151936 stride=151936", (EVIDENCE / "logs/decode-0.log").read_text(errors="replace"))
    assert len(dispatch) == 4
    for worker in range(3):
        lines = (EVIDENCE / f"prefill-r32-dispatch-{worker}.txt").read_text().splitlines()
        assert sum("prewarm done" in line for line in lines) == 4 and sum("using candidate" in line for line in lines) == 4


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


@pytest.mark.parametrize("run,job,score,requests", [("41078642203", "27682", 6321.12, 4019), ("41093781934", "27709", 6339.31, 4024)])
def test_basic_evidence_checksums_and_local_official_parser(tmp_path, run, job, score, requests):
    evidence = PACKAGE / "basic/results" / run
    basic = json.loads((ROOT / "configs/firmus-basic.json").read_text())
    checked = set()
    for line in (evidence / "SHA256SUMS").read_text().splitlines():
        digest, path = line.split("  ", 1)
        assert hashlib.sha256((evidence / path).read_bytes()).hexdigest() == digest
        checked.add(evidence / path)
    assert checked == {p for p in evidence.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
    output = tmp_path / "score.json"
    subprocess.run(["python3", str(PACKAGE / "run/parse-aiperf-result.py"), str(evidence / "profile_export_aiperf.json"), str(output)], check=True)
    recorded = json.loads((evidence / "score.json").read_text())
    assert json.loads(output.read_text()).items() <= recorded.items()
    assert recorded["concurrency"] == basic["concurrency"]
    assert round(recorded["good_output_tokens_per_second"], 2) == score and recorded["request_count"] == requests and recorded["error_request_count"] == 0
    assert (evidence / "logs/slurm-accounting.txt").read_text().startswith(f"{job}|COMPLETED|0:0|")


def test_basic_build_record_matches_pinned_image():
    sidecar = json.loads((PACKAGE / "basic/build/provenance.json").read_text())
    basic = json.loads((ROOT / "configs/firmus-basic.json").read_text())
    assert sidecar["container_artifact"]["sha256"] == basic["container_sha256"] and sidecar["container_artifact"]["size_bytes"] == basic["container_size_bytes"]
    assert sidecar["source_commits"]["qwen"] == basic["container_source_commit"] and sidecar["applied_patches"] == []
    assert f"Successfully tagged localhost/apac-qwen-basic:{basic['container_source_commit']}" in (PACKAGE / "basic/build/build.log").read_text()