"""Basic launch/readback boundaries: isolate the public stack from Bonus."""
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/firmus-basic.json").read_text())


@pytest.mark.parametrize("role,index", [("prefill", 0), ("prefill", 1), ("prefill", 2), ("decode", 0)])
def test_public_launch_has_capacity_and_isolated_port_ranges(config, role, index):
    argv, env = load("basic-worker").launch(config, role, index, "10.0.0.1")
    value = lambda key: argv[argv.index(key) + 1]
    assert value("--attn-dp-size") == "4"
    assert "--dp-size" not in argv and "--enable-dp-attention" not in argv
    assert value("--ep-size") == value("--tp-size") == "4"
    assert value("--disaggregation-mode") == role
    assert value("--disaggregation-transfer-backend") == "nixl"
    assert value("--moe-a2a-backend") == "megamoe"
    assert value("--load-balance-method") == config["load_balance_method"]
    assert value("--load-snapshot-publish-interval") == str(config["load_snapshot_publish_interval"])
    assert value("--speculative-eagle-topk") == "1"
    if role == "prefill":
        assert value("--prefill-max-requests") == str(config["prefill"]["prefill_max_requests"])
    expected = config["prefill"]["chunked_prefill_size"] // 4 if role == "prefill" else max(config["decode_cuda_graph_bs"]) * config["speculation"]["draft_tokens"]
    assert int(env["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK"]) >= expected
    assert int(value("--dist-init-addr").split(":")[1]) < 32768
    assert not any(key.startswith("DG_SM90_") for key in env)
    assert "PYTHONPATH" not in env


def test_request_cap_and_client_concurrency_are_independent(config):
    module = load("basic-worker")
    before, _ = module.launch(config, "decode", 0, "127.0.0.1")
    config["concurrency"] += 16
    after, _ = module.launch(config, "decode", 0, "127.0.0.1")
    assert before == after
    prefill, _ = module.launch(config, "prefill", 0, "127.0.0.1")
    assert int(prefill[prefill.index("--max-running-requests") + 1]) == config["prefill"]["max_running_requests"]


def test_readback_rejects_other_version_and_clipped_pool(config):
    checker = load("check-basic-server-info")
    fixture = json.loads((ROOT / "qwen-experiment-metrics/tests/fixtures/firmus-server-info-38369214512.json").read_text())
    fixture.update(version=config["sglang_version"], tp_size=4, attn_dp_size=4, ep_size=4, disaggregation_mode="decode", context_length=8192, kv_cache_dtype="fp8_e4m3", page_size=64, attention_backend="fa3", moe_a2a_backend="megamoe", speculative_algorithm="EAGLE3", speculative_num_steps=3, speculative_num_draft_tokens=4, speculative_eagle_topk=1, max_running_requests=128, max_total_tokens=config["decode_max_total_tokens"], max_total_num_tokens=config["decode_max_total_tokens"], ep_num_redundant_experts=0)
    fixture.update(load_balance_method=config["load_balance_method"], load_snapshot_publish_interval=config["load_snapshot_publish_interval"])
    fixture["internal_states"] = [{"memory_usage": {"token_capacity": config["decode_max_total_tokens"]}, "effective_max_running_requests_per_dp": 32, "decode_cuda_graph_bs": config["decode_cuda_graph_bs"]} for _ in range(4)]
    checker.check(config, fixture, "decode")
    fixture["internal_states"][2]["memory_usage"]["token_capacity"] -= 64
    with pytest.raises(ValueError, match="DP2.*token capacity"):
        checker.check(config, fixture, "decode")
    fixture["internal_states"][2]["memory_usage"]["token_capacity"] += 64
    fixture["version"] = "0.5.22"
    with pytest.raises(ValueError, match="version"):
        checker.check(config, fixture, "decode")
    fixture["version"] = config["sglang_version"]
    fixture["max_total_tokens"] //= 2
    with pytest.raises(ValueError, match="max_total_tokens"):
        checker.check(config, fixture, "decode")
    fixture["max_total_tokens"] = config["decode_max_total_tokens"]
    fixture["max_total_num_tokens"] //= 2
    with pytest.raises(ValueError, match="max_total_num_tokens"):
        checker.check(config, fixture, "decode")


def test_preparation_rejects_bonus_image_before_creating_runtime(tmp_path):
    image = tmp_path / "1.sqsh"
    image.write_bytes(b"not a Basic image")
    image.with_suffix(".container-sidecar.json").write_text(json.dumps({"tier": "bonus", "applied_patches": ["changed-model.patch"]}))
    runtime = tmp_path / "run"
    result = subprocess.run(["bash", str(ROOT / "submission/basic/prepare.sh"), str(runtime), str(image), str(tmp_path), str(tmp_path), str(ROOT / "configs/firmus-basic.json")], text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert not runtime.exists()


@pytest.mark.parametrize("disconnect", [False, True])
def test_preparation_reaches_draft_without_new_unsquashfs_options(tmp_path, config, disconnect):
    image = tmp_path / "1.sqsh"
    image.write_bytes(b"pinned fixture image")
    config.update(container_sha256=hashlib.sha256(image.read_bytes()).hexdigest(), container_size_bytes=image.stat().st_size, container_source_commit="a" * 40)
    image.with_suffix(".container-sidecar.json").write_text(json.dumps({"tier": "basic", "applied_patches": [], "source_commits": {"sglang": config["sglang_commit"], "qwen": config["container_source_commit"]}, "container_artifact": {"size_bytes": config["container_size_bytes"], "sha256": config["container_sha256"]}}))
    settings = tmp_path / "config.json"
    settings.write_text(json.dumps(config))
    model = tmp_path / "model"
    metadata = model / ".cache/huggingface/download"
    metadata.mkdir(parents=True)
    (metadata / "config.json.metadata").write_text("e156cb4efae43fbee1a1ab073f946a1377e6b969\n")
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3_moe", "quantization_config": {"quant_method": "fp8"}}))
    shards = {f"weight-{i}": f"shard-{i}.safetensors" for i in range(24)}
    for name in shards.values():
        (model / name).write_bytes(b"fixture")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": shards}))
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    for name in ("config.json", "generation_config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
        (tokenizer / name).write_text("{}")
    binary = tmp_path / "bin"
    binary.mkdir()
    marker = tmp_path / "draft-stage"
    child = tmp_path / "download-child"
    download = f"touch '{marker}'; sleep 600 & echo $! >'{child}'; wait" if disconnect else f"touch '{marker}'; exit 78"
    for name, content in {"unsquashfs": "exit 99", "uvx": download}.items():
        path = binary / name
        path.write_text("#!/bin/bash\n" + content + "\n")
        path.chmod(0o755)
    runtime = tmp_path / "run"
    process = subprocess.Popen(["bash", str(ROOT / "submission/basic/prepare.sh"), str(runtime), str(image), str(model), str(tokenizer), str(settings)], env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if disconnect:
        deadline = time.monotonic() + 10
        while not child.exists():
            assert time.monotonic() < deadline
            time.sleep(0.02)
        os.kill(process.pid, signal.SIGHUP)
    _, stderr = process.communicate(timeout=10)
    assert process.returncode == (129 if disconnect else 78), stderr
    if disconnect:
        state = Path(f"/proc/{child.read_text().strip()}/stat")
        assert not state.exists() or state.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    assert marker.exists()
    assert not runtime.exists()
