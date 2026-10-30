"""Basic launch/readback boundaries: isolate the public stack from Bonus."""
import importlib.util
import json
import subprocess
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
    assert value("--speculative-eagle-topk") == "1"
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
    checker.check(config, fixture, "decode")
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
