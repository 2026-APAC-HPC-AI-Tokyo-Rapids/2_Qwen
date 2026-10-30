"""Check the declared Basic layout against upstream server readback."""
import json
import sys
from pathlib import Path


def check(config, info, role):
    expected = {
        "version": config["sglang_version"], "tp_size": 4, "attn_dp_size": 4,
        "ep_size": 4, "disaggregation_mode": role, "context_length": 8192,
        "kv_cache_dtype": config["kv_cache_dtype"], "page_size": config["page_size"],
        "attention_backend": config["attention_backend"],
        "moe_a2a_backend": config["moe_a2a_backend"],
        "speculative_algorithm": "EAGLE3",
        "speculative_num_steps": config["speculation"]["steps"],
        "speculative_num_draft_tokens": config["speculation"]["draft_tokens"],
        "speculative_eagle_topk": 1,
        "max_running_requests": config["prefill"]["max_running_requests"] if role == "prefill" else config["decode_max_running_requests"],
        "max_total_tokens": config["prefill"]["max_total_tokens"] if role == "prefill" else config["decode_max_total_tokens"],
        "ep_num_redundant_experts": 0,
    }
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f"{role}: {key} expected {value!r}, observed {info.get(key)!r}")
    if role == "prefill":
        assert info["chunked_prefill_size"] == config["prefill"]["chunked_prefill_size"] // 4


if __name__ == "__main__":
    check(json.loads(Path(sys.argv[1]).read_text()), json.loads(Path(sys.argv[2]).read_text()), sys.argv[3])
