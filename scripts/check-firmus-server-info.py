"""Check archived effective settings for every Attention-DP scheduler."""
import argparse
import json
from pathlib import Path


def check(config, info, role):
    aspire = "workers" in config
    if aspire:
        config = {**config, "decode_dp_size": config["dp_size"], "prefill": {"dp": config["dp_size"], "ep": 4, "chunked_prefill_size": config["prefill_chunked_size"], "moe_a2a_backend": "megamoe", "moe_runner_backend": "deep_gemm", "expert_location": config["prefill_expert_location"]}}
    prefill = role == "prefill"
    p = config["prefill"]
    expected = {"tp_size": 4, "dp_size": p["dp"] if prefill else config["decode_dp_size"], "ep_size": p["ep"] if prefill else 4, "enable_dp_attention": True, "kv_cache_dtype": "fp8_e4m3", "context_length": 8192, "page_size": 64, "attention_backend": "fa3", "max_total_tokens": config.get("prefill_max_total_tokens", 131072) if prefill else config["decode_max_total_tokens"], "chunked_prefill_size": p["chunked_prefill_size"] // p["dp"] if prefill else 8192 // config["decode_dp_size"], "max_running_requests": config["concurrency"] if prefill else config["decode_max_running_requests"], "speculative_algorithm": "EAGLE3", "speculative_draft_model_path": "/draft", "speculative_num_steps": config["speculation"]["steps"], "speculative_num_draft_tokens": config["speculation"]["draft_tokens"], "speculative_eagle_topk": 1, "speculative_draft_model_quantization": None, "enable_two_batch_overlap": False, "enable_symm_mem": False}
    if prefill:
        expected.update(moe_a2a_backend=p["moe_a2a_backend"], moe_runner_backend=p["moe_runner_backend"])
        if p["moe_a2a_backend"] == "deepep":
            expected.update(deepep_mode="normal", disable_cuda_graph=True)
    else:
        # SGLang resolves an omitted draft runner to the target runner before serving this endpoint.
        expected.update(disaggregation_decode_extra_slots=config["decode_extra_slots"], moe_a2a_backend="megamoe", moe_runner_backend="deep_gemm", enable_dp_lm_head=True, moe_dense_tp_size=1, speculative_moe_a2a_backend=None, speculative_moe_runner_backend="deep_gemm")
    placed = prefill and p["expert_location"] != "trivial"
    # A non-trivial --init-expert-location makes SGLang dispatch through its static logical-to-physical table.
    expected.update(ep_dispatch_algorithm="static" if placed else None)
    expected.update(load_balance_method=config[f"{role}_load_balance_method"] if aspire else "total_requests", load_snapshot_publish_interval=1)
    if aspire:
        expected["mem_fraction_static"] = 0.90 if prefill else config["decode_mem_fraction_static"]
        if prefill:
            expected["disable_cuda_graph"] = True
    states = info.get("internal_states", [])
    if len(states) != expected["dp_size"]:
        raise ValueError(f"{role}: expected {expected['dp_size']} DP states, got {len(states)}")
    for index, state in enumerate([info, *states]):
        for name, value in expected.items():
            if name not in state or state[name] != value:
                raise ValueError(f"{role} state {index}: {name}={state.get(name)!r}, expected {value!r}")
        location = state.get("init_expert_location")
        if not ((location or "").endswith("/source/prefill-expert-location.json") if placed else location == "trivial"):
            raise ValueError(f"{role} state {index}: init_expert_location={location!r}")
    if not prefill and info.get("cuda_graph_bs_decode") != config["decode_cuda_graph_bs"]:
        raise ValueError("Decode CUDA graph request list changed")
    if not prefill and any(state.get("effective_max_running_requests_per_dp") != config["decode_max_running_requests"] // config["decode_dp_size"] for state in states):
        raise ValueError("Decode effective scheduler limit differs from requested batch cap")
    if aspire and any(state.get("memory_usage", {}).get("token_capacity") != expected["max_total_tokens"] for state in states):
        raise ValueError(f"{role}: actual KV capacity differs from requested pool; inspect memory profiling logs")
    print(f"PASS: {role} TP4, DP{expected['dp_size']}, EP{expected['ep_size']}; all DP schedulers checked")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("info", type=Path)
    parser.add_argument("role", choices=["prefill", "decode"])
    args = parser.parse_args()
    check(json.loads(args.config.read_text()), json.loads(args.info.read_text()), args.role)
