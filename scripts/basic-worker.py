"""Launch an unmodified upstream worker using only public controls."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def launch(config, role, index, host):
    assert config["tier"] == "basic" and role in ("prefill", "decode")
    assert config["prefill"]["workers"] == 3
    assert all(config["prefill"][key] == 4 for key in ("tp", "dp", "ep"))
    assert config["decode_dp_size"] == 4
    assert config["speculation"]["algorithm"] == "EAGLE3"
    port = 30100 + index if role == "prefill" else 30200
    args = [
        "/opt/sglang/bin/python", "-m", "sglang.launch_server",
        "--model-path", "/model", "--served-model-name", "qwen",
        "--context-length", "8192", "--tp-size", "4", "--attn-dp-size", "4",
        "--ep-size", "4", "--host", "0.0.0.0", "--port", str(port),
        "--dist-init-addr", f"127.0.0.1:{32000 + index * 100 if role == 'prefill' else 29000}",
        "--nccl-port", str(port + 1000),
        "--mem-fraction-static", str(config["mem_fraction_static"]),
        "--page-size", str(config["page_size"]),
        "--kv-cache-dtype", config["kv_cache_dtype"],
        "--attention-backend", config["attention_backend"],
        "--moe-a2a-backend", config["moe_a2a_backend"],
        "--moe-runner-backend", config["moe_runner_backend"],
        "--disaggregation-mode", role, "--disaggregation-transfer-backend", "nixl",
        "--speculative-algorithm", "EAGLE3", "--speculative-draft-model-path", "/draft",
        "--speculative-num-steps", str(config["speculation"]["steps"]),
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", str(config["speculation"]["draft_tokens"]),
    ]
    if role == "prefill":
        p = config["prefill"]
        args += ["--disaggregation-bootstrap-port", str(9000 + index), "--chunked-prefill-size", str(p["chunked_prefill_size"]), "--max-total-tokens", str(p["max_total_tokens"]), "--max-running-requests", str(p["max_running_requests"]), "--optimistic-prefill-attempts", str(p["optimistic_prefill_attempts"]), "--disable-cuda-graph"]
        capacity = p["chunked_prefill_size"] // p["dp"]
    else:
        args += ["--max-total-tokens", str(config["decode_max_total_tokens"]), "--max-running-requests", str(config["decode_max_running_requests"]), "--disaggregation-decode-extra-slots", str(config["decode_extra_slots"]), "--cuda-graph-bs-decode", *map(str, config["decode_cuda_graph_bs"])]
        capacity = max(256, max(config["decode_cuda_graph_bs"]) * config["speculation"]["draft_tokens"])
    env = {
        "SGLANG_HOST_IP": host,
        "SGLANG_RUST_BUILD_MODE": "never",
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": str(capacity),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }
    return args, env


def main():
    root, role, index, host = sys.argv[1:]
    root = Path(root)
    args, env = launch(json.loads((root / "source/firmus-framework.json").read_text()), role, int(index), host)
    name = f"{role}-{index}"
    env.update({"TRITON_CACHE_DIR": f"/tmp/qwen-basic-{os.environ['SLURM_JOB_ID']}-{name}/triton", "TORCHINDUCTOR_CACHE_DIR": f"/tmp/qwen-basic-{os.environ['SLURM_JOB_ID']}-{name}/inductor", "SGLANG_DG_CACHE_DIR": str(root / "home/deep-gemm")})
    for path in (env["TRITON_CACHE_DIR"], env["TORCHINDUCTOR_CACHE_DIR"]):
        Path(path).mkdir(parents=True, exist_ok=True)
    hcas = [f"{p.name}:1" for p in sorted(Path("/sys/class/infiniband").glob("*")) if (p / "ports/1/state").read_text().strip().startswith("4:")]
    if not hcas:
        raise RuntimeError("No active InfiniBand HCA")
    env["UCX_NET_DEVICES"] = ",".join(hcas)
    env["UCX_POSIX_USE_PROC_LINK"] = "n"
    (root / "source" / f"{name}-launch.json").write_text(json.dumps({"argv": args, "environment": env}, indent=2) + "\n")
    subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version,power.limit,clocks.max.sm", "--format=csv"], check=True)
    print(json.dumps({"argv": args, "environment": env}), flush=True)
    os.execvpe(args[0], args, {**os.environ, **env})


if __name__ == "__main__":
    main()
