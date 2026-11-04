"""Start one unmodified SGLang 0.5.23 worker with public launch_server flags only; submit.sbatch runs it through srun.

Every value comes from source/firmus-framework.json (configs/firmus-basic.json). Each worker is TP4 with DP
attention 4 and EP4 on the four GPUs of its srun step, with EAGLE3 speculation and NIXL KV transfer.

usage: worker.py RUN_DIR ROLE INDEX HOST_IP   (ROLE is prefill or decode)
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def launch_args(config, role, index):
    port = 30100 + index if role == "prefill" else 30200
    # DP attention derives TCP endpoints from this address: each worker on a node gets its own range.
    dist_port = 32000 + index * 100 if role == "prefill" else 29000
    speculation = config["speculation"]
    args = ["/opt/sglang/bin/python", "-m", "sglang.launch_server", "--model-path", "/model", "--served-model-name", "qwen"]
    args += ["--context-length", "8192", "--tp-size", "4", "--attn-dp-size", "4", "--ep-size", "4"]
    args += ["--host", "0.0.0.0", "--port", str(port), "--dist-init-addr", f"127.0.0.1:{dist_port}", "--nccl-port", str(port + 1000)]
    args += ["--mem-fraction-static", str(config["mem_fraction_static"]), "--page-size", str(config["page_size"]), "--kv-cache-dtype", config["kv_cache_dtype"]]
    args += ["--attention-backend", config["attention_backend"], "--moe-a2a-backend", config["moe_a2a_backend"], "--moe-runner-backend", config["moe_runner_backend"]]
    args += ["--disaggregation-mode", role, "--disaggregation-transfer-backend", "nixl"]
    args += ["--load-balance-method", config["load_balance_method"], "--load-snapshot-publish-interval", str(config["load_snapshot_publish_interval"])]
    args += ["--speculative-algorithm", "EAGLE3", "--speculative-draft-model-path", "/draft", "--speculative-num-steps", str(speculation["steps"])]
    args += ["--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", str(speculation["draft_tokens"])]
    if role == "prefill":
        prefill = config["prefill"]
        args += ["--disaggregation-bootstrap-port", str(9000 + index), "--chunked-prefill-size", str(prefill["chunked_prefill_size"])]
        args += ["--max-total-tokens", str(prefill["max_total_tokens"]), "--max-running-requests", str(prefill["max_running_requests"])]
        args += ["--optimistic-prefill-attempts", str(prefill["optimistic_prefill_attempts"]), "--disable-cuda-graph"]
        # The MegaMoE buffer holds one chunk per DP rank.
        megamoe_tokens = prefill["chunked_prefill_size"] // prefill["dp"]
    else:
        args += ["--max-total-tokens", str(config["decode_max_total_tokens"]), "--max-running-requests", str(config["decode_max_running_requests"])]
        args += ["--disaggregation-decode-extra-slots", str(config["decode_extra_slots"]), "--cuda-graph-bs-decode", *map(str, config["decode_cuda_graph_bs"])]
        # The MegaMoE buffer holds the largest CUDA graph batch times the draft tokens per request, at least 256.
        megamoe_tokens = max(256, max(config["decode_cuda_graph_bs"]) * speculation["draft_tokens"])
    return args, megamoe_tokens


def main():
    root, role, index, host = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
    name = f"{role}-{index}"
    config = json.loads((root / "source/firmus-framework.json").read_text())
    args, megamoe_tokens = launch_args(config, role, index)

    # NIXL over UCX on every active InfiniBand port of this node.
    hcas = [f"{device.name}:1" for device in sorted(Path("/sys/class/infiniband").glob("*")) if (device / "ports/1/state").read_text().startswith("4:")]
    cache = f"/tmp/qwen-basic-{os.environ['SLURM_JOB_ID']}-{name}"
    env = {
        "SGLANG_HOST_IP": host,
        "SGLANG_RUST_BUILD_MODE": "never",
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": str(megamoe_tokens),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TRITON_CACHE_DIR": f"{cache}/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{cache}/inductor",
        "SGLANG_DG_CACHE_DIR": str(root / "home/deep-gemm"),
        "UCX_NET_DEVICES": ",".join(hcas),
        "UCX_POSIX_USE_PROC_LINK": "n",
    }
    Path(env["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    # Record the exact launch, then replace this process with the server.
    launch = {"argv": args, "environment": env}
    (root / "source" / f"{name}-launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version,power.limit,clocks.max.sm", "--format=csv"], check=True)
    print(json.dumps(launch), flush=True)
    os.execvpe(args[0], args, {**os.environ, **env})


if __name__ == "__main__":
    main()
