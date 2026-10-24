"""Run existing EP4 bench + current two-warpgroup timeline on Genkai, no model."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_worker():
    import torch
    import torch.distributed as dist

    b = module("megamoe_bench", ROOT / "bench-firmus-megamoe.py")
    # Same 140-slot geometry as current Prefill; old measured median-hot layer.
    b.CASES = b.TIMELINE_CASES = [("prefill", "real43-replica4", 4096)]
    b.TIMELINE_CALLS = 5
    b.WARMUP, b.ITERATIONS = 5, 30
    original = b.bench_symm_buffer

    def checked_buffer(*args, **kwargs):
        assert torch.cuda.device_count() == 4
        assert all(torch.cuda.can_device_access_peer(b.RANK, i) for i in range(4) if i != b.RANK)
        return original(*args, **kwargs)

    b.bench_symm_buffer = checked_buffer
    b.main()
    assert not dist.is_initialized()


def summary(path):
    import numpy as np

    s = module("timeline_summary", ROOT / "summarize-megamoe-kernel-bench.py")
    data = np.load(path)
    assert data.shape[:2] == (5, 4)
    assert np.all(data[:, :, 1] == 2), "Expected the two-warpgroup timeline ABI"
    # Existing decoder handles WG0; v2 adds WG1 events40..46 and counters24..31.
    calls = [[s.parse(rank) for rank in call] for call in data]
    chosen = sorted(calls[1:], key=lambda call: max(r["end"] for r in call))[len(calls[1:]) // 2]
    for rank, r in enumerate(chosen):
        t, c = r["times"], r["counters"]
        assert r["header"]["epilogue_threads"] == 256
        assert np.isfinite(t[:, [25, 27, 42, 44]]).all(), "Both WGs must record FC1 and FC2"
        def stat(values):
            return [round(float(np.nanmedian(values)), 2), round(float(np.nanmax(values)), 2)]
        result = {"rank": rank, "kernel_us": round(float(r["end"]), 2), "sm_ghz": round(float(np.nanmedian(r["ghz"])), 3), "geometry": r["header"]}
        for label, begin, end in s.PHASES:
            result[label] = stat(t[:,s.E[end]]-t[:,s.E[begin]])
        for i, name in enumerate(s.COUNTERS):
            result[name] = stat(c[:,i])
        result["WG1_FC1_span_us"] = stat(t[:,43]-t[:,42])
        result["WG1_FC2_span_us"] = stat(t[:,45]-t[:,44])
        for i, name in enumerate(("FC1_blocks","FC2_blocks","FC1_full_wait_us","FC2_full_wait_us","FC1_busy_us","FC2_busy_us","FC1_loop_us","FC2_loop_us"),24):
            values=c[:,i] if i<26 else c[:,i]/r["ghz"]/1000
            result["WG1_"+name]=stat(values)
        print("TIMELINE " + json.dumps(result), flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix="megamoe-", dir=os.environ["RUNTIME_DIR"]+"/results") as tmp:
        work = Path(tmp)
        variants=["baseline"] + (["candidate"] if (ROOT/"candidate").is_dir() else []) + ["timeline"]
        rows={}
        for variant in variants:
            env=os.environ.copy()
            env["PYTHONPATH"]=str(ROOT/variant)+os.pathsep+str(ROOT)
            env["NCCL_CUMEM_ENABLE"]="1"
            env["OMP_NUM_THREADS"]="1"
            cmd=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc-per-node=4",str(__file__),"worker","--variant",variant,"--output",str(work/(variant+".jsonl")),"--scratch",str(work)]
            if variant=="timeline":cmd += ["--timeline-dir",str(work/"timeline")]
            if variant!="baseline":cmd += ["--reference",str(work/"baseline.jsonl")]
            subprocess.run(cmd,env=env,check=True)
            rows[variant]=[json.loads(line) for line in (work/(variant+".jsonl")).read_text().splitlines()]
        for variant, entries in rows.items():
            for r in entries:
                assert r["deterministic"]
                if variant!="baseline":assert r["output_sha256_matches_reference"],(variant,r["case"])
                if variant=="timeline":assert r["bitwise_equal_to_reference"]
                print("MEGAMOE " + json.dumps({k:r[k] for k in ("case","variant","median_ms","p10_ms","p90_ms","rank_load_max_over_mean","local_slot_fraction","sm_clock_mhz_before","sm_clock_mhz_after","power_w_before","power_w_after")}),flush=True)
        for path in (work/"timeline").glob("*.npy"):summary(path)
        print("PASS full EP4 MegaMoE: output hashes, instrumented bit equality, both Prefill warpgroups and transfer timeline",flush=True)


if __name__ == "__main__":
    if sys.argv[1:2]==["worker"]:
        del sys.argv[1]
        run_worker()
    else:
        main()
