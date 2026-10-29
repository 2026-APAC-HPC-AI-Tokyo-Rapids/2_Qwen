"""CPU gate for the actual EP1 host selector, TMA layouts and CUDA template.

Also covers the opt-in four-stage Prefill switch (DG_SM90_PREFILL_4STAGE) on
EP1 and EP4: only the stage count and shared memory size may change, and only
for the two-warpgroup Prefill geometry. Its tail split (DG_SM90_PREFILL_TAIL_SPLIT)
may only append the trailing kPrefillTailSplit template argument to that kernel.

Reuse the native-source extractor from the EP4 contract audit. No GPU API is
executed. The original source checkout is read-only; all patches use --check.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    return subprocess.run(*args, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--control-header", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("contract", ROOT / "scripts/check-sm90-independent-wg-host.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    source = out / "source"
    m.prepare_source(args.source_repo, source)
    # Record function and descriptor encoder are unchanged. Only the input
    # matrix changes to cover the requested EP1/EP4 layouts and opt-in switch.
    m.HOST_MAIN = m.HOST_MAIN.split("int main() {")[0] + r'''
int main() {
    using namespace deep_gemm;
    unsetenv("DG_JIT_DEBUG"); unsetenv("DG_PRINT_CONFIGS");
    setenv("DG_SM90_DECODE_INDEPENDENT_WG","0",1);
    for(int ranks : {1,4}) for(int experts : {128,140}) {
        if(ranks==1 && experts!=128) continue;
        for(int tokens : {0,16,256,1023,1024,4001,4096,8002,8192}) {
            if(ranks==4 && tokens>4096) continue;
            for(int local : {0,1}) for(int stage4 : {0,1,2}) {
                prefill_env=1; independent_env=local+2*stage4;
                setenv("DG_SM90_PREFILL_2WG","1",1);
                setenv("DG_SM90_PREFILL_LOCAL_EP1",local?"1":"0",1);
                setenv("DG_SM90_PREFILL_4STAGE",stage4?"1":"0",1);
                setenv("DG_SM90_PREFILL_TAIL_SPLIT",stage4==2?"1":"0",1);
                int cap=ranks==1?8448:4224;
                int pool=layout::get_num_max_pool_tokens(ranks,cap,8,experts/ranks);
                int sf=layout::get_num_sf_ring_tokens(pool,layout::kMinCandidateBlockM);
                torch::Tensor y{tokens,4096,torch::kBFloat16}, l1{pool,4096,torch::kFloat8_e4m3fn}, l1sf{sf,32,torch::kFloat}, l2{pool,1536,torch::kFloat8_e4m3fn}, l2sf{sf,24,torch::kFloat}, w1{experts/ranks*3072,4096,torch::kFloat8_e4m3fn}, w2{experts/ranks*4096,1536,torch::kFloat8_e4m3fn}, wsf{0,0,torch::kFloat};
                sm90_fp8_mega_moe(y,l1,l1sf,l2,l2sf,w1,w2,wsf,wsf,std::nullopt,std::vector<int64_t>(ranks),0,cap,experts/ranks,tokens,8,4096,1536,std::numeric_limits<float>::infinity(),true);
            }
        }
    }
}
'''
    cpp = out / "host.cpp"
    cpp.write_text(m.host_translation_unit(source))
    fmt = args.source_repo / "third-party/fmt/include"
    run(["g++", "-std=c++20", "-O0", "-I" + str(fmt), str(cpp), "-o", str(out / "host")])
    rows = []
    for line in subprocess.check_output([str(out / "host")], text=True).splitlines():
        fields = line.split("\t")
        numbers = list(map(int, fields[:-1]))
        assert len(numbers) == 4 + len(m.HOST_FIELDS) + 7 * len(m.MAP_NAMES)
        template = fields[-1]
        ranks = int(template.split(",")[16])
        host = dict(zip(m.HOST_FIELDS, numbers[4:23], strict=True))
        tokens, experts, _, switches = numbers[:4]
        enabled, stage4 = switches & 1, switches >> 1
        assert host["launch_threads"] == sum(host[x] for x in ("dispatch_threads", "loader_threads", "math_threads"))
        assert host["launch_smem"] == host["smem"] <= 232448
        if ranks == 1 and tokens >= 1024:
            assert host["math_threads"] == (256 if enabled else 512)
            assert host["block_m"] == 128 and host["block_n"] == 256
            assert host["arrival_counter"] and host["reuse_accum"]
            assert host["experts_per_wave"] == 128 and not host["swap_ab"]
        rows.append({"ranks": ranks, "tokens": tokens, "experts": experts, "enabled": enabled, "stage4": stage4, "host": host, "jit_template_args": template, "descriptor_fields": numbers[23:]})
    three = [r for r in rows if not r["stage4"]]
    for before, after in zip(three[::2], three[1::2], strict=True):
        if before["ranks"] == 4 or before["tokens"] < 1024:
            assert before["host"] == after["host"]
            assert before["jit_template_args"] == after["jit_template_args"]
            assert before["descriptor_fields"] == after["descriptor_fields"]
    # The four-stage switch changes only the stage count and shared memory of
    # the two-warpgroup Prefill geometry (18 KiB per-warp CD, four 50,192-byte
    # stages); every other configuration, field and descriptor is unchanged.
    for before, after, split in zip(rows[::3], rows[1::3], rows[2::3], strict=True):
        assert (before["stage4"], after["stage4"], split["stage4"]) == (0, 1, 2)
        assert split["host"] == after["host"] and split["descriptor_fields"] == after["descriptor_fields"]
        two_wg = before["host"]["math_threads"] == 256 and before["host"]["block_m"] == 128
        assert split["jit_template_args"] == after["jit_template_args"] + (",true" if two_wg else "")
        assert before["descriptor_fields"] == after["descriptor_fields"]
        if before["host"]["math_threads"] == 256 and before["host"]["block_m"] == 128:
            assert before["host"]["stages"] == 3 and after["host"]["stages"] == 4
            assert after["host"]["smem"] == after["host"]["launch_smem"] == 1024 + 8192 + 18432 + 144 + 4 * 50192
            assert {k: v for k, v in before["host"].items() if k not in ("stages", "smem", "launch_smem")} == {k: v for k, v in after["host"].items() if k not in ("stages", "smem", "launch_smem")}
            fields = before["jit_template_args"].split(",")
            fields[11] = "4"
            assert ",".join(fields) == after["jit_template_args"]
        else:
            assert before["host"] == after["host"]
            assert before["jit_template_args"] == after["jit_template_args"]
    overlay = out / "overlay"
    target = overlay / m.KERNEL
    target.parent.mkdir(parents=True)
    shutil.copy2(args.control_header, target)
    for patch in ("patches/deep-gemm-bench/prefill-local-ep1/local-ep1.patch", "patches/deep-gemm-bench/prefill-4stage/sm90-prefill-4stage.patch", "patches/deep-gemm-bench/prefill-tail-split/sm90-prefill-tail-split.patch"):
        run(["git", "apply", "--check", str(ROOT / patch)], cwd=overlay)
        run(["git", "apply", str(ROOT / patch)], cwd=overlay)
    names = list(m.KERNEL_NAMES)
    names.insert(names.index("kPrefill2WG") + 1, "kPrefillWarpCD")
    names.insert(names.index("kPrefillWarpCD") + 1, "kPrefillWarpCDRowBytes")
    m.KERNEL_NAMES = ("kQwenPrefillRanks",) + tuple(names)
    selected = [r for r in rows if r["enabled"] and ((r["ranks"] == 1 and r["tokens"] == 8002) or (r["ranks"] == 4 and r["experts"] == 140 and r["tokens"] == 4096))]
    templates = [r["jit_template_args"] for r in selected]
    cpp = out / "layout.cpp"
    cpp.write_text(m.kernel_translation_unit(source, target.read_text(), templates))
    run(["g++", "-std=c++20", "-O0", "-I" + str(fmt), str(cpp), "-o", str(out / "layout")])
    contracts = subprocess.check_output([str(out / "layout")], text=True).splitlines()
    for row, contract in zip(selected, contracts, strict=True):
        values = list(map(int, contract.split()))
        assert values[0] == row["host"]["smem"], (row, values)
        assert values[8] == (18432 if row["stage4"] else 65536), (row, values)
        assert row["jit_template_args"].endswith(",true") == (row["stage4"] == 2), row
        row["kernel_layout"] = values
    if args.nvcc:
        for row in selected:
            name = f"ep{row['ranks']}-stages{row['host']['stages']}" + ("-split" if row["stage4"] == 2 else "")
            cu = out / f"{name}.cu"
            cu.write_text('#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>\nextern "C" void* instantiate() { return reinterpret_cast<void*>(&deep_gemm::sm90_fp8_mega_moe_impl<' + row["jit_template_args"] + '>); }\n')
            includes = [overlay / "deep_gemm/include", args.source_repo / "deep_gemm/include", args.source_repo / "third-party/cutlass/include"]
            with (out / f"{name}-build.log").open("w") as log:
                run([str(args.nvcc), "--cubin", "--std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3", "--ptxas-options=-v,-warn-spills", "-gencode", "arch=compute_90a,code=sm_90a", *["-I" + str(p) for p in includes], str(cu), "-o", str(out / f"{name}.cubin")], stdout=log, stderr=subprocess.STDOUT)
    result = {"source_commit": m.REVISION, "control_header_sha256": hashlib.sha256(args.control_header.read_bytes()).hexdigest(), "candidate_header_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "gpu_executed": False, "native_cuda_compiled": bool(args.nvcc), "cases": rows, "selected_kernels": selected}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS: {len(rows)} native host configurations; EP4 unchanged; EP1 descriptor/layout agreement; four-stage switch changes only the 2WG Prefill stages and the tail split only its template flag; CUDA compilation={bool(args.nvcc)}")


if __name__ == "__main__":
    main()
