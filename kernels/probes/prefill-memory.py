"""Bounded head4 QKV memory-layout gate; no serving throughput claim.

Mechanism: BM8 x 4 heads retains 256 threads and 32 eight-lane groups.
Unlike rejected run 40225743220 (BM4 x 32 heads, 97 registers), it retains
one independent head/token per group and the deployed arithmetic. The exact
candidate was CPU-compiled previously: PDL control/candidate 57 registers,
zero spills; this script does not claim a measured GPU improvement.

Preparation needs the current image's rope.cuh after serving overlays. It does
not load CUDA or change serving files. --run is an explicit diagnostic only;
it needs the existing SGLang JIT environment and never submits a scheduler job.
"""
import argparse
import difflib
import hashlib
import json
import statistics
from pathlib import Path

HEADER_PATH = "python/sglang/kernels/jit/csrc/elementwise/rope.cuh"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one current-source match: {old[:80]!r}")
    return source.replace(old, new, 1)


def tiled_header(source, heads):
    """Change only head/row ownership; keep each lane's arithmetic untouched."""
    if heads not in (4, 8):
        raise ValueError("Only four/eight heads per CTA are supported")
    assert ".k = pointer::offset(k.data_ptr(), -static_cast<int64_t>(num_qo_heads) * kHeadBytes)" in source
    source = replace_once(source, "template<bool kUsePDL, typename IdType, bool kVImage=false>\n__global__ __launch_bounds__(256) void qknorm_rope_prefill_fast_kernel(", "template<bool kUsePDL, typename IdType, bool kVImage=false, int kHeadTile=1>\n__global__ __launch_bounds__(256) void qknorm_rope_prefill_fast_kernel(")
    source = replace_once(source, "    constexpr int BM=32, THREADS=256;\n    const int head=blockIdx.y, first_row=blockIdx.x*BM, rows=p.num_tokens;", "    static_assert(kHeadTile==1 || kHeadTile==4 || kHeadTile==8);\n    static_assert(!kVImage || kHeadTile==1);\n    constexpr int BM=32/kHeadTile, THREADS=256;\n    const int head=blockIdx.y*kHeadTile+(threadIdx.x/8)%kHeadTile;\n    const int first_row=blockIdx.x*BM, rows=p.num_tokens;\n    const bool valid_head=head<p.num_qo_heads+2*p.num_kv_heads;")
    source = replace_once(source, "    for (int row = threadIdx.x / 8; row < BM; row += THREADS / 8) {", "    for (int row = (threadIdx.x / 8) / kHeadTile; row < BM; row += THREADS / (8*kHeadTile)) {")
    start = source.index("void qknorm_rope_prefill_fast_kernel(")
    end = source.index("// Decode: four independent H128 heads", start)
    kernel = source[start:end]
    assert kernel.count("if(token<rows)") == 2
    kernel = kernel.replace("if(token<rows)", "if(valid_head && token<rows)")
    source = source[:start] + kernel + source[end:]
    launch = f"""    if (contiguous_k && !v_image && num_qo_heads==64 && num_kv_heads==4) {{
      // 32 independent head/row groups per CTA, eight lanes each, as control.
      // Q/K/V boundaries are warp-aligned; each warp executes one role.
      constexpr int kHeadTile={heads};
      const auto tiled = id_type.is_type<int32_t>() ? qknorm_rope_prefill_fast_kernel<kUsePDL, int32_t, false, kHeadTile> : qknorm_rope_prefill_fast_kernel<kUsePDL, int64_t, false, kHeadTile>;
      LaunchKernel(dim3(div_ceil(num_tokens, 32u/kHeadTile), div_ceil(num_qo_heads+2*num_kv_heads, unsigned(kHeadTile))), 256, device.unwrap()).enable_pdl(kUsePDL)(tiled, params, contiguous_k->data_ptr(), contiguous_v->data_ptr(), nullptr, 0);
      return;
    }}
"""
    source = replace_once(source, "    if (contiguous_k) {\n      const auto prefill = v_image", launch + "    if (contiguous_k) {\n      const auto prefill = v_image")
    return source


def check_mapping(heads, tokens):
    """Every (token, head) has exactly eight lanes and one owner CTA."""
    counts = {}
    rows = 32 // heads
    for bx in range((tokens + rows - 1) // rows):
        for by in range((72 + heads - 1) // heads):
            for group in range(32):
                token, head = bx * rows + group // heads, by * heads + group % heads
                if token < tokens and head < 72:
                    key = token, head
                    counts[key] = counts.get(key, 0) + 1
            # Each warp stays entirely Q, K or V: full-mask XOR is unchanged.
            for warp in range(8):
                warp_heads = [by * heads + group % heads for group in range(warp * 4, warp * 4 + 4)]
                roles = {0 if head < 64 else 1 if head < 68 else 2 for head in warp_heads}
                assert len(roles) == 1
    assert len(counts) == tokens * 72 and all(n == 1 for n in counts.values())


def prepare(header, output):
    source = header.read_text()
    output.mkdir(parents=True, exist_ok=True)
    headers = {"control": output / "rope-heads1.cuh"}
    headers["control"].write_text(source)
    for heads in (4,):
        for tokens in (0, 1, 7, 8, 9, 31, 32, 33, 3986, 4032, 4096):
            check_mapping(heads, tokens)
        candidate = tiled_header(source, heads)
        headers[f"heads{heads}"] = output / f"rope-heads{heads}.cuh"
        headers[f"heads{heads}"].write_text(candidate)
        patch = "".join(difflib.unified_diff(source.splitlines(keepends=True), candidate.splitlines(keepends=True), fromfile="a/" + HEADER_PATH, tofile="b/" + HEADER_PATH))
        (output / f"prefill-qkv-heads{heads}.patch").write_text(patch)
    metadata = {"source_sha256": hashlib.sha256(source.encode()).hexdigest(), "headers": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in headers.items()}, "mapping_check": "PASS", "gpu_executed": False}
    (output / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return headers


def run(headers, output):
    import torch
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args

    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0)
    candidate_name, = (name for name in headers if name != "control")
    funcs = {name: load_jit("prefill_memory_" + name, cuda_files=[str(path.resolve())], cuda_wrappers=[("run", f"QKNormRopeFP8Kernel<{make_cpp_args(is_arch_support_pdl())}>::run_prefill")]).run for name, path in headers.items()}
    torch.manual_seed(9261027)
    angles = torch.randn(8192, 64, device="cuda")
    trig = torch.cat((angles.cos(), angles.sin()), dim=1)
    qw, kw = [(torch.randn(128, device="cuda") * .25 + v).bfloat16() for v in (1, .75)]
    qw[::4], kw[1::4] = 512, -2048
    result = {"scope": "bounded one-GPU exact QKV gate, not serving throughput", "candidate_name": candidate_name, "rotating_input_sets": 4, "cases": []}

    def make_case(rows, dtype, special=False):
        packed = torch.randn(rows, 72, 128, device="cuda", dtype=torch.bfloat16) * 512
        if rows and special:
            # BF16 signed zero, subnormal, finite extreme, +/-Inf and NaN.
            patterns = torch.tensor([0, -32768, 1, -32767, 127, -32641, 32639, -129, 32640, -128, 32704], device="cuda", dtype=torch.int16)
            bits = packed.view(torch.int16).view(-1)[::257]
            bits.copy_(patterns.repeat((bits.numel() + len(patterns) - 1) // len(patterns))[:bits.numel()])
        pos = torch.randint(8192, (rows,), device="cuda", dtype=dtype)
        loc = (torch.randperm(8192, device="cuda")[:rows] + 1).to(dtype)
        if rows:
            loc[0] = 0
            pos[0], pos[-1] = 0, 8191
        shapes = ((rows, 64, 128), (8193, 4, 128), (8193, 4, 128), (rows, 4, 128), (rows, 4, 128))
        buffers = {name: [torch.full(shape, 0xa5, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn) for shape in shapes] for name in funcs}
        calls = {}
        for name, fn in funcs.items():
            oq, kc, vc, ck, cv = buffers[name]
            calls[name] = lambda fn=fn, args=(packed[:, :64], packed[:, 64:68], qw, kw, trig, pos, 1e-6, oq, packed[:, 68:], kc, vc, loc, ck, cv): fn(*args)
        return packed, pos, loc, buffers, calls

    def equal(buffers, label):
        for name in funcs:
            assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(buffers[name], buffers["control"])), f"{name}: byte mismatch {label}"

    for rows in (0, 1, 7, 8, 9, 31, 32, 33, 129, 3986, 4032, 4096):
        for dtype in (torch.int32, torch.int64):
            for special in (False, True):
                packed, pos, loc, buffers, calls = make_case(rows, dtype, special)
                for call in calls.values():
                    call()
                torch.cuda.synchronize()
                equal(buffers, (rows, str(dtype), special))
                if rows in (33, 3986, 4032, 4096):
                    graphs = {}
                    for name in funcs:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            calls[name]()
                        graphs[name] = graph
                    for _ in range(2):
                        packed.neg_()
                        pos.copy_(pos.roll(1))
                        loc.copy_(loc.roll(3))
                        for name in funcs:
                            for tensor in buffers[name]:
                                tensor.view(torch.uint8).fill_(0xa5)
                            graphs[name].replay()
                        torch.cuda.synchronize()
                        equal(buffers, "changed-input graph")
            entry = {"rows": rows, "dtype": str(dtype), "finite_and_nonfinite_five_buffers_bitwise": True, "changed_input_graph_bitwise": rows in (33, 3986, 4032, 4096)}
            result["cases"].append(entry)
            print(json.dumps(entry), flush=True)

    # Four independent finite input/output sets rotate inside each timed graph.
    # Paired A/B order reverses each round, followed by a terminal A control.
    for rows in (3986, 4032, 4096):
        sets = [make_case(rows, torch.int64) for _ in range(4)]
        graphs = {}
        for name in funcs:
            for case in sets:
                case[4][name]()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(2):
                    for case in sets:
                        case[4][name]()
            graphs[name] = graph
        for case in sets:
            equal(case[3], "rotating-input warmup")
        for _ in range(3):
            for graph in graphs.values():
                graph.replay()
        samples = {name: [] for name in funcs}
        order = []
        def measure(name):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / 8)
            order.append(name)
        for pair in range(12):
            for name in tuple(funcs) if pair % 2 == 0 else tuple(funcs)[::-1]:
                measure(name)
        measure("control")
        medians = {name: statistics.median(values) for name, values in samples.items()}
        entry = {"rows": rows, "dtype": "torch.int64", "samples_us": samples, "order": order, "median_us": medians, "ratio": medians[candidate_name]/medians["control"]}
        result.setdefault("timing", []).append(entry)
        print(json.dumps(entry), flush=True)
    # Selection requires a useful consistent reduction and bounded control drift.
    result["candidate_eligible"] = all(case["ratio"] < .97 and abs(case["samples_us"]["control"][-1] / statistics.median(case["samples_us"]["control"][:-1]) - 1) < .05 for case in result["timing"])
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS exact QKV {candidate_name} gate; eligible=" + str(result["candidate_eligible"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run", action="store_true", help="Explicitly run the one-GPU diagnostic after CPU preparation")
    args = parser.parse_args()
    prepared = prepare(args.header, args.output)
    if args.run:
        run(prepared, args.output)
    else:
        print("PASS CPU source/mapping preparation only; no GPU execution")
