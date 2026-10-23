"""Kernel-only SM90 MegaMoE EP4 benchmark on the exact Qwen serving shapes.

Run under torchrun with four local GPUs inside the serving image. Weight and
scale setup follow scripts/check-firmus-megamoe.py; the symmetric buffer uses
the serving cap (4096 tokens/rank, as SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK
in sglang.sbatch) so DeepGEMM's host heuristic selects the serving geometries.
The buffer is allocated with extra tail bytes that only the bench-only timeline
patch writes. Each timed iteration is pre-dispatch + fused kernel, exactly the
serving call pair; the reported value per iteration is the maximum over ranks.

Routing generators (top-8 distinct logical experts, EP4):
  balanced: per token, the 8 largest of 128 uniform randoms (uniform subsets).
  local:    rank r's tokens choose 8 of its own experts 32r..32r+31 (no NVLink payload).
  mixed50/mixed64: token t has floor((t+1)*8*p)-floor(t*8*p) local slots,
            computed with integer percentages. Remaining slots cycle through the
            other three ranks by (8*t+slot) modulo 3. Every source rank uses the
            same pattern relative to itself, so aggregate destination-rank loads
            are exactly equal. Within each destination, random top-8 candidates
            supply distinct experts (candidate index = slot); weights are random
            and normalized as usual. Local fraction differs from p by less than
            one slot over the case. This measures locality at 32 experts/rank,
            not replication's extra tiles, hot-expert skew or memory footprint.
  skew:     Gumbel top-8 over logits log(f[r]) - 0.25 * log(j + 1) for expert j of
            rank r, f = (1.0, 1.1, 1.2, 1.3). Calibrated on 16384 tokens to rank loads
            1 : 1.11 : 1.21 : 1.31 and per-expert max/mean about 2.0; the realized
            loads of each case are reported.
  real<L>:  Gumbel top-8 with log(logical_count) from run 39051152456, using
            independent source-rank RNG streams. The median/p90/p10 hot-8 layers
            and their exact static maps are embedded in firmus_megamoe_replicas.
            Each real<L>/replica4 comparison uses the same seed, hidden states,
            logical top-8, routing weights and per-logical-expert weight/scale
            seeds. Only placement changes: 128 static slots versus 140 with
            local-copy preference. Zero counts have zero selection probability.
            Sampling without replacement changes the marginal slot frequencies
            from the input weights; JSONL records the realized hot/local shares
            and logical-routing hash, not just the original hot-8 share.

Each rank reads its GPU's SM clock and power through NVML right before and right
after the timed iterations (outside the CUDA event pair), because Prefill power
throttling (1.47-1.67 GHz) moves the kernel time by several percent. The samples
are instantaneous and flag throttled cases; they are too noisy to normalize by.
With --reference (another variant's bench.jsonl), rows carry the reference
median and whether the output hash matches it; the caller judges mismatches.

--tune-l2-prefetch writes one JSON result for the pre-serving Decode probe.
It requires --decode-routing JSON with source.phase ("decode", or
"prefill-proxy" for measured Prefill frequencies while no Decode dump exists),
source.run_id and layers[{layer, counts[128], physical_to_logical_map[128]}].
Counts are logical-expert frequencies, resampled with Gumbel top-8; Prefill
replica maps are not accepted (Decode keeps the trivial placement). The probe captures 0/8/16 via the
rebuilt native host and measures ABCABC cold replays (128 MiB read inside each
graph), 30 samples per setting per block, two blocks, plus continuous replays.
Only the MegaMoE interval is timed in the cold graphs. Per-step HBM reference
measurements and all samples are retained; selected_stages defaults to 0 unless
every case gains >=2.13 us and both blocks improve. The caller exports the
selected env to the Decode worker; this script does not launch serving.
"""
import argparse
import ctypes
import hashlib
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from firmus_megamoe_replicas import REAL_ROUTING_LAYERS, replica_weights

RANK = int(os.environ["LOCAL_RANK"])
if RANK == 0:
    os.environ["DG_PRINT_CONFIGS"] = "1"
import deep_gemm
import torch.distributed._symmetric_memory as symm_mem

NUM_RANKS, NUM_EXPERTS, TOPK, HIDDEN, INTERMEDIATE, CAP = 4, 128, 8, 4096, 1536, 4096
TIMELINE_MAGIC = 0x314C544D4F45474D
TIMELINE_EXTRA_BYTES = 1 << 20
TIMELINE_CALLS = 8
WARMUP, ITERATIONS = 5, 30
L2_PREFETCH_ENV = "DG_SM90_DECODE_L2_PREFETCH_STAGES"
L2_PREFETCH_STAGES = (0, 8, 16)
HBM_READ_BYTES = 128 << 20
CASES = [("decode", "balanced", n) for n in (4, 16, 32, 64, 96, 128)] + [("decode", "local", 128)]
CASES += [("prefill", "balanced", n) for n in (1024, 2048, 4096)] + [("prefill", "local", 4096), ("prefill", "skew", 4096)]
CASES += [("prefill", f"real{layer}{suffix}", 4096) for layer in REAL_ROUTING_LAYERS for suffix in ("", "-replica4")]
TIMELINE_CASES = [("decode", "balanced", 128), ("prefill", "balanced", 4096)]


def case_name(case):
    return f"{case[0]}-{case[1]}-{case[2]}"


def bench_symm_buffer(group, num_experts=NUM_EXPERTS):
    # Mirrors deep_gemm.SM90SymmBuffer; only the allocation grows by the tail.
    buf = deep_gemm.SM90SymmBuffer.__new__(deep_gemm.SM90SymmBuffer)
    buf.group, buf.num_experts, buf.num_topk, buf.hidden, buf.intermediate_hidden = group, num_experts, TOPK, HIDDEN, INTERMEDIATE
    buf.num_max_tokens_per_rank = (CAP + deep_gemm._C.get_token_alignment_for_mega_moe() - 1) // deep_gemm._C.get_token_alignment_for_mega_moe() * deep_gemm._C.get_token_alignment_for_mega_moe()
    num_bytes, slice_input_buffers = deep_gemm._C.get_symm_buffer_size_for_sm90_mega_moe(NUM_RANKS, num_experts, buf.num_max_tokens_per_rank, TOPK, HIDDEN, INTERMEDIATE, True, "swiglu")
    buf.buffer = symm_mem.empty(num_bytes + TIMELINE_EXTRA_BYTES, dtype=torch.int8, device="cuda")
    buf.handle = symm_mem.rendezvous(buf.buffer, group=group)
    buf.buffer.zero_()
    group.barrier()
    torch.cuda.synchronize()
    views = slice_input_buffers(buf.buffer)
    dtypes = (torch.float8_e4m3fn, None, None, None, torch.float8_e4m3fn, None, torch.float8_e4m3fn, None)
    (buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights, buf.l1_acts, buf.l1_acts_sf, buf.l2_acts, buf.l2_acts_sf) = (deep_gemm._from_dlpack_if_needed(view, dtype) for view, dtype in zip(views, dtypes))
    return buf, num_bytes


def routing(kind, count, generator, device):
    rank = dist.get_rank()
    if kind == "balanced":
        weights, ids = torch.rand(count, NUM_EXPERTS, generator=generator, device=device).topk(TOPK, dim=1)
    elif kind == "local":
        weights, ids = torch.rand(count, NUM_EXPERTS // NUM_RANKS, generator=generator, device=device).topk(TOPK, dim=1)
        ids += rank * (NUM_EXPERTS // NUM_RANKS)
    elif kind in ("mixed50", "mixed64"):
        percent = int(kind.removeprefix("mixed"))
        token = torch.arange(count, device=device)[:, None]
        slot = torch.arange(TOPK, device=device)[None, :]
        local_slots = ((token + 1) * TOPK * percent) // 100 - (token * TOPK * percent) // 100
        destination = torch.where(slot < local_slots, rank, (rank + 1 + (token * TOPK + slot) % (NUM_RANKS - 1)) % NUM_RANKS)
        candidates = torch.rand(count, NUM_RANKS, NUM_EXPERTS // NUM_RANKS, generator=generator, device=device).topk(TOPK, dim=2).indices
        ids = candidates.transpose(1, 2).gather(2, destination[:, :, None]).squeeze(2) + destination * (NUM_EXPERTS // NUM_RANKS)
        weights = torch.rand(count, TOPK, generator=generator, device=device)
    else:
        local = torch.arange(NUM_EXPERTS // NUM_RANKS, device=device, dtype=torch.float32)
        factor = torch.tensor([1.0, 1.1, 1.2, 1.3], device=device).repeat_interleave(NUM_EXPERTS // NUM_RANKS)
        logits = factor.log() - 0.25 * (local + 1).log().repeat(NUM_RANKS)
        uniform = torch.rand(count, NUM_EXPERTS, generator=generator, device=device).clamp_min(1e-20)
        _, ids = (logits - (-uniform.log()).log()).topk(TOPK, dim=1)
        weights = torch.rand(count, TOPK, generator=generator, device=device)
    return ids.to(torch.int32).contiguous(), (weights / weights.sum(-1, keepdim=True)).contiguous()


def case_seed(case, rank):
    kind = case[1]
    if kind.startswith("real"):
        return 100000 + int(kind.split("-")[0].removeprefix("real")) * NUM_RANKS + rank
    return CASES.index(case) * 8 + rank


def real_routing(layer, replicas, count, generator, device):
    data = REAL_ROUTING_LAYERS[layer]
    logits = torch.tensor(data["counts"], dtype=torch.float64, device=device).log()
    uniform = torch.rand(count, NUM_EXPERTS, generator=generator, device=device, dtype=torch.float64).clamp_min(1e-20)
    logical = (logits - (-uniform.log()).log()).topk(TOPK, dim=1).indices
    weights = torch.rand(count, TOPK, generator=generator, device=device)
    mapping = data[{0: "baseline_map", 4: "replica4_map"}[replicas]]
    width = len(mapping) // NUM_RANKS
    rank = dist.get_rank()
    table = [next((p for p in range(rank * width, (rank + 1) * width) if mapping[p] == expert), mapping.index(expert)) for expert in range(NUM_EXPERTS)]
    ids = torch.tensor(table, dtype=torch.int32, device=device)[logical]
    return ids.contiguous(), (weights / weights.sum(-1, keepdim=True)).contiguous(), logical.to(torch.int32)


def gather(value):
    parts = [torch.empty_like(value) for _ in range(NUM_RANKS)]
    dist.all_gather(parts, value)
    return parts


def output_digest(actual, count):
    digest = hashlib.sha256(actual[:count].contiguous().view(torch.uint8).cpu().numpy().tobytes()).digest()
    parts = gather(torch.tensor(list(digest), dtype=torch.uint8, device=actual.device))
    return hashlib.sha256(b"".join(bytes(part.cpu().tolist()) for part in parts)).hexdigest()


def locate_timeline(buf, num_bytes):
    # The patch writes at the first 4096-byte address boundary after the combine buffer.
    start = max(0, num_bytes - (1 << 16)) // 4096 * 4096
    window = buf.buffer[start:num_bytes + 8192].view(torch.int64).cpu()
    base = buf.buffer.data_ptr()
    found = [start + i * 8 for i in range(window.numel()) if int(window[i]) == TIMELINE_MAGIC and (base + start + i * 8) % 4096 == 0]
    assert len(found) == 1, ("timeline header", found)
    offset = found[0]
    header = buf.buffer[offset:offset + 128].view(torch.int64).cpu().tolist()
    words = 16 + header[2] * header[3] * 2 + header[2] * header[4]
    assert offset + words * 8 <= buf.buffer.numel(), ("timeline exceeds tail", offset, words)
    return offset, words


def sample_summary(samples):
    ordered = sorted(samples)
    return {"median_us": statistics.median(ordered), "p10_us": ordered[int(0.1 * (len(ordered) - 1))], "p90_us": ordered[round(0.9 * (len(ordered) - 1))], "rank_max_us": samples}


def select_l2_prefetch(cases, min_gain_us=2.13):
    """Require >=0.2 ms/94 layers in every case and improvement in both blocks."""
    candidates = []
    gains = {}
    for stages in (8, 16):
        case_gains = []
        eligible = bool(cases)
        for case in cases:
            rows = {row["stages"]: row for row in case["measurements"] if row["cache"] == "cold"}
            control, candidate = rows[0], rows[stages]
            gain = control["median_us"] - candidate["median_us"]
            case_gains.append(gain)
            eligible &= gain >= min_gain_us and all(a["median_us"] > b["median_us"] for a, b in zip(control["blocks"], candidate["blocks"], strict=True))
        gains[str(stages)] = case_gains
        if eligible:
            candidates.append((min(case_gains), -stages, stages))
    selected = max(candidates)[2] if candidates else 0
    return {"selected_stages": selected, "min_gain_us": min_gain_us, "per_case_gain_us": gains, "rule": "gain >= threshold in every cold case; both blocks faster; maximize worst-case gain; ties prefer 8"}


def load_decode_routing(path):
    """Measured logical-expert counts. No Decode dump exists yet, so the serving
    job passes the measured Prefill frequencies (REAL_ROUTING_LAYERS) labelled
    phase="prefill-proxy"; the label is carried into the result JSON."""
    data = json.loads(path.read_text())
    assert data["source"]["phase"] in ("decode", "prefill-proxy") and data["source"]["run_id"]
    assert data["layers"]
    seen = set()
    for layer in data["layers"]:
        assert layer["layer"] not in seen
        seen.add(layer["layer"])
        counts = layer["counts"]
        assert len(counts) == NUM_EXPERTS and all(type(n) is int and n >= 0 for n in counts)
        assert sum(n > 0 for n in counts) >= TOPK
        assert sorted(layer["physical_to_logical_map"]) == list(range(NUM_EXPERTS))
    return data


@triton.jit
def hbm_read_kernel(source, checksum, ELEMENTS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(source + offsets, offsets < ELEMENTS, 0)
    tl.store(checksum + tl.program_id(0), tl.sum(values, 0))


def make_hbm_read(device):
    # Independent of all weights/activations. Read every byte and retain checksums.
    data = torch.ones(HBM_READ_BYTES // 4, dtype=torch.int32, device=device)
    checksum = torch.empty(triton.cdiv(data.numel(), 4096), dtype=torch.int32, device=device)

    def read():
        hbm_read_kernel[(checksum.numel(),)](data, checksum, data.numel(), 4096)

    read()
    torch.cuda.synchronize()
    assert checksum.long().sum().item() == data.numel()
    return read, data, checksum


def hbm_reference(read, data, checksum, device):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        read()
    for _ in range(WARMUP):
        graph.replay()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERATIONS)]
    torch.cuda.synchronize()
    clocks_before = gather(torch.tensor([torch.cuda.clock_rate(device)], device=device))
    dist.barrier()
    for start, end in events:
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    clocks_after = gather(torch.tensor([torch.cuda.clock_rate(device)], device=device))
    assert checksum.long().sum().item() == data.numel()
    samples = torch.tensor([a.elapsed_time(b) * 1000 for a, b in events], device=device)
    dist.all_reduce(samples, op=dist.ReduceOp.MAX)
    return {**sample_summary(samples.tolist()), "read_bytes": HBM_READ_BYTES, "sm_clock_mhz_before": [x.item() for x in clocks_before], "sm_clock_mhz_after": [x.item() for x in clocks_after]}


def tune_l2_prefetch(args, device, read, read_data, checksum, clock_start):
    started = time.monotonic()
    host_record = json.loads(Path("/opt/apac/deep-gemm-host-build.json").read_text())
    host_patch_sha = host_record["patches"]["sm90-decode-l2-prefetch.patch"]
    routing_data = load_decode_routing(args.decode_routing)
    count = 128
    cases = [{"name": "decode-balanced-128", "layer": None}, *[{"name": f"decode-measured-layer{row['layer']}-128", **row} for row in routing_data["layers"]]]
    buf, _ = bench_symm_buffer(dist.group.WORLD)
    results = []
    previous_env = os.environ.get(L2_PREFETCH_ENV)
    try:
        for case in cases:
            generator = torch.Generator(device=device).manual_seed(28000 + RANK + (0 if case["layer"] is None else case["layer"] * NUM_RANKS))
            hidden = (torch.randn(count, HIDDEN, generator=generator, dtype=torch.bfloat16, device=device) * 0.25).contiguous()
            mapping = list(range(NUM_EXPERTS)) if case["layer"] is None else case["physical_to_logical_map"]
            w13, w2, s13, s2 = replica_weights(RANK, device, mapping)
            l1, l2 = deep_gemm.transform_weights_for_mega_moe_sm90((w13, s13), (w2, s2))
            if case["layer"] is None:
                top_ids, top_weights = routing("balanced", count, generator, device)
            else:
                logits = torch.tensor(case["counts"], device=device, dtype=torch.float64).log()
                uniform = torch.rand(count, NUM_EXPERTS, generator=generator, device=device, dtype=torch.float64).clamp_min(1e-20)
                logical_ids = (logits - (-uniform.log()).log()).topk(TOPK, dim=1).indices
                inverse = torch.tensor([mapping.index(expert) for expert in range(NUM_EXPERTS)], device=device, dtype=torch.int32)
                top_ids = inverse[logical_ids].contiguous()
                top_weights = torch.rand(count, TOPK, generator=generator, device=device)
                top_weights = (top_weights / top_weights.sum(-1, keepdim=True)).contiguous()
            actual = torch.empty(count, HIDDEN, dtype=torch.bfloat16, device=device)

            def pre_dispatch(hidden=hidden, top_ids=top_ids, top_weights=top_weights, buf=buf, count=count):
                deep_gemm.mega_moe_pre_dispatch_sm90(hidden, top_ids, top_weights, buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights, num_tokens=count, group_size=128, routed_scaling_factor=1.0)

            def moe(actual=actual, l1=l1, l2=l2, buf=buf):
                deep_gemm.fp8_mega_moe(actual, l1, l2, buf, recipe=(128, 128, 128), activation="swiglu", fast_math=True)

            graphs = {}
            expected = None
            for stages in L2_PREFETCH_STAGES:
                os.environ[L2_PREFETCH_ENV] = str(stages)
                for _ in range(WARMUP):
                    pre_dispatch()
                    moe()
                torch.cuda.synchronize()
                assert torch.isfinite(actual).all(), (case["name"], stages)
                if expected is None:
                    expected = actual.clone()
                else:
                    assert torch.equal(expected, actual), (case["name"], stages, "prefetch changed output")
                for cache in ("warm", "cold"):
                    start, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
                    graph = torch.cuda.CUDAGraph()
                    dist.barrier()
                    with torch.cuda.graph(graph):
                        if cache == "cold":
                            read()
                        pre_dispatch()
                        start.record()
                        moe()
                        end.record()
                    graphs[stages, cache] = (graph, start, end)
                    for _ in range(WARMUP):
                        graph.replay()
                    torch.cuda.synchronize()
            samples = {(stages, cache): [] for stages in L2_PREFETCH_STAGES for cache in ("warm", "cold")}
            for block in range(2):
                for cache in ("warm", "cold"):
                    block_samples = {stages: [] for stages in L2_PREFETCH_STAGES}
                    if cache == "warm":
                        # Continuous same-variant replays retain the original bench mode.
                        for stages in L2_PREFETCH_STAGES:
                            graph, _, _ = graphs[stages, cache]
                            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERATIONS)]
                            dist.barrier()
                            for start, end in events:
                                start.record()
                                graph.replay()
                                end.record()
                            torch.cuda.synchronize()
                            block_samples[stages] = [a.elapsed_time(b) * 1000 for a, b in events]
                    else:
                        # Capture fixes each native JIT kernel; replay order is ABCABC...
                        for _ in range(ITERATIONS):
                            for stages in L2_PREFETCH_STAGES:
                                graph, start, end = graphs[stages, cache]
                                dist.barrier()
                                graph.replay()
                                end.synchronize()
                                block_samples[stages].append(start.elapsed_time(end) * 1000)
                                assert torch.equal(expected, actual), (case["name"], stages, block)
                    for stages in L2_PREFETCH_STAGES:
                        durations = torch.tensor(block_samples[stages], device=device)
                        dist.all_reduce(durations, op=dist.ReduceOp.MAX)
                        samples[stages, cache].append(sample_summary(durations.tolist()))
            measurements = []
            for (stages, cache), blocks in samples.items():
                combined = [value for block in blocks for value in block["rank_max_us"]]
                spread = abs(blocks[0]["median_us"] - blocks[1]["median_us"])
                measurements.append({"stages": stages, "cache": cache, **sample_summary(combined), "blocks": blocks, "block_median_spread_us": spread, "timed_region": "MegaMoE" if cache == "cold" else "pre_dispatch+MegaMoE graph replay"})
            assert checksum.long().sum().item() == read_data.numel()
            loads = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device=device)
            for ids in gather(top_ids):
                loads += torch.bincount(ids.flatten().long(), minlength=NUM_EXPERTS)
            seeds = [28000 + rank + (0 if case["layer"] is None else case["layer"] * NUM_RANKS) for rank in range(NUM_RANKS)]
            results.append({"case": case["name"], "tokens_per_rank": count, "physical_experts": NUM_EXPERTS, "experts_per_rank": 32, "replicas": 0, "routing_method": "uniform-top8" if case["layer"] is None else "gumbel-top8-without-replacement-from-decode-counts", "routing_seed_per_rank": seeds, "sampled_expert_recv_pairs": loads.tolist(), "routing_sha256": output_digest(top_ids, count), "output_sha256": output_digest(actual, count), "measurements": measurements})
    finally:
        if previous_env is None:
            os.environ.pop(L2_PREFETCH_ENV, None)
        else:
            os.environ[L2_PREFETCH_ENV] = previous_env
    clock_end = hbm_reference(read, read_data, checksum, device)
    selection = select_l2_prefetch(results)
    result = {"schema_version": "decode-l2-prefetch-probe/v1", **selection, "env": {L2_PREFETCH_ENV: str(selection["selected_stages"])}, "host": {"source_commit": host_record["source_commit"], "extension_sha256": host_record["extension_sha256"], "prefetch_patch_sha256": host_patch_sha}, "warmup": WARMUP, "blocks": 2, "iterations_per_block": ITERATIONS, "cold_read_bytes": HBM_READ_BYTES, "cold_order": list(L2_PREFETCH_STAGES), "routing_source": routing_data["source"], "routing_file_sha256": hashlib.sha256(args.decode_routing.read_bytes()).hexdigest(), "clock_reference_start": clock_start, "clock_reference_end": clock_end, "clock_reference_ratio": clock_end["median_us"] / clock_start["median_us"], "elapsed_s": time.monotonic() - started, "cases": results}
    if RANK == 0:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "selected_stages": selection["selected_stages"], "elapsed_s": result["elapsed_s"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--timeline-dir", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--tune-l2-prefetch", action="store_true")
    parser.add_argument("--decode-routing", type=Path)
    args = parser.parse_args()
    if args.tune_l2_prefetch and (args.decode_routing is None or args.timeline_dir is not None):
        parser.error("--tune-l2-prefetch requires --decode-routing and excludes --timeline-dir")
    timeline = args.timeline_dir is not None
    torch.cuda.set_device(RANK)
    assert torch.cuda.get_device_capability() == (9, 0)
    dist.init_process_group("nccl", timeout=timedelta(seconds=300), device_id=torch.device("cuda", RANK))
    assert dist.get_world_size() == NUM_RANKS
    device = torch.device("cuda", RANK)
    libc = ctypes.CDLL(None)
    torch.cuda.clock_rate(device)  # Initialize NVML before the first timed case.
    if RANK == 0:
        print(json.dumps({"deep_gemm": deep_gemm.__file__, "variant": args.variant, "timeline": timeline}), flush=True)
    read, read_data, checksum = make_hbm_read(device)
    clock_start = hbm_reference(read, read_data, checksum, device)
    if args.tune_l2_prefetch:
        tune_l2_prefetch(args, device, read, read_data, checksum, clock_start)
        dist.destroy_process_group()
        return
    torch.manual_seed(713 + RANK)
    w13 = (torch.randn(32, 3072, 4096, dtype=torch.bfloat16, device=device) * 4).to(torch.float8_e4m3fn)
    w2 = (torch.randn(32, 4096, 1536, dtype=torch.bfloat16, device=device) * 4).to(torch.float8_e4m3fn)
    s13 = torch.rand(32, 24, 32, device=device) * 0.005 + 0.005
    s2 = torch.rand(32, 32, 12, device=device) * 0.005 + 0.005
    l1, l2 = deep_gemm.transform_weights_for_mega_moe_sm90((w13, s13), (w2, s2))
    buf, num_bytes = bench_symm_buffer(dist.group.WORLD)
    buffers = {NUM_EXPERTS: (buf, num_bytes)}
    reference = {}
    if args.reference is not None:
        reference = {row["case"]: row for row in map(json.loads, args.reference.read_text().splitlines())}
    region = None
    output = args.output.open("w") if RANK == 0 else None
    for case in TIMELINE_CASES if timeline else CASES:
        name = case_name(case)
        stage, kind, count = case
        real = kind.startswith("real")
        if real:
            layer = int(kind.split("-")[0].removeprefix("real"))
            replicas = int(kind.rsplit("-replica", 1)[1]) if "-replica" in kind else 0
            data = REAL_ROUTING_LAYERS[layer]
            mapping = data[{0: "baseline_map", 4: "replica4_map"}[replicas]]
            w13, w2, s13, s2 = replica_weights(RANK, device, mapping)
            l1, l2 = deep_gemm.transform_weights_for_mega_moe_sm90((w13, s13), (w2, s2))
            if len(mapping) not in buffers:
                buffers[len(mapping)] = bench_symm_buffer(dist.group.WORLD, len(mapping))
            buf, num_bytes = buffers[len(mapping)]
        generator = torch.Generator(device=device)
        generator.manual_seed(case_seed(case, RANK))
        hidden = (torch.randn(count, HIDDEN, generator=generator, device=device) * 0.25).to(torch.bfloat16)
        if real:
            top_ids, top_weights, logical_ids = real_routing(layer, replicas, count, generator, device)
        else:
            top_ids, top_weights = routing(kind, count, generator, device)
        actual = torch.empty(count, HIDDEN, dtype=torch.bfloat16, device=device)
        if RANK == 0:
            print(f"### case {name} variant {args.variant}", flush=True)

        def forward(hidden=hidden, top_ids=top_ids, top_weights=top_weights, count=count, actual=actual, buf=buf, l1=l1, l2=l2):
            deep_gemm.mega_moe_pre_dispatch_sm90(hidden, top_ids, top_weights, buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights, num_tokens=count, group_size=128, routed_scaling_factor=1.0)
            deep_gemm.fp8_mega_moe(actual, l1, l2, buf, recipe=(128, 128, 128), activation="swiglu", fast_math=True)

        for _ in range(WARMUP):
            forward()
        torch.cuda.synchronize()
        libc.fflush(None)
        call = forward
        if stage == "decode":
            # Serving replays Decode MegaMoE inside CUDA graphs.
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(graph):
                forward()
            call = graph.replay
            for _ in range(WARMUP):
                call()
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERATIONS)]
        torch.cuda.synchronize()
        gpu_before = [torch.cuda.clock_rate(device), torch.cuda.power_draw(device) / 1000]
        dist.barrier()
        for start, end in events:
            start.record()
            call()
            end.record()
        torch.cuda.synchronize()
        gpu_state = gather(torch.tensor(gpu_before + [torch.cuda.clock_rate(device), torch.cuda.power_draw(device) / 1000], dtype=torch.float64, device=device))
        assert torch.isfinite(actual).all(), (name, RANK)
        durations = torch.tensor([start.elapsed_time(end) for start, end in events], device=device)
        dist.all_reduce(durations, op=dist.ReduceOp.MAX)
        ordered = sorted(durations.tolist())
        first = actual.clone()
        call()
        torch.cuda.synchronize()
        deterministic = torch.tensor(int(torch.equal(first, actual)), device=device)
        dist.all_reduce(deterministic, op=dist.ReduceOp.MIN)
        loads = torch.zeros(buf.num_experts, dtype=torch.int64, device=device)
        local_pairs = 0
        for source, ids in enumerate(gather(top_ids)):
            loads += torch.bincount(ids.flatten().long(), minlength=buf.num_experts)
            local_pairs += (ids // (buf.num_experts // NUM_RANKS) == source).sum().item()
        rank_loads = loads.view(NUM_RANKS, -1).sum(1).tolist()
        row = {"case": name, "variant": args.variant, "stage": stage, "routing": kind, "tokens_per_rank": count, "cuda_graph": stage == "decode", "warmup": WARMUP, "iterations": ITERATIONS, "median_ms": ordered[len(ordered) // 2], "p10_ms": ordered[int(0.1 * (len(ordered) - 1))], "p90_ms": ordered[round(0.9 * (len(ordered) - 1))], "min_ms": ordered[0], "max_ms": ordered[-1], "max_rank_ms": durations.tolist(), "recv_pairs_per_rank": rank_loads, "rank_load_max_over_mean": max(rank_loads) / (sum(rank_loads) / NUM_RANKS), "expert_load_max_over_mean": loads.max().item() / loads.float().mean().item(), "deterministic": bool(deterministic.item()), "output_sha256": output_digest(actual, count)}
        row.update({key: [part[index].item() for part in gpu_state] for index, key in enumerate(("sm_clock_mhz_before", "power_w_before", "sm_clock_mhz_after", "power_w_after"))})
        row.update({"physical_experts": buf.num_experts, "local_slot_fraction": local_pairs / (count * NUM_RANKS * TOPK)})
        if real:
            hot_mask = torch.tensor([expert in data["hot8"] for expert in mapping], device=device)
            row.update({"logical_layer": layer, "layer_selection": data["selection"], "routing_seed_per_rank": [case_seed(case, rank) for rank in range(NUM_RANKS)], "source_hot8_slot_fraction": data["hot8_share"], "sampled_hot8_slot_fraction": loads[hot_mask].sum().item() / (count * NUM_RANKS * TOPK), "logical_routing_sha256": output_digest(logical_ids, count)})
            replicated_ids = data["hot8"][:replicas]
            replicated_mask = torch.tensor([expert in replicated_ids for expert in mapping], device=device)
            row.update({"replicated_logical_experts": replicas, "source_replicated_slot_fraction": sum(data["counts"][expert] for expert in replicated_ids) / sum(data["counts"]), "sampled_replicated_slot_fraction": loads[replicated_mask].sum().item() / (count * NUM_RANKS * TOPK)})
        if name in reference:
            row["reference_variant"] = reference[name]["variant"]
            row["reference_median_ms"] = reference[name]["median_ms"]
            row["delta_median_vs_reference"] = row["median_ms"] / reference[name]["median_ms"] - 1
            row["output_sha256_matches_reference"] = row["output_sha256"] == reference[name]["output_sha256"]
        saved = args.scratch / f"{name}-rank{RANK}.pt"
        if case in TIMELINE_CASES and not timeline:
            torch.save(actual.cpu(), saved)
        if timeline:
            if region is None:
                offset, words = locate_timeline(buf, num_bytes)
                region = buf.buffer[offset:offset + words * 8].view(torch.int64)
                row["timeline_offset_minus_layout_bytes"] = offset - num_bytes
            snapshots = []
            for _ in range(TIMELINE_CALLS):
                region.zero_()
                call()
                snapshots.append(region.clone())
            torch.cuda.synchronize()
            dumps = gather(torch.stack(snapshots))
            if RANK == 0:
                import numpy

                args.timeline_dir.mkdir(parents=True, exist_ok=True)
                numpy.save(args.timeline_dir / f"{name}.npy", torch.stack(dumps, 1).cpu().numpy())
            expected = torch.load(saved)
            difference = torch.tensor([float(not torch.equal(expected, actual.cpu())), (expected.float() - actual.cpu().float()).abs().max().item()], device=device)
            dist.all_reduce(difference, op=dist.ReduceOp.MAX)
            row["bitwise_equal_to_reference"] = difference[0].item() == 0
            row["max_abs_diff_to_reference"] = difference[1].item()
        if RANK == 0:
            libc.fflush(None)
            print(json.dumps(row), flush=True)
            output.write(json.dumps(row) + "\n")
            output.flush()
        dist.barrier()
    if timeline:
        mismatched = [row for row in map(json.loads, args.output.read_text().splitlines()) if not row["bitwise_equal_to_reference"]] if RANK == 0 else []
        assert not mismatched, ("instrumented output differs", mismatched)
    clock_end = hbm_reference(read, read_data, checksum, device)
    if RANK == 0:
        args.output.with_suffix(".clock.json").write_text(json.dumps({"start": clock_start, "end": clock_end}, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
