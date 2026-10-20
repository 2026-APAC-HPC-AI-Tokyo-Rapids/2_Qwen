"""Synthetic hot-8 placement shared by the kernel gate and benchmark.

Logical experts 0..7 occur once on each GPU; cold experts 8..127 are split
round-robin, 30 per GPU. Each rank owns 38 consecutive physical IDs. Static
dispatch chooses the local copy when present, otherwise the sole cold owner.
This tests physical geometry, not the measured serving frequency distribution:
uniform logical routing has expected local fraction 8/128 + 120/128/4 = 0.296875.
"""
import hashlib
import json

import torch
import torch.distributed as dist


def physical_to_logical():
    return [expert for rank in range(4) for expert in [*range(8), *range(8 + rank, 128, 4)]]


def logical_to_physical(rank):
    mapping = physical_to_logical()
    return [next((p for p in range(rank * 38, (rank + 1) * 38) if mapping[p] == expert), mapping.index(expert)) for expert in range(128)]


def remap(ids, rank):
    table = torch.tensor(logical_to_physical(rank), dtype=torch.int32, device=ids.device)
    return torch.where(ids >= 0, table[ids.clamp_min(0).long()], -1).contiguous()


def replica_weights(rank, device):
    # Identical seeds reproduce the checkpoint's FP8 bytes and FP32 scales on
    # every copy; no rank-specific RNG participates in replica initialization.
    mapping = physical_to_logical()[rank * 38:(rank + 1) * 38]
    tensors = [torch.empty(shape, dtype=dtype, device=device) for shape, dtype in (((38, 3072, 4096), torch.float8_e4m3fn), ((38, 4096, 1536), torch.float8_e4m3fn), ((38, 24, 32), torch.float32), ((38, 32, 12), torch.float32))]
    for slot, expert in enumerate(mapping):
        generator = torch.Generator(device=device).manual_seed(19000 + expert)
        for tensor in tensors[:2]:
            tensor[slot] = (torch.randn(tensor.shape[1:], generator=generator, dtype=torch.bfloat16, device=device) * 4).to(torch.float8_e4m3fn)
        for tensor in tensors[2:]:
            tensor[slot] = torch.rand(tensor.shape[1:], generator=generator, device=device) * 0.005 + 0.005
    return tensors


def check_replicas(quantized, dequantized):
    """Compare physical dispatch with a logical-128 FP32 reference, once/expert.

    The first physical owner computes the independent reference using logical
    IDs (never remapped IDs); all-reduce adds each logical contribution once.
    Raw tiny cases cover SwapAB and non-SwapAB shapes. Padded cases mirror
    replicated Prefill's minimum 304 output rows, including live tiny ranks.
    """
    import deep_gemm

    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    mapping = physical_to_logical()
    w13, w2, s13, s2 = replica_weights(rank, device)
    digest = hashlib.sha256()
    for tensor in (w13, w2, s13, s2):
        digest.update(tensor[:8].contiguous().view(torch.uint8).cpu().numpy().tobytes())
    hashes = [None] * 4
    dist.all_gather_object(hashes, digest.hexdigest())
    assert len(set(hashes)) == 1, ("replica weight/scale mismatch", hashes)
    l1, l2 = deep_gemm.transform_weights_for_mega_moe_sm90((w13, s13), (w2, s2))
    buf = deep_gemm.get_symm_buffer_for_mega_moe(dist.group.WORLD, 152, 4096, 8, 4096, 1536, use_fp8_dispatch=True, activation="swiglu")
    generator = torch.Generator(device=device).manual_seed(23000 + rank)
    patterns = torch.randn(4, 4096, generator=generator, dtype=torch.bfloat16, device=device) * 0.25
    weights = torch.rand(4, 8, generator=generator, device=device)
    weights /= weights.sum(-1, keepdim=True)
    references = {}
    for kind in ("hot", "cold", "mixed"):
        if kind == "hot":
            ids = torch.arange(8, device=device).repeat(4, 1)
        elif kind == "cold":
            # Last cold slots include physical lane boundaries 31/32/37 and151.
            ids = torch.tensor([[mapping[r * 38 + slot] for r in range(4) for slot in slots] for slots in ((31, 32), (32, 37), (31, 37), (32, 37))], device=device)
        else:
            ids = torch.stack([torch.randperm(128, generator=generator, device=device)[:8] for _ in range(4)])
        ids = ids.to(torch.int32)
        ids[0, -1] = -1
        gathered = []
        for value in (patterns, ids, weights):
            parts = [torch.empty_like(value) for _ in range(4)]
            dist.all_gather(parts, value)
            gathered.append(torch.cat(parts))
        all_x, all_ids, all_weights = gathered
        x = quantized(all_x, 128)
        for group in (64, 128):
            reference = torch.zeros(16, 4096, device=device)
            for slot, expert in enumerate(mapping[rank * 38:(rank + 1) * 38]):
                if mapping.index(expert) // 38 != rank:
                    continue
                rows, slots = torch.where(all_ids == expert)
                if not rows.numel():
                    continue
                gate, up = (x[rows] @ dequantized(w13[slot], s13[slot]).T).chunk(2, dim=-1)
                activation = torch.nn.functional.silu(gate) * up
                out = quantized(activation, group) @ dequantized(w2[slot], s2[slot]).T
                reference.index_add_(0, rows, out.to(torch.bfloat16).float() * all_weights[rows, slots, None])
            dist.all_reduce(reference)
            references[kind, group] = reference
        references[kind, "ids"] = ids
    cases = [(f"tiny-{n}", [n] * 4, "hot", False) for n in (1, 8, 9, 17, 25, 33, 41, 49, 57, 65, 127, 128)]
    cases += [("mixed-tiny", [7, 0, 65, 129], "cold", False), ("boundary-303", [303] * 4, "mixed", False), ("boundary-304", [304] * 4, "mixed", False), ("boundary-305", [305] * 4, "mixed", False)]
    cases += [("full-hot", [4096] * 4, "hot", True), ("full-cold", [4096] * 4, "cold", True), ("idle-prefill-geometry", [4096, 4096, 4096, 0], "mixed", True), ("mixed-prefill-geometry", [4096, 1, 0, 303], "mixed", True), ("all-idle", [0] * 4, "hot", True), ("live-after-idle", [4096] * 4, "mixed", True)]
    for case, counts, kind, padded in cases:
        count = counts[rank]
        indices = torch.arange(count, device=device) % 4
        hidden = patterns[indices].contiguous()
        top_ids = remap(references[kind, "ids"][indices], rank)
        top_weights = weights[indices].contiguous()
        output_rows = max(count, 304 if padded else 1)
        actual = torch.empty(output_rows, 4096, dtype=torch.bfloat16, device=device)
        # Two calls exercise cleanup/reuse after the preceding routing case.
        for replay in range(2):
            deep_gemm.mega_moe_pre_dispatch_sm90(hidden, top_ids, top_weights, buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights, num_tokens=count, group_size=128, routed_scaling_factor=1.0)
            deep_gemm.fp8_mega_moe(actual, l1, l2, buf, recipe=(128, 128, 128), activation="swiglu", fast_math=True)
            torch.cuda.synchronize()
            group = 128 if padded or min(counts) >= 304 else 64
            expected = references[kind, group][rank * 4 + indices].to(torch.bfloat16).float()
            observed = actual[:count].float()
            assert torch.isfinite(actual).all(), (case, rank)
            relative_l2 = ((observed - expected).norm() / expected.norm().clamp_min(1e-10)).item()
            max_scaled_error = ((observed - expected).abs().max() / expected.abs().max().clamp_min(1e-10)).item() if count else 0.0
            assert relative_l2 < 0.025 and max_scaled_error < 0.08, (case, rank, relative_l2, max_scaled_error)
            assert torch.count_nonzero(actual[count:]).item() == 0, (case, rank, "masked padding")
            print(json.dumps({"case": f"replica8-{case}", "rank": rank, "tokens": count, "output_rows": output_rows, "physical_experts": 152, "replay": replay, "relative_l2": relative_l2, "max_scaled_error": max_scaled_error, "hot_weight_scale_sha256": hashes[rank], "workspace_bytes": buf.buffer.numel()}), flush=True)
            dist.barrier()
    if rank == 0:
        print("PASS: SM90 MegaMoE replica8 logical-128 reference / physical-152 gate", flush=True)
