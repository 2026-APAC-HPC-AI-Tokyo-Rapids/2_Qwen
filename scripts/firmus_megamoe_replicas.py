"""Hot-8 placements shared by the kernel gate and benchmark.

Logical experts 0..7 occur once on each GPU; cold experts 8..127 are split
round-robin, 30 per GPU. Each rank owns 38 consecutive physical IDs. Static
dispatch chooses the local copy when present, otherwise the sole cold owner.
This tests physical geometry, not the measured serving frequency distribution:
uniform logical routing has expected local fraction 8/128 + 120/128/4 = 0.296875.
REAL_ROUTING_LAYERS separately embeds measured counts and exact serving maps for
the paired real-frequency benchmarks; the numerical gate keeps its synthetic map.
"""
import hashlib
import json

import torch
import torch.distributed as dist

# BEGIN GENERATED REAL ROUTING
# Run 39051152456 logical_count, summed over passes and all three workers.
# Nearest-rank hot-8 quantiles (94 layers, zero-based layer IDs): median, p90, p10.
# Baseline placement SHA256: c31515f7ceeb5c26fe750bbb4098f15801c7ca8175a49a6d43e6365379cc9aee
# Replica-B placement SHA256: 40f7be52aac506a52dbee7d9c379207864c016dbc912271f63f62246f8a853f4
# Counts are Gumbel sampling weights; without-replacement slot marginals differ.
REAL_ROUTING_LAYERS = {
    43: {
        'selection': 'median',
        'ascending_position': 47,
        'hot8_share': 0.531342174673829,
        'hot8': [11, 118, 34, 3, 64, 98, 45, 15],
        'counts': [1426971, 514966, 6008045, 12095442, 65, 405909, 94848, 7, 1933283, 1444, 97817, 23979019, 172, 9094, 510500, 6554617, 320932, 995417, 22714, 4285811, 440, 575286, 606710, 3, 1136452, 175628, 615221, 90763, 52219, 6083, 1432348, 1335355, 6166348, 4139, 16743089, 378310, 665, 630121, 1632340, 8563, 48696, 55695, 36446, 1075937, 536113, 7277224, 337330, 697, 11274, 107825, 952937, 146218, 5559, 4240016, 0, 336486, 280714, 7001, 78854, 7310, 31239, 48152, 130360, 84600, 11889980, 2894688, 30285, 4166525, 1013916, 879997, 2255068, 62, 2364629, 477680, 1033832, 6590, 54, 768795, 11415, 1820, 865286, 1364879, 4455, 273488, 4603159, 37, 988518, 3128437, 1621561, 662116, 64960, 38171, 1094, 16642, 64542, 1084310, 274, 48377, 11202934, 0, 484, 12250, 54, 1808266, 3686094, 1140360, 2, 536248, 4697297, 2146796, 51887, 10401, 1541756, 846244, 27101, 662, 65405, 122996, 19365529, 23965, 436, 92517, 6516, 495147, 3608070, 4482975, 12773, 49178],
        'baseline_map': [11, 15, 125, 104, 65, 103, 30, 105, 43, 86, 113, 22, 1, 5, 55, 25, 49, 27, 41, 127, 61, 18, 78, 39, 122, 33, 9, 100, 96, 7, 23, 106, 118, 45, 2, 53, 124, 109, 88, 81, 95, 17, 80, 37, 107, 123, 46, 83, 117, 121, 116, 28, 110, 91, 119, 93, 13, 75, 82, 79, 20, 12, 76, 85, 34, 98, 108, 19, 87, 70, 38, 0, 24, 68, 69, 89, 21, 14, 35, 56, 62, 10, 63, 94, 40, 60, 66, 126, 48, 59, 29, 47, 36, 120, 54, 99, 3, 64, 32, 84, 67, 72, 8, 112, 31, 74, 50, 77, 26, 44, 73, 16, 51, 6, 58, 90, 97, 42, 114, 101, 111, 57, 52, 92, 115, 4, 71, 102],
        'replica_map': [11, 118, 34, 3, 64, 98, 45, 15, 32, 67, 65, 109, 112, 81, 43, 86, 80, 89, 107, 1, 5, 25, 51, 10, 63, 94, 127, 91, 114, 126, 101, 13, 75, 82, 120, 4, 71, 102, 11, 118, 34, 3, 64, 98, 45, 15, 2, 53, 87, 8, 88, 31, 95, 17, 113, 77, 44, 73, 55, 83, 62, 27, 90, 28, 40, 42, 18, 111, 57, 29, 33, 7, 23, 106, 54, 99, 11, 118, 34, 3, 64, 98, 45, 15, 108, 19, 104, 72, 38, 30, 105, 74, 50, 26, 22, 123, 46, 16, 49, 6, 58, 110, 61, 66, 93, 78, 39, 52, 79, 47, 36, 100, 96, 85, 11, 118, 34, 3, 64, 98, 45, 15, 84, 125, 124, 70, 103, 0, 24, 68, 69, 37, 21, 14, 35, 56, 117, 121, 116, 41, 97, 60, 119, 48, 59, 122, 9, 92, 115, 20, 12, 76],
    },
    62: {
        'selection': 'upper10',
        'ascending_position': 85,
        'hot8_share': 0.6245242418393845,
        'hot8': [125, 49, 67, 80, 110, 20, 60, 27],
        'counts': [2224, 418030, 422289, 102788, 323185, 1303, 611540, 90, 7347, 353406, 1566476, 580, 929362, 99199, 185556, 3891, 1379953, 1434824, 306605, 525, 12760713, 1480162, 432087, 660264, 8, 553386, 25, 6218792, 1143782, 22019, 7, 28037, 62, 699872, 43071, 182013, 18996, 906725, 7345, 12831, 18141, 4641, 8, 59, 267, 92026, 975178, 2, 6768, 23511875, 23451, 3579, 1928189, 1544810, 430785, 11726, 451681, 108269, 10515, 1832733, 7709748, 1476247, 762543, 1408464, 4165, 2349097, 2218236, 21747499, 75468, 3930357, 1349246, 922957, 127, 4292, 74847, 4, 13679, 61, 234173, 1119038, 18416980, 129, 579, 72167, 182580, 245180, 79949, 3825376, 7903, 2254436, 100576, 3212908, 1, 12403, 1930409, 29122, 4829, 19415, 10933, 16171, 72066, 354658, 161813, 3259565, 1692010, 42494, 1081703, 223, 27924, 2333507, 12889648, 1041964, 75155, 2448603, 3383981, 773, 4281389, 178303, 6083, 713004, 36351, 0, 0, 3828436, 721238, 24986941, 4746, 1624879],
        'baseline_map': [125, 27, 116, 91, 89, 59, 10, 21, 16, 12, 62, 33, 56, 2, 85, 14, 102, 68, 83, 105, 29, 40, 55, 8, 118, 126, 51, 77, 43, 30, 47, 122, 49, 60, 69, 103, 65, 66, 53, 63, 79, 111, 71, 6, 22, 1, 18, 117, 3, 86, 74, 120, 108, 99, 93, 88, 48, 73, 15, 24, 42, 75, 92, 121, 67, 20, 87, 113, 94, 104, 61, 28, 106, 37, 23, 54, 101, 4, 35, 90, 45, 100, 95, 50, 36, 39, 58, 96, 64, 5, 115, 82, 44, 107, 81, 7, 80, 110, 123, 114, 109, 52, 127, 17, 70, 46, 124, 119, 25, 9, 78, 84, 57, 13, 112, 34, 31, 97, 76, 98, 38, 41, 0, 11, 19, 72, 32, 26],
        'replica_map': [125, 49, 67, 80, 110, 20, 60, 27, 116, 113, 65, 94, 127, 61, 70, 71, 119, 6, 54, 101, 85, 117, 90, 86, 83, 95, 108, 40, 98, 38, 48, 15, 115, 11, 19, 107, 77, 24, 125, 49, 67, 80, 110, 20, 60, 27, 69, 91, 89, 52, 21, 16, 79, 12, 37, 25, 22, 1, 14, 35, 3, 45, 100, 31, 97, 99, 93, 8, 118, 73, 0, 72, 32, 43, 42, 30, 125, 49, 67, 80, 110, 20, 60, 27, 123, 103, 109, 59, 53, 63, 106, 46, 62, 33, 2, 4, 18, 102, 13, 112, 34, 105, 50, 36, 55, 88, 96, 41, 51, 75, 47, 92, 121, 122, 125, 49, 67, 80, 110, 20, 60, 27, 87, 114, 66, 104, 10, 17, 28, 111, 124, 23, 56, 9, 78, 84, 57, 68, 74, 120, 29, 76, 39, 58, 126, 64, 5, 82, 44, 81, 7, 26],
    },
    7: {
        'selection': 'lower10',
        'ascending_position': 10,
        'hot8_share': 0.38885017549882583,
        'hot8': [32, 86, 119, 72, 79, 101, 93, 67],
        'counts': [38846, 1941, 188846, 561042, 4809346, 2168, 3197443, 1461419, 382, 982446, 390, 202892, 223, 2998235, 360718, 116276, 1241141, 50303, 440278, 0, 10012, 1171055, 91239, 2798963, 235110, 4275103, 15523, 304941, 853, 5617, 4967789, 307609, 14722748, 1119782, 2441611, 9263, 79161, 796741, 68488, 4325592, 869, 2409641, 692629, 314890, 5571375, 5310451, 68723, 587380, 342187, 288663, 3128577, 5176702, 1240678, 11194, 1884978, 621180, 2019, 50491, 15631, 4339497, 903426, 508324, 80358, 3087146, 5775827, 536525, 2868, 6196149, 363522, 2831179, 6, 48461, 9851732, 675021, 698049, 22813, 5892, 7630, 6447, 8845278, 145914, 17, 1455415, 163040, 5623249, 406, 13733481, 2193475, 52, 6729, 2131122, 5, 1610152, 7026121, 90480, 5700822, 1641, 550643, 902427, 702160, 23407, 8662199, 43883, 32338, 2703127, 467, 696635, 366236, 465, 367344, 1056386, 1833266, 6337, 1738006, 310109, 16327, 48202, 47240, 1083266, 10810274, 77695, 183184, 638710, 4950493, 3539, 32, 675019, 26344],
        'baseline_map': [32, 67, 44, 123, 25, 13, 104, 54, 92, 21, 110, 99, 74, 126, 3, 18, 114, 49, 121, 94, 62, 57, 116, 103, 115, 35, 76, 56, 1, 8, 91, 19, 86, 93, 84, 30, 39, 63, 34, 90, 7, 52, 118, 98, 73, 47, 65, 68, 43, 31, 2, 22, 36, 38, 102, 75, 53, 20, 78, 124, 28, 85, 12, 81, 119, 101, 95, 51, 59, 50, 23, 41, 113, 16, 33, 37, 106, 122, 97, 109, 14, 24, 11, 80, 46, 71, 0, 127, 58, 89, 112, 66, 40, 105, 10, 70, 72, 79, 64, 45, 4, 6, 69, 87, 111, 82, 9, 60, 42, 55, 61, 107, 48, 27, 83, 15, 120, 17, 117, 100, 26, 77, 29, 5, 96, 108, 88, 125],
        'replica_map': [32, 86, 119, 72, 79, 101, 93, 67, 64, 123, 59, 6, 104, 87, 113, 21, 110, 98, 73, 55, 61, 18, 114, 24, 11, 80, 46, 71, 0, 75, 53, 89, 112, 5, 96, 85, 88, 91, 32, 86, 119, 72, 79, 101, 93, 67, 95, 30, 4, 13, 34, 41, 7, 82, 9, 99, 74, 126, 65, 109, 14, 49, 121, 94, 36, 57, 116, 103, 26, 77, 78, 124, 40, 105, 10, 81, 32, 86, 119, 72, 79, 101, 93, 67, 84, 51, 39, 50, 23, 90, 92, 52, 118, 60, 42, 47, 97, 68, 43, 31, 2, 22, 62, 38, 117, 100, 115, 35, 76, 66, 28, 108, 8, 19, 32, 86, 119, 72, 79, 101, 93, 67, 44, 45, 25, 63, 69, 54, 111, 16, 33, 37, 106, 122, 3, 107, 48, 27, 83, 15, 120, 17, 102, 127, 58, 20, 29, 56, 1, 12, 125, 70],
    },
}
# END GENERATED REAL ROUTING


def physical_to_logical():
    return [expert for rank in range(4) for expert in [*range(8), *range(8 + rank, 128, 4)]]


def logical_to_physical(rank):
    mapping = physical_to_logical()
    return [next((p for p in range(rank * 38, (rank + 1) * 38) if mapping[p] == expert), mapping.index(expert)) for expert in range(128)]


def remap(ids, rank):
    table = torch.tensor(logical_to_physical(rank), dtype=torch.int32, device=ids.device)
    return torch.where(ids >= 0, table[ids.clamp_min(0).long()], -1).contiguous()


def replica_weights(rank, device, mapping=None):
    # Identical seeds reproduce the checkpoint's FP8 bytes and FP32 scales on
    # every copy; no rank-specific RNG participates in replica initialization.
    mapping = physical_to_logical() if mapping is None else mapping
    local_experts = len(mapping) // 4
    mapping = mapping[rank * local_experts:(rank + 1) * local_experts]
    tensors = [torch.empty(shape, dtype=dtype, device=device) for shape, dtype in (((local_experts, 3072, 4096), torch.float8_e4m3fn), ((local_experts, 4096, 1536), torch.float8_e4m3fn), ((local_experts, 24, 32), torch.float32), ((local_experts, 32, 12), torch.float32))]
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
