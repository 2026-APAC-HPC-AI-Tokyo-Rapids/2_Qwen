"""Per-step Decode kernel time by category from an nsys SQLite export.

Usage: analyze-decode-step-profile.py <nsys-decode-0.sqlite> <output.json>

A step starts at a TARGET_VERIFY CUDA-graph launch holding 94 MegaMoE calls and
ends at the next one. Kernels up to the last MegaMoE are the 94 layers; the rest
of that graph is the verify tail (target LM head, logits cast); then the eager
kernels (sampling and verification, the eager draft extend forward, batch
bookkeeping) and the draft CUDA graph (two draft decode steps). n = verify rows /
4 tokens per request per rank. "idle:<section>" is the time between the end of the
previous kernel and the start of the next one. For each category: mean us per step
at each n and least-squares a + b*n over all n and over n >= 36 (the FA3 split2
three-round regime, where the serving runs).
"""
import collections
import json
import sqlite3
import statistics
import sys


def layer_category(name, demangled, dense_index):
    # Layer 0 has no residual yet: a plain RMSNorm and a group quant replace the fused kernel.
    for key, label in (("mega_moe_impl", "megamoe"), ("pre_dispatch", "pre_dispatch"), ("add_rmsnorm_group_quant", "norm_quant"), ("FusedAddRMSNorm", "post_attn_norm"), ("RMSNormKernel", "norm_quant"), ("per_token_group_quant", "norm_quant"), ("qknorm_rope", "qknorm_rope"), ("combine_fixed", "fa3_combine"), ("prepare_varlen", "fa3_prepare"), ("router_triton", "router_topk"), ("splitKreduce", "gate"), ("nvjet", "gate")):
        if key in name:
            return label
    if "gemm_1d2d" in name:
        return ("dense_qkv", "dense_o")[dense_index % 2]
    if "FlashAttnFwdSm90" in demangled:
        return "fa3_verify"
    return "layer_other"


def draft_category(name, demangled, grid):
    """Draft forward kernels (eager draft extend and draft graph)."""
    if "FlashAttnFwd" in demangled or "prepare_varlen" in name:
        return "draft_attn"
    if name.startswith("nvjet") and ("512x" in name or "coopA" in name):
        return "draft_lm_head"
    if name.startswith("nvjet") or "cutlass_3x_gemm" in demangled or "splitKreduce" in name:
        return "draft_gemm"
    if "argmax" in name or ("reduce_kernel" in name and "ArgMax" in demangled):
        return "draft_argmax"
    return "draft_small"


def eager_category(name, demangled, grid, in_draft):
    """Eager kernels: sampling and verification, the draft extend forward, bookkeeping."""
    if in_draft:
        return draft_category(name, demangled, grid)
    if "reduce_kernel" in name and "ArgMax" in demangled:
        return "sample_argmax"
    if grid >= 2000 and ("elementwise" in name or "where" in demangled):
        return "sample_penalty"
    if name in ("VerifyTreeGreedy", "fill_bonus_tokens"):
        return "sample_verify"
    return "eager_bookkeeping"


def main(db, output):
    c = sqlite3.connect(db)
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for dev in range(4):
        rows = list(c.execute("select k.start,k.end,s.value,k.gridX,k.graphId,d.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id join StringIds d on k.demangledName=d.id where deviceId=? order by start", (dev,)))
        moe_graphs = collections.Counter(r[4] for r in rows if "mega_moe_impl" in r[2] and r[4] is not None)
        starts = []
        i = 0
        while i < len(rows):
            g = rows[i][4]
            if g is not None and moe_graphs.get(g):
                j = i
                while j < len(rows) and rows[j][4] == g:
                    j += 1
                if sum("mega_moe_impl" in rows[k][2] for k in range(i, j)) == 94:
                    starts.append((i, j))
                i = j
            else:
                i += 1
        for (a, b), (a2, _) in zip(starts, starts[1:]):
            n = next(r[3] for r in rows[a:b] if "add_rmsnorm_group_quant" in r[2]) // 4
            last_moe = max(k for k in range(a, b) if "mega_moe_impl" in rows[k][2])
            # The eager draft extend starts at its first GEMM (EAGLE3 fc) after verification.
            draft_start = next((k for k in range(b, a2) if rows[k][4] is None and rows[k][2].startswith("nvjet")), a2)
            # ... and ends at its argmax (the first draft token); batch bookkeeping follows.
            draft_end = next((k for k in range(draft_start, a2) if "reduce_kernel" in rows[k][2] and "ArgMax" in rows[k][5]), a2)
            tot = collections.defaultdict(float)
            prev = rows[a][0]
            dense_index = 0
            for k in range(a, a2):
                start, end, name, grid, graph, demangled = rows[k]
                if k < b:
                    if k <= last_moe:
                        sec, key = "layers", layer_category(name, demangled, dense_index)
                        dense_index += "gemm_1d2d" in name
                    else:
                        sec = "verify"
                        key = "target_lm_head" if name.startswith("nvjet") else "logits_cast" if grid >= 2000 else "verify_tail_other"
                elif graph is None:
                    sec, key = "eager", eager_category(name, demangled, grid, draft_start <= k <= draft_end)
                else:
                    sec, key = "draft_graph", draft_category(name, demangled, grid)
                    key = "graph_" + key
                tot[key] += (end - start) / 1e3
                tot["idle:" + sec] += max(0, start - prev) / 1e3
                prev = max(prev, end)
            tot["period"] = (rows[a2][0] - rows[a][0]) / 1e3
            for key, v in tot.items():
                per[n][key].append(v)
    keys = sorted({k for d in per.values() for k in d})
    ns = sorted(per)
    counts = {n: len(per[n]["period"]) for n in ns}
    out = {"steps_by_n": counts}
    print("n:", ns, "steps:", [counts[n] for n in ns])

    def fit(key, lo):
        xs, ys = [], []
        for n in ns:
            if n < lo:
                continue
            v = per[n].get(key, [])
            v = v + [0.0] * (counts[n] - len(v))
            xs += [n] * len(v)
            ys += v
        mx, my = statistics.mean(xs), statistics.mean(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        return slope, my - slope * mx

    for key in keys:
        at = {n: sum(per[n].get(key, [])) / counts[n] for n in ns}
        slope, intercept = fit(key, 0)
        slope36, intercept36 = fit(key, 36)
        out[key] = {"slope_us_per_request": slope, "intercept_us": intercept, "slope_us_per_request_n36": slope36, "intercept_us_n36": intercept36, "mean_us_by_n": at}
        print(f"{key:26s} b {slope:7.1f} a {intercept:8.1f} | n>=36 b {slope36:7.1f} a {intercept36:8.1f} | n40 {at.get(40, 0):8.1f} n42 {at.get(42, 0):8.1f} n44 {at.get(44, 0):8.1f}")
    with open(output, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
