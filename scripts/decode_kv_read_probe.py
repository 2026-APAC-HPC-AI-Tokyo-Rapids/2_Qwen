"""Diagnostic for the Decode FA3 selection: read the verify KV cache at 128, 256 and 512 B per token row.

The FA3 verify tile reads one KV head, i.e. 128 B of every 512 B token row
([token][4 heads][128] FP8 in 64-token pages), and the kernel stays near 2.7 TB/s
however many CTAs run. This read-only probe walks the same pages through the same
page table with 1, 2 or 4 heads per program (128/256/512 contiguous bytes per
row) and the same total bytes and program count, so the selection run records
whether the read granularity alone moves the bandwidth. Nothing is selected from it.
"""
import triton
import triton.language as tl

GRANULARITIES = (1, 2, 4)
REPEATS = 10


@triton.jit
def _read_rows(K, V, PAGES, LENGTHS, OUT, columns, HEADS: tl.constexpr, CHUNKS: tl.constexpr):
    # One program: request b, heads [g * HEADS, (g + 1) * HEADS), the c-th of CHUNKS token ranges.
    pid = tl.program_id(0)
    b = pid // ((4 // HEADS) * CHUNKS)
    g = (pid // CHUNKS) % (4 // HEADS)
    c = pid % CHUNKS
    length = tl.load(LENGTHS + b)
    pages = (length + 63) // 64
    first = c * pages // CHUNKS
    last = (c + 1) * pages // CHUNKS
    rows = tl.arange(0, 64)
    cols = tl.max_contiguous(tl.multiple_of(g * HEADS * 32 + tl.arange(0, HEADS * 32), 32), HEADS * 32)
    acc = tl.zeros((64, HEADS * 32), dtype=tl.int32)
    for p in range(first, last):
        page = tl.load(PAGES + b * columns + p).to(tl.int64)
        offsets = tl.multiple_of(page * 8192, 8192) + rows[:, None] * 128 + cols[None, :]
        acc ^= tl.load(K + offsets) + tl.load(V + offsets)
    tl.store(OUT + pid, tl.sum(acc, axis=None))


def measure(torch, common_sets):
    """TB/s per granularity over the four rotating KV sets of one batch (KV bytes / median launch time)."""
    lengths = common_sets[0]["cache_seqlens"]
    batch = lengths.numel()
    columns = common_sets[0]["page_table"].shape[1]
    kv_bytes = 2 * 512 * 64 * int(((lengths + 63) // 64).sum().item())
    result = {"kv_bytes": kv_bytes}
    for heads in GRANULARITIES:
        chunks = 2 * heads
        grid = (batch * (4 // heads) * chunks,)
        out = torch.empty(grid[0], dtype=torch.int32, device="cuda")
        args = [(c["k_cache"].view(torch.int32), c["v_cache"].view(torch.int32), c["page_table"], c["cache_seqlens"]) for c in common_sets]
        for k, v, pages, seqlens in args:
            _read_rows[grid](k, v, pages, seqlens, out, columns, HEADS=heads, CHUNKS=chunks, num_warps=4)
        torch.cuda.synchronize()
        times = []
        for _ in range(REPEATS):
            for k, v, pages, seqlens in args:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                _read_rows[grid](k, v, pages, seqlens, out, columns, HEADS=heads, CHUNKS=chunks, num_warps=4)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) * 1e-3)
        times.sort()
        result[f"{128 * heads}B"] = {"programs": grid[0], "median_us": times[len(times) // 2] * 1e6, "tb_per_s": kv_bytes / times[len(times) // 2] / 1e12}
    return result
