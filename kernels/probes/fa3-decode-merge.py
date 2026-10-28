"""One-forward Decode merge diagnostic; analysis imports on CPU without torch."""
import argparse
import ctypes
import json
import runpy
import time
from pathlib import Path

import numpy as np

RECORD_FIELDS = ("cta_x", "cta_y", "cta_z", "m", "head", "batch", "raw_split", "count", "split", "arrival_old", "last", "valid")
ABI_FIELDS = ("counters", "partial", "partial_lse", "output", "lse", "fp8", "scales", "scale_stride", "total_q", "epilogue_partial", "epilogue_partial_lse", "epilogue_output", "epilogue_lse", "epilogue_scales", "epilogue_scale_stride", "partial_row_stride", "partial_dim_stride", "partial_head_stride", "partial_batch_stride", "partial_split_stride", "lse_row_stride", "lse_head_stride", "lse_batch_stride", "lse_split_stride", "epilogue_total_q", "output_row_stride", "output_head_stride", "cu_q")
STAGES = ("records", "workspace", "merge_snapshot", "merged_fp32", "fp8_bytes_scales")


def fp32_merge(partial, plse):
    """Head-major partials and natural-log LSE -> token-major FP32 O."""
    maximum = np.max(plse, axis=0)
    weights = np.exp(plse - maximum[None]).astype(np.float32)
    weights /= np.sum(weights, axis=0, dtype=np.float32)[None]
    out = np.zeros_like(partial[0])
    for split in range(len(partial)):
        out += weights[split, ..., None] * partial[split]
    return out.transpose(1, 0, 2)


def fp8_quantize(merged):
    """E4M3FN round-to-nearest/even, no BF16 staging or extra rounding point."""
    raw = np.arange(127, dtype=np.uint8)
    exponent, fraction = (raw >> 3).astype(np.int32), (raw & 7).astype(np.float32)
    levels = np.where(exponent == 0, fraction * 2.0**-9, (1 + fraction / 8) * np.exp2(exponent - 7)).astype(np.float32)
    amax = np.maximum(np.max(np.abs(merged), axis=-1), np.float32(1e-10))
    gain = np.float32(448) / amax
    values = np.minimum(np.abs(merged * gain[..., None]), np.float32(448))
    high = np.searchsorted(levels, values).clip(1, 126)
    low = high - 1
    dl, dh = values - levels[low], levels[high] - values
    choose_high = (dh < dl) | ((dh == dl) & ((high & 1) == 0))
    codes = np.where(choose_high, high, low).astype(np.uint8)
    codes |= np.signbit(merged).astype(np.uint8) * 128
    return codes, (amax * np.float32(1 / 448)).T.copy(), amax.T.copy()


def analyze(control, merge):
    issues = {stage: [] for stage in STAGES}
    comparisons = []

    def check(stage, label, actual, expected, exact=False, coordinates=None):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if exact:
            same = actual == expected
        else:
            same = np.isclose(actual, expected, rtol=2e-6, atol=2e-7, equal_nan=False)
        bad = np.argwhere(~same)
        item = {"check": label, "differing": len(bad)}
        if len(bad):
            item["first"] = [{"index": i.tolist(), "actual": str(actual[tuple(i)]), "expected": str(expected[tuple(i)])} for i in bad[:5]]
            if coordinates:
                item["coordinates"] = coordinates
            issues[stage].append(item)
        comparisons.append(item)

    batch, heads, splits, _ = merge["records"].shape
    for name, data in (("control", control), ("merge", merge)):
        records = data["records"]
        b, h, s = np.indices((batch, heads, splits))
        expected = np.stack((np.zeros_like(b), h, b, (splits << 16) + s, np.full_like(b, splits), s), axis=-1)
        check("records", name + ".coordinates", records[..., 3:9], expected, exact=True, coordinates="batch, KV head, split, field[m,head,batch,raw,count,index]")
        check("records", name + ".valid", records[..., 11], np.ones_like(b), exact=True)
        check("records", name + ".cta_nonnegative", records[..., :3] >= 0, np.ones((*b.shape, 3), dtype=bool), exact=True)
        check("records", name + ".abi", data["abi"], np.broadcast_to(data["expected_abi"], data["abi"].shape), exact=True, coordinates="batch, KV head, split, ABI_FIELDS")
        check("records", name + ".prepare_counts", data["prepare"][:batch], np.full(batch, splits), exact=True)
        check("records", name + ".prepare_m", data["prepare"][batch:2 * batch], np.ones(batch), exact=True)
        check("records", name + ".prepare_order", np.sort(data["prepare"][2 * batch:3 * batch]), np.arange(batch), exact=True)
        check("records", name + ".prepare_counters", data["prepare"][4 * batch:], np.zeros(1 + batch * heads), exact=True)
        check("records", name + ".metadata_preserved", data["metadata"][:4 * batch], data["prepare"][:4 * batch], exact=True)
        check("records", name + ".final_counters", data["metadata"][4 * batch:], np.zeros(1 + batch * heads), exact=True)
    check("records", "prepare_metadata", merge["prepare"], control["prepare"], exact=True)
    old = merge["records"][..., 9]
    last = merge["records"][..., 10]
    check("records", "arrival_permutation", np.sort(old, axis=-1), np.broadcast_to(np.arange(splits), old.shape), exact=True)
    check("records", "elected_last", last, (old == splits - 1).astype(np.int32), exact=True)
    check("workspace", "partial_O", merge["partial"], control["partial"], coordinates="split, Q head, token, dim")
    check("workspace", "partial_LSE", merge["partial_lse"], control["partial_lse"], coordinates="split, Q head, token")
    check("merge_snapshot", "loaded_O_vs_own_workspace", merge["loaded"], merge["partial"], coordinates="split, Q head, token, dim")
    for pass_id in range(2):
        check("merge_snapshot", "loaded_LSE_pass" + str(pass_id), merge["loaded_lse"][pass_id], merge["partial_lse"], coordinates="split, Q head, token")
    for name, data in (("control", control), ("merge", merge)):
        expected = fp32_merge(data["partial"], data["partial_lse"])
        check("merged_fp32", name + ".host_merge", data["merged"], expected, coordinates="token, Q head, dim")
    check("merged_fp32", "merged_O", merge["merged"], control["merged"], coordinates="token, Q head, dim")
    for name, data in (("control", control), ("merge", merge)):
        _, scales, amax = fp8_quantize(data["merged"])
        check("fp8_bytes_scales", name + ".amax", data["amax"], amax, coordinates="Q head, token")
        check("fp8_bytes_scales", name + ".scales_from_fp32", data["quant_scales"], scales)
        check("fp8_bytes_scales", name + ".final_codes_vs_kernel", data["output"], data["codes"], exact=True, coordinates="token, Q head, dim")
        check("fp8_bytes_scales", name + ".final_scales_vs_kernel", data["scales"], data["quant_scales"], exact=True)
    check("fp8_bytes_scales", "amax", merge["amax"], control["amax"])
    check("fp8_bytes_scales", "scales", merge["scales"], control["scales"])
    check("fp8_bytes_scales", "FP8_bytes", merge["output"], control["output"], exact=True, coordinates="token, Q head, dim")
    # CPU/GPU division and exp implementations may straddle FP8 midpoints.
    # Report ties separately; only differences away from a midpoint are errors.
    for name, data in (("control", control), ("merge", merge)):
        codes, _, _ = fp8_quantize(data["merged"])
        actual = data["codes"]
        mismatch = actual != codes
        mag_actual, mag_expected = actual & 127, codes & 127
        adjacent = np.abs(mag_actual.astype(int) - mag_expected.astype(int)) == 1
        def decode(c):
            e, f = (c >> 3).astype(int), (c & 7).astype(np.float32)
            return np.where(e == 0, f * 2.0**-9, (1 + f / 8) * np.exp2(e - 7))
        midpoint = (decode(mag_actual) + decode(mag_expected)) * .5
        scaled = np.abs(data["merged"] * (np.float32(448) / np.maximum(np.max(np.abs(data["merged"]), axis=-1), np.float32(1e-10)))[..., None])
        ties = mismatch & adjacent & ((actual >> 7) == (codes >> 7)) & np.isclose(scaled, midpoint, rtol=2e-6, atol=0)
        check("fp8_bytes_scales", name + ".host_quant_non_ties", mismatch & ~ties, np.zeros_like(mismatch), exact=True)
        comparisons.append({"check": name + ".host_quant_ties", "differing": int(ties.sum())})
    first = next((stage for stage in STAGES if issues[stage]), None)
    return {"first_divergence": first, "issues": issues, "comparisons": comparisons, "record_fields": RECORD_FIELDS, "abi_fields": ABI_FIELDS, "tolerances": {"rtol": 2e-6, "atol": 2e-7, "bytes": "exact; CPU reference permits adjacent midpoint ties"}}


def run_gpu(directory):
    import torch

    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0)
    helper = runpy.run_path(str(Path(__file__).with_name("check-firmus-decode-kernels.py")))
    common, lengths = helper["verify_inputs"](40, 93141, False)
    columns = common["page_table"].shape[1]
    lib = ctypes.CDLL(str(Path(__file__).with_name("fa3-decode-merge.so")))
    pointers = ctypes.POINTER(ctypes.c_void_p)
    lib.decode_probe_prepare.argtypes = [pointers, ctypes.c_int, ctypes.c_void_p]
    lib.decode_probe_run.argtypes = [ctypes.c_int, pointers, ctypes.c_int, ctypes.c_void_p]
    lib.decode_probe_prepare.restype = lib.decode_probe_run.restype = ctypes.c_int
    stream = torch.cuda.current_stream().cuda_stream
    dumps = []
    for merge in (False, True):
        def floats(shape):
            return torch.full(shape, float("nan"), device="cuda", dtype=torch.float32)
        def integers(shape, dtype=torch.int32):
            return torch.full(shape, -1, device="cuda", dtype=dtype)
        output = torch.full((160, 64, 128), 0xA5, device="cuda", dtype=torch.uint8)
        lse, partial, plse = floats((64, 160)), floats((3, 64, 160, 128)), floats((3, 64, 160))
        metadata, scales = integers((321,)), floats((64, 160))
        records, abi = integers((40, 4, 3, 12)), integers((40, 4, 3, 28), torch.int64)
        loaded, loaded_lse = floats(partial.shape), floats((2, 3, 64, 160))
        merged, amax = floats(output.shape), floats(scales.shape)
        codes, quant_scales = torch.full_like(output, 0xA5), floats(scales.shape)
        args = [common[k] for k in ("q", "k_cache", "v_cache", "page_table", "cache_seqlens", "cu_seqlens_q")]
        args += [output, lse, partial, plse, metadata, scales, records, abi, loaded, loaded_lse, merged, amax, codes, quant_scales]
        ptr = (ctypes.c_void_p * len(args))(*(t.data_ptr() for t in args))
        assert lib.decode_probe_prepare(ptr, columns, stream) == 0
        prepared = metadata.cpu().numpy().copy()
        assert lib.decode_probe_run(int(merge), ptr, columns, stream) == 0
        torch.cuda.synchronize()
        names = ("output", "lse", "partial", "partial_lse", "metadata", "scales", "records", "abi", "loaded", "loaded_lse", "merged", "amax", "codes", "quant_scales")
        data = {name: t.cpu().numpy().copy() for name, t in zip(names, args[6:])}
        data["prepare"] = prepared
        ptrs = [t.data_ptr() for t in args]
        data["expected_abi"] = np.array([ptrs[10] + 161 * 4, ptrs[8], ptrs[9], ptrs[6], ptrs[7], ptrs[6], ptrs[11], 160, 160, ptrs[8], ptrs[9], ptrs[6], ptrs[7], ptrs[11], 160, 128, 1, 160 * 128, 0, 64 * 160 * 128, 1, 160, 0, 64 * 160, 160, 8192, 128, ptrs[5]], dtype=np.int64)
        dumps.append(data)
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory / "fa3-decode-merge-dumps.npz", **{prefix + "_" + key: value for prefix, dump in zip(("control", "merge"), dumps) for key, value in dump.items()})
    result = analyze(*dumps)
    result.update({"batch": 40, "seed": 93141, "lengths": lengths, "device": torch.cuda.get_device_name(), "torch": torch.__version__, "provenance": json.loads(Path(__file__).with_name("provenance.json").read_text()), "scope": "one forward per candidate, no graph replay, no performance measurement"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", type=Path, help="Reanalyze an NPZ dump on CPU")
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    args = parser.parse_args()
    started = time.monotonic()
    if args.analyze:
        with np.load(args.analyze) as dump:
            data = [{key[len(prefix) + 1:]: dump[key] for key in dump.files if key.startswith(prefix + "_")} for prefix in ("control", "merge")]
        result = analyze(*data)
    else:
        result = run_gpu(args.results)
    result["elapsed_seconds"] = time.monotonic() - started
    args.results.mkdir(exist_ok=True)
    (args.results / "fa3-decode-merge.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["first_divergence=" + str(result["first_divergence"])]
    for stage in STAGES:
        for issue in result["issues"][stage]:
            lines.append(stage + ": " + json.dumps(issue))
    text = "\n".join(lines) + "\n"
    (args.results / "fa3-decode-merge.log").write_text(text)
    print(text, end="", flush=True)
    # A divergent result is a successful diagnostic, not a numerical gate PASS.
    print("PASS kernel=fa3_decode_merge diagnostic_completed=true first_divergence=" + str(result["first_divergence"]), flush=True)


if __name__ == "__main__":
    main()
