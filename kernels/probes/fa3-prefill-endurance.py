"""Endurance check of the Prefill FA3 rows: thousands of consecutive eager calls over varlen shapes.

Rows come from fa3-prefill-pipe.so (index 0 is the pinned n192-direct-o-varlen). For each packed-length
set every row is called repeatedly with a small unrelated kernel between calls (as in serving, the
tile-counter memset and the PDL launch follow other work); the first and the last output of every row
must equal the pin's bytes. A hang shows as the last ENDURANCE line before the trial timeout.
"""
import argparse
import ctypes
import json
import math
import time
from pathlib import Path

import torch

SCALE = 1 / math.sqrt(128)
# Serving packs 1..4 requests of 3584..4096 tokens; L % 8 == 1 leaves a one-token last M-tile.
SHAPES = ((3986, 4027), (3986, 3986), (4000, 4064), (4096, 4096), (3585, 4089), (4001,), (3600, 3700, 3800), (3585, 3593, 4001, 4089), (3584, 4096))
CALLS = {"pin": 60, "pipe": 250}
OTHER_CALLS = 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--names", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    begin = time.monotonic()
    torch.cuda.set_device(0)
    names = args.names.split(",")
    lib = ctypes.CDLL(str(args.library))
    lib.pf_run.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p]
    assert lib.pf_count() == len(names), names
    noise = torch.zeros(1 << 20, device="cuda")
    result = {"shapes": {}, "calls": {name: 0 for name in names}}
    for shape in SHAPES:
        generator = torch.Generator(device="cuda").manual_seed(sum(shape))
        rows = sum(shape)
        q = (torch.randn(rows, 64, 128, device="cuda", generator=generator) * 2).to(torch.float8_e4m3fn)
        k = torch.randn(rows, 4, 128, device="cuda", generator=generator).to(torch.float8_e4m3fn)
        v = torch.randn(rows, 4, 128, device="cuda", generator=generator).to(torch.float8_e4m3fn)
        cu = torch.tensor([0, *torch.tensor(shape).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
        padded = (rows + 3) // 4 * 4
        record = {}
        reference = None
        for index, name in enumerate(names):
            print(f"ENDURANCE start shape={'+'.join(map(str, shape))} row={name}", flush=True)
            out = torch.zeros(rows, 64, 128, dtype=torch.float8_e4m3fn, device="cuda")
            scales = torch.zeros(64, padded, dtype=torch.float32, device="cuda")
            metadata = torch.zeros(17, dtype=torch.int32, device="cuda")
            stream = torch.cuda.current_stream().cuda_stream
            snapshots = []
            calls = CALLS.get(name, OTHER_CALLS)
            for call in range(calls):
                noise.add_(1.0)
                status = lib.pf_run(index, rows, len(shape), max(shape), q.data_ptr(), k.data_ptr(), v.data_ptr(), cu.data_ptr(), out.data_ptr(), scales.data_ptr(), scales.stride(0), metadata.data_ptr(), SCALE, stream)
                assert status == 0, (name, shape, call, status)
                if call in (0, calls - 1):
                    snapshots.append((out.view(torch.uint8).clone(), scales.clone()))
            torch.cuda.synchronize()
            result["calls"][name] += calls
            if reference is None:
                reference = snapshots[0]
            equal = [bool(torch.equal(o, reference[0]) and torch.equal(s, reference[1])) for o, s in snapshots]
            record[name] = {"calls": calls, "first_last_equal_pin": equal}
            print(f"ENDURANCE done shape={'+'.join(map(str, shape))} row={name} calls={calls} equal_pin={equal}", flush=True)
        result["shapes"]["+".join(map(str, shape))] = record
    result["elapsed_seconds"] = time.monotonic() - begin
    result["all_equal_pin"] = all(all(r["first_last_equal_pin"]) for shape in result["shapes"].values() for r in shape.values())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("ENDURANCE-RESULT " + json.dumps({"calls": result["calls"], "all_equal_pin": result["all_equal_pin"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
