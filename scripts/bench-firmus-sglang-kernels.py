"""Route-check stub for the SGLang JIT-kernel bench contract (experiment branch only).

python bench-firmus-sglang-kernels.py --output <jsonl> --variant <image|candidate>
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from sglang.kernels.ops.moe.decode_silu_quant import decode_silu_quant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    torch.manual_seed(1)
    with args.output.open("w") as output:
        for rows in (128, 4096):
            x = torch.randn(rows, 768, device="cuda", dtype=torch.bfloat16) * 4
            actual, scales = decode_silu_quant(x)
            torch.cuda.synchronize()
            digest = hashlib.sha256(actual.view(torch.uint8).cpu().numpy().tobytes() + scales.cpu().numpy().tobytes()).hexdigest()
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(100)]
            for start, end in events:
                start.record()
                decode_silu_quant(x)
                end.record()
            torch.cuda.synchronize()
            ordered = sorted(start.elapsed_time(end) * 1000 for start, end in events)
            row = {"case": f"decode-silu-quant-{rows}", "variant": args.variant, "median_us": ordered[50], "p10_us": ordered[10], "p90_us": ordered[90], "output_sha256": digest, "sm_clock_mhz": torch.cuda.clock_rate()}
            print(json.dumps(row), flush=True)
            output.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
