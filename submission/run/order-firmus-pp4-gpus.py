"""Print the CUDA_VISIBLE_DEVICES order of one PP4 Prefill worker: the slowest GPU hosts the last stage.

SGLang puts PP stage s on visible device s. Every GPU of a PP4 pipeline sits at
the 700 W cap, so its period is set by the slowest GPU that runs a full 24-layer
stage. The last stage (22 layers + LM head + EAGLE draft extend) is the
lightest: in run 40883389891 its GPU ran 1-2% above the clock the same GPU
reaches under symmetric EP4 load, the others 2-5% below. Moving the last stage
onto the worker's slowest GPU (configs/firmus-gpu-sm-clock.json, EP4 means)
leaves the 24-layer stages on the faster three.

stdin: `nvidia-smi --query-gpu=index,uuid --format=csv,noheader`.
argv: the clock table and the current CUDA_VISIBLE_DEVICES (integer indices).
Output: the same entries, with the slowest moved to the end if it is at least
MARGIN_MHZ slower than the current last GPU; otherwise, or if any GPU is not
in the table, the list unchanged.
"""
import json
import sys
from pathlib import Path

MARGIN_MHZ = 5.0


def order(table, smi, visible):
    uuid = {}
    for line in smi.strip().splitlines():
        index, value = (x.strip() for x in line.split(","))
        uuid[index] = value
    devices = visible.split(",")
    clock = {d: table["gpus"].get(uuid.get(d), {}).get("sm_mhz") for d in devices}
    if len(devices) != 4 or None in clock.values():
        return visible
    slowest = min(devices, key=lambda d: clock[d])
    if clock[slowest] + MARGIN_MHZ > clock[devices[-1]]:
        return visible
    return ",".join([d for d in devices if d != slowest] + [slowest])


if __name__ == "__main__":
    print(order(json.loads(Path(sys.argv[1]).read_text()), sys.stdin.read(), sys.argv[2]))
