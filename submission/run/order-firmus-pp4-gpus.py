"""Print the CUDA_VISIBLE_DEVICES order of one PP4 Prefill worker, with its slowest GPU running the last stage.

SGLang runs pipeline stage s on visible device s. Every GPU of a PP4 pipeline runs at the 700 W cap, so the pipeline
runs at the pace of its slowest GPU that holds a full 24-layer stage. The last stage (22 layers, the LM head and the
EAGLE draft extend) is the lightest, so it goes on the worker's slowest GPU, using the measured SM clocks in
config/firmus-gpu-sm-clock.json. The order changes only when that GPU is at least MARGIN_MHZ slower than the
current last GPU, and only for GPUs the table measured.

usage: nvidia-smi --query-gpu=index,uuid --format=csv,noheader | order-firmus-pp4-gpus.py CLOCK_TABLE CUDA_VISIBLE_DEVICES
"""
import json
import sys
from pathlib import Path

MARGIN_MHZ = 5.0


def order(table, smi, visible):
    uuid = dict((field.strip() for field in line.split(",")) for line in smi.strip().splitlines())
    devices = visible.split(",")
    if not all(uuid[device] in table["gpus"] for device in devices):
        return visible
    clock = {device: table["gpus"][uuid[device]]["sm_mhz"] for device in devices}
    slowest = min(devices, key=clock.get)
    if clock[slowest] + MARGIN_MHZ > clock[devices[-1]]:
        return visible
    return ",".join([device for device in devices if device != slowest] + [slowest])


if __name__ == "__main__":
    print(order(json.loads(Path(sys.argv[1]).read_text()), sys.stdin.read(), sys.argv[2]))
