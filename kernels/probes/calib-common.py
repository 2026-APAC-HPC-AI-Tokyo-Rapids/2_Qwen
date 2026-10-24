"""Summary-only reporting shared by the two bounded calibration probes."""

import json
import math
import statistics


def distribution(values):
    assert values and all(math.isfinite(v) and v > 0 for v in values)
    values = sorted(values)
    return statistics.median(values), values[len(values) // 10], values[len(values) * 9 // 10]


def report(shape, name, blocks, stock_blocks, numeric):
    med, p10, p90 = distribution([v for block in blocks for v in block])
    stock = distribution([v for block in stock_blocks for v in block])[0]
    ratios = [distribution(b)[0] / distribution(s)[0] for b, s in zip(blocks, stock_blocks, strict=True)]
    print(json.dumps({"shape": shape, "candidate": name, "numeric": numeric, "blocks": blocks, "stock_blocks": stock_blocks}), flush=True)
    print(f"CALIB | {shape} | {name} | {med:.3f} | {p10:.3f} | {p90:.3f} | {med / stock:.5f} | {ratios[0]:.5f} | {ratios[1]:.5f} |", flush=True)


def device_info(torch):
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability() == (9, 0)
    p = torch.cuda.get_device_properties(0)
    # The image's dense override is guarded by num_sms == 132. Fail rather
    # than accidentally label the unchanged stock kernel as a tuned candidate.
    assert p.multi_processor_count == 132, p
    print(json.dumps({"device": p.name, "sm_count": p.multi_processor_count, "torch": torch.__version__, "cuda": torch.version.cuda, "uuid": str(p.uuid)}), flush=True)
