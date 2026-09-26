"""Validate and summarize SGLang's complete-model one-batch measurements."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def summarize(root):
    plans = list(root.rglob('forward-plan.json'))
    if len(plans) != 1:
        raise ValueError('exactly one forward plan required')
    plan = json.loads(plans[0].read_text())
    rows = []
    for worker in plan['workers']:
        files = list(root.rglob(worker['name'] + '.jsonl'))
        if len(files) != 1:
            raise ValueError(f'missing or duplicate worker {worker["name"]}')
        measurements = [json.loads(line) for line in files[0].read_text().splitlines()]
        expected = {(bs, length) for bs in worker.get('batch_sizes', plan['batch_sizes']) for length in plan['input_lengths'] if bs * (length + plan['output_length']) <= 131072}
        grouped = {}
        for measurement in measurements:
            key = (measurement['batch_size'], measurement['input_len'])
            if measurement['run_name'] != worker['name'] or measurement['output_len'] != plan['output_length'] or key not in expected:
                raise ValueError('unexpected workload')
            for field in ('prefill_latency', 'median_decode_latency'):
                value = measurement[field]
                if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                    raise ValueError('finite positive seconds required')
            grouped.setdefault(key, []).append(measurement)
        if set(grouped) != expected or any(len(v) != plan['repetitions'] for v in grouped.values()):
            raise ValueError(f'incomplete measurement coverage for {worker["name"]}')
        for (bs, length), samples in sorted(grouped.items()):
            rows.append({**worker, 'batch_size': bs, 'input_len': length, 'output_len': plan['output_length'], 'prefill_ms_samples': [m['prefill_latency'] * 1000 for m in samples], 'decode_ms_samples': [m['median_decode_latency'] * 1000 for m in samples], 'prefill_ms': statistics.median(m['prefill_latency'] * 1000 for m in samples), 'decode_ms': statistics.median(m['median_decode_latency'] * 1000 for m in samples), 'raw_sha256': hashlib.sha256(files[0].read_bytes()).hexdigest()})
    return {'schema_version': 'qwen-forward-summary/v1', 'plan': plan, 'measurements': rows, 'timing_boundary': 'SGLang one_batch ModelRunner extend/decode with synchronization; includes sampling, excludes serving queues and NIXL', 'official_result': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'workers': len(result['plan']['workers']), 'measurement_cells': len(result['measurements'])}))


if __name__ == '__main__':
    main()
