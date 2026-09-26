"""Compare generic and measured forward predictions on an independent allocation."""
import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

from aisimulate_core.sdk import RustForwardPassPerfModel
from aisimulate_core.sdk.errors import PerfDataNotAvailableError
from calibrate import VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fpm', type=Path)
    parser.add_argument('validation_summary', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.fpm.with_suffix('.metadata.json').read_text())
    validation_sha = hashlib.sha256(args.validation_summary.read_bytes()).hexdigest()
    if metadata['summary_sha256'] == validation_sha:
        raise ValueError('validation summary must differ from training summary')
    points = json.loads(args.validation_summary.read_text())['measurements']
    os.environ['AIC_ALLOW_UNLISTED_VERSIONS'] = '1'
    rows = []
    for tp, ep in sorted({(p['tp'], p['ep']) for p in points}):
        for role in ('prefill', 'decode'):
            for method in ('generic', 'measured'):
                config = {'model': str(Path('research/simulation/model').resolve()), 'system': 'h100_sxm', 'backend': 'sglang', 'backend_version': VERSION if method == 'measured' else '0.5.14', 'worker_type': role, 'tp': tp, 'pp': 1, 'attention_dp': 1, 'moe_tp_size': tp // ep, 'moe_ep_size': ep, 'gemm_quant_mode': 'fp8_block', 'moe_quant_mode': 'fp8_block', 'fmha_quant_mode': 'bfloat16', 'kvcache_quant_mode': 'bfloat16', 'estimation_mode': 'fpm_interpolation' if method == 'measured' else 'op_level', 'fallback_policy': 'deny'}
                if method == 'measured':
                    config['estimator_config'] = {'fpm_interpolation': {'fpm_parquet_path': str(args.fpm.resolve())}}
                model = RustForwardPassPerfModel.best_available(config)
                try:
                    for point in points:
                        if (point['tp'], point['ep']) != (tp, ep):
                            continue
                        b, n = point['batch_size'], point['input_len']
                        query = {'scheduled_requests': {'num_prefill_requests': b if role == 'prefill' else 0, 'sum_prefill_tokens': b * n if role == 'prefill' else 0, 'sum_prefill_kv_tokens': 0, 'num_decode_requests': b if role == 'decode' else 0, 'sum_decode_kv_tokens': b * (n + 16) if role == 'decode' else 0}}
                        row = {'method': method, 'worker': point['name'], 'tp': tp, 'ep': ep, 'role': role, 'batch_size': b, 'input_len': n, 'actual_ms': point[role + '_ms']}
                        try:
                            predicted = model.estimate_forward_pass_time_ms(query)
                            row.update(predicted_ms=predicted, absolute_percentage_error=abs(predicted / row['actual_ms'] - 1) * 100)
                        except PerfDataNotAvailableError as error:
                            row['unsupported'] = str(error)
                        rows.append(row)
                finally:
                    model.close()
    metrics = {}
    for method in ('generic', 'measured'):
        errors = [r['absolute_percentage_error'] for r in rows if r['method'] == method and 'absolute_percentage_error' in r]
        metrics[method] = {'covered': len(errors), 'median_ape': statistics.median(errors) if errors else None, 'max_ape': max(errors) if errors else None}
    result = {'official_result': False, 'scope': 'Independent allocation ModelRunner timing; not official AIPerf accuracy. Generic release slot differs from deployed source commit.', 'training_summary_sha256': metadata['summary_sha256'], 'validation_summary_sha256': validation_sha, 'metrics': metrics, 'measurements': rows}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
