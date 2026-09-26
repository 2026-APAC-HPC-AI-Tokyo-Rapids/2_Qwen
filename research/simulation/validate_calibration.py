"""Evaluate held-out shapes through AISimulate's public performance model API."""
import argparse
import json
import os
import statistics
from pathlib import Path

from aisimulate_core.sdk import RustForwardPassPerfModel
from aisimulate_core.sdk.errors import PerfDataNotAvailableError
from calibrate import VERSION, export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    parquet = export(args.summary, args.output / 'training', held_out_length=4000)
    # Real measured source commit, absent from the generic database's release slots.
    os.environ['AIC_ALLOW_UNLISTED_VERSIONS'] = '1'
    summary = json.loads(args.summary.read_text())
    rows = []
    for tp, ep in sorted({(p['tp'], p['ep']) for p in summary['measurements']}):
        for role in ('prefill', 'decode'):
            config = {'model': str(Path('research/simulation/model').resolve()), 'system': 'h100_sxm', 'backend': 'sglang', 'backend_version': VERSION, 'worker_type': role, 'tp': tp, 'pp': 1, 'attention_dp': 1, 'moe_tp_size': tp // ep, 'moe_ep_size': ep, 'fmha_quant_mode': 'bfloat16', 'kvcache_quant_mode': 'bfloat16', 'estimation_mode': 'fpm_interpolation', 'fallback_policy': 'deny', 'estimator_config': {'fpm_interpolation': {'fpm_parquet_path': str(parquet.resolve())}}}
            model = RustForwardPassPerfModel.best_available(config)
            try:
                for point in summary['measurements']:
                    if (point['tp'], point['ep']) != (tp, ep) or point['input_len'] != 4000:
                        continue
                    batch = point['batch_size']
                    metrics = {'scheduled_requests': {'num_prefill_requests': batch if role == 'prefill' else 0, 'sum_prefill_tokens': batch * 4000 if role == 'prefill' else 0, 'sum_prefill_kv_tokens': 0, 'num_decode_requests': batch if role == 'decode' else 0, 'sum_decode_kv_tokens': batch * 4016 if role == 'decode' else 0}}
                    try:
                        predicted = model.estimate_forward_pass_time_ms(metrics)
                    except PerfDataNotAvailableError as error:
                        rows.append({'worker': point['name'], 'role': role, 'tp': tp, 'ep': ep, 'batch_size': batch, 'input_len': 4000, 'unsupported': str(error)})
                        continue
                    actual = point[role + '_ms']
                    rows.append({'worker': point['name'], 'role': role, 'tp': tp, 'ep': ep, 'batch_size': batch, 'input_len': 4000, 'predicted_ms': predicted, 'measured_ms': actual, 'absolute_percentage_error': abs(predicted / actual - 1) * 100})
            finally:
                model.close()
    errors = [r['absolute_percentage_error'] for r in rows if 'absolute_percentage_error' in r]
    result = {'predicted_cells': len(errors), 'unsupported_cells': len(rows) - len(errors), 'official_result': False, 'validation': 'input length 4000 excluded from training, other lengths from same allocation; independent AIPerf validation still required', 'median_absolute_percentage_error': statistics.median(errors) if errors else None, 'max_absolute_percentage_error': max(errors) if errors else None, 'measurements': rows}
    (args.output / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'measurements'}, indent=2))


if __name__ == '__main__':
    main()
