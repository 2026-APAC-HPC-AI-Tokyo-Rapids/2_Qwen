"""Export measured SGLang forward times to AISimulate's external FPM schema."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

POLICY = 'sglang_one_batch_three_repetitions_v1'
VERSION = 'fc275674a97d78553b22597ee72c8fd48f5420f4'


def rows_from_summary(summary, model_path, held_out_length=None):
    groups = {}
    for point in summary['measurements']:
        if point['input_len'] == held_out_length:
            continue
        key = (point['tp'], point['ep'], point['batch_size'], point['input_len'])
        groups.setdefault(key, []).append(point)
    rows = []
    for (tp, ep, batch, length), points in sorted(groups.items()):
        for kind in ('prefill', 'decode'):
            samples = [x for point in points for x in point[kind + '_ms_samples']]
            if any(len(point[kind + '_ms_samples']) != 3 for point in points):
                raise ValueError('three repetitions required')
            # Decode reports the median of 31 forwards at contexts ISL+1..ISL+31.
            # Assign its center coordinate and retain that approximation explicitly.
            kv = 0 if kind == 'prefill' else batch * (length + 16)
            rows.append({'cell_id': f'tp{tp}-ep{ep}', 'model_path': model_path, 'system': 'h100_sxm', 'backend': 'sglang', 'backend_version': VERSION, 'weight_quantization': 'fp8_block', 'gemm_quant_mode': 'fp8_block', 'moe_quant_mode': 'fp8_block', 'fmha_quant_mode': 'bfloat16', 'comm_quant_mode': 'half', 'kv_cache_dtype': 'bfloat16', 'tp': tp, 'pp': 1, 'dp': 1, 'moe_tp': tp // ep, 'moe_ep': ep, 'cp': 1, 'moe_backend': 'auto', 'attention_backend': 'auto', 'enable_wideep': False, 'enable_eplb': False, 'workload_kind': kind, 'batch_size': batch, 'total_prefill_tokens': batch * length if kind == 'prefill' else 0, 'total_kv_read_tokens': kv, 'partition_policy': 'balanced_v1', 'latency_ms': float(statistics.median(samples)), '_samples': samples})
    for point in summary.get('chunk_measurements', []):
        if point['kind'] == 'decode' and point['input_len'] == held_out_length:
            continue
        base = next(row for row in rows if row['tp'] == point['tp'] and row['moe_ep'] == point['ep'] and row['workload_kind'] == point['kind'])
        batch, length = point['batch_size'], point['input_len']
        kind = point['kind']
        row = {**base, 'batch_size': batch, 'total_prefill_tokens': batch * length if kind == 'prefill' else 0, 'total_kv_read_tokens': batch * (point['prefix_len'] if kind == 'prefill' else length + 16), 'latency_ms': float(statistics.median(point['latency_ms_samples'])), '_samples': point['latency_ms_samples']}
        rows.append(row)
    # Multiple TP4 replicas and full/chunked warmup can measure the same physical
    # coordinate. Aggregate those observations before the native unique-cell gate.
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in ('tp', 'moe_ep', 'workload_kind', 'batch_size', 'total_prefill_tokens', 'total_kv_read_tokens'))
        grouped.setdefault(key, []).append(row)
    return [{**{k: v for k, v in group[0].items() if k != '_samples'}, 'latency_ms': float(statistics.median(x for row in group for x in row['_samples']))} for group in grouped.values()]


def export(summary_path, output, held_out_length=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    summary = json.loads(summary_path.read_text())
    model = str(Path('research/simulation/model').resolve())
    rows = rows_from_summary(summary, model, held_out_length)
    output.mkdir(parents=True, exist_ok=False)
    parquet = output / 'fpm_forward_perf.parquet'
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    metadata = {'schema_name': 'aic_fpm_forward_perf', 'schema_version': 6, 'coordinate_system': 'iteration_totals_balanced_v1', 'measurement_policy': POLICY, 'row_count': len(rows), 'parquet_sha256': hashlib.sha256(parquet.read_bytes()).hexdigest(), 'system': 'h100_sxm', 'backend': 'sglang', 'backend_version': VERSION, 'producer': 'research/simulation/calibrate.py', 'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(), 'held_out_input_length': held_out_length, 'aggregation': 'median of three repetitions per worker; both measured TP4 replicas pooled', 'decode_coordinate': 'median of 31 decode forwards assigned to ISL+16; not per-iteration telemetry', 'timing_boundary': summary['timing_boundary'], 'model_revision': summary['plan']['model_revision'], 'model_config_sha256': hashlib.sha256((Path(model) / 'config.json').read_bytes()).hexdigest(), 'official_result': False}
    parquet.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return parquet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--held-out-input-length', type=int)
    args = parser.parse_args()
    print(export(args.summary, args.output, args.held_out_input_length))
