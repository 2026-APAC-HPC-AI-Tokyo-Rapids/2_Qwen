"""CPU-only target-forward sensitivity screen; never an official AIPerf score."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(rows, start=None, end=None):
    good = sum(r['terminal_status'] == 'completed' and r['ttft_ms'] <= 3000 and r['itl_ms'] <= 20 for r in rows)
    result = {'requests': len(rows), 'joint_good_requests': good, 'joint_good_fraction': good / len(rows), 'joint_sla_gate': good / len(rows) >= .9}
    for field in ('ttft_ms', 'itl_ms'):
        values = np.array([r[field] for r in rows])
        result[field] = {'mean': float(values.mean()), 'p90': float(np.percentile(values, 90)), 'p99': float(np.percentile(values, 99))}
    if start is not None:
        seconds = (end - start) / 1000
        result.update({'window_start_ms': start, 'window_end_ms': end, 'output_tok_s_completed_in_window': sum(r['output_length'] for r in rows) / seconds, 'joint_good_tok_s_completed_in_window': sum(r['output_length'] for r in rows if r['terminal_status'] == 'completed' and r['ttft_ms'] <= 3000 and r['itl_ms'] <= 20) / seconds})
    return result


def summarize(output, config_path, provenance):
    report = json.loads((output / 'prediction/prediction.json').read_text())
    rows = [json.loads(s) for s in (output / 'prediction/requests.jsonl').read_text().splitlines()]
    start, end = report['duration_ms'] * .1, report['duration_ms'] * .9
    steady = [r for r in rows if start <= r['terminal_time_ms'] < end]
    all_metrics = metrics(rows, 0, report['duration_ms'])
    all_metrics['reported_output_tok_s'] = report['output_throughput_tok_s']
    result = {'official_result': False, 'condition': config_path.stem, 'provenance': provenance, 'all_requests_initial_transient_included': all_metrics, 'steady_window_diagnostic': metrics(steady, start, end), 'steady_window_definition': 'Completions in [10%, 90%) of simulated duration; full latency of those requests; not official AIPerf warmup/measurement/cooldown.', 'actual_forward_batch_sizes': {'status': 'unavailable', 'reason': 'Whole-model FPM provider exports no operation/forward-batch diagnostics; admissions are not forward batch evidence.'}, 'report_metrics': {k: report[k] for k in ('duration_ms', 'request_throughput_rps', 'mean_ttft_ms', 'p90_ttft_ms', 'p99_ttft_ms', 'mean_itl_ms', 'p90_itl_ms', 'p99_itl_ms', 'wall_time_ms')}, 'limitations': ['Target-only BF16 KV/attention calibration; deployed EAGLE3 FP8-KV serving is not simulated.', 'Synthetic 4000/200 lengths and fixed c40; no tokenizer, official prompt or AIPerf phases.', 'No SGLang 512-token admission reservation or preallocated PD queue model.', 'Measured whole-forward timing excludes serving CPU overhead; same FPM cannot rank alternate MoE backends.', 'Fixed KV blocks bypass memory-component capacity estimation; doubled pool is a sensitivity input, not evidence it fits device memory.']}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--fpm', required=True, type=Path)
    parser.add_argument('--reuse-prediction', type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    repo = Path(__file__).resolve().parents[3]
    config = yaml.safe_load(args.config.read_text())
    native_sglang = config.pop('native_sglang', None)
    # External FPM identity includes its exact collected model path.
    if not Path(config['engine']['model']).is_absolute():
        config['engine']['model'] = str(repo / config['engine']['model'])
    for worker in config['engine']['workers'].values():
        worker['timing']['fpm_parquet_path'] = str(args.fpm.resolve())
    effective = args.output / 'effective-config.yaml'
    effective.write_text(yaml.safe_dump(config, sort_keys=False))
    provenance = {'native_sglang': native_sglang, 'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), 'condition_config_sha256': sha(args.config), 'effective_config_sha256': sha(effective), 'fpm_sha256': sha(args.fpm), 'fpm_metadata_sha256': {p.name: sha(p) for p in args.fpm.parent.glob('*metadata*')}, 'model_config_sha256': sha(Path(config['engine']['model']) / 'config.json'), 'python': platform.python_version(), 'aisimulate': importlib.metadata.version('aisimulate'), 'environment': {'AIC_ALLOW_UNLISTED_VERSIONS': '1'}, 'reuse_prediction': str(args.reuse_prediction) if args.reuse_prediction else None}
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    if args.reuse_prediction:
        (args.output / 'prediction').symlink_to(args.reuse_prediction.resolve(), target_is_directory=True)
    else:
        command = [str(Path(sys.executable).with_name('aisimulate')), 'predict', '--config', str(effective), '--stack', 'engine', '--output-dir', str(args.output / 'prediction'), '--capture-per-request', '--detail', 'all']
        if native_sglang:
            native_path = args.output / 'native-sglang.json'
            native_path.write_text(json.dumps(native_sglang, indent=2) + '\n')
            command = [sys.executable, str(Path(__file__).with_name('native_run.py')), str(effective), str(args.output / 'prediction'), str(native_path)]
        provenance['command'] = command
        before = time.monotonic()
        with (args.output / 'predict.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, AIC_ALLOW_UNLISTED_VERSIONS='1'), check=False)
        provenance.update({'returncode': result.returncode, 'wall_seconds_including_cli_startup': time.monotonic() - before})
        (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        if result.returncode:
            (args.output / 'summary.json').write_text(json.dumps({'official_result': False, 'condition': args.config.stem, 'provenance': provenance, 'unsupported_or_failed': (args.output / 'predict.log').read_text()[-8000:]}, indent=2) + '\n')
            return result.returncode
    result = summarize(args.output, args.config, provenance)
    print(json.dumps({'condition': result['condition'], 'commit': provenance['commit'], 'all': result['all_requests_initial_transient_included'], 'steady': result['steady_window_diagnostic']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
