"""Run a bounded CPU search; rank joint-SLA goodput, retaining every config."""
import argparse
import concurrent.futures
import copy
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from calibrate import VERSION


def simulate(item):
    directory, config = item
    directory.mkdir(parents=True)
    config_path = directory / 'prediction.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    command = [str(Path(sys.executable).with_name('aisimulate')), 'predict', '--config', str(config_path), '--stack', 'engine', '--output-dir', str(directory / 'prediction'), '--capture-per-request', '--detail', 'all']
    with (directory / 'predict.log').open('w') as log:
        env = dict(os.environ)
        if config["engine"]["estimation_mode"] == "fpm_interpolation":
            env["AIC_ALLOW_UNLISTED_VERSIONS"] = "1"
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False, env=env)
    summary = {'candidate': directory.name, 'exit_code': result.returncode, 'official_result': False}
    if result.returncode == 0:
        report = json.loads((directory / 'prediction/prediction.json').read_text())
        requests = [json.loads(line) for line in (directory / 'prediction/requests.jsonl').read_text().splitlines()]
        good = sum(r['terminal_status'] == 'completed' and r['ttft_ms'] <= 3000 and r['itl_ms'] <= 20 for r in requests)
        summary.update({'joint_sla_fraction': good / len(requests), 'joint_sla_gate': good / len(requests) >= .9, 'simulated_goodput_tok_s': good * 200 / (report['duration_ms'] / 1000), 'metrics': {key: report[key] for key in ('duration_ms', 'request_throughput_rps', 'mean_ttft_ms', 'p99_ttft_ms', 'mean_itl_ms', 'p99_itl_ms')}})
    else:
        summary['error_tail'] = (directory / 'predict.log').read_text()[-5000:]
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--fpm', type=Path)
    parser.add_argument('--nixl-summary', type=Path)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[4, 8, 12, 16, 24, 32])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = yaml.safe_load(Path('research/simulation/prediction.yaml').read_text())
    base['engine']['model'] = str(Path(base['engine']['model']).resolve())
    base['traffic']['stop']['requests'] = 320
    base['engine'].update({'gemm_quant_mode': 'fp8_block', 'moe_quant_mode': 'fp8_block'})
    backends = [('sglang', VERSION)] if args.fpm else [('sglang', '0.5.14'), ('vllm', '0.24.0'), ('trtllm', '1.3.0rc20')]
    measured_link = None
    if args.nixl_summary:
        measured_link = json.loads(args.nixl_summary.read_text())['nixl_segmented_writes']
    tasks = []
    for (backend, version), prefill_tp, decode_tp, concurrency in itertools.product(backends, (4, 8), (4, 8), args.concurrency):
        config = copy.deepcopy(base)
        engine = config['engine']
        engine.update({'backend': backend, 'backend_version': version, 'estimation_mode': 'fpm_interpolation' if args.fpm else 'op_level'})
        config['traffic']['load']['concurrency'] = concurrency
        for role, tp in [('prefill', prefill_tp), ('decode', decode_tp)]:
            worker = engine['workers'][role]
            worker['parallelism'].update({'replicas': 8 // tp, 'tensor': tp, 'moe_tensor': 4 if tp == 4 else 1, 'moe_expert': 1 if tp == 4 else 8})
            worker['kv_cache'].update({'block_size': 1 if backend == 'sglang' else 16, 'capacity': {'type': 'fixed', 'blocks': 131072 if backend == 'sglang' else 8192}})
            if args.fpm:
                worker['timing'] = {'estimation_mode': 'fpm_interpolation', 'fallback_policy': 'deny', 'fpm_parquet_path': str(args.fpm.resolve())}
        if measured_link and decode_tp == 8:
            point = next(p for p in measured_link if p['source_tp'] == prefill_tp and p['prompt_tokens'] == 4000)
            engine['kv_transfer']['bandwidth_gb_per_second'] = point['effective_unique_KV_GB_per_second']
        elif args.fpm:
            # No measured TP4 receiver transfer. Do not invent its link calibration.
            continue
        name = f'{backend}-p{8 // prefill_tp}x{prefill_tp}-d{8 // decode_tp}x{decode_tp}-c{concurrency}'
        tasks.append((args.output / name, config))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(simulate, tasks))
    results.sort(key=lambda r: (r.get('joint_sla_gate', False), r.get('simulated_goodput_tok_s', -1)), reverse=True)
    provenance = {'official_result': False, 'forward_calibration': str(args.fpm) if args.fpm else None, 'nixl_calibration': str(args.nixl_summary) if args.nixl_summary else None, 'limitations': ['Synthetic lengths and finite request count; official AIPerf needed', 'Generic backend data cannot rank deployed versions conclusively'] if not args.fpm else ['SGLang ModelRunner timing excludes serving CPU overhead', 'Uniform sequence lengths and median decode context approximation', 'NIXL isolated transfer fit does not model queueing/overlap'], 'candidates': results}
    (args.output / 'search.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in r.items() if k != 'error_tail'} for r in results[:10]], indent=2))


if __name__ == '__main__':
    main()
