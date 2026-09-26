"""Persist each stock SGLang measurement before the next shape is attempted."""
import json
from pathlib import Path

from sglang.benchmark import one_batch

_latency_test = one_batch.latency_test


def recording_latency_test(server_args, port_args, bench_args, gpu_id, tp_rank):
    destination = bench_args.result_filename
    plan_path = Path(destination).parent.parent / 'source/forward-plan.json'
    plan = json.loads(plan_path.read_text())
    if 'random_seed' in plan:
        one_batch.np.random.seed(plan['random_seed'])
    if 'chunk_batch_sizes' in plan and server_args.cuda_graph_config is not None:
        server_args.cuda_graph_config[one_batch.Phase.DECODE].max_bs = max(plan['decode_batch_sizes'])
    bench_args.result_filename = ''
    measure_once = one_batch.latency_test_run_once
    calls = 0

    def record(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = measure_once(*args, **kwargs)
        # Stock latency_test's first call is its warmup, not a measured repeat.
        if calls > 1 and tp_rank == 0 and result is not None:
            with open(destination, 'a') as stream:
                stream.write(json.dumps(result) + '\n')
        expected_calls = 1 + len(bench_args.batch_size) * len(bench_args.input_len) * len(bench_args.output_len)
        if calls == expected_calls and 'chunk_batch_sizes' in plan:
            from chunk_probe import measure

            measure(args[1], plan, str(Path(destination).with_suffix('.chunks.jsonl')), tp_rank)
        return result

    one_batch.latency_test_run_once = record
    try:
        _latency_test(server_args, port_args, bench_args, gpu_id, tp_rank)
    finally:
        one_batch.latency_test_run_once = measure_once


if __name__ == '__main__':
    one_batch.latency_test = recording_latency_test
    one_batch.cli_main()
