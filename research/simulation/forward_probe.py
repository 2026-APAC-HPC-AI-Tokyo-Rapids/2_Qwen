"""Persist each stock SGLang measurement before the next shape is attempted."""
import json

from sglang.benchmark import one_batch

_latency_test = one_batch.latency_test


def recording_latency_test(server_args, port_args, bench_args, gpu_id, tp_rank):
    destination = bench_args.result_filename
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
        return result

    one_batch.latency_test_run_once = record
    try:
        _latency_test(server_args, port_args, bench_args, gpu_id, tp_rank)
    finally:
        one_batch.latency_test_run_once = measure_once


if __name__ == '__main__':
    one_batch.latency_test = recording_latency_test
    one_batch.cli_main()
