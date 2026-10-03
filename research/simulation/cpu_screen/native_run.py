"""Use existing native SGLang config fields omitted by the public compiler."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from aisimulate import _runtime
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig
from aisimulate.detail import build_prediction_details
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper.replay import ReplayOutputRequirements


class RecordingRuntime:
    def __init__(self, output):
        self.output = output

    def run_replay_json(self, execution_spec_json):
        (self.output / 'execution-spec.json').write_text(execution_spec_json + '\n')
        return _runtime.run_replay_json(execution_spec_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('native_sglang', type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    overrides = json.loads(args.native_sglang.read_text())
    assert set(overrides) == {'prefill'}
    assert set(overrides['prefill']) == {'max_prefill_tokens', 'chunked_prefill_size'}
    config = CorePredictionConfig.from_yaml(args.config)
    spec = prediction_to_replay_spec(config)
    prefill = dict(spec.backend_deployment.prefill_engine_args)
    prefill['sglang'] = overrides['prefill']
    spec = replace(spec, backend_deployment=replace(spec.backend_deployment, prefill_engine_args=prefill))
    before = time.monotonic()
    runner = EngineReplayRunnerFactory(runtime=RecordingRuntime(args.output)).create(0)
    report = runner.run(spec, output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_per_request=True, capture_memory_diagnostics=True, capture_performance_diagnostics=True))
    native = report.metadata['native_report']
    native.setdefault('wall_time_ms', (time.monotonic() - before) * 1000)
    native['details'] = build_prediction_details(native, ['summary', 'memory', 'time', 'energy', 'source'])
    (args.output / 'prediction.json').write_text(json.dumps(native) + '\n')
    with (args.output / 'requests.jsonl').open('w') as requests:
        for row in native['per_request']:
            requests.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    main()
