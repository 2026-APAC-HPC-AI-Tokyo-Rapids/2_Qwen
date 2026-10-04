import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('probe', ROOT / 'scripts/benchmark-prefill-moe-routing.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def record(tokens, pattern, ratio=.99, wins=20, quality=True):
    return {'tokens': tokens, 'pattern': pattern, 'graph_changed_ids_reset_pass': True,
            'standalone': {'stock': [1]*20, 'tiled': [1]*20},
            'whole_moe': {'stock': [1]*20, 'tiled': [1]*20, 'quality_eligible': quality,
                          'comparison': {'median_ratio': ratio, 'wins': wins}}}


def records():
    return [record(8192, 'balanced'), record(8192, 'skewed')], [record(4096, 'skewed'), record(8192, 'hotonly'), record(65, 'balanced')]


def test_predeclared_gate_needs_whole_moe_and_no_heldout_regression():
    training, heldout = records()
    assert probe.evaluate(training, heldout)
    for key, value in [('median_ratio', .997), ('wins', 14)]:
        training, heldout = records()
        training[0]['whole_moe']['comparison'][key] = value
        assert not probe.evaluate(training, heldout)
    training, heldout = records()
    heldout[0]['whole_moe']['comparison']['median_ratio'] = 1.001
    assert not probe.evaluate(training, heldout)
    training, heldout = records()
    training[0]['whole_moe']['quality_eligible'] = False
    assert not probe.evaluate(training, heldout)


def test_incomplete_results_cannot_pass():
    training, heldout = records()
    assert not probe.evaluate(training[:1], heldout)
    assert not probe.evaluate(training, heldout[:1])
    training[0]['whole_moe']['tiled'].pop()
    assert not probe.evaluate(training, heldout)
    training, heldout = records()
    heldout[0]['graph_changed_ids_reset_pass'] = False
    assert not probe.evaluate(training, heldout)


def test_pairing_and_reverse_order():
    assert probe.order(0) == ['stock', 'tiled']
    assert probe.order(1) == ['tiled', 'stock']
    x = probe.summarize([.98, 1.98], [1, 2])
    assert x['wins'] == 2 and x['paired_ratios'] == [.98, .99]


def test_finite_one_gpu_and_graph_changed_ids():
    source = (ROOT / 'scripts/benchmark-prefill-moe-routing.py').read_text()
    for contract in ['torch.cuda.device_count() == 1', 'warmup_replays <= 65536', 'while warmup < 200', 'graph_changed_ids_reset_pass', 'empty_helper_and_whole_moe_pass', 'independent_oracle_pass', "'heldout': []", "'serving_adoption': False"]:
        assert contract in source
    assert 'prefill_tile_modes=' not in source and 'grouped_mm' not in source


def test_torch_loads_before_prebuilt_extension():
    source = (ROOT / 'scripts/benchmark-prefill-moe-routing.py').read_text()
    main = source[source.index('def main():'):]
    assert main.index('import torch') < main.index('import apac_prefill_moe_routing as native')
    # Prevent import sorting from regrouping native ahead of torch.
    assert main.index('import torch') < main.index('assert torch.__version__') < main.index('import apac_prefill_moe_routing as native')
