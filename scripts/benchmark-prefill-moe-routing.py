#!/usr/bin/env python3
"""Finite external one-GPU routing probe; no runtime tuning or auto-adoption."""
import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

GATE = {'training_ratio_max': .995, 'training_wins_min': 15, 'rounds': 20,
        'heldout_ratio_max': 1.0, 'relative_l2_max': .001,
        'actual_router_histogram': False, 'automatic_adoption': False}


def summarize(candidate, stock):
    ratios = [a / b for a, b in zip(candidate, stock)]
    return {'median_ratio': statistics.median(candidate) / statistics.median(stock),
            'paired_median': statistics.median(ratios), 'wins': sum(r < 1 for r in ratios),
            'paired_range': [min(ratios), max(ratios)], 'paired_ratios': ratios}


def evaluate(training, heldout):
    if {(c['tokens'], c['pattern']) for c in training} != {(8192, 'balanced'), (8192, 'skewed')} or len(training) != 2:
        return False
    if {(c['tokens'], c['pattern']) for c in heldout} != {(4096, 'skewed'), (8192, 'hotonly'), (65, 'balanced')} or len(heldout) != 3:
        return False
    for c in training + heldout:
        if not c.get('graph_changed_ids_reset_pass', False):
            return False
        if any(len(c['whole_moe'].get(name, [])) != 20 or len(c['standalone'].get(name, [])) != 20 for name in ['stock', 'tiled']):
            return False
    return all(c['whole_moe']['quality_eligible'] and c['whole_moe']['comparison']['median_ratio'] <= GATE['training_ratio_max'] and c['whole_moe']['comparison']['wins'] >= GATE['training_wins_min'] for c in training) and all(c['whole_moe']['quality_eligible'] and c['whole_moe']['comparison']['median_ratio'] <= GATE['heldout_ratio_max'] for c in heldout)


def order(round_index):
    return ['stock', 'tiled'] if round_index % 2 == 0 else ['tiled', 'stock']


def main():
    import torch

    # Loading torch first supplies libc10/libtorch to the prebuilt CUDA extension.
    assert torch.__version__
    import apac_prefill_moe_routing as native
    from sglang.srt.layers.moe import cutlass_moe as moe

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0)
    root = Path('/opt/apac/prefill-moe-routing')
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['schema'] == 'prefill-moe-routing/v1'
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() == manifest['module_sha256']
    assert hashlib.sha256((root / 'moe_routing.cu').read_bytes()).hexdigest() == manifest['source_sha256']
    image_sha = os.environ['QWEN_GRID_IMAGE_SHA']
    assert len(image_sha) == 64 and all(c in '0123456789abcdef' for c in image_sha)
    assert hasattr(moe, '_prefill_tiled_routing') and hasattr(moe, '_prefill_fused_shuffle')
    moe._prefill_fused_shuffle = True
    moe.get_server_args = lambda: SimpleNamespace(disaggregation_mode='prefill')
    moe.get_parallel = lambda: SimpleNamespace(moe_ep_size=1)
    prop = torch.cuda.get_device_properties(0)
    result = {'schema': 'prefill-moe-routing-probe/v1', 'image_sha256': image_sha, 'manifest': manifest,
              'gate': GATE, 'cases': [], 'heldout': [], 'serving_adoption': False,
              'device': {'name': prop.name, 'uuid': str(prop.uuid), 'sm_count': prop.multi_processor_count, 'visible_count': 1},
              'bounds': {'variants': 2, 'training_shapes': 2, 'heldout_shapes': 3, 'oracle_shapes': 1,
                         'rounds': 20, 'timed_replays': 100, 'continuous_warmup_ms': 200, 'max_contiguous_warmup_replays_per_measurement': 65536, 'maximum_measurements': 404, 'maximum_timed_replays': 40400,
                         'termination': 'finite cases exhausted or any unknown/semantic/quality error; timeout720s'},
              'objective': 'wholeMoE wheel GEMMs/shuffleon, helper-only difference; standalone secondary'}
    def save():
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    def quality(value, reference):
        a, b = value.float(), reference.float()
        delta = (a - b).abs()
        denom = torch.linalg.vector_norm(b).clamp_min(1e-30)
        return {'finite': bool(torch.isfinite(a).all()), 'relative_l2': float(torch.linalg.vector_norm(a-b) / denom),
                'max_abs': float(delta.max()) if delta.numel() else 0, 'max_rel': float((delta/b.abs().clamp_min(1e-8)).max()) if delta.numel() else 0,
                'bitwise_equal': bool(torch.equal(value, reference))}
    def check_maps(ids, offsets, p1, p2, a_map, c_map):
        flat = ids.flatten().long()
        counts = torch.bincount(flat, minlength=128)
        expect_offsets = torch.cat([torch.zeros(1, device='cuda', dtype=torch.long), counts.cumsum(0)])
        assert torch.equal(offsets.long(), expect_offsets)
        assert torch.equal(p1[:, 0].long(), counts) and torch.equal(p2[:, 0].long(), counts)
        assert bool((p1[:, 1] == 768).all() and (p1[:, 2] == 4096).all())
        assert bool((p2[:, 1] == 4096).all() and (p2[:, 2] == 384).all())
        rows = torch.arange(flat.numel(), device='cuda')
        assert torch.equal(c_map.long().sort().values, rows)
        assert torch.equal(a_map[c_map.long()].long(), rows // 8)
        assert bool(((c_map >= offsets[flat]) & (c_map < offsets[flat+1])).all())
    def measure(run, reference, semantics=None):
        for _ in range(3):
            actual = run()
        torch.cuda.synchronize()
        eager = quality(actual, reference) if reference is not None else None
        if semantics:
            semantics()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        warmup = 0
        warmup_replays = 400
        while warmup < 200:
            start.record()
            assert warmup_replays <= 65536, "warmup bounded replay limit exceeded"
            for _ in range(warmup_replays):
                graph.replay()
            end.record(); end.synchronize()
            warmup = start.elapsed_time(end)
            if warmup < 200:
                warmup_replays *= 2
        if semantics:
            semantics()
        q = quality(captured, reference) if reference is not None else None
        for test in [eager, q]:
            if test is not None:
                assert test['finite'] and test['relative_l2'] <= GATE['relative_l2_max'], test
        start.record()
        for _ in range(100):
            graph.replay()
        end.record(); end.synchronize()
        timing = start.elapsed_time(end) / 100
        del graph, captured, actual
        gc.collect()
        return {'ms': timing, 'warmup_ms': warmup, 'warmup_replays': warmup_replays, 'eager': eager, 'graph': q}
    torch.manual_seed(20261005)
    torch.backends.cuda.matmul.allow_tf32 = False
    e, h, intermediate = 128, 4096, 384
    w1 = (torch.randn((e, intermediate*2, h), device='cuda', dtype=torch.bfloat16)*8).to(torch.float8_e4m3fn)
    w2 = (torch.randn((e, h, intermediate), device='cuda', dtype=torch.bfloat16)*8).to(torch.float8_e4m3fn)
    s1 = torch.full((e, 6, 32), .0025, device='cuda')
    s2 = torch.full((e, 32, 3), .0025, device='cuda')
    strides = [torch.full((e,), v, dtype=torch.int64, device='cuda') for v in (h, intermediate*2, intermediate, h)]
    workspace = torch.empty(1048576, dtype=torch.uint8, device='cuda')
    pointers = [torch.empty(e, dtype=torch.int64, device='cuda') for _ in range(5)]
    offsets = torch.empty(e+1, dtype=torch.int32, device='cuda')
    p1, p2 = [torch.empty((e, 3), dtype=torch.int32, device='cuda') for _ in range(2)]
    def case(tokens, pattern, seed, destination, rounds=20):
        torch.manual_seed(seed)
        x = torch.randn((tokens, h), device='cuda', dtype=torch.bfloat16)
        row = torch.arange(tokens, device='cuda').unsqueeze(1)
        lane = torch.arange(8, device='cuda').unsqueeze(0)
        ids = ((row*8+lane)%128 if pattern == 'balanced' else (row+lane)%8 if pattern == 'hotonly' else torch.where(row%4 == 0, (row*17+lane*13)%128, (row+lane)%16)).int().contiguous()
        weights = torch.softmax(torch.randn((tokens, 8), device='cuda'), -1)
        a_map = torch.empty(tokens*8, dtype=torch.int32, device='cuda')
        c_map = torch.empty_like(a_map)
        counts = torch.empty(((tokens*8+511)//512, 128), dtype=torch.int32, device='cuda')
        tile_offsets = torch.empty_like(counts)
        def helper(name):
            if name == 'stock':
                moe.prepare_moe_input(ids, offsets, p1, p2, a_map, c_map, 128, 384, 4096, None)
            else:
                native.prepare(ids, offsets, p1, p2, a_map, c_map, counts, tile_offsets)
        def whole(name):
            moe._prefill_tiled_routing = name == 'tiled'
            return moe.cutlass_fused_experts_fp8(x, w1.transpose(1, 2), w2.transpose(1, 2), s1.transpose(1, 2), s2.transpose(1, 2), weights, ids, *strides, workspace, *pointers, offsets, p1, p2, enable_es=(False, False))
        reference = whole('stock').clone()
        record = {'tokens': tokens, 'pattern': pattern, 'seed': seed, 'expert_counts': torch.bincount(ids.flatten().long(), minlength=128).cpu().tolist(), 'utc_start_epoch': time.time(), 'standalone': {}, 'whole_moe': {'quality_eligible': True}}
        destination.append(record); save()
        dense = {}
        if tokens == 32:
            for dtype in [torch.bfloat16, torch.float32]:
                value = torch.zeros_like(x, dtype=torch.float32)
                for expert in torch.unique(ids).tolist():
                    rows, slots = torch.where(ids == expert)
                    dw1 = w1[expert].float()*s1[expert].repeat_interleave(128, 0).repeat_interleave(128, 1)
                    dw2 = w2[expert].float()*s2[expert].repeat_interleave(128, 0).repeat_interleave(128, 1)
                    up = x[rows].to(dtype) @ dw1.to(dtype).T
                    activation = torch.nn.functional.silu(up[:, :intermediate])*up[:, intermediate:]
                    down = activation @ dw2.to(dtype).T
                    value.index_add_(0, rows, down.float()*weights[rows, slots].to(torch.bfloat16).float().unsqueeze(1))
                dense[str(dtype)] = value
            record['oracle_baseline'] = {key: quality(reference, value)['relative_l2'] for key, value in dense.items()}
            candidate = whole('tiled')
            record['oracle_candidate'] = {key: quality(candidate, value)['relative_l2'] for key, value in dense.items()}
            assert all(value <= record['oracle_baseline'][key]*1.25+.005 for key, value in record['oracle_candidate'].items())
            record['independent_oracle_pass'] = True
        for index in range(rounds):
            for name in order(index):
                result['in_progress'] = {'tokens': tokens, 'pattern': pattern, 'round': index, 'variant': name}
                save()
                hm = measure(lambda name=name: helper(name), None, lambda: check_maps(ids, offsets, p1, p2, a_map, c_map))
                record['standalone'].setdefault(name, []).append(hm)
                wm = measure(lambda name=name: whole(name), reference)
                record['whole_moe'].setdefault(name, []).append(wm)
                save()
        record['standalone']['comparison'] = summarize([r['ms'] for r in record['standalone']['tiled']], [r['ms'] for r in record['standalone']['stock']])
        record['whole_moe']['comparison'] = summarize([r['ms'] for r in record['whole_moe']['tiled']], [r['ms'] for r in record['whole_moe']['stock']])
        # Same captured pointers, changed route contents: catches stale scratch/cursor accumulation.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            helper('tiled')
        original = ids.clone()
        for change in [torch.zeros_like(ids), (original+37)%128, original]:
            ids.copy_(change)
            graph.replay(); torch.cuda.synchronize()
            check_maps(ids, offsets, p1, p2, a_map, c_map)
        del graph
        record['graph_changed_ids_reset_pass'] = True
        record['utc_end_epoch'] = time.time()
        result.pop('in_progress', None); save()
    for pattern in ['balanced', 'skewed']:
        case(8192, pattern, 20261010, result['cases'])
    for tokens, pattern in [(4096, 'skewed'), (8192, 'hotonly'), (65, 'balanced')]:
        case(tokens, pattern, 20261011, result['heldout'])
    result['oracle'] = []
    case(32, 'hotonly', 20261012, result['oracle'], rounds=1)
    empty_ids = torch.empty((0, 8), dtype=torch.int32, device='cuda')
    empty_map = torch.empty(0, dtype=torch.int32, device='cuda')
    empty_scratch = torch.empty((0, 128), dtype=torch.int32, device='cuda')
    native.prepare(empty_ids, offsets, p1, p2, empty_map, empty_map, empty_scratch, empty_scratch)
    assert bool((offsets == 0).all() and (p1[:, 0] == 0).all() and (p2[:, 0] == 0).all())
    for enabled in [False, True]:
        moe._prefill_tiled_routing = enabled
        empty_output = moe.cutlass_fused_experts_fp8(torch.empty((0, h), dtype=torch.bfloat16, device='cuda'), w1.transpose(1, 2), w2.transpose(1, 2), s1.transpose(1, 2), s2.transpose(1, 2), torch.empty((0, 8), device='cuda'), empty_ids, *strides, workspace, *pointers, offsets, p1, p2, enable_es=(False, False))
        assert empty_output.shape == (0, h)
    result['empty_helper_and_whole_moe_pass'] = True
    result['e2e_evidence_gate_pass'] = evaluate(result['cases'], result['heldout']) and result['oracle'][0]['independent_oracle_pass']
    save()
    print('Finite routing probe complete; no candidate auto-adoption.', flush=True)


if __name__ == '__main__':
    main()
