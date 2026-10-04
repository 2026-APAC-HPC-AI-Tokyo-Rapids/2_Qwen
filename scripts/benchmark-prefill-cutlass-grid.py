"""Bounded offline prebuilt CUTLASS grid; never selects a serving configuration."""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib
import itertools
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace


def validate_manifest(manifest, resolved, source_root, condition_source):
    assert manifest['schema'] == 'prefill-moe-grid/v2'
    for name, digest in manifest['source_hashes'].items():
        assert hashlib.sha256((source_root / name).read_bytes()).hexdigest() == digest, ('source hash mismatch', name)
    entries = manifest['configurations']
    assert len({entry['mode'] for entry in entries}) == len(entries)
    assert len({entry['id'] for entry in entries}) == len(entries)
    actual = {entry['mode']: entry for entry in resolved}
    assert set(actual) == {entry['mode'] for entry in entries}
    tree = ast.parse(condition_source)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_prefill_tile_modes')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<installed finite mode map>', 'exec'), namespace)  # noqa: S102
    for entry in entries:
        for key, value in actual[entry['mode']].items():
            assert entry[key] == value, ('native metadata mismatch', entry['id'], key)
        modes = namespace['_prefill_tile_modes'](entry['id'], entry['id'], True, True, 1, False)
        assert modes == (entry['mode'], entry['mode']), ('production name/ID mismatch', entry['id'])


def unique_configurations(entries):
    seen, unique, duplicates = set(), [], []
    for entry in sorted(entries, key=lambda x: x['mode']):
        key = (tuple(entry['tile_mnk']), tuple(entry['cluster_mnk']), entry['schedule'], entry['resolved_stages'])
        if key in seen:
            duplicates.append(entry['mode'])
        else:
            seen.add(key)
            unique.append(entry)
    return unique, duplicates


def rotated_order(names, index):
    order = names[index % len(names):] + names[:index % len(names)]
    return list(reversed(order)) if index % 2 else order


def shortlist(records, role, limit=2):
    scores = {}
    for case in records:
        stock = statistics.median(case['variants']['stock']['rounds_ms'])
        for variant in case['variants'].values():
            if variant.get('role') != role or variant.get('status') != 'measured' or not variant['quality_eligible'] or variant['mode'] == 0:
                continue
            if len(variant['rounds_ms']) == 5 and all(a < b for a, b in zip(variant['rounds_ms'], case['variants']['stock']['rounds_ms'])):
                scores.setdefault(variant['mode'], []).append(statistics.median(variant['rounds_ms']) / stock)
    # Require both training patterns and no median regression; no production adoption.
    ranked = [(statistics.mean(ratios), mode) for mode, ratios in scores.items() if len(ratios) == len(records) and max(ratios) < 1]
    return [mode for _, mode in sorted(ranked)[:limit]]


def main():
    import sglang.srt.layers.moe.cutlass_moe as moe
    import torch
    native = importlib.import_module('apac_prefill_moe_tiles')
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    manifest = json.loads(Path('/opt/apac/prefill-moe-tiles/manifest.json').read_text())
    assert manifest['schema'] == 'prefill-moe-grid/v2'
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() == manifest['module_sha256']
    assert torch.cuda.device_count() == 1, 'finite grid requires exactly one visible allocated GPU'
    assert torch.cuda.get_device_capability() == (9, 0)
    image_sha = os.environ['QWEN_GRID_IMAGE_SHA']
    assert len(image_sha) == 64 and all(char in '0123456789abcdef' for char in image_sha), 'valid image SHA256 required'
    properties = torch.cuda.get_device_properties(0)
    validate_manifest(manifest, native.configurations(), Path('/opt/apac/prefill-moe-tiles'), Path('/opt/sglang/source/python/sglang/srt/layers/quantization/fp8.py').read_text())
    configs, duplicates = unique_configurations(manifest['configurations'])
    moe._prefill_fused_shuffle = True
    moe.get_server_args = lambda: SimpleNamespace(disaggregation_mode='prefill')
    moe.get_parallel = lambda: SimpleNamespace(moe_ep_size=1)
    moe._prefill_tile_mm = lambda mode, *inputs: moe.fp8_blockwise_scaled_grouped_mm(*inputs) if mode < 0 else native.grouped_mm(*inputs, mode)
    torch.manual_seed(20261004)
    torch.backends.cuda.matmul.allow_tf32 = False
    e, h, i, topk = 128, 4096, 384, 8
    w1 = (torch.randn((e, i*2, h), device='cuda', dtype=torch.bfloat16)*8).to(torch.float8_e4m3fn)
    w2 = (torch.randn((e, h, i), device='cuda', dtype=torch.bfloat16)*8).to(torch.float8_e4m3fn)
    s1 = torch.full((e, i*2//128, h//128), .0025, device='cuda')
    s2 = torch.full((e, h//128, i//128), .0025, device='cuda')
    strides = [torch.full((e,), value, dtype=torch.int64, device='cuda') for value in (h,i*2,i,h)]
    workspace = torch.empty(1048576, dtype=torch.uint8, device='cuda')
    pointers = [torch.empty(e, dtype=torch.int64, device='cuda') for _ in range(5)]
    offsets = torch.empty(e+1, dtype=torch.int32, device='cuda')
    problems = [torch.empty((e,3), dtype=torch.int32, device='cuda') for _ in range(2)]
    result = {'schema': 'prefill-cutlass-grid-probe/v1', 'manifest': manifest, 'duplicate_modes_excluded': duplicates, 'actual_router_histogram': False, 'shuffle_on': True, 'objective': 'whole MoE graph CUDA-event median; role variant with partner stock, not isolated GEMM', 'rounds': 5, 'replays': 100, 'cases': [], 'joint': [], 'serving_adoption': False, 'image_sha256': image_sha, 'device': {'name': properties.name, 'uuid': str(getattr(properties, 'uuid', 'unavailable')), 'total_memory': properties.total_memory, 'sm_count': properties.multi_processor_count, 'visible_count': torch.cuda.device_count(), 'torch': torch.__version__, 'cuda': torch.version.cuda}}
    result['search_bounds'] = {'maximum_prebuilt_configurations': 16, 'maximum_training_variants_per_shape': 34, 'training_shapes': 2, 'heldout_shapes': 3, 'maximum_joint_variants_per_shape': 6, 'maximum_graph_measurements': 430, 'maximum_timed_replays': 43000, 'rounds': 5, 'replays_per_measurement': 100, 'eager_warmup_calls_per_measurement': 3, 'graph_warmup_replays_per_measurement': 3, 'shortlist_per_role': 2, 'termination': 'finite grid and heldout exhausted; known unsupported candidate skipped; unknown runtime or quality failure stops; external timeout720seconds; no serving adoption'}
    assert len(manifest['configurations']) <= 16 and len(configs) <= 16
    def save():
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    def quality(value, reference):
        a,b=value.float(),reference.float()
        error=(a-b).abs()
        l2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))
        return {'finite': bool(torch.isfinite(a).all()), 'relative_l2': l2, 'max_abs': float(error.max()), 'max_rel': float((error/b.abs().clamp_min(1e-8)).max()), 'bitwise_equal': bool(torch.equal(value,reference))}
    def case(tokens, distribution, seed, variants, destination):
        torch.manual_seed(seed)
        x=torch.randn((tokens,h),device='cuda',dtype=torch.bfloat16)
        row=torch.arange(tokens,device='cuda').unsqueeze(1)
        lane=torch.arange(topk,device='cuda').unsqueeze(0)
        ids=(row*topk+lane)%e if distribution=='balanced' else (row+lane)%16 if distribution=='hotonly' else torch.where(row%4==0,(row*17+lane*13)%e,(row+lane)%16)
        ids=ids.to(torch.int32).contiguous()
        weights=torch.softmax(torch.randn((tokens,topk),device='cuda'),-1)
        def run(modes):
            return moe.cutlass_fused_experts_fp8(x,w1.transpose(1,2),w2.transpose(1,2),s1.transpose(1,2),s2.transpose(1,2),weights,ids,*strides,workspace,*pointers,offsets,*problems,enable_es=(False,False),prefill_tile_modes=modes)
        reference=run((-1,-1)).clone()
        torch.cuda.synchronize()
        record={'tokens': tokens,'distribution': distribution,'seed': seed,'expert_counts': torch.bincount(ids.flatten().long(),minlength=e).cpu().tolist(),'variants': {}}
        destination.append(record)
        dense={}
        if tokens==32:
            for dtype in [torch.bfloat16,torch.float32]:
                out=torch.zeros_like(x,dtype=torch.float32)
                for expert in torch.unique(ids).tolist():
                    rows,slots=torch.where(ids==expert)
                    dw1=w1[expert].float()*s1[expert].repeat_interleave(128,0).repeat_interleave(128,1)
                    dw2=w2[expert].float()*s2[expert].repeat_interleave(128,0).repeat_interleave(128,1)
                    up=x[rows].to(dtype)@dw1.to(dtype).T
                    activation=torch.nn.functional.silu(up[:,:i])*up[:,i:]
                    down=activation@dw2.to(dtype).T
                    out.index_add_(0,rows,down.float()*weights[rows,slots].to(torch.bfloat16).float().unsqueeze(1))
                dense[str(dtype)]=out
            record['oracle_baseline']={name:quality(reference,value)['relative_l2'] for name,value in dense.items()}
        save()
        for round_index in range(5):
            for name in rotated_order(list(variants),round_index):
                modes,role,mode=variants[name]
                entry=record['variants'].setdefault(name,{'modes': modes,'role': role,'mode': mode,'rounds_ms': [],'status': 'in_progress'})
                if entry['status']=='unsupported':
                    continue
                result['in_progress']={'tokens': tokens,'distribution': distribution,'round': round_index,'variant': name}
                save()
                try:
                    for _ in range(3):
                        actual=run(modes)
                    torch.cuda.synchronize()
                except RuntimeError as error:
                    if name not in ('stock','wheel') and 'Failed to implement GEMM' in str(error):
                        entry.update(status='unsupported',error=str(error));save();continue
                    raise
                eager=quality(actual,reference)
                assert eager['finite'] and eager['relative_l2']<.1,(name,eager)
                if name=='stock' and tokens>32:
                    assert eager['relative_l2']<=1e-3,('clone equivalence',eager)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,capture_error_mode='global'):
                    captured=run(modes)
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
                q=quality(captured,reference)
                assert q['finite'] and q['relative_l2']<.1,(name,q)
                if name=='stock' and tokens>32:
                    assert q['relative_l2']<=1e-3,('clone graph equivalence',q)
                oracle={key:quality(captured,value)['relative_l2'] for key,value in dense.items()}
                for key,value in oracle.items():
                    assert value<=record['oracle_baseline'][key]*1.25+.005,(name,key,value)
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100):
                    graph.replay()
                end.record();end.synchronize()
                entry['rounds_ms'].append(start.elapsed_time(end)/100)
                entry.update(status='measured',eager_quality=eager,graph_quality=q,oracle_relative_l2=oracle,quality_eligible=entry.get('quality_eligible',True) and eager['relative_l2']<=1e-3 and q['relative_l2']<=1e-3)
                del graph,captured,actual
                gc.collect()
                save()
        result.pop('in_progress',None)
        save()
    variants={'wheel':((-1,-1),'control',-1),'stock':((0,0),'control',0)}
    for config in configs:
        mode=config['mode']
        for role,modes in [('up',(mode,0)),('down',(0,mode))]:
            variants[f'{role}:{config["id"]}']=(modes,role,mode)
    for distribution in ['balanced','skewed']:
        case(8192,distribution,20261004,variants,result['cases'])
    up,down=shortlist(result['cases'],'up'),shortlist(result['cases'],'down')
    result['shortlist']={'up': up,'down': down,'criterion': 'five paired rounds faster in both training patterns and quality eligible; external heldout shortlist only, no adoption'}
    joint={'wheel':((-1,-1),'control',-1),'stock':((0,0),'control',0)}
    for u,d in itertools.product(up or [0],down or [0]):
        if (u,d)!=(0,0):
            joint[f'joint:{u},{d}']=((u,d),'joint',[u,d])
    case(4096,'skewed',20261005,joint,result['joint'])
    case(8192,'hotonly',20261005,joint,result['joint'])
    # Independent dense oracle; small geometry is forced only in this external probe.
    case(32,'hotonly',20261006,joint,result['joint'])
    print('Finite grid completed. Review stock/quality/paired rounds/heldout before E2E; no runtime adoption.',flush=True)

if __name__=='__main__':
    main()
