"""Predeclared same-image confirmation; original grid verdict remains unchanged."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

CANDIDATE_IDS = ('coop128_c1x2_s4', 'coop128_c1x1_s4')
GATE = {'training_median_ratio_max': {'balanced': .995, 'skewed': .999}, 'minimum_faster_rounds': 15, 'rounds': 20, 'heldout_median_ratio_max': 1.0, 'maximum_relative_l2': 1e-3, 'bootstrap_interval': '95% diagnostic only; rounds are not independent runs', 'initial_grid_independent_dense_oracle': 'passed; not rerun in this large-shape confirmation', 'original_grid_verdict': 'no robust winner; unchanged', 'automatic_adoption': False}

def summarize(values, baseline):
    ratios=[a/b for a,b in zip(values,baseline)]
    randomizer=random.Random(20261007)
    bootstrap=sorted(statistics.median(randomizer.choices(ratios,k=len(ratios))) for _ in range(2000))
    return {'median_time_ratio':statistics.median(values)/statistics.median(baseline),'paired_ratios':ratios,'faster_rounds':sum(r<1 for r in ratios),'paired_median':statistics.median(ratios),'paired_range':[min(ratios),max(ratios)],'diagnostic_bootstrap95':[bootstrap[49],bootstrap[1949]]}

def evaluate(cases,heldout):
    evaluated={}
    for name in CANDIDATE_IDS:
        training={c['distribution']:summarize(c['variants'][name]['rounds_ms'],c['variants']['wheel']['rounds_ms']) for c in cases if c['variants'][name]['status']=='measured'}
        validation={str(c['tokens'])+':'+c['distribution']:summarize(c['variants'][name]['rounds_ms'],c['variants']['wheel']['rounds_ms']) for c in heldout if c['variants'][name]['status']=='measured'}
        quality=all(c['variants'][name].get('quality_eligible',False) for c in cases+heldout)
        passed=quality and len(training)==2 and len(validation)==2 and all(len(v['paired_ratios'])==20 and v['faster_rounds']>=15 and v['median_time_ratio']<=GATE['training_median_ratio_max'][key] for key,v in training.items()) and all(v['median_time_ratio']<=1 for v in validation.values())
        evaluated[name]={'training':training,'heldout':validation,'quality_all_rounds':quality,'e2e_evidence_gate_pass':passed,'serving_adoption':False}
    return evaluated

def main():
    spec=importlib.util.spec_from_file_location('cutlass_grid_shared',Path(__file__).with_name('benchmark-prefill-cutlass-grid.py'))
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
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
    helper.validate_manifest(manifest, native.configurations(), Path('/opt/apac/prefill-moe-tiles'), Path('/opt/sglang/source/python/sglang/srt/layers/quantization/fp8.py').read_text())
    ids = {entry['id']: entry['mode'] for entry in manifest['configurations']}
    selected = {name: ids[name] for name in CANDIDATE_IDS}
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
    result = {'schema': 'prefill-cutlass-confirm-probe/v1', 'manifest': manifest, 'actual_router_histogram': False, 'shuffle_on': True, 'objective': 'whole MoE graph CUDA-event median; down variant with up wheel fixed, not isolated GEMM', 'rounds': 20, 'replays': 100, 'cases': [], 'heldout': [], 'serving_adoption': False, 'image_sha256': image_sha, 'device': {'name': properties.name, 'uuid': str(getattr(properties, 'uuid', 'unavailable')), 'total_memory': properties.total_memory, 'sm_count': properties.multi_processor_count, 'visible_count': torch.cuda.device_count(), 'torch': torch.__version__, 'cuda': torch.version.cuda}}
    result['predeclared_gate'] = GATE
    result['candidate_modes'] = selected
    result['search_bounds'] = {'candidates': 2, 'controls': 2, 'training_shapes': 2, 'heldout_shapes': 2, 'rounds': 20, 'replays_per_measurement': 100, 'continuous_gpu_warmup_ms': 200, 'minimum_contiguous_graph_warmup_replays': 400, 'maximum_measurements': 320, 'maximum_timed_replays': 32000, 'termination': 'four shapes exhausted or unknown failure; external timeout720; no automatic adoption'}
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
        ids=(row*topk+lane)%e if distribution=='balanced' else torch.where(row%2==0,(row*17+lane*13)%e,(row+lane)%32) if distribution=='moderate' else torch.where(row%4==0,(row*17+lane*13)%e,(row+lane)%16)
        ids=ids.to(torch.int32).contiguous()
        weights=torch.softmax(torch.randn((tokens,topk),device='cuda'),-1)
        def run(modes):
            return moe.cutlass_fused_experts_fp8(x,w1.transpose(1,2),w2.transpose(1,2),s1.transpose(1,2),s2.transpose(1,2),weights,ids,*strides,workspace,*pointers,offsets,*problems,enable_es=(False,False),prefill_tile_modes=modes)
        reference=run((-1,-1)).clone()
        torch.cuda.synchronize()
        record={'tokens': tokens,'distribution': distribution,'seed': seed,'expert_counts': torch.bincount(ids.flatten().long(),minlength=e).cpu().tolist(),'variants': {},'utc_start_epoch':time.time()}
        destination.append(record)
        dense={}
        save()
        for round_index in range(20):
            for name in helper.rotated_order(list(variants),round_index):
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
                warm_start,warm_end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                warm_start.record()
                elapsed=0.0
                while elapsed<200:
                    for _ in range(400):
                        graph.replay()
                    warm_end.record()
                    warm_end.synchronize()
                    elapsed=warm_start.elapsed_time(warm_end)
                entry.setdefault('continuous_warmup_ms',[]).append(elapsed)
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100):
                    graph.replay()
                end.record();end.synchronize()
                entry['rounds_ms'].append(start.elapsed_time(end)/100)
                entry.setdefault('quality_rounds',[]).append({'eager':eager,'graph':q})
                entry.update(status='measured',eager_quality=eager,graph_quality=q,oracle_relative_l2=oracle,quality_eligible=entry.get('quality_eligible',True) and eager['relative_l2']<=1e-3 and q['relative_l2']<=1e-3)
                del graph,captured,actual
                gc.collect()
                save()
        record['utc_end_epoch']=time.time()
        result.pop('in_progress',None)
        save()
    variants={'wheel':((-1,-1),'control',-1),'stock':((0,0),'control',0)}
    for name,mode in selected.items():
        variants[name]=((-1,mode),'down',mode)
    for distribution in ['balanced','skewed']:
        case(8192,distribution,20261007,variants,result['cases'])
    case(4096,'skewed',20261008,variants,result['heldout'])
    case(8192,'moderate',20261009,variants,result['heldout'])
    result['confirmation']=evaluate(result['cases'],result['heldout'])
    save()
    print('Independent bounded confirmation completed; no automatic serving adoption.',flush=True)

if __name__=='__main__':
    main()
