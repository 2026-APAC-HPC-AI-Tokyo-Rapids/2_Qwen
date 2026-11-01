"""Separate, bounded Decode g4:4-7 gate; failures keep the control env."""
import argparse
import hashlib
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = 'firmus-decode-greedy/v1'
KEY = 'SGLANG_DECODE_GREEDY_VERIFY'
BATCHES = (1,32,44,46,48)
STATES = ('initial','boundary','changed','special','restored')
REPLAYS, BLOCKS, ROUNDS, REPS = 10000, 3, 5, 16
WORKER_SECONDS, WALL_SECONDS = 85, 90


def read(path):
    return json.loads(path.read_text())


def expected_cases():
    return {f'{b}:{int(pad)}' for b in BATCHES for pad in (False,True)}


def check_record(record, run_id, expected, image_sha256, deadline):
    assert record['schema']==SCHEMA and record['run_id']==run_id, 'stale record'
    assert record['uuid'] in expected and 'H200' in record['device'], 'wrong Decode GPU'
    assert record['image_sha256']==image_sha256, 'wrong image'
    assert record['passed'] is True, record.get('error','worker failed')
    assert 0<record['elapsed_seconds']<WORKER_SECONDS and record['finished_at']<=deadline, 'late worker'
    assert record['replays']>=REPLAYS and record['endurance_equal'] is True, 'endurance'
    assert record['sanitize_modes']==[False,True], 'missing sanitize modes'
    assert set(record['cases'])==expected_cases(), 'incomplete cases'
    for case in record['cases'].values():
        assert case['states']==list(STATES) and case['equal'] is True and case['no_penalty_equal'] is True, 'bit mismatch'
    assert set(record['timings_ms'])=={f'{b}:{p}' for b in (44,46,48) for p in (0,1)}, 'incomplete timing shapes'
    for blocks in record['timings_ms'].values():
        assert len(blocks)==BLOCKS, 'timing blocks'
        for block in blocks:
            for order in ('forward','reverse'):
                samples=block[order]
                for mode in ('control','candidate'):
                    assert len(samples[mode])==ROUNDS and all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in samples[mode]), 'invalid timing'
                assert statistics.median(samples['control'])-statistics.median(samples['candidate'])>=.05, 'below 0.05 ms saving'


def aggregate(directory, env_output, run_id, image_sha256, deadline, workers_ok):
    result={'schema':SCHEMA,'run_id':run_id,'adopted':False,'gpus':[]}
    try:
        assert workers_ok and time.time()<=deadline, 'workers failed or timeout'
        inventory=read(directory/'inventory.json')
        assert inventory['run_id']==run_id and inventory['host']=='g4', 'wrong inventory'
        assert set(inventory['gpus'])=={'4','5','6','7'}, 'wrong Decode indices'
        expected=set(inventory['gpus'].values())
        assert len(expected)==4, 'duplicate inventory UUID'
        seen=set(); binaries=set()
        for rank in range(4):
            record=read(directory/f'rank-{rank}.json')
            check_record(record,run_id,expected,image_sha256,deadline)
            assert record['rank']==rank and record['uuid'] not in seen, 'duplicate/rank mismatch'
            assert record['uuid']==inventory['gpus'][str(rank+4)], 'wrong rank UUID'
            seen.add(record['uuid']); binaries.add(record['binary_sha256'])
            result['gpus'].append(record)
        assert seen==expected and len(binaries)==1 and len(next(iter(binaries)))==64, 'incomplete GPU/binary set'
        lines=env_output.read_text().splitlines(keepends=True)
        assert [l.strip() for l in lines if l.startswith(KEY+'=')]==[KEY+'=0'], 'control env required'
        tmp=env_output.with_suffix('.greedy.tmp')
        try:
            tmp.write_text(''.join(KEY+'=1\n' if l.startswith(KEY+'=') else l for l in lines))
            assert time.time()<=deadline, 'aggregate timeout'
            tmp.replace(env_output)
        finally:
            tmp.unlink(missing_ok=True)
        result.update(adopted=True,reason='all four Decode GPUs passed')
    except Exception as error:  # noqa: BLE001 - refusal is a measured gate outcome; control stays
        result['reason']=f'{type(error).__name__}: {error}'
    return result


def inventory(output, run_id):
    assert socket.gethostname().split('.')[0]=='g4', 'Decode gate requires g4'
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    gpus={i.strip():u.strip().removeprefix('GPU-') for i,u in (line.split(',') for line in rows.strip().splitlines()) if i.strip() in ('4','5','6','7')}
    output.mkdir(parents=True,exist_ok=True)
    (output/'inventory.json').write_text(json.dumps({'run_id':run_id,'host':'g4','gpus':gpus})+'\n')


def capture(torch, step):
    step(); torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): step()
    return graph


class Case:
    """Exercise actual update_penalties, logits buffer and eagle_sample on both paths."""
    def __init__(self, torch, b, padding):
        from types import SimpleNamespace as NS

        from sglang.srt.layers.logits_processor import (
            LogitsProcessor,
            LogitsProcessorOutput,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.sampling.penaltylib.min_new_tokens import (
            BatchedMinNewTokensPenalizer,
        )
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
        from sglang.srt.speculative import greedy_verify
        from sglang.srt.speculative.eagle_utils import eagle_sample
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        self.torch=torch; self.b=b; self.padding=padding
        # Graph bucket has one padded request. Only live rows reach sampling,
        # exactly as DecodeCudaGraphRunner replay narrows the output.
        bucket=b+int(padding)
        self.logits=torch.empty((bucket*4,152064+64*int(padding)),dtype=torch.bfloat16,device='cuda')[:,:152064]
        self.minimum=torch.full((b,1),2,dtype=torch.int32,device='cuda')
        self.lengths=torch.empty_like(self.minimum)
        self.stops=torch.tensor([[0,151643,151645,152064]]*b,dtype=torch.int64,device='cuda')
        self.penalty=BatchedMinNewTokensPenalizer.__new__(BatchedMinNewTokensPenalizer)
        self.penalty._is_prepared=True; self.penalty.greedy_stop_ids=self.stops
        self.penalty.min_new_tokens=self.minimum; self.penalty.len_output_tokens=self.lengths
        self.stop_storage=torch.zeros((b,152065),device='cuda')
        self.orchestrator=NS(is_required=True,penalizers={BatchedMinNewTokensPenalizer:self.penalty},accumulate_additive_penalties=lambda x:self.penalty.apply(x),accumulate_scaling_penalties=lambda:None)
        self.info=NS(temperatures=self.minimum,is_all_greedy=True,vocab_size=152064,penalizer_orchestrator=self.orchestrator,acc_additive_penalties=None,acc_scaling_penalties=None,logit_bias=None,greedy_minimum=None)
        self.batch=NS(device='cuda',seq_lens=torch.ones(b,dtype=torch.int32,device='cuda'),sampling_info=self.info,spec_algorithm=SpeculativeAlgorithm.EAGLE3,forward_mode=ForwardMode.TARGET_VERIFY,return_logprob=False)
        indices=torch.arange(b*4,dtype=torch.int64,device='cuda').reshape(b,4)
        self.verify=NS(draft_token_num=4,max_tree_depth=4,tree_topk=1,draft_token=torch.zeros((b,4),dtype=torch.int64,device='cuda'),retrieve_index=indices,retrieve_next_token=torch.tensor([[1,2,3,-1]]*b,dtype=torch.int64,device='cuda'),retrieve_next_sibling=torch.full((b,4),-1,dtype=torch.int64,device='cuda'))
        self.buffers={mode:torch.empty((bucket*4,152064),dtype=dtype,device='cuda') for mode,dtype in (('control',torch.float32),('candidate',torch.bfloat16))}
        self.processor=NS(vocab_size=152064,final_logit_softcapping=None)
        self.outputs={}; self.tokens={}; self.graphs={}
        self.update=SamplingBatchInfo.update_penalties; self.copy=LogitsProcessor._copy_logits_to_buffer
        self.sample=eagle_sample; self.output_class=LogitsProcessorOutput; self.greedy=greedy_verify
        self.NS=NS
        self.set_state('initial')

    def set_state(self, state):
        t=self.torch; b=self.b
        self.penalty._is_prepared=state!='no_penalty'
        self.orchestrator.is_required=state!='no_penalty'
        self.logits.fill_(-2)
        self.minimum.fill_(2); self.stops.copy_(t.tensor([[0,151643,151645,152064]]*b,device='cuda'))
        self.lengths.fill_(1 if state in ('initial','restored') else 2 if state=='boundary' else 3)
        # Cover NaNs at both ends, +/-Inf, all equal, signed zeros and ties
        # crossing both warp and 4096-column tile boundaries.
        live=self.logits[:b*4]
        live[::4,0]=float('nan'); live[::4,-1]=float('nan')
        live[1::4,151645]=float('inf'); live[1::4,4096]=float('-inf')
        live[2::4].fill_(0)
        live[3::4,4095]=8; live[3::4,4096]=8; live[3::4,-1]=8
        if state=='special':
            live[::4].fill_(float('-inf')); live[1::4].fill_(float('nan'))
            live[2::4].fill_(0); live[2::4,17]=2.0**-133
            live[3::4].fill_(0); live[3::4,0]=-0.0
        if state=='changed':
            live[::4,0]=10; live[::4,-1]=10
            live.fill_(-2); live[:,4095]=10
            self.stops[:,0]=4096; self.minimum[::2]=4
        self.stop_storage.zero_().scatter_add_(1,self.stops,t.full_like(self.stops,float('-inf'),dtype=t.float32))
        self.penalty.stop_token_penalties=self.stop_storage[:,:152064]
        # Vary accepted depth, from mismatch to a full chain and bonus.
        self.verify.draft_token.fill_(0)
        self.verify.draft_token[:,1:]=4095 if state=='changed' else 0

    def step(self, mode):
        from unittest.mock import patch
        t=self.torch
        # Recording references adds no GPU work and leaves production calls intact.
        argmax=t.argmax; target=self.greedy.target_argmax
        def record_argmax(*args,**kwargs):
            out=argmax(*args,**kwargs); self.tokens[mode]=out; return out
        def record_target(*args,**kwargs):
            out=target(*args,**kwargs); self.tokens[mode]=out; return out
        self.update(self.info)
        metadata=self.NS(forward_mode=self.batch.forward_mode,next_token_logits_buffer=self.buffers[mode])
        logits=self.copy(self.processor,self.logits,metadata)[:self.b*4]
        with patch.object(t,'argmax',record_argmax), patch.object(self.greedy,'target_argmax',record_target):
            predict,length,index=self.sample(self.verify,self.batch,self.output_class(next_token_logits=logits))
        bonus=predict[index[t.arange(self.b,device='cuda'),length-1]]
        self.outputs[mode]=(self.tokens[mode],index,length,bonus)

    def build_graphs(self, sanitize):
        os.environ['SGLANG_SANITIZE_NAN_LOGITS']='1' if sanitize else '0'
        for mode in ('control','candidate'):
            os.environ[KEY]='1' if mode=='candidate' else '0'
            self.graphs[mode]=capture(self.torch,lambda mode=mode:self.step(mode))

    def equal(self):
        for graph in self.graphs.values(): graph.replay()
        self.torch.cuda.synchronize()
        assert all(self.torch.equal(a,b) for a,b in zip(self.outputs['control'],self.outputs['candidate'])), 'argmax/index/length/bonus mismatch'

    def timing(self):
        t=self.torch; blocks=[]
        for _ in range(BLOCKS):
            block={}
            for order,modes in (('forward',('control','candidate')),('reverse',('candidate','control'))):
                samples={m:[] for m in modes}
                for _ in range(ROUNDS):
                    for mode in modes:
                        start,end=t.cuda.Event(enable_timing=True),t.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(REPS): self.graphs[mode].replay()
                        end.record(); end.synchronize()
                        samples[mode].append(start.elapsed_time(end)/REPS)
                block[order]=samples
            blocks.append(block)
        return blocks


def run(output, run_id, image_sha256, deadline):
    started=time.monotonic(); rank=int(os.environ['LOCAL_RANK'])
    result={'schema':SCHEMA,'run_id':run_id,'rank':rank,'passed':False,'image_sha256':image_sha256,'cases':{},'timings_ms':{},'sanitize_modes':[False,True]}
    try:
        import torch
        assert int(os.environ['WORLD_SIZE'])==4
        torch.set_num_threads(1); torch.cuda.set_device(rank)
        result.update(uuid=str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-'),device=torch.cuda.get_device_name(rank))
        identities=read(output/'inventory.json')
        assert identities['run_id']==run_id and result['uuid']==identities['gpus'][str(rank+4)], 'wrong physical Decode GPU'
        binary=Path('/opt/apac/fa3-verify/libapac_fa3_verify.so')
        with binary.open('rb') as f: result['binary_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
        assert result['binary_sha256']==read(binary.with_name('manifest.json'))['binary_sha256']
        for b in BATCHES:
            for pad in (False,True):
                case=Case(torch,b,pad)
                for sanitize in (False,True):
                    case.set_state('initial'); case.build_graphs(sanitize)
                    initial=None
                    for state in STATES:
                        assert time.time()<deadline, 'deadline'
                        case.set_state(state); case.equal()
                        if state=='initial': initial=tuple(x.clone() for x in case.outputs['candidate'])
                        if state=='restored': assert all(torch.equal(x,y) for x,y in zip(initial,case.outputs['candidate'])), 'initial input not restored'
                    if not sanitize and b in (44,46,48):
                        # Finite, realistic logits for performance (not pathological NaNs).
                        case.logits.normal_(); case.lengths.fill_(1)
                        result['timings_ms'][f'{b}:{int(pad)}']=case.timing()
                    case.set_state('no_penalty'); case.build_graphs(sanitize); case.equal()
                result['cases'][f'{b}:{int(pad)}']={'equal':True,'states':list(STATES),'no_penalty_equal':True}
        case.set_state('initial'); case.build_graphs(True); case.equal()
        failed=torch.zeros((),dtype=torch.bool,device='cuda')
        def endurance_step():
            case.graphs['candidate'].replay()
            for left,right in zip(case.outputs['control'],case.outputs['candidate']): failed.logical_or_(torch.any(left!=right))
        endurance=capture(torch,endurance_step)
        failed.zero_()
        for _ in range(REPLAYS): endurance.replay()
        torch.cuda.synchronize()
        assert not failed.item(), 'transient endurance mismatch'
        assert all(torch.equal(a,b) for a,b in zip(case.outputs['control'],case.outputs['candidate'])), 'endurance mismatch'
        result.update(replays=REPLAYS,endurance_equal=True)
        result.update(passed=True,elapsed_seconds=time.monotonic()-started,finished_at=time.time())
        check_record(result,run_id,set(identities['gpus'].values()),image_sha256,deadline)
    except Exception as error:  # noqa: BLE001 - any rank failure is recorded and refuses the gate
        result.update(passed=False,error=f'{type(error).__name__}: {error}')
    result.update(elapsed_seconds=time.monotonic()-started,finished_at=time.time())
    tmp=output/f'rank-{rank}.tmp'; tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); tmp.replace(output/f'rank-{rank}.json')
    print(json.dumps(result,allow_nan=False),flush=True)
    if not result['passed']: raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--run-id',required=True)
    p.add_argument('--image-sha256',required=True); p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--launch',action='store_true'); p.add_argument('--aggregate',type=Path)
    p.add_argument('--env-output',type=Path); p.add_argument('--workers-ok',type=int,default=0)
    a=p.parse_args()
    if a.aggregate:
        a.output.write_text(json.dumps(aggregate(a.aggregate,a.env_output,a.run_id,a.image_sha256,a.deadline,bool(a.workers_ok)),indent=2)+'\n')
    elif a.launch:
        inventory(a.output,a.run_id)
        os.environ['CUDA_VISIBLE_DEVICES']=','.join('GPU-'+read(a.output/'inventory.json')['gpus'][str(i)] for i in range(4,8))
        subprocess.run([str(Path(sys.executable).with_name('torchrun')),'--standalone','--nproc-per-node=4',__file__,'--output',str(a.output),'--run-id',a.run_id,'--image-sha256',a.image_sha256,'--deadline',str(a.deadline)],check=True)
    else:
        run(a.output,a.run_id,a.image_sha256,a.deadline)


if __name__=='__main__': main()
