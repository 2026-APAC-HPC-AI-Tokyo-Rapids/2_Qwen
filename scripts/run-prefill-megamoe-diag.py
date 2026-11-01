"""Diagnostic only. One EP1 process on each of the twelve Prefill GPUs; no env output."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

TOKENS, HIDDEN, INTERMEDIATE, EXPERTS, TOPK = 7988, 4096, 1536, 128, 8
MAGIC, CTAS, WARPS, CAPACITY = 0x18314D4F45444941, 132, 12, 2048
WORDS = 32 + CAPACITY * 8
EXTRA_BYTES = 4096 + CTAS * WARPS * WORDS * 8


def routing_cases(np, fixture):
    rng = np.random.default_rng(181101)
    # Exactly balanced apart from the unavoidable remainder, eight distinct experts/token.
    balanced = (np.arange(TOKENS)[:, None]*8 + np.arange(8)[None, :]) % EXPERTS
    data = np.load(fixture)
    # Two historical 4001-token microbatches on the same old EP rank. Trim each
    # to 3994 rather than calling the 8002-row capture current serving routing.
    captured = data['logical_ids'][:, 0, 0, :3994, :].reshape(TOKENS, TOPK).astype(np.int32)
    edges = np.argsort(rng.random((TOKENS, EXPERTS-16)), axis=1)[:, -TOPK:] + 16
    # Explicitly include empty expert 0 and 1/64/65/127/128/129-row experts.
    offset = 0
    for expert, count in enumerate((1, 64, 65, 127, 128, 129), start=1):
        edges[offset:offset+count, -1] = expert
        offset += count
    return {'balanced':balanced.astype(np.int32), 'historical-routing':captured, 'edges':edges.astype(np.int32)}


def parse_trace(np, raw):
    """raw starts at align_up(buffer.data_ptr() + host_size, 4096)."""
    assert raw.size >= CTAS*WARPS*WORDS, 'truncated trace buffer'
    assert int(raw[0]) == MAGIC, 'trace header missing at private tail'
    words = raw[:CTAS*WARPS*WORDS].reshape(CTAS, WARPS, WORDS)
    assert np.all(words[:, :, 0] == MAGIC), 'missing warp/CTA records'
    assert np.all(words[:, :, 3] == TOKENS)
    assert not np.any(words[:, :, 4]), 'trace overflow; never truncate silently'
    records = []
    for cta in range(CTAS):
        for warp in range(WARPS):
            row = words[cta, warp]
            count = int(row[1])
            assert 2 <= count <= CAPACITY
            events = row[32:32+count*8].reshape(count, 8).astype(object).tolist()
            assert events[0][1] == 1 and events[-1][1] == 15
            records.append({'cta':cta, 'warp':warp, 'sm':int(row[2]), 'events':events, 'wait_ns':{'fc1_full':int(row[16]), 'fc1_empty':int(row[17]), 'fc2_full':int(row[20]), 'fc2_empty':int(row[21])}, 'wait_count':{'fc1_full':int(row[24]), 'fc1_empty':int(row[25]), 'fc2_full':int(row[28]), 'fc2_empty':int(row[29])}})
    return records


def trace_layout(base, size, allocation_bytes):
    address = (base + size + 4095)//4096*4096
    offset = address-base
    trace_bytes = CTAS*WARPS*WORDS*8
    assert offset % 8 == 0 and offset+trace_bytes <= allocation_bytes
    return {'buffer_address':base, 'buffer_bytes':allocation_bytes, 'host_size_bytes':size, 'combine_end_address':base+size, 'trace_address':address, 'trace_offset_bytes':offset, 'trace_bytes':trace_bytes, 'page_bytes':4096, 'legacy_read_offset_bytes':max(0,size-4096)//4096*4096}


def tensor_info(tensor):
    return {'address':tensor.data_ptr(), 'shape':list(tensor.shape), 'stride':list(tensor.stride()), 'dtype':str(tensor.dtype), 'bytes':tensor.numel()*tensor.element_size()}


def jit_evidence(template):
    """Read the compiler's artifacts, without changing its cache or enabling profiling."""
    root = Path(os.environ.get('DG_JIT_CACHE_DIR', str(Path.home()/'.deep_gemm')))/'cache'
    result = {'cache':str(root), 'kernels':[]}
    try:
        for source in sorted(root.glob('kernel.sm90_fp8_mega_moe.*/kernel.cu')):
            code = source.read_text()
            match = re.search(r'&(sm90_fp8_mega_moe(?:_diag)?_impl)\s*<([\s\S]*?)>', code)
            if not match:
                continue
            entry = {'source':str(source), 'source_text':code, 'kernel_name':match[1], 'cubin':str(source.with_suffix('.cubin'))}
            entry['matches_build_template'] = ''.join(template.split()) in ''.join(code.split())
            result['kernels'].append(entry)
            # Same symbol inventory used by KernelRuntime when loading this CUBIN.
            if not entry['matches_build_template']:
                continue
            try:
                symbols = subprocess.run([str(Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))/'bin/cuobjdump'), '-symbols', str(source.with_suffix('.cubin'))], capture_output=True, text=True, timeout=1, check=False)
                entry.update(symbols=symbols.stdout, symbols_stderr=symbols.stderr, symbols_exit=symbols.returncode)
            except (OSError, subprocess.TimeoutExpired) as exc:
                entry['symbols_error'] = str(exc)
    except OSError as exc:
        result['error'] = str(exc)
    return result


def record_trace(np, raw, case, host, save):
    # Save bytes and host arguments before parsing, including on success. uint64
    # view preserves the device's bytes; hex is never a formatted integer dump.
    case['trace_buffer'] = {'head_hex':raw[:512].tobytes().hex(), 'head_bytes':min(raw.nbytes,4096), 'dtype':str(raw.dtype), 'magic_hex_le':MAGIC.to_bytes(8,'little').hex(), 'magic_words_first_1024':np.flatnonzero(raw[:1024] == MAGIC).tolist(), 'host':host}
    save()
    try:
        case['trace'] = parse_trace(np, raw)
    except (AssertionError, ValueError, IndexError) as exc:
        case['trace_error'] = f'{type(exc).__name__}: {exc}'
        save()
        case['trace_buffer']['jit'] = jit_evidence(host['template_build'])


def collect_cases(record, cases, measure, capture, save):
    """A parser failure is case-local and must not fail torchrun's other ranks."""
    for name, ids in cases.items():
        record['active_case'] = name
        case = record['cases'][name] = {'samples_us':[], 'byte_equal':False}
        save()
        measure(ids, case)
        save()  # Complete times and byte oracle survive a later copy/parse failure.
        capture(case)
        save()
    record.pop('active_case', None)
    record['passed'] = all(case['byte_equal'] and case.get('finite') is True and 'trace' in case and 'trace_error' not in case for case in record['cases'].values())


def run(output, fixture, run_id):
    started = time.monotonic()
    rank = int(os.environ['LOCAL_RANK'])
    output.mkdir(parents=True, exist_ok=True)
    record = {'schema':'prefill-megamoe-diag/v1', 'run_id':run_id, 'rank':rank, 'tokens':TOKENS, 'cases':{}, 'passed':False, 'stage':'imports'}

    def save():
        record['elapsed_seconds'] = time.monotonic()-started
        (output/f'rank-{rank}.json').write_text(json.dumps(record)+'\n')

    save()
    try:
        run_rank(record, fixture, rank, save)
    except Exception:  # noqa: BLE001 - diagnostic ranks must not terminate their peers.
        # This is diagnostic-only: report failure in the fragment, return normally
        # so elastic does not SIGTERM independent EP1 ranks on the same worker.
        record['error'] = traceback.format_exc()
        record['passed'] = False
        traceback.print_exc()
    finally:
        save()


def run_rank(record, fixture, rank, save):
    import deep_gemm
    import numpy as np
    import torch
    import torch.distributed as dist

    record.update(uuid=str(torch.cuda.get_device_properties(rank).uuid), fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(), routing_is_current_serving=False, build=json.loads(Path('/opt/apac/prefill-megamoe-diag-build.json').read_text()), stage='setup')
    save()
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (9, 0)
    assert torch.cuda.get_device_properties(rank).multi_processor_count == CTAS
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
    assert dist.get_world_size() == 4
    singles = [dist.new_group([i]) for i in range(4)]
    group = singles[rank]
    os.environ.update(DG_SM90_PREFILL_2WG='1', DG_SM90_PREFILL_LOCAL_EP1='1', DG_SM90_PREFILL_4STAGE='1', DG_SM90_PREFILL_LEAN='1', DG_SM90_PREFILL_SPLIT_ACCUM='1', DG_SM90_PREFILL_TAIL_PAIR='0')
    device = torch.device('cuda', rank)
    generator = torch.Generator(device=device).manual_seed(181101)
    alignment = deep_gemm._C.get_token_alignment_for_mega_moe()
    cap = (8192+alignment-1)//alignment*alignment
    assert cap == 8448, ('unexpected serving capacity', cap)
    size, slicer = deep_gemm._C.get_symm_buffer_size_for_sm90_mega_moe(1, EXPERTS, cap, TOPK, HIDDEN, INTERMEDIATE, True, 'swiglu')
    # Same EP1 layout/addresses for both variants; only the diagnostic uses the extra tail.
    buffer = torch.zeros(size+EXTRA_BYTES, dtype=torch.int8, device=device)
    layout = trace_layout(buffer.data_ptr(), size, buffer.numel())
    record.update(capacity=cap, trace_layout=layout)
    save()
    views = [deep_gemm._from_dlpack_if_needed(v, dtype) for v, dtype in zip(slicer(buffer), (torch.float8_e4m3fn,None,None,None,torch.float8_e4m3fn,None,torch.float8_e4m3fn,None), strict=True)]
    buf = SimpleNamespace(buffer=buffer, group=group, handle=SimpleNamespace(buffer_ptrs=[buffer.data_ptr()]), num_max_tokens_per_rank=cap, num_experts=EXPERTS, num_topk=TOPK)
    w13 = (torch.randn(EXPERTS, 2*INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=device, generator=generator)*4).to(torch.float8_e4m3fn)
    w2 = (torch.randn(EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, device=device, generator=generator)*4).to(torch.float8_e4m3fn)
    s13 = torch.rand(EXPERTS, 2*INTERMEDIATE//128, HIDDEN//128, device=device, generator=generator)*.005+.005
    s2 = torch.rand(EXPERTS, HIDDEN//128, INTERMEDIATE//128, device=device, generator=generator)*.005+.005
    l1, l2 = deep_gemm.transform_weights_for_mega_moe_sm90((w13,s13),(w2,s2))
    del w13, w2
    hidden = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device=device, generator=generator)*.25
    weights = torch.rand(TOKENS, TOPK, device=device, generator=generator)
    weights = weights/weights.sum(-1, keepdim=True)
    outputs = {name:torch.empty_like(hidden) for name in ('control','diagnostic')}
    host = {**layout, 'api':'deep_gemm._C.fp8_mega_moe_diag', 'kernel_name':'deep_gemm::sm90_fp8_mega_moe_diag_impl', 'template_build':record['build']['template'], 'kernel_name_provenance':'explicit diagnostic API source; failure JIT artifacts include compiled symbols', 'arguments':{'y':tensor_info(outputs['diagnostic']), 'l1_weights':list(map(tensor_info,l1)), 'l2_weights':list(map(tensor_info,l2)), 'activation_views':list(map(tensor_info,views)), 'sym_buffer_ptrs':[buffer.data_ptr()], 'rank_idx':0, 'num_ranks':1, 'num_tokens':TOKENS, 'num_max_tokens_per_rank':cap, 'num_experts':EXPERTS, 'num_topk':TOPK, 'recipe':[128,128,128], 'activation':'swiglu', 'activation_clamp':None, 'fast_math':True, 'cumulative_local_expert_recv_stats':None}, 'launch':{'grid':[CTAS,1,1], 'block':[WARPS*32,1,1], 'stream':torch.cuda.current_stream().cuda_stream}, 'prefill_environment':{key:os.environ[key] for key in ('DG_SM90_PREFILL_2WG','DG_SM90_PREFILL_LOCAL_EP1','DG_SM90_PREFILL_4STAGE','DG_SM90_PREFILL_LEAN','DG_SM90_PREFILL_SPLIT_ACCUM','DG_SM90_PREFILL_TAIL_PAIR')}}

    def call(mode):
        if mode == 'control':
            deep_gemm.fp8_mega_moe(outputs[mode], l1, l2, buf, recipe=(128,128,128), activation='swiglu', fast_math=True)
        else:
            deep_gemm._C.fp8_mega_moe_diag(outputs[mode], *l1, *l2, None, buffer, [buffer.data_ptr()], 0, cap, EXPERTS, TOPK, (128,128,128), 'swiglu', None, True)

    def measure(cpu_ids, case):
        ids = torch.as_tensor(cpu_ids, dtype=torch.int32, device=device)
        deep_gemm.mega_moe_pre_dispatch_sm90(hidden, ids, weights, *views[:4], num_tokens=TOKENS, group_size=128, routed_scaling_factor=1.0)
        for _ in range(2):
            for mode in ('control','diagnostic'):
                call(mode)
        torch.cuda.synchronize()
        case['expert_rows'] = np.bincount(cpu_ids.ravel(), minlength=EXPERTS).tolist()
        samples = case['samples_us']
        equal, finite = [], []
        for iteration in range(6):
            order = ('control','diagnostic') if iteration % 2 == 0 else ('diagnostic','control')
            pair = {'order':list(order)}
            for mode in order:
                # Like the existing MegaMoE gate, keep CPU launch latency outside the span.
                torch.cuda._sleep(1_000_000)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(); call(mode); end.record(); end.synchronize()
                pair[mode] = begin.elapsed_time(end)*1000
            equal.append(torch.equal(outputs['control'].view(torch.uint8), outputs['diagnostic'].view(torch.uint8)))
            finite.append(bool(torch.isfinite(outputs['control']).all()))
            samples.append(pair)
        case.update(byte_equal=all(equal), finite=all(finite), byte_equal_pairs=equal, finite_pairs=finite)

    def capture(case):
        # Capture one call per routing outside timing, then wait for ALL stores.
        call('diagnostic'); torch.cuda.synchronize()
        raw = buffer[layout['trace_offset_bytes']:layout['trace_offset_bytes']+layout['trace_bytes']].view(torch.int64).cpu().numpy().view(np.uint64)
        record_trace(np, raw, case, host, save)

    try:
        record['stage'] = 'cases'
        collect_cases(record, routing_cases(np, fixture), measure, capture, save)
        record['stage'] = 'done'
    finally:
        save()
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    a = p.parse_args()
    run(a.output, a.fixture, a.run_id)


if __name__ == '__main__':
    main()
