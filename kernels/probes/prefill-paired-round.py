"""Prepare exact BF16 paired-RN Prefill QKV source without importing CUDA."""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path

HEADER_PATH = 'python/sglang/kernels/jit/csrc/elementwise/rope.cuh'
OLD = '''                float a[2], b[2];
                #pragma unroll
                for(int h=0;h<2;++h) {
                    const int d=j*2+h;
                    const float u=__bfloat162float(__float2bfloat16_rn(x[d]*scale*wx[d]));
                    const float v=__bfloat162float(__float2bfloat16_rn(y[d]*scale*wy[d]));
                    const float c=reinterpret_cast<const float*>(cos4)[d], s=reinterpret_cast<const float*>(sin4)[d];
                    a[h]=__bfloat162float(__float2bfloat16_rn(u*c-v*s));
                    b[h]=__bfloat162float(__float2bfloat16_rn(u*s+v*c));
                }
                ox[j]=qkv_prefill_fp8(make_float2(a[0],a[1]));
                oy[j]=qkv_prefill_fp8(make_float2(b[0],b[1]));'''
HELPER = r'''__device__ __forceinline__ float2 qkv_prefill_paired_rn(float a, float b) {
    uint32_t bits;
    // Same operand order as CUDA13 cuda_bf16.hpp: low16=a, high16=b.
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(bits) : "f"(a), "f"(b));
    return make_float2(__uint_as_float(bits << 16), __uint_as_float(bits & 0xffff0000u));
}

'''
NEW = '''                // Pair adjacent BF16 RN-even conversions at the same two
                // rounding points; preserve each scalar expression and FMA.
                const float2 u=qkv_prefill_paired_rn(x[j*2]*scale*wx[j*2],x[j*2+1]*scale*wx[j*2+1]);
                const float2 v=qkv_prefill_paired_rn(y[j*2]*scale*wy[j*2],y[j*2+1]*scale*wy[j*2+1]);
                float a[2], b[2];
                #pragma unroll
                for(int h=0;h<2;++h) {
                    const int d=j*2+h;
                    const float uh=h==0?u.x:u.y, vh=h==0?v.x:v.y;
                    const float c=reinterpret_cast<const float*>(cos4)[d], s=reinterpret_cast<const float*>(sin4)[d];
                    a[h]=uh*c-vh*s;
                    b[h]=uh*s+vh*c;
                }
                const float2 ar=qkv_prefill_paired_rn(a[0],a[1]);
                const float2 br=qkv_prefill_paired_rn(b[0],b[1]);
                ox[j]=qkv_prefill_fp8(ar);
                oy[j]=qkv_prefill_fp8(br);'''


def paired_header(source):
    start = source.index('void qknorm_rope_prefill_fast_kernel(')
    end = source.index('// Decode: four independent H128 heads', start)
    kernel = source[start:end]
    if kernel.count(OLD) != 1:
        raise ValueError('Expected the exact current scalar BF16 rounding loop once')
    template_start = source.rfind("template<bool kUsePDL, typename IdType, bool kVImage=false>", 0, start)
    if template_start < 0:
        raise ValueError('Expected current Prefill template signature')
    return source[:template_start] + HELPER + source[template_start:start] + kernel.replace(OLD, NEW, 1) + source[end:]


def prepare(header, output):
    source = header.read_text()
    candidate = paired_header(source)
    output.mkdir(parents=True, exist_ok=True)
    headers = {'control': output/'rope-control.cuh', 'paired': output/'rope-paired.cuh'}
    headers['control'].write_text(source)
    headers['paired'].write_text(candidate)
    patch = ''.join(difflib.unified_diff(source.splitlines(keepends=True), candidate.splitlines(keepends=True), fromfile='a/'+HEADER_PATH, tofile='b/'+HEADER_PATH))
    (output/'prefill-qkv-paired-round.patch').write_text(patch)
    metadata = {'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'headers': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name,path in headers.items()}, 'gpu_executed':False, 'mechanism':'same BM32/head1 mapping, pair BF16 RN conversions only'}
    (output/'preparation.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return headers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--header', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args=parser.parse_args()
    headers = prepare(args.header,args.output)
    if args.run:
        spec = importlib.util.spec_from_file_location('prefill_memory', Path(__file__).with_name('prefill-memory.py'))
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        probe.run(headers, args.output)
    else:
        print('PASS CPU paired-round source preparation; no GPU execution')
