// BF16 -> exact FP32 penalty -> full-vocabulary argmax. No GEMM change.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <climits>
#include <cmath>

#include "greedy_verify_math.cuh"
__device__ GreedyPair greedy_reduce(GreedyPair x) {
    __shared__ GreedyPair warps[8];
    for (int d=16; d; d/=2) x=greedy_better(x,{__shfl_down_sync(0xffffffff,x.value,d),__shfl_down_sync(0xffffffff,x.index,d)});
    if ((threadIdx.x&31)==0) warps[threadIdx.x/32]=x;
    __syncthreads();
    x=threadIdx.x<8 ? warps[threadIdx.x] : GreedyPair{-INFINITY,INT_MAX};
    if (threadIdx.x<32) for (int d=16; d; d/=2) x=greedy_better(x,{__shfl_down_sync(0xffffffff,x.value,d),__shfl_down_sync(0xffffffff,x.index,d)});
    return x;
}
__global__ void greedy_tiles(const __nv_bfloat16* logits, int stride, int vocab, const long* stops, int nstop, const int* minimum, const int* lengths, bool penalty, bool sanitize, GreedyPair* partial) {
    const int row=blockIdx.y, chunk=blockIdx.x, req=row/4;
    const bool active=penalty && lengths[req]<minimum[req];
    GreedyPair best{-INFINITY,INT_MAX};
    for (int k=0; k<16; ++k) {
        const int i=chunk*4096+k*256+threadIdx.x;
        if (i>=vocab) break;
        float x=__bfloat162float(logits[static_cast<long>(row)*stride+i]);
        bool stop=false;
        if (active) for (int s=0; s<nstop; ++s) stop |= stops[req*nstop+s]==i;
        // Same FP32 operation as zero accumulator + where, then logits.add_.
        x=greedy_value(x,sanitize,penalty,stop);
        best=greedy_better(best,{x,i});
    }
    best=greedy_reduce(best);
    if (threadIdx.x==0) partial[row*gridDim.x+chunk]=best;
}
__global__ void greedy_finish(const GreedyPair* partial, int chunks, long* output) {
    GreedyPair best{-INFINITY,INT_MAX};
    for (int c=threadIdx.x; c<chunks; c+=256) best=greedy_better(best,partial[blockIdx.x*chunks+c]);
    best=greedy_reduce(best);
    if (threadIdx.x==0) output[blockIdx.x]=best.index;
}
void greedy_verify_launch(const void* logits, int rows, int stride, int vocab, const long* stops, int nstop, const int* minimum, const int* lengths, bool penalty, bool sanitize, void* partial, long* output, cudaStream_t stream) {
    const int chunks=(vocab+4095)/4096;
    greedy_tiles<<<dim3(chunks,rows),256,0,stream>>>(static_cast<const __nv_bfloat16*>(logits),stride,vocab,stops,nstop,minimum,lengths,penalty,sanitize,static_cast<GreedyPair*>(partial));
    greedy_finish<<<rows,256,0,stream>>>(static_cast<const GreedyPair*>(partial),chunks,output);
}
