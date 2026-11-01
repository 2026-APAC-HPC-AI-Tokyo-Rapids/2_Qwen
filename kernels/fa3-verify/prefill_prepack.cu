// Production pack: preserve FP8 bytes, stage contiguous reads, gather the
// inverse physical map in shared memory, and issue aligned coalesced stores.
// The independent GPU oracle remains prefill_native_image.cu (LDSM/PRMT/STSM).
#include <cuda_runtime.h>
#include <cstdint>
namespace apac_fa3_verify {
__host__ __device__ inline int prefill_v_logical_offset(int physical) {
    int x=0;
    x |= ((physical>>8)&1)<<0;
    x |= ((physical>>5)&1)<<1;
    x |= ((physical>>6)&1)<<2;
    x |= ((physical>>7)&1)<<3;
    x |= ((physical>>9)&1)<<4;
    x |= ((physical>>10)&1)<<5;
    x |= ((physical>>11)&1)<<6;
    x |= ((physical>>0)&1)<<7;
    x |= ((physical>>2)&1)<<8;
    x |= ((physical>>3)&1)<<9;
    x |= ((physical>>1)&1)<<10;
    x |= (((physical>>4)^(physical>>7))&1)<<11;
    x |= physical&12288;
    return x;
}
__global__ void prefill_prepack_kernel(const uint8_t* __restrict__ v, uint8_t* __restrict__ image, const int* __restrict__ cu) {
    __shared__ __align__(16) uint8_t contiguous[96*128], native[96*128];
    const int start=cu[blockIdx.z], length=cu[blockIdx.z+1]-start;
    #pragma unroll
    for(int i=0;i<6;++i) {
        int x=(i*128+threadIdx.x)*16, local=blockIdx.x*96+x/128;
        uint4 value{};
        if(local<length) value=*reinterpret_cast<const uint4*>(v+((start+local)*4+blockIdx.y)*128+x%128);
        *reinterpret_cast<uint4*>(contiguous+x)=value;
    }
    __syncthreads();
    #pragma unroll
    for(int i=0;i<6;++i) {
        int x=(i*128+threadIdx.x)*16;
        const int logical=prefill_v_logical_offset(x);
        uint4 value;
        #pragma unroll
        for(int b=0;b<16;++b) reinterpret_cast<uint8_t*>(&value)[b]=contiguous[logical+(b&1)*128+(b&2)*512+(b&4)*64+(b&8)*64];
        *reinterpret_cast<uint4*>(native+x)=value;
    }
    __syncthreads();
    auto dst=image+((blockIdx.z*4+blockIdx.y)*gridDim.x+blockIdx.x)*96*128;
    #pragma unroll
    for(int i=0;i<6;++i) {
        int x=(i*128+threadIdx.x)*16;
        *reinterpret_cast<uint4*>(dst+x)=*reinterpret_cast<const uint4*>(native+x);
    }
}
void prefill_prepack_launch(const void* v,void* image,const int* cu,int blocks,cudaStream_t stream) {
    prefill_prepack_kernel<<<dim3(blocks,4,2),128,0,stream>>>(static_cast<const uint8_t*>(v),static_cast<uint8_t*>(image),cu);
}
}
