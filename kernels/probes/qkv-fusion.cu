// Experimental epilogue only: the GEMM body is generated from the pinned image.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>

struct QKVParams {
    const __nv_bfloat16 *qw, *kw;
    const float* trig;
    const int64_t *pos, *loc;
    uint8_t *q, *k, *v, *kc, *vc;
};

template<int BM, int THREADS>
__device__ __forceinline__ void qkv_epilogue(const __nv_bfloat16* tile, int first_row, int head, int rows, const QKVParams& p) {
    const int lane = threadIdx.x % 8;
    // Same eight-lane/head reduction tree as the deployed SGLang prologue.
    for (int row = threadIdx.x / 8; row < BM; row += THREADS / 8) {
        const int token = first_row + row;
        float x[8], y[8], partial[4];
        uint16_t ox[4], oy[4];
        const int col = lane * 8;
        const int offset = row * 64 + (col ^ ((row % 8) * 8));
        const uint4 vx = *reinterpret_cast<const uint4*>(tile + offset);
        const uint4 vy = *reinterpret_cast<const uint4*>(tile + BM * 64 + offset);
        #pragma unroll
        for (int i=0;i<8;++i) {
            x[i] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&vx)[i]);
            y[i] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&vy)[i]);
        }
        if (head < 68) {
            #pragma unroll
            for (int j=0;j<4;++j) {
                float sum=0.f;
                sum += x[j*2]*x[j*2]; sum += x[j*2+1]*x[j*2+1];
                sum += y[j*2]*y[j*2]; sum += y[j*2+1]*y[j*2+1];
                partial[j]=sum;
            }
            #pragma unroll
            for (int mask=4;mask>=1;mask>>=1) {
                #pragma unroll
                for(int j=0;j<4;++j) partial[j] += __shfl_xor_sync(0xffffffff,partial[j],mask,8);
            }
            const float sum02=partial[0]+partial[2], sum13=partial[1]+partial[3];
            const float scale=rsqrtf((sum02+sum13)/128.f+1e-6f);
            const auto w=head<64?p.qw:p.kw;
            const int64_t pos=token<rows?p.pos[token]:0;
            #pragma unroll
            for(int j=0;j<4;++j) {
                float a[2], b[2];
                #pragma unroll
                for(int h=0;h<2;++h) {
                    const int d=col+j*2+h;
                    const float u=__bfloat162float(__float2bfloat16_rn(x[j*2+h]*scale*__bfloat162float(w[d])));
                    const float v=__bfloat162float(__float2bfloat16_rn(y[j*2+h]*scale*__bfloat162float(w[64+d])));
                    const float c=p.trig[pos*128+d], s=p.trig[pos*128+64+d];
                    a[h]=__bfloat162float(__float2bfloat16_rn(u*c-v*s));
                    b[h]=__bfloat162float(__float2bfloat16_rn(u*s+v*c));
                }
                ox[j]=__nv_fp8x2_e4m3(make_float2(a[0],a[1])).__x;
                oy[j]=__nv_fp8x2_e4m3(make_float2(b[0],b[1])).__x;
            }
        } else {
            #pragma unroll
            for(int j=0;j<4;++j) {
                ox[j]=__nv_fp8x2_e4m3(make_float2(x[j*2],x[j*2+1])).__x;
                oy[j]=__nv_fp8x2_e4m3(make_float2(y[j*2],y[j*2+1])).__x;
            }
        }
        if(token<rows) {
            const uint2 lo=*reinterpret_cast<const uint2*>(ox), hi=*reinterpret_cast<const uint2*>(oy);
            uint8_t* dst=head<64?p.q+(int64_t(token)*64+head)*128:(head<68?p.k:p.v)+(int64_t(token)*4+head%4)*128;
            *reinterpret_cast<uint2*>(dst+col)=lo;
            *reinterpret_cast<uint2*>(dst+64+col)=hi;
            if(head>=64 && p.loc[token]!=0) {
                dst=(head<68?p.kc:p.vc)+(p.loc[token]*4+head%4)*128;
                *reinterpret_cast<uint2*>(dst+col)=lo;
                *reinterpret_cast<uint2*>(dst+64+col)=hi;
            }
        }
    }
}

#include "qkv-fused.cuh"
using namespace deep_gemm;
constexpr int BM=128, BN=128, STAGES=5, THREADS=384;
constexpr int SMEM=BM*BN*2+STAGES*(BM*128+BN*128+BM*4)+32*4+STAGES*16;
#define GEMM_ARGS cute::UMMA::Major::K,0,9216,4096,1,BM,BN,128,128,128,128,STAGES,128,256,1,false,132,GemmType::Normal,cutlass::bfloat16_t,epilogue::transform::EpilogueIdentity
static auto plain_kernel=&sm90_fp8_gemm_1d2d_impl<GEMM_ARGS>;
static auto fused_kernel=&qkv_fused_impl<GEMM_ARGS>;

static CUtensorMap descriptor(void* ptr, CUtensorMapDataType type, uint64_t x, uint64_t y, uint32_t bx, uint32_t by, unsigned bytes, CUtensorMapSwizzle swizzle) {
    CUtensorMap map{};
    uint64_t dims[2]={x,y}, stride[1]={x*bytes};
    uint32_t box[2]={bx,by}, elem[2]={1,1};
    CUresult result=cuTensorMapEncodeTiled(&map,type,2,ptr,dims,stride,box,elem,CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(result!=CUDA_SUCCESS) { std::fprintf(stderr,"tensor map error %d\n",int(result)); std::abort(); }
    return map;
}
extern "C" int qkv_init() {
    int sms=0; cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0);
    if(sms!=132) return -1;
    for(auto fn:{reinterpret_cast<const void*>(plain_kernel),reinterpret_cast<const void*>(fused_kernel)}) {
        auto status=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);
        if(status!=cudaSuccess) return int(status);
        cudaFuncAttributes attr{}; cudaFuncGetAttributes(&attr,fn);
        int resident=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,fn,THREADS,SMEM);
        std::printf("qkv_resources regs=%d local=%zu smem=%d resident=%d\n",attr.numRegs,attr.localSizeBytes,SMEM,resident);
        if(resident<1 || attr.localSizeBytes) return -2;
    }
    return 0;
}
extern "C" int qkv_run(int fused,int rows,int sf_stride,void* a,void* sa,void* b,void* sb,void* out,QKVParams p,void* stream_ptr) {
    const auto ta=descriptor(a,CU_TENSOR_MAP_DATA_TYPE_UINT8,4096,rows,128,BM,1,CU_TENSOR_MAP_SWIZZLE_128B);
    const auto tb=descriptor(b,CU_TENSOR_MAP_DATA_TYPE_UINT8,4096,9216,128,BN,1,CU_TENSOR_MAP_SWIZZLE_128B);
    const auto td=descriptor(out,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,9216,rows,64,BM,2,CU_TENSOR_MAP_SWIZZLE_128B);
    const auto ts=descriptor(sa,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,sf_stride,32,BM,1,4,CU_TENSOR_MAP_SWIZZLE_NONE);
    const auto ca=*reinterpret_cast<const cute::TmaDescriptor*>(&ta), cb=*reinterpret_cast<const cute::TmaDescriptor*>(&tb);
    const auto cd=*reinterpret_cast<const cute::TmaDescriptor*>(&td), cs=*reinterpret_cast<const cute::TmaDescriptor*>(&ts);
    auto stream=static_cast<cudaStream_t>(stream_ptr);
    if(fused) qkv_fused_impl<GEMM_ARGS><<<132,THREADS,SMEM,stream>>>(static_cast<float*>(sb),nullptr,rows,9216,4096,ca,cb,cd,cs,p);
    else sm90_fp8_gemm_1d2d_impl<GEMM_ARGS><<<132,THREADS,SMEM,stream>>>(static_cast<float*>(sb),nullptr,rows,9216,4096,ca,cb,cd,cs);
    return int(cudaGetLastError());
}
