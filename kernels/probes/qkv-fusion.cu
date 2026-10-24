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

template<bool Fast>
__device__ __forceinline__ uint16_t qkv_fp8(float2 x) {
    if constexpr (!Fast) return __nv_cvt_float2_to_fp8x2(x, __NV_NOSAT, __NV_E4M3);
    // NOSAT in CUDA 12.9 falls back to double/integer conversion. Use the
    // hardware conversion and restore its exact overflow/canonical-NaN rules.
    const uint16_t packed=__nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
    const unsigned a=__float_as_uint(x.x), b=__float_as_uint(x.y);
    const unsigned aa=a&0x7fffffff, bb=b&0x7fffffff;
    const unsigned lo=aa>0x43e80000 ? (aa>0x7f800000 ? 0x7f : ((a>>24)&0x80)|0x7f) : packed&0xff;
    const unsigned hi=bb>0x43e80000 ? (bb>0x7f800000 ? 0x7f : ((b>>24)&0x80)|0x7f) : packed>>8;
    return lo|(hi<<8);
}

__global__ void check_fp8(uint16_t* reference,uint16_t* candidate) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    const float2 x=make_float2(__bfloat162float(__ushort_as_bfloat16(i)),__bfloat162float(__ushort_as_bfloat16(i^0x8000)));
    reference[i]=qkv_fp8<false>(x);
    candidate[i]=qkv_fp8<true>(x);
}
extern "C" int qkv_check_fp8(void* reference,void* candidate,void* stream) {
    check_fp8<<<256,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<uint16_t*>(reference),static_cast<uint16_t*>(candidate));
    return int(cudaGetLastError());
}

template<int BM, int THREADS>
__device__ __forceinline__ void qkv_epilogue(const __nv_bfloat16* tile, int first_row, int head, int rows, const QKVParams& p) {
    const int lane = threadIdx.x % 8;
    const int col = lane * 8;
    float wx[8], wy[8];
    if (head < 68) {
        const auto w = head < 64 ? p.qw : p.kw;
        const uint4 vx = *reinterpret_cast<const uint4*>(w + col);
        const uint4 vy = *reinterpret_cast<const uint4*>(w + 64 + col);
        #pragma unroll
        for (int i=0;i<8;++i) {
            wx[i] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&vx)[i]);
            wy[i] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&vy)[i]);
        }
    }
    // Same eight-lane/head reduction tree as the deployed SGLang prologue.
    for (int row = threadIdx.x / 8; row < BM; row += THREADS / 8) {
        const int token = first_row + row;
        float x[8], y[8], partial[4];
        uint16_t ox[4], oy[4];
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
            const int64_t pos=token<rows?p.pos[token]:0;
            const float4 cos4[2] = {*reinterpret_cast<const float4*>(p.trig+pos*128+col), *reinterpret_cast<const float4*>(p.trig+pos*128+col+4)};
            const float4 sin4[2] = {*reinterpret_cast<const float4*>(p.trig+pos*128+64+col), *reinterpret_cast<const float4*>(p.trig+pos*128+64+col+4)};
            #pragma unroll
            for(int j=0;j<4;++j) {
                float a[2], b[2];
                #pragma unroll
                for(int h=0;h<2;++h) {
                    const int d=j*2+h;
                    const float u=__bfloat162float(__float2bfloat16_rn(x[d]*scale*wx[d]));
                    const float v=__bfloat162float(__float2bfloat16_rn(y[d]*scale*wy[d]));
                    const float c=reinterpret_cast<const float*>(cos4)[d], s=reinterpret_cast<const float*>(sin4)[d];
                    a[h]=__bfloat162float(__float2bfloat16_rn(u*c-v*s));
                    b[h]=__bfloat162float(__float2bfloat16_rn(u*s+v*c));
                }
                ox[j]=qkv_fp8<(BM==256)>(make_float2(a[0],a[1]));
                oy[j]=qkv_fp8<(BM==256)>(make_float2(b[0],b[1]));
            }
        } else {
            #pragma unroll
            for(int j=0;j<4;++j) {
                ox[j]=qkv_fp8<(BM==256)>(make_float2(x[j*2],x[j*2+1]));
                oy[j]=qkv_fp8<(BM==256)>(make_float2(y[j*2],y[j*2+1]));
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
constexpr int THREADS=384;
#define GEMM_ARGS cute::UMMA::Major::K,0,9216,4096,1,BM,BN,128,128,128,128,STAGES,128,256,CLUSTER,(CLUSTER>1),132,GemmType::Normal,cutlass::bfloat16_t,epilogue::transform::EpilogueIdentity
template<int BM, int BN, int STAGES>
constexpr int smem_bytes = BM*BN*2+STAGES*(BM*128+BN*128+BM*4)+32*4*(BN/128)+STAGES*16;

static CUtensorMap descriptor(void* ptr, CUtensorMapDataType type, uint64_t x, uint64_t y, uint32_t bx, uint32_t by, unsigned bytes, CUtensorMapSwizzle swizzle) {
    CUtensorMap map{};
    uint64_t dims[2]={x,y}, stride[1]={x*bytes};
    uint32_t box[2]={bx,by}, elem[2]={1,1};
    CUresult result=cuTensorMapEncodeTiled(&map,type,2,ptr,dims,stride,box,elem,CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(result!=CUDA_SUCCESS) { std::fprintf(stderr,"tensor map error %d\n",int(result)); std::abort(); }
    return map;
}
template<int BM, int BN, int STAGES, int CLUSTER>
int init_variant() {
    constexpr int SMEM=smem_bytes<BM,BN,STAGES>;
    int kind=0;
    for(auto fn:{reinterpret_cast<const void*>(&sm90_fp8_gemm_1d2d_impl<GEMM_ARGS>),reinterpret_cast<const void*>(&qkv_fused_impl<GEMM_ARGS>)}) {
        auto status=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);
        if(status!=cudaSuccess) return int(status);
        cudaFuncAttributes attr{}; cudaFuncGetAttributes(&attr,fn);
        int resident=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,fn,THREADS,SMEM);
        std::printf("qkv_resources bm=%d bn=%d stages=%d cluster=%d kind=%d regs=%d local=%zu smem=%d resident=%d\n",BM,BN,STAGES,CLUSTER,kind++,attr.numRegs,attr.localSizeBytes,SMEM,resident);
        if(resident<1) return -2;
    }
    return 0;
}
extern "C" int qkv_init() {
    int sms=0; cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0);
    if(sms!=132) return -1;
    int status=init_variant<128,128,5,1>();
    return status ? status : init_variant<256,128,3,2>();
}
template<int BM, int BN, int STAGES, int CLUSTER>
int run_variant(int kind,int rows,int sf_stride,void* a,void* sa,void* b,void* sb,void* out,QKVParams p,void* stream_ptr) {
    const auto ta=descriptor(a,CU_TENSOR_MAP_DATA_TYPE_UINT8,4096,rows,128,BM,1,CU_TENSOR_MAP_SWIZZLE_128B);
    const auto tb=descriptor(b,CU_TENSOR_MAP_DATA_TYPE_UINT8,4096,9216,128,BN,1,CU_TENSOR_MAP_SWIZZLE_128B);
    const auto td=descriptor(out,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,9216,rows,64,BM,2,CU_TENSOR_MAP_SWIZZLE_128B);
    const auto ts=descriptor(sa,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,sf_stride,32,BM,1,4,CU_TENSOR_MAP_SWIZZLE_NONE);
    const auto ca=*reinterpret_cast<const cute::TmaDescriptor*>(&ta), cb=*reinterpret_cast<const cute::TmaDescriptor*>(&tb);
    const auto cd=*reinterpret_cast<const cute::TmaDescriptor*>(&td), cs=*reinterpret_cast<const cute::TmaDescriptor*>(&ts);
    auto stream=static_cast<cudaStream_t>(stream_ptr);
    cudaLaunchAttribute attr{};
    attr.id=cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x=CLUSTER; attr.val.clusterDim.y=1; attr.val.clusterDim.z=1;
    cudaLaunchConfig_t config{};
    config.gridDim=dim3(132); config.blockDim=dim3(THREADS);
    config.dynamicSmemBytes=smem_bytes<BM,BN,STAGES>; config.stream=stream;
    config.attrs=&attr; config.numAttrs=1;
    if(kind==1) return int(cudaLaunchKernelEx(&config,qkv_fused_impl<GEMM_ARGS>,static_cast<float*>(sb),static_cast<int*>(nullptr),unsigned(rows),9216u,4096u,ca,cb,cd,cs,p));
    return int(cudaLaunchKernelEx(&config,sm90_fp8_gemm_1d2d_impl<GEMM_ARGS>,static_cast<float*>(sb),static_cast<int*>(nullptr),unsigned(rows),9216u,4096u,ca,cb,cd,cs));
}
extern "C" int qkv_run(int mode,int rows,int sf_stride,void* a,void* sa,void* b,void* sb,void* out,QKVParams p,void* stream_ptr) {
    if(mode<2) return run_variant<128,128,5,1>(mode%2,rows,sf_stride,a,sa,b,sb,out,p,stream_ptr);
    return run_variant<256,128,3,2>(mode%2,rows,sf_stride,a,sa,b,sb,out,p,stream_ptr);
}
