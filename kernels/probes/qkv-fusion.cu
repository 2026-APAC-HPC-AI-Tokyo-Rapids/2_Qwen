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

template<int BM, int BN, int THREADS>
__device__ __forceinline__ void qkv_epilogue(const __nv_bfloat16* tile, int first_row, int block_n, int rows, const QKVParams& p) {
    constexpr int HEADS = BN / 128, HEAD_THREADS = THREADS / HEADS;
    const int local_head = threadIdx.x / HEAD_THREADS;
    const int head = block_n * HEADS + local_head;
    tile += local_head * BM * 128;
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
    for (int row = (threadIdx.x % HEAD_THREADS) / 8; row < BM; row += HEAD_THREADS / 8) {
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
                ox[j]=__nv_cvt_float2_to_fp8x2(make_float2(a[0],a[1]), __NV_NOSAT, __NV_E4M3);
                oy[j]=__nv_cvt_float2_to_fp8x2(make_float2(b[0],b[1]), __NV_NOSAT, __NV_E4M3);
            }
        } else {
            #pragma unroll
            for(int j=0;j<4;++j) {
                ox[j]=__nv_cvt_float2_to_fp8x2(make_float2(x[j*2],x[j*2+1]), __NV_NOSAT, __NV_E4M3);
                oy[j]=__nv_cvt_float2_to_fp8x2(make_float2(y[j*2],y[j*2+1]), __NV_NOSAT, __NV_E4M3);
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

// Each four-lane WGMMA row owns pairs at columns (lane*2 + 8*t).
// Fold the eight chunks in the same XOR-4/2/1 order as the deployed
// eight-lane norm, then combine its four partial sums across these lanes.
template<int BN>
__device__ __forceinline__ void qkv_register_epilogue(const float* accum, int first_row, int block_n, int rows, const QKVParams& p) {
    const int lane = threadIdx.x % 4;
    const int first = first_row + (threadIdx.x / 32) * 16 + (threadIdx.x % 32) / 4;
    #pragma unroll
    for(int h=0;h<BN/128;++h) {
        const int head=block_n*(BN/128)+h;
        #pragma unroll
        for(int r=0;r<2;++r) {
            const int token=first+r*8;
            float x[16],y[16],part[8];
            #pragma unroll
            for(int t=0;t<8;++t) {
                const int base=(h*16+t)*4+r*2;
                x[t*2]=__bfloat162float(__float2bfloat16_rn(accum[base]));
                x[t*2+1]=__bfloat162float(__float2bfloat16_rn(accum[base+1]));
                y[t*2]=__bfloat162float(__float2bfloat16_rn(accum[base+32]));
                y[t*2+1]=__bfloat162float(__float2bfloat16_rn(accum[base+33]));
                float sum=0.f;
                sum+=x[t*2]*x[t*2]; sum+=x[t*2+1]*x[t*2+1];
                sum+=y[t*2]*y[t*2]; sum+=y[t*2+1]*y[t*2+1];
                part[t]=sum;
            }
            float scale=1.f;
            if(head<68) {
                float sum=((part[0]+part[4])+(part[2]+part[6]))+((part[1]+part[5])+(part[3]+part[7]));
                sum+=__shfl_xor_sync(0xffffffff,sum,2,4);
                sum+=__shfl_xor_sync(0xffffffff,sum,1,4);
                scale=rsqrtf(sum/128.f+1e-6f);
            }
            const int64_t pos=token<rows?p.pos[token]:0;
            const int64_t loc=token<rows && head>=64?p.loc[token]:0;
            const auto w=head<64?p.qw:p.kw;
            #pragma unroll
            for(int t=0;t<8;++t) {
                const int col=t*8+lane*2;
                float2 a=make_float2(x[t*2],x[t*2+1]),b=make_float2(y[t*2],y[t*2+1]);
                if(head<68) {
                    const float2 wx=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(w+col));
                    const float2 wy=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(w+64+col));
                    const float2 c=*reinterpret_cast<const float2*>(p.trig+pos*128+col);
                    const float2 s=*reinterpret_cast<const float2*>(p.trig+pos*128+64+col);
                    const float u0=__bfloat162float(__float2bfloat16_rn(a.x*scale*wx.x));
                    const float u1=__bfloat162float(__float2bfloat16_rn(a.y*scale*wx.y));
                    const float v0=__bfloat162float(__float2bfloat16_rn(b.x*scale*wy.x));
                    const float v1=__bfloat162float(__float2bfloat16_rn(b.y*scale*wy.y));
                    a=__bfloat1622float2(__float22bfloat162_rn(make_float2(u0*c.x-v0*s.x,u1*c.y-v1*s.y)));
                    b=__bfloat1622float2(__float22bfloat162_rn(make_float2(u0*s.x+v0*c.x,u1*s.y+v1*c.y)));
                }
                if(token<rows) {
                    const uint16_t lo=__nv_cvt_float2_to_fp8x2(a,__NV_NOSAT,__NV_E4M3);
                    const uint16_t hi=__nv_cvt_float2_to_fp8x2(b,__NV_NOSAT,__NV_E4M3);
                    uint8_t* dst=head<64?p.q+(int64_t(token)*64+head)*128:(head<68?p.k:p.v)+(int64_t(token)*4+head%4)*128;
                    *reinterpret_cast<uint16_t*>(dst+col)=lo;
                    *reinterpret_cast<uint16_t*>(dst+64+col)=hi;
                    if(head>=64 && loc!=0) {
                        dst=(head<68?p.kc:p.vc)+(loc*4+head%4)*128;
                        *reinterpret_cast<uint16_t*>(dst+col)=lo;
                        *reinterpret_cast<uint16_t*>(dst+64+col)=hi;
                    }
                }
            }
        }
    }
}

#include "qkv-fused.cuh"
#include "qkv-register.cuh"
using namespace deep_gemm;
constexpr int BM=128, THREADS=384;
#define GEMM_ARGS cute::UMMA::Major::K,0,9216,4096,1,BM,BN,128,128,128,128,STAGES,128,256,CLUSTER,(CLUSTER>1),132,GemmType::Normal,cutlass::bfloat16_t,epilogue::transform::EpilogueIdentity
template<int BN, int STAGES>
constexpr int smem_bytes = BM*BN*2+STAGES*(BM*128+BN*128+BM*4)+32*4*(BN/128)+STAGES*16;

static CUtensorMap descriptor(void* ptr, CUtensorMapDataType type, uint64_t x, uint64_t y, uint32_t bx, uint32_t by, unsigned bytes, CUtensorMapSwizzle swizzle) {
    CUtensorMap map{};
    uint64_t dims[2]={x,y}, stride[1]={x*bytes};
    uint32_t box[2]={bx,by}, elem[2]={1,1};
    CUresult result=cuTensorMapEncodeTiled(&map,type,2,ptr,dims,stride,box,elem,CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(result!=CUDA_SUCCESS) { std::fprintf(stderr,"tensor map error %d\n",int(result)); std::abort(); }
    return map;
}
template<int BN, int STAGES, int CLUSTER>
int init_variant() {
    constexpr int SMEM=smem_bytes<BN,STAGES>;
    int kind=0;
    for(auto fn:{reinterpret_cast<const void*>(&sm90_fp8_gemm_1d2d_impl<GEMM_ARGS>),reinterpret_cast<const void*>(&qkv_fused_impl<GEMM_ARGS>),reinterpret_cast<const void*>(&qkv_register_impl<GEMM_ARGS>)}) {
        auto status=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);
        if(status!=cudaSuccess) return int(status);
        cudaFuncAttributes attr{}; cudaFuncGetAttributes(&attr,fn);
        int resident=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,fn,THREADS,SMEM);
        std::printf("qkv_resources bn=%d stages=%d cluster=%d kind=%d regs=%d local=%zu smem=%d resident=%d\n",BN,STAGES,CLUSTER,kind++,attr.numRegs,attr.localSizeBytes,SMEM,resident);
        if(resident<1) return -2;
    }
    return 0;
}
extern "C" int qkv_init() {
    int sms=0; cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0);
    if(sms!=132) return -1;
    int status=init_variant<128,5,1>();
    return status ? status : init_variant<256,3,2>();
}
template<int BN, int STAGES, int CLUSTER>
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
    config.dynamicSmemBytes=smem_bytes<BN,STAGES>; config.stream=stream;
    config.attrs=&attr; config.numAttrs=1;
    if(kind==2) return int(cudaLaunchKernelEx(&config,qkv_register_impl<GEMM_ARGS>,static_cast<float*>(sb),static_cast<int*>(nullptr),unsigned(rows),9216u,4096u,ca,cb,cd,cs,p));
    if(kind==1) return int(cudaLaunchKernelEx(&config,qkv_fused_impl<GEMM_ARGS>,static_cast<float*>(sb),static_cast<int*>(nullptr),unsigned(rows),9216u,4096u,ca,cb,cd,cs,p));
    return int(cudaLaunchKernelEx(&config,sm90_fp8_gemm_1d2d_impl<GEMM_ARGS>,static_cast<float*>(sb),static_cast<int*>(nullptr),unsigned(rows),9216u,4096u,ca,cb,cd,cs));
}
extern "C" int qkv_run(int mode,int rows,int sf_stride,void* a,void* sa,void* b,void* sb,void* out,QKVParams p,void* stream_ptr) {
    if(mode<2) return run_variant<128,5,1>(mode%2,rows,sf_stride,a,sa,b,sb,out,p,stream_ptr);
    if(mode<4) return run_variant<256,3,2>(mode%2,rows,sf_stride,a,sa,b,sb,out,p,stream_ptr);
    if(mode==4) return run_variant<128,5,1>(2,rows,sf_stride,a,sa,b,sb,out,p,stream_ptr);
    return run_variant<256,3,2>(2,rows,sf_stride,a,sa,b,sb,out,p,stream_ptr);
}
