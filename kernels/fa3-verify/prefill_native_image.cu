// GPU gate only: dump the deployed LDSM/PRMT/STSM byte image, independently
// of the host-generated map used by the QKV producer. Never timed in selection.
#include "verify.h"
#include "utils.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
namespace apac_fa3_verify {
using namespace cute;
template<int N> using NativeLayout = flash::CollectiveMainloopFwdSm90<2,Shape<_1,_1,_1>,Shape<Int<N==96?192:128>,Int<N>,Int<128>>,128,cutlass::float_e4m3_t,float,cutlass::arch::Sm90,true,false,false,true,false,false,false,false,true,true,true,false,false,cutlass::bfloat16_t>;
template<int N> __global__ void prefill_dump_native_kernel(const uint8_t* v, uint8_t* out, const int* cu) {
    using M=NativeLayout<N>;
    __shared__ __align__(1024) uint8_t vt[N*128], vv[N*128];
    auto sVt=as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(vt),take<0,2>(typename M::SmemLayoutVt{})));
    auto sV=as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(vv),take<0,2>(typename M::SmemLayoutVtMma{})));
    for(int x=threadIdx.x;x<N*128;x+=128) {
        int local=blockIdx.x*N+x/128, token=cu[blockIdx.z]+local, dim=x%128;
        sVt(dim,x/128)=token<cu[blockIdx.z+1]?v[(token*4+blockIdx.y)*128+dim]:0;
    }
    __syncthreads();
    // Add the stage axis used by the original copy partitions.
    auto st=as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(vt),typename M::SmemLayoutVt{}));
    auto sv=as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(vv),typename M::SmemLayoutVtMma{}));
    auto ts=typename M::S2RTiledCopyVt{}.get_thread_slice(threadIdx.x).partition_S(flat_divide(st,typename M::LDSM_divide_shape{}));
    auto td=typename M::R2STiledCopyV{}.get_thread_slice(threadIdx.x).partition_D(flat_divide(sv,typename M::STSM_divide_shape{}));
    constexpr int ilp=(size<2>(ts)*size<3>(ts))%2==0?2:1;
    auto ss=logical_divide(group_modes<1,rank(ts)-1>(ts),Shape<Underscore,Int<ilp>>{});
    auto dd=logical_divide(group_modes<1,rank(td)-1>(td),Shape<Underscore,Int<ilp>>{});
    #pragma unroll
    for(int i=0;i<size<1,1>(ss);++i) {
        auto reg=make_fragment_like(dd(_,make_coord(_,_0{}),_0{}));
        copy(typename M::S2RTiledCopyVt{},ss(_,make_coord(_,i),0),reg);
        auto r64=recast<uint2>(reg);
        #pragma unroll
        for(int j=0;j<size(r64);++j) {
            uint32_t a=r64[j].x,b=r64[j].y;
            r64[j].x=__byte_perm(a,b,0x6420);r64[j].y=__byte_perm(a,b,0x7531);
        }
        copy(typename M::R2STiledCopyV{},reg,dd(_,make_coord(_,i),0));
    }
    __syncthreads();
    auto dst=out+((blockIdx.z*4+blockIdx.y)*gridDim.x+blockIdx.x)*N*128;
    for(int x=threadIdx.x;x<N*128;x+=128) dst[x]=vv[x];
}
void prefill_dump_native_launch(const void* v,void* out,const int* cu,int batch,int blocks,int n,cudaStream_t stream) {
    if(n==96) prefill_dump_native_kernel<96><<<dim3(blocks,4,batch),128,0,stream>>>(static_cast<const uint8_t*>(v),static_cast<uint8_t*>(out),cu);
    else prefill_dump_native_kernel<192><<<dim3(blocks,4,batch),128,0,stream>>>(static_cast<const uint8_t*>(v),static_cast<uint8_t*>(out),cu);
}
}
