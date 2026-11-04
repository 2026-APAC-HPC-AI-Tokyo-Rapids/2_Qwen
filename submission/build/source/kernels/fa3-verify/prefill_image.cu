// GPU gate only: dump the deployed LDSM/PRMT/STSM byte image, independently
// of the host-generated map used by the QKV producer. Never timed in selection.
#include "verify.h"
#include "prefill_layout.cuh"
namespace apac_fa3_verify {
using namespace cute;
using M = PrefillLayout;
__global__ void prefill_dump_v_kernel(const uint8_t* v, uint8_t* out, int rows) {
    __shared__ __align__(1024) uint8_t vt[224*128], vv[224*128];
    auto sVt=as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(vt),take<0,2>(M::SmemLayoutVt{})));
    auto sV=as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(vv),take<0,2>(M::SmemLayoutVtMma{})));
    for(int x=threadIdx.x;x<224*128;x+=128) {
        int token=blockIdx.x*224+x/128, dim=x%128;
        sVt(dim,x/128)=token<rows?v[(token*4+blockIdx.y)*128+dim]:0;
    }
    __syncthreads();
    // Add the stage axis used by the original copy partitions.
    auto st=as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(vt),M::SmemLayoutVt{}));
    auto sv=as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(vv),M::SmemLayoutVtMma{}));
    auto ts=M::S2RTiledCopyVt{}.get_thread_slice(threadIdx.x).partition_S(flat_divide(st,M::LDSM_divide_shape{}));
    auto td=M::R2STiledCopyV{}.get_thread_slice(threadIdx.x).partition_D(flat_divide(sv,M::STSM_divide_shape{}));
    constexpr int ilp=(size<2>(ts)*size<3>(ts))%2==0?2:1;
    auto ss=logical_divide(group_modes<1,rank(ts)-1>(ts),Shape<Underscore,Int<ilp>>{});
    auto dd=logical_divide(group_modes<1,rank(td)-1>(td),Shape<Underscore,Int<ilp>>{});
    #pragma unroll
    for(int i=0;i<size<1,1>(ss);++i) {
        auto reg=make_fragment_like(dd(_,make_coord(_,_0{}),_0{}));
        copy(M::S2RTiledCopyVt{},ss(_,make_coord(_,i),0),reg);
        auto r64=recast<uint2>(reg);
        #pragma unroll
        for(int j=0;j<size(r64);++j) {
            uint32_t a=r64[j].x,b=r64[j].y;
            r64[j].x=__byte_perm(a,b,0x6420);r64[j].y=__byte_perm(a,b,0x7531);
        }
        copy(M::R2STiledCopyV{},reg,dd(_,make_coord(_,i),0));
    }
    __syncthreads();
    auto dst=out+(blockIdx.y*gridDim.x+blockIdx.x)*224*128;
    for(int x=threadIdx.x;x<224*128;x+=128) dst[x]=vv[x];
}
void prefill_dump_v_launch(const void* v,void* out,int rows,cudaStream_t stream) {
    prefill_dump_v_kernel<<<dim3((rows+223)/224,4),128,0,stream>>>(static_cast<const uint8_t*>(v),static_cast<uint8_t*>(out),rows);
}
}
