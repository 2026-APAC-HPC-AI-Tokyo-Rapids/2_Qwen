#include "verify.h"
#include "flash_fwd_launch_template.h"
namespace apac_fa3_verify {
void prefill_fixed(Flash_fwd_params& p, cudaStream_t stream, int* resource) {
    run_flash_fwd<90,128,128,1,cutlass::float_e4m3_t,cutlass::bfloat16_t,
        true,false,false,true,false,false,false,false,true,false,false,false,false,224,2,true>(p,stream,resource);
}
}
