#include "verify.h"
#include "flash_fwd_launch_template.h"
namespace apac_fa3_verify {
// build.py supplies these directly from candidates.json, once per unique main.
void VERIFY_NAME(Flash_fwd_params& p, cudaStream_t stream, int* resource) {
    run_flash_fwd<90, 128, 128, 1, cutlass::float_e4m3_t, cutlass::bfloat16_t, true, false, false, true, true, false, false, false, true, VERIFY_SPLIT, false, true, false, VERIFY_BLOCK_N, VERIFY_STAGES, false, 0, VERIFY_MERGE>(p, stream, resource);
}
}
