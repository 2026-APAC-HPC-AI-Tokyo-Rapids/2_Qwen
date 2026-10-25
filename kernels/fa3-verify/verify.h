// Fixed companion instantiations; upstream FlashAttention retains its license.
#pragma once
#include "flash.h"
#include <cuda_runtime.h>
namespace apac_fa3_verify {
using Launch = void (*)(Flash_fwd_params&, cudaStream_t, int*);
void combine(Flash_fwd_params&, cudaStream_t);
void prefill_fixed(Flash_fwd_params&, cudaStream_t, int*);
void prefill(Flash_fwd_params&, cudaStream_t, int*);
}
