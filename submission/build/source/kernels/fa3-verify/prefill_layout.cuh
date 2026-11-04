// Same layouts and copy partitions as the pinned producer's LDSM/PRMT/STSM.
#pragma once
#include "utils.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
namespace apac_fa3_verify {
using PrefillLayout = flash::CollectiveMainloopFwdSm90<2, cute::Shape<cute::_1,cute::_1,cute::_1>,
    cute::Shape<cute::Int<128>,cute::Int<224>,cute::Int<128>>,128,cutlass::float_e4m3_t,float,
    cutlass::arch::Sm90,true,false,false,true,false,false,false,false,true,true,true,false,false,cutlass::bfloat16_t>;
}
