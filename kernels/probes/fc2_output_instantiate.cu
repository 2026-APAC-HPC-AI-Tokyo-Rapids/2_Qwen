// CPU-only compilation of the complete, unmodified MegaMoE entry point.
// This catches integration/register regressions missed by the output-only probe.
#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>
#define INSTANTIATE(NAME,E,W,BM,BN,S,T,REUSE,ARRIVAL,SYNC,SWAP) extern "C" void* NAME() { return reinterpret_cast<void*>(&deep_gemm::sm90_fp8_mega_moe_impl<4224,4096,1536,E,8,W,BM,BN,128,141312,2260992,S,64,64,T,132,4,cute::numeric_limits<float>::infinity(),true,0,REUSE,ARRIVAL,SYNC,false,SWAP>); }
INSTANTIATE(prefill128,128,32,128,256,3,256,true,true,false,false)
INSTANTIATE(prefill140,140,35,128,256,3,256,true,true,false,false)
INSTANTIATE(decode_independent,128,32,64,128,6,256,false,false,true,true)
INSTANTIATE(decode_shared,128,32,64,128,8,256,false,false,true,true)
INSTANTIATE(masked_nonswap,140,35,64,128,8,256,false,false,true,false)
INSTANTIATE(partial_wave,128,16,64,128,8,256,false,false,true,true)
