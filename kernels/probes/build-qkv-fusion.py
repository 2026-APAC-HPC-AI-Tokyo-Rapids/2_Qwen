"""Add a Qwen QKV epilogue to the pinned DeepGEMM kernel for this experiment."""
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
source = source.replace('void dispatch_num_former_iters(', 'void qkv_dispatch_num_former_iters(').replace('dispatch_num_former_iters<', 'qkv_dispatch_num_former_iters<')
source = source.replace('sm90_fp8_gemm_1d2d_impl(', 'qkv_fused_impl(')
old = 'const __grid_constant__ cute::TmaDescriptor tensor_map_sfa) {'
assert source.count(old) == 1
source = source.replace(old, 'const __grid_constant__ cute::TmaDescriptor tensor_map_sfa, const __grid_constant__ QKVParams qkv) {')
start = source.index('            // Use TMA store to write back to global memory')
end = source.index('            __syncwarp();', start) + len('            __syncwarp();')
source = source[:start] + '''            static_assert(BLOCK_N == 128 && kSwizzleDMode == 128);
            qkv_epilogue<BLOCK_M, kNumMathThreads>(smem_d, m_block_idx * BLOCK_M, n_block_idx, shape_m, qkv);
            // The next tile may overwrite smem_d only after all readers finish.
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);''' + source[end:]
Path(sys.argv[2]).write_text(source)
