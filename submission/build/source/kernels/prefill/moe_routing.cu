// EP1 TP4/TP2 routing only. No GEMM/quantization templates or FP arithmetic.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {
constexpr int E = 128;
constexpr int B = 512;
constexpr int THREADS = 256;
constexpr int MAX_ROWS = 8192 * 8;

// All lanes call ballot. Only the identical valid mask participates in match/shuffle.
__device__ void grouped_increment(int expert, bool valid, int* counters, int& start, int& rank) {
  unsigned mask = __ballot_sync(0xffffffffu, valid);
  if (valid) {
    unsigned peers = __match_any_sync(mask, expert);
    int lane = threadIdx.x & 31;
    int leader = __ffs(peers) - 1;
    int base = 0;
    if (lane == leader) base = atomicAdd(counters + expert, __popc(peers));
    start = __shfl_sync(peers, base, leader);
    rank = __popc(peers & ((1u << lane) - 1));
  }
}

__global__ void tile_histogram(const int* ids, int* counts, int rows) {
  __shared__ int hist[E];
  if (threadIdx.x < E) hist[threadIdx.x] = 0;
  __syncthreads();
  for (int j = threadIdx.x; j < B; j += THREADS) {
    int i = blockIdx.x * B + j;
    int expert = i < rows ? ids[i] : -1;
    int unused_start = 0, unused_rank = 0;
    grouped_increment(expert, i < rows && expert >= 0 && expert < E, hist, unused_start, unused_rank);
  }
  __syncthreads();
  if (threadIdx.x < E) counts[blockIdx.x * E + threadIdx.x] = hist[threadIdx.x];
}

// One thread per expert: at most 128 tiles for the production bound.
__global__ void prefix_metadata(const int* counts, int* tile_offsets, int* offsets, int* p1, int* p2, int tiles, int intermediate_size) {
  __shared__ int totals[E];
  int expert = threadIdx.x;
  int total = 0;
  for (int tile = 0; tile < tiles; ++tile) {
    tile_offsets[tile * E + expert] = total;
    total += counts[tile * E + expert];
  }
  totals[expert] = total;
  __syncthreads();
  int begin = 0;
  for (int other = 0; other < expert; ++other) begin += totals[other];
  offsets[expert] = begin;
  if (expert == E - 1) offsets[E] = begin + total;
  p1[3 * expert] = total; p1[3 * expert + 1] = 2 * intermediate_size; p1[3 * expert + 2] = 4096;
  p2[3 * expert] = total; p2[3 * expert + 1] = 4096; p2[3 * expert + 2] = intermediate_size;
  for (int tile = 0; tile < tiles; ++tile) tile_offsets[tile * E + expert] += begin;
}

__global__ void tile_scatter(const int* ids, const int* tile_offsets, int* a_map, int* c_map, int rows) {
  __shared__ int cursor[E];
  if (threadIdx.x < E) cursor[threadIdx.x] = 0;
  __syncthreads();
  for (int j = threadIdx.x; j < B; j += THREADS) {
    int i = blockIdx.x * B + j;
    int expert = i < rows ? ids[i] : -1;
    bool valid = i < rows && expert >= 0 && expert < E;
    int start = 0, rank = 0;
    grouped_increment(expert, valid, cursor, start, rank);
    if (valid) {
      int dst = tile_offsets[blockIdx.x * E + expert] + start + rank;
      a_map[dst] = i / 8;
      c_map[i] = dst;
    }
  }
}

void check_int(const torch::Tensor& x, const torch::Tensor& ids) {
  TORCH_CHECK(x.is_cuda() && x.device() == ids.device() && x.scalar_type() == at::kInt && x.is_contiguous(), "routing expects same-device contiguous CUDA int32 tensors");
}
}  // namespace

void prepare(torch::Tensor ids, torch::Tensor offsets, torch::Tensor p1, torch::Tensor p2, torch::Tensor a_map, torch::Tensor c_map, torch::Tensor counts, torch::Tensor tile_offsets, int intermediate_size) {
  TORCH_CHECK(intermediate_size == 384 || intermediate_size == 768, "routing supports TP4/TP2 intermediate sizes 384/768");
  TORCH_CHECK(ids.is_cuda() && ids.dim() == 2 && ids.size(1) == 8 && ids.scalar_type() == at::kInt && ids.is_contiguous(), "ids must be CUDA contiguous int32[T,8]; production IDs must all be valid 0..127");
  int64_t rows64 = ids.numel();
  TORCH_CHECK(rows64 <= MAX_ROWS, "routing supports at most T8192/top8");
  int rows = static_cast<int>(rows64);
  int tiles = (rows + B - 1) / B;
  for (auto& x : {offsets, p1, p2, a_map, c_map, counts, tile_offsets}) check_int(x, ids);
  TORCH_CHECK(offsets.dim() == 1 && offsets.numel() == E + 1, "offsets shape must be [129]");
  TORCH_CHECK(p1.sizes() == c10::IntArrayRef({E, 3}) && p2.sizes() == c10::IntArrayRef({E, 3}), "problem sizes must be [128,3]");
  TORCH_CHECK(a_map.dim() == 1 && c_map.dim() == 1 && a_map.numel() == rows && c_map.numel() == rows, "maps must be [8T]");
  TORCH_CHECK(counts.sizes() == c10::IntArrayRef({tiles, E}) && tile_offsets.sizes() == c10::IntArrayRef({tiles, E}), "scratch must be [ceil(8T/512),128]");
  c10::cuda::CUDAGuard guard(ids.device());
  auto stream = at::cuda::getCurrentCUDAStream(ids.get_device());
  if (tiles) {
    tile_histogram<<<tiles, THREADS, 0, stream>>>(ids.data_ptr<int>(), counts.data_ptr<int>(), rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  // Also explicitly zeros metadata for helper-only T0 tests; caller skips whole MoE.
  prefix_metadata<<<1, E, 0, stream>>>(counts.data_ptr<int>(), tile_offsets.data_ptr<int>(), offsets.data_ptr<int>(), p1.data_ptr<int>(), p2.data_ptr<int>(), tiles, intermediate_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (tiles) {
    tile_scatter<<<tiles, THREADS, 0, stream>>>(ids.data_ptr<int>(), tile_offsets.data_ptr<int>(), a_map.data_ptr<int>(), c_map.data_ptr<int>(), rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &prepare);
}
