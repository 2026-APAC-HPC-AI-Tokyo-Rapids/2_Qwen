#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/impl/norm.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>

#include <cassert>
#include <numeric>

namespace {

struct FusedRopeParams {
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;  // NOTE: this k is pre-offset in host code to reduce computation in kernel
  const void* __restrict__ cos_sin_cache_ptr;
  const void* __restrict__ positions;
  int64_t q_stride_bytes;
  int64_t k_stride_bytes;
  int64_t head_stride_bytes;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t num_tokens;
  void* __restrict__ q_fp8 = nullptr;
};

struct FusedRopeStoreParams {
  FusedRopeParams base_params;
  void* v_ptr;
  void* __restrict__ k_cache;
  void* __restrict__ v_cache;
  const void* __restrict__ out_loc;
  int64_t v_stride_bytes;
  int64_t cache_stride_bytes;
};

constexpr uint32_t kBlockSize = 128;

[[maybe_unused]]
constexpr auto next_pow2(uint32_t target, uint32_t factor = 1) {
  uint32_t power = 1;
  while (power * factor < target)
    power *= 2;
  return power;
}

// Rotate one head row in place, reading and writing as T. Factored out so the
// q and k branches can run different (same-size) dtypes -- e.g. fp16 q emitted
// by a fused QK-norm while k stays bf16. When cache_out is non-null the rotated
// row is additionally stored there (the fused KV-store k path); predicated so
// the q and k branches keep one instruction stream per instantiation.
template <typename T, bool kIsNeox, int64_t kRopeDim, int64_t kVecSize, bool kFP8Output = false>
__device__ __forceinline__ void
rope_rotate_head(void* input, const void* cos_ptr, const void* sin_ptr, uint32_t lane_id, void* cache_out = nullptr, void* fp8_out = nullptr) {
  using namespace device;
  using T2 = packed_t<T>;
  using Storage = AlignedVector<T2, kVecSize>;
  if constexpr (kIsNeox) {
    using CacheStorage = AlignedVector<fp32x2_t, kVecSize>;
    const auto input_x = input;
    const auto input_y = pointer::offset(input, (kRopeDim / 2) * sizeof(T));
    auto input_vec_x = load_as<Storage>(input_x, lane_id);
    auto input_vec_y = load_as<Storage>(input_y, lane_id);
    const auto cos_pair = load_as<CacheStorage>(cos_ptr, lane_id);
    const auto sin_pair = load_as<CacheStorage>(sin_ptr, lane_id);
#pragma unroll
    for (int64_t j = 0; j < kVecSize; ++j) {
      const auto [x0, x1] = cast<fp32x2_t>(input_vec_x[j]);
      const auto [y0, y1] = cast<fp32x2_t>(input_vec_y[j]);
      const auto [cos_0, cos_1] = cos_pair[j];
      const auto [sin_0, sin_1] = sin_pair[j];
      const auto out_x0 = x0 * cos_0 - y0 * sin_0;
      const auto out_y0 = x0 * sin_0 + y0 * cos_0;
      const auto out_x1 = x1 * cos_1 - y1 * sin_1;
      const auto out_y1 = x1 * sin_1 + y1 * cos_1;
      input_vec_x[j] = cast<T2, fp32x2_t>({out_x0, out_x1});
      input_vec_y[j] = cast<T2, fp32x2_t>({out_y0, out_y1});
    }
    if constexpr (kFP8Output) {
      // Match BF16 RoPE followed by torch.to(float8_e4m3fn): retain
      // the BF16 rounding above and use non-saturating FP8 conversion.
      using FP8Storage = AlignedVector<uint16_t, kVecSize>;
      FP8Storage fp8_x, fp8_y;
#ifdef USE_ROCM
      static_assert(!kFP8Output, "FP8 Q fusion requires CUDA");
#else
#pragma unroll
      for (int64_t j = 0; j < kVecSize; ++j) {
        fp8_x[j] = __nv_cvt_float2_to_fp8x2(cast<fp32x2_t>(input_vec_x[j]), __NV_NOSAT, __NV_E4M3);
        fp8_y[j] = __nv_cvt_float2_to_fp8x2(cast<fp32x2_t>(input_vec_y[j]), __NV_NOSAT, __NV_E4M3);
      }
#endif
      store_as<FP8Storage>(fp8_out, fp8_x, lane_id);
      store_as<FP8Storage>(pointer::offset(fp8_out, kRopeDim / 2), fp8_y, lane_id);
    } else {
      store_as<Storage>(input_x, input_vec_x, lane_id);
      store_as<Storage>(input_y, input_vec_y, lane_id);
    }
    if (cache_out != nullptr) {
      store_as<Storage>(cache_out, input_vec_x, lane_id);
      const auto cache_out_y = pointer::offset(cache_out, (kRopeDim / 2) * sizeof(T));
      store_as<Storage>(cache_out_y, input_vec_y, lane_id);
    }
  } else {
    static_assert(!kFP8Output, "FP8 Q fusion currently supports NeoX RoPE only");
    using CacheStorage = AlignedVector<float, kVecSize>;
    auto input_vec = load_as<Storage>(input, lane_id);
    const auto cos_vec = load_as<CacheStorage>(cos_ptr, lane_id);
    const auto sin_vec = load_as<CacheStorage>(sin_ptr, lane_id);
#pragma unroll
    for (int64_t j = 0; j < kVecSize; ++j) {
      const auto [x, y] = cast<fp32x2_t>(input_vec[j]);
      const auto cos = cos_vec[j];
      const auto sin = sin_vec[j];
      const auto out_x = x * cos - y * sin;
      const auto out_y = x * sin + y * cos;
      input_vec[j] = cast<T2, fp32x2_t>({out_x, out_y});
    }
    store_as<Storage>(input, input_vec, lane_id);
    if (cache_out != nullptr) {
      store_as<Storage>(cache_out, input_vec, lane_id);
    }
  }
}

template <
    bool kIsNeox,
    int64_t kRopeDim,
    bool kUsePDL,
    typename DType,
    typename QDType,
    typename IdType,
    uint32_t kWorkThreads,
    bool kFP8Output = false>
__global__ void fused_rope_kernel(const __grid_constant__ FusedRopeParams params) {
  using namespace device;

  constexpr int64_t kCosSinStrideBytes = kRopeDim * sizeof(float);
  constexpr int64_t kVecSize = next_pow2(kRopeDim, (2 * kWorkThreads * (1 + kIsNeox)));
  constexpr int64_t kDimPerThread = kVecSize * 2 * (1 + kIsNeox);
  constexpr uint32_t kLaneCount = kRopeDim / kDimPerThread;
  static_assert(kRopeDim % kDimPerThread == 0 && kLaneCount <= kWorkThreads);
  static_assert(sizeof(QDType) == sizeof(DType), "q/k must be same-size dtypes");

  const auto &[
    q, k, cos_sin_cache_ptr, positions, // pointers
    q_stride_bytes, k_stride_bytes, head_stride_bytes,  // strides
    num_qo_heads, num_kv_heads, num_tokens, // dimensions
    q_fp8 // optional FP8 Q output
  ] = params;

  const auto num_blks = gridDim.x;
  constexpr auto kWorkersPerBlock = kBlockSize / kWorkThreads;
  const auto num_workers = num_blks * kWorkersPerBlock;
  const auto num_q_and_k_heads = num_qo_heads + num_kv_heads;
  const auto num_works = num_q_and_k_heads * num_tokens;
  const auto start_worker_id = (blockIdx.x * kBlockSize + threadIdx.x) / kWorkThreads;
  const auto cos_cache_ptr = cos_sin_cache_ptr;
  const auto sin_cache_ptr = pointer::offset(cos_sin_cache_ptr, kCosSinStrideBytes / 2);

  uint32_t lane_id = threadIdx.x % kWorkThreads;
  if constexpr (kLaneCount < kWorkThreads) {
    if (lane_id >= kLaneCount) return;
  }

  PDLWaitPrimary<kUsePDL>();

  for (auto idx = start_worker_id; idx < num_works; idx += num_workers) {
    const int64_t token_id = idx / num_q_and_k_heads;
    const int64_t head_id = idx % num_q_and_k_heads;
    const auto pos = static_cast<const IdType*>(positions)[token_id];
    const auto load_q = head_id < num_qo_heads;
    const auto input_ = load_q ? pointer::offset(q, token_id * q_stride_bytes)  //
                               : pointer::offset(k, token_id * k_stride_bytes);
    const auto input = pointer::offset(input_, head_id * head_stride_bytes);
    const auto cos_ptr = pointer::offset(cos_cache_ptr, pos * kCosSinStrideBytes);
    const auto sin_ptr = pointer::offset(sin_cache_ptr, pos * kCosSinStrideBytes);
    if (load_q) {
      void* output = nullptr;
      if constexpr (kFP8Output)
        output = pointer::offset(q_fp8, (token_id * num_qo_heads + head_id) * kRopeDim);
      rope_rotate_head<QDType, kIsNeox, kRopeDim, kVecSize, kFP8Output>(input, cos_ptr, sin_ptr, lane_id, nullptr, output);
    } else {
      rope_rotate_head<DType, kIsNeox, kRopeDim, kVecSize>(input, cos_ptr, sin_ptr, lane_id);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <
    bool kIsNeox,
    int64_t kRopeDim,
    bool kUsePDL,
    typename DType,
    typename QDType,
    typename IdType,
    uint32_t kWorkThreads>
__global__ void fused_rope_store_kernel(const __grid_constant__ FusedRopeStoreParams params) {
  using namespace device;

  constexpr int64_t kCosSinStrideBytes = kRopeDim * sizeof(float);
  constexpr int64_t kVecSize = kRopeDim / (2 * kWorkThreads * (1 + kIsNeox));
  constexpr int64_t kDimPerThread = kVecSize * 2 * (1 + kIsNeox);
  static_assert(kRopeDim == kDimPerThread * kWorkThreads);
  static_assert(sizeof(QDType) == sizeof(DType), "q/k must be same-size dtypes");

  const auto& [base_params, v_ptr, k_cache, v_cache, out_loc, v_stride_bytes, cache_stride_bytes] = params;
  const auto &[
    q, k, cos_sin_cache_ptr, positions, // pointers
    q_stride_bytes, k_stride_bytes, head_stride_bytes,  // strides
    num_qo_heads, num_kv_heads, num_tokens, // dimensions
    q_fp8 // optional FP8 Q output
  ] = base_params;

  const auto num_blks = gridDim.x;
  constexpr auto kWorkersPerBlock = kBlockSize / kWorkThreads;
  const auto num_workers = num_blks * kWorkersPerBlock;
  const auto num_q_and_k_heads = num_qo_heads + num_kv_heads;
  const auto num_works = num_q_and_k_heads * num_tokens;
  const auto num_extra_works = num_kv_heads * num_tokens;  // rope works + v store works
  const auto start_worker_id = (blockIdx.x * kBlockSize + threadIdx.x) / kWorkThreads;
  const auto lane_id = threadIdx.x % kWorkThreads;
  const auto cos_cache_ptr = cos_sin_cache_ptr;
  const auto sin_cache_ptr = pointer::offset(cos_sin_cache_ptr, kCosSinStrideBytes / 2);

  auto idx = start_worker_id;

  PDLWaitPrimary<kUsePDL>();
  // in this case, head_dim = rope_dim must be true
  __builtin_assume(head_stride_bytes == kRopeDim * sizeof(DType));

  for (; idx < num_works; idx += num_workers) {
    const int64_t token_id = idx / num_q_and_k_heads;
    const int64_t head_id = idx % num_q_and_k_heads;
    const auto pos = static_cast<const IdType*>(positions)[token_id];
    const auto loc = static_cast<const IdType*>(out_loc)[token_id];
    const auto load_q = head_id < num_qo_heads;
    const auto input_ = load_q ? pointer::offset(q, token_id * q_stride_bytes)  //
                               : pointer::offset(k, token_id * k_stride_bytes);
    const auto input = pointer::offset(input_, head_id * head_stride_bytes);
    const auto cos_ptr = pointer::offset(cos_cache_ptr, pos * kCosSinStrideBytes);
    const auto sin_ptr = pointer::offset(sin_cache_ptr, pos * kCosSinStrideBytes);
    if (load_q) {
      // q rotates in place only (no cache store); may be a different dtype.
      rope_rotate_head<QDType, kIsNeox, kRopeDim, kVecSize>(input, cos_ptr, sin_ptr, lane_id);
    } else {
      const auto k_out = pointer::offset(k_cache, loc * cache_stride_bytes, head_id * head_stride_bytes);
      rope_rotate_head<DType, kIsNeox, kRopeDim, kVecSize>(input, cos_ptr, sin_ptr, lane_id, k_out);
    }
  }

  __syncwarp();  // to avoid warp divergence
  idx -= num_works;
  for (; idx < num_extra_works; idx += num_workers) {
    using VStorage = AlignedVector<DType, kRopeDim / kWorkThreads>;
    const int64_t token_id = idx / num_kv_heads;
    const int64_t head_id = idx % num_kv_heads;
    const auto loc = static_cast<const IdType*>(out_loc)[token_id];
    const auto input = pointer::offset(v_ptr, token_id * v_stride_bytes, head_id * head_stride_bytes);
    const auto input_vec = load_as<VStorage>(input, lane_id);
    const auto output = pointer::offset(v_cache, loc * cache_stride_bytes, head_id * head_stride_bytes);
    store_as<VStorage>(output, input_vec, lane_id);
  }
  PDLTriggerSecondary<kUsePDL>();
}

// Qwen3 attention prologue for BF16 QKV, head_dim 128, NeoX RoPE: per-head
// RMSNorm, cached RoPE, FP8 Q (no BF16 Q writeback), and either BF16 K in place or, with
// kStoreKV, the unscaled FP8 K/V cache rows that set_kv_buffer would write.
//
// Every head is still processed by one full warp with the original 32-lane
// layout (lane l holds elements 2l, 2l+1 and 64+2l, 64+2l+1), so the RMSNorm
// sum-of-squares order, the warp reduction tree and every rounding step are
// those of the previous per-head kernel. Only the work distribution changed:
// a warp owns one token's heads {g, g + G, g + 2G, ...}, loads the position,
// cos/sin and both norm weights once, and issues a chunk of head loads before
// any reduction so each warp keeps several 256-byte head rows in flight.
constexpr uint32_t kQKNormRopeWarps = 4;
constexpr uint32_t kQKNormRopeChunk = 6;

struct QKNormRopeFP8Params {
  const void* __restrict__ q;
  void* __restrict__ k;
  const void* __restrict__ v;
  const void* __restrict__ cos_sin_cache;
  const void* __restrict__ positions;
  const void* __restrict__ q_weight;
  const void* __restrict__ k_weight;
  void* __restrict__ q_fp8;
  void* __restrict__ k_cache;
  void* __restrict__ v_cache;
  const void* __restrict__ cache_loc;
  int64_t q_stride_bytes;
  int64_t k_stride_bytes;
  int64_t v_stride_bytes;
  int64_t cache_slots;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t num_tokens;
  uint32_t head_groups;
  float eps;
};

// Identical expressions to the former kQKNorm branch of rope_rotate_head.
SGL_DEVICE void qknorm_rope_head(
    bf16x2_t& x, bf16x2_t& y, const bf16x2_t wx, const bf16x2_t wy, const fp32x2_t cos_pair, const fp32x2_t sin_pair, const float eps) {
  using namespace device;
  AlignedVector<bf16x2_t, 2> input_vec, weight_vec;
  input_vec[0] = x;
  input_vec[1] = y;
  weight_vec[0] = wx;
  weight_vec[1] = wy;
  const auto normalized = norm::apply_norm_warp<128>(input_vec, weight_vec, eps);
  const auto [x0, x1] = cast<fp32x2_t>(normalized[0]);
  const auto [y0, y1] = cast<fp32x2_t>(normalized[1]);
  const auto [cos_0, cos_1] = cos_pair;
  const auto [sin_0, sin_1] = sin_pair;
  const auto out_x0 = x0 * cos_0 - y0 * sin_0;
  const auto out_y0 = x0 * sin_0 + y0 * cos_0;
  const auto out_x1 = x1 * cos_1 - y1 * sin_1;
  const auto out_y1 = x1 * sin_1 + y1 * cos_1;
  x = cast<bf16x2_t, fp32x2_t>({out_x0, out_x1});
  y = cast<bf16x2_t, fp32x2_t>({out_y0, out_y1});
}

// BF16 -> FP8 E4M3 exactly as torch.Tensor.to(torch.float8_e4m3fn).
SGL_DEVICE uint16_t to_fp8x2(const bf16x2_t value) {
#ifdef USE_ROCM
  static_assert(sizeof(bf16x2_t) == 0, "FP8 QK-norm RoPE requires CUDA");
  return 0;
#else
  // Same exact NOSAT repair as qkv_prefill_fp8; keep the BF16 input rounding.
  const auto x = device::cast<fp32x2_t>(value);
  const uint16_t packed = __nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
  const unsigned a = __float_as_uint(x.x), b = __float_as_uint(x.y);
  const unsigned aa = a & 0x7fffffff, bb = b & 0x7fffffff;
  // 464 is the RNE midpoint above 448: the tie stays finite (even code).
  // Overflow and infinity retain sign; input NaNs become positive canonical NaN.
  const unsigned lo = aa > 0x43e80000 ? (aa > 0x7f800000 ? 0x7f : ((a >> 24) & 0x80) | 0x7f) : packed & 0xff;
  const unsigned hi = bb > 0x43e80000 ? (bb > 0x7f800000 ? 0x7f : ((b >> 24) & 0x80) | 0x7f) : packed >> 8;
  return lo | (hi << 8);
#endif
}

template <bool kUsePDL, typename IdType, bool kStoreKV>
__global__ void __launch_bounds__(kQKNormRopeWarps* device::kWarpThreads)
    qknorm_rope_fp8_kernel(const __grid_constant__ QKNormRopeFP8Params params) {
  using namespace device;
  using Pair = AlignedVector<bf16x2_t, 1>;
  using CachePair = AlignedVector<fp32x2_t, 1>;
  constexpr int64_t kDim = 128;
  constexpr int64_t kHeadBytes = kDim * sizeof(bf16_t);
  constexpr int64_t kHalfBytes = kHeadBytes / 2;
  constexpr int64_t kCosSinStrideBytes = kDim * sizeof(float);

  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const uint32_t warp_id = blockIdx.x * kQKNormRopeWarps + threadIdx.x / kWarpThreads;
  const uint32_t token = warp_id / params.head_groups;
  const uint32_t group = warp_id % params.head_groups;
  if (token >= params.num_tokens) return;  // warp-uniform

  PDLWaitPrimary<kUsePDL>();

  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_heads = num_qo_heads + params.num_kv_heads;
  const int64_t pos = static_cast<const IdType*>(params.positions)[token];
  const auto cos_ptr = pointer::offset(params.cos_sin_cache, pos * kCosSinStrideBytes);
  const auto cos_pair = load_as<CachePair>(cos_ptr, lane_id)[0];
  const auto sin_pair = load_as<CachePair>(pointer::offset(cos_ptr, kCosSinStrideBytes / 2), lane_id)[0];
  const auto q_wx = load_as<Pair>(params.q_weight, lane_id)[0];
  const auto q_wy = load_as<Pair>(pointer::offset(params.q_weight, kHalfBytes), lane_id)[0];
  const auto k_wx = load_as<Pair>(params.k_weight, lane_id)[0];
  const auto k_wy = load_as<Pair>(pointer::offset(params.k_weight, kHalfBytes), lane_id)[0];
  const auto q_row = pointer::offset(params.q, static_cast<int64_t>(token) * params.q_stride_bytes);
  // k is pre-offset by -num_qo_heads heads on the host, as in fused_rope_kernel.
  const auto k_row = pointer::offset(params.k, static_cast<int64_t>(token) * params.k_stride_bytes);

  const uint32_t stride = params.head_groups * kQKNormRopeChunk;
  for (uint32_t first = group; first < num_heads; first += stride) {
    bf16x2_t xs[kQKNormRopeChunk], ys[kQKNormRopeChunk];
#pragma unroll
    for (uint32_t i = 0; i < kQKNormRopeChunk; ++i) {
      const uint32_t head = first + i * params.head_groups;
      if (head < num_heads) {
        const auto input = head < num_qo_heads ? pointer::offset(q_row, head * kHeadBytes) : pointer::offset(static_cast<const void*>(k_row), head * kHeadBytes);
        xs[i] = load_as<Pair>(input, lane_id)[0];
        ys[i] = load_as<Pair>(pointer::offset(input, kHalfBytes), lane_id)[0];
      }
    }
#pragma unroll
    for (uint32_t i = 0; i < kQKNormRopeChunk; ++i) {
      const uint32_t head = first + i * params.head_groups;
      if (head >= num_heads) break;  // warp-uniform; heads increase with i
      auto x = xs[i];
      auto y = ys[i];
      if (head < num_qo_heads) {
        qknorm_rope_head(x, y, q_wx, q_wy, cos_pair, sin_pair, params.eps);
        const auto out = pointer::offset(params.q_fp8, (static_cast<int64_t>(token) * num_qo_heads + head) * kDim);
        store_as<uint16_t>(out, to_fp8x2(x), lane_id);
        store_as<uint16_t>(pointer::offset(out, kDim / 2), to_fp8x2(y), lane_id);
        continue;
      }
      qknorm_rope_head(x, y, k_wx, k_wy, cos_pair, sin_pair, params.eps);
      if constexpr (!kStoreKV) {
        // Without the fused store the caller writes the rotated BF16 K to the
        // cache itself. With it, the K input stays unmodified (komoriyuta's
        // fused KV-store contract, checked by check_rope_fp8_kv): nothing reads
        // the BF16 K after the FP8 cache row is written.
        const auto k_out = pointer::offset(k_row, head * kHeadBytes);
        store_as<bf16x2_t>(k_out, x, lane_id);
        store_as<bf16x2_t>(pointer::offset(k_out, kHalfBytes), y, lane_id);
      }
      if constexpr (kStoreKV) {
        // Mirror store_cache: device-asserted slot bound and the reserved
        // CUDA-graph padding slot 0, which padded rows target and must not
        // race on.
        const int64_t slot = static_cast<const IdType*>(params.cache_loc)[token];
        assert(slot >= 0 && slot < params.cache_slots);
        if (slot != 0) {
          const uint32_t kv_head = head - num_qo_heads;
          const int64_t row = (slot * params.num_kv_heads + kv_head) * kDim;
          const auto k_dst = pointer::offset(params.k_cache, row);
          store_as<uint16_t>(k_dst, to_fp8x2(x), lane_id);
          store_as<uint16_t>(pointer::offset(k_dst, kDim / 2), to_fp8x2(y), lane_id);
          using VQuad = AlignedVector<bf16x2_t, 2>;
          using FP8Quad = AlignedVector<uint16_t, 2>;
          const auto v_src = pointer::offset(params.v, static_cast<int64_t>(token) * params.v_stride_bytes + kv_head * kHeadBytes);
          const auto value = load_as<VQuad>(v_src, lane_id);
          FP8Quad fp8;
          fp8[0] = to_fp8x2(value[0]);
          fp8[1] = to_fp8x2(value[1]);
          store_as<FP8Quad>(pointer::offset(params.v_cache, row), fp8, lane_id);
        }
      }
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}



// Prefill: 32 tokens per CTA, eight lanes per head; exact hardware FP8 conversion.
__device__ __forceinline__ uint16_t qkv_prefill_fp8(float2 x) {
    // NOSAT in CUDA 12.9 falls back to double/integer conversion. Use the
    // hardware conversion and restore its exact overflow/canonical-NaN rules.
    const uint16_t packed=__nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
    const unsigned a=__float_as_uint(x.x), b=__float_as_uint(x.y);
    const unsigned aa=a&0x7fffffff, bb=b&0x7fffffff;
    const unsigned lo=aa>0x43e80000 ? (aa>0x7f800000 ? 0x7f : ((a>>24)&0x80)|0x7f) : packed&0xff;
    const unsigned hi=bb>0x43e80000 ? (bb>0x7f800000 ? 0x7f : ((b>>24)&0x80)|0x7f) : packed>>8;
    return lo|(hi<<8);
}

// Generated by prefill_layout_host.cu from pinned CuTe copy partitions.
__host__ __device__ inline int prefill_v_byte_offset(int token, int dim) {
    int x=(token%32)*128+dim;
    int out=(token/32)*4096;
    out ^= ((x >> 0) & 1) * 256;
    out ^= ((x >> 1) & 1) * 32;
    out ^= ((x >> 2) & 1) * 64;
    out ^= ((x >> 3) & 1) * 144;
    out ^= ((x >> 4) & 1) * 512;
    out ^= ((x >> 5) & 1) * 1024;
    out ^= ((x >> 6) & 1) * 2048;
    out ^= ((x >> 7) & 1) * 1;
    out ^= ((x >> 8) & 1) * 4;
    out ^= ((x >> 9) & 1) * 8;
    out ^= ((x >> 10) & 1) * 2;
    out ^= ((x >> 11) & 1) * 16;
    return out;
}

__device__ __forceinline__ float2 qkv_prefill_paired_rn(float a, float b) {
    uint32_t bits;
    // Same operand order as CUDA13 cuda_bf16.hpp: low16=a, high16=b.
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(bits) : "f"(a), "f"(b));
    return make_float2(__uint_as_float(bits << 16), __uint_as_float(bits & 0xffff0000u));
}

template<bool kUsePDL, typename IdType, bool kVImage=false>
__global__ __launch_bounds__(256) void qknorm_rope_prefill_fast_kernel(
    const __grid_constant__ QKNormRopeFP8Params p, void* contiguous_k, void* contiguous_v, void* v_image, int image_rows) {
    device::PDLWaitPrimary<kUsePDL>();
    constexpr int BM=32, THREADS=256;
    const int head=blockIdx.y, first_row=blockIdx.x*BM, rows=p.num_tokens;
    const int q_heads=p.num_qo_heads, kv_heads=p.num_kv_heads;
    const int lane = threadIdx.x % 8;
    const int col = lane * 8;
    float wx[8], wy[8];
    if (head < q_heads + kv_heads) {
        const auto w = static_cast<const __nv_bfloat16*>(head < q_heads ? p.q_weight : p.k_weight);
        const uint4 vx = *reinterpret_cast<const uint4*>(w + col);
        const uint4 vy = *reinterpret_cast<const uint4*>(w + 64 + col);
        #pragma unroll
        for (int i=0;i<8;++i) {
            wx[i] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&vx)[i]);
            wy[i] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&vy)[i]);
        }
    }
    // Same eight-lane/head reduction tree as the deployed SGLang prologue.
    for (int row = threadIdx.x / 8; row < BM; row += THREADS / 8) {
        const int token = first_row + row;
        float x[8], y[8], partial[4];
        uint16_t ox[4], oy[4];
        const bool is_q=head<q_heads, is_k=head<q_heads+kv_heads;
        const int h=is_k ? head : head-q_heads-kv_heads;
        const auto src=static_cast<const char*>(is_q ? p.q : (is_k ? p.k : p.v));
        const int64_t stride=is_q ? p.q_stride_bytes : (is_k ? p.k_stride_bytes : p.v_stride_bytes);
        const auto tile=reinterpret_cast<const __nv_bfloat16*>(src+int64_t(token)*stride+int64_t(h)*256);
        uint4 vx{}, vy{};
        if(token<rows) {
            vx=*reinterpret_cast<const uint4*>(tile+col);
            vy=*reinterpret_cast<const uint4*>(tile+64+col);
        }
        #pragma unroll
        for (int i=0;i<8;++i) {
            x[i] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&vx)[i]);
            y[i] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&vy)[i]);
        }
        if (head < q_heads + kv_heads) {
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
            const float scale=rsqrtf((sum02+sum13)/128.f+p.eps);
            const int64_t pos=token<rows?static_cast<const IdType*>(p.positions)[token]:0;
            const float4 cos4[2] = {*reinterpret_cast<const float4*>(static_cast<const float*>(p.cos_sin_cache)+pos*128+col), *reinterpret_cast<const float4*>(static_cast<const float*>(p.cos_sin_cache)+pos*128+col+4)};
            const float4 sin4[2] = {*reinterpret_cast<const float4*>(static_cast<const float*>(p.cos_sin_cache)+pos*128+64+col), *reinterpret_cast<const float4*>(static_cast<const float*>(p.cos_sin_cache)+pos*128+64+col+4)};
            #pragma unroll
            for(int j=0;j<4;++j) {
                // Pair adjacent BF16 RN-even conversions at the same two
                // rounding points; preserve each scalar expression and FMA.
                const float2 u=qkv_prefill_paired_rn(x[j*2]*scale*wx[j*2],x[j*2+1]*scale*wx[j*2+1]);
                const float2 v=qkv_prefill_paired_rn(y[j*2]*scale*wy[j*2],y[j*2+1]*scale*wy[j*2+1]);
                float a[2], b[2];
                #pragma unroll
                for(int h=0;h<2;++h) {
                    const int d=j*2+h;
                    const float uh=h==0?u.x:u.y, vh=h==0?v.x:v.y;
                    const float c=reinterpret_cast<const float*>(cos4)[d], s=reinterpret_cast<const float*>(sin4)[d];
                    a[h]=uh*c-vh*s;
                    b[h]=uh*s+vh*c;
                }
                const float2 ar=qkv_prefill_paired_rn(a[0],a[1]);
                const float2 br=qkv_prefill_paired_rn(b[0],b[1]);
                ox[j]=qkv_prefill_fp8(ar);
                oy[j]=qkv_prefill_fp8(br);
            }
        } else {
            #pragma unroll
            for(int j=0;j<4;++j) {
                ox[j]=qkv_prefill_fp8(make_float2(x[j*2],x[j*2+1]));
                oy[j]=qkv_prefill_fp8(make_float2(y[j*2],y[j*2+1]));
            }
        }
        if constexpr (kVImage) {
            if (head >= q_heads+kv_heads && token < ((image_rows+223)/224)*224) {
                // Every V CTA writes its full 32-token tile, including zero
                // padding through ceil(rows/224)*224. Row-major and paged V
                // below remain present for the fallback and Decode transfer.
                int blocks=(image_rows+223)/224;
                auto image=static_cast<uint8_t*>(v_image)+(h*blocks+token/224)*224*128;
                #pragma unroll
                for(int j=0;j<8;++j) {
                    image[prefill_v_byte_offset(token%224,col+j)]=token<image_rows?reinterpret_cast<uint8_t*>(ox)[j]:0;
                    image[prefill_v_byte_offset(token%224,64+col+j)]=token<image_rows?reinterpret_cast<uint8_t*>(oy)[j]:0;
                }
            }
        }
        if(token<rows) {
            const uint2 lo=*reinterpret_cast<const uint2*>(ox), hi=*reinterpret_cast<const uint2*>(oy);
            const int kv_head=head<q_heads+kv_heads ? head-q_heads : head-q_heads-kv_heads;
            uint8_t* dst=head<q_heads
                ? static_cast<uint8_t*>(p.q_fp8)+(int64_t(token)*q_heads+head)*128
                : static_cast<uint8_t*>(head<q_heads+kv_heads ? contiguous_k : contiguous_v)+(int64_t(token)*kv_heads+kv_head)*128;
            *reinterpret_cast<uint2*>(dst+col)=lo;
            *reinterpret_cast<uint2*>(dst+64+col)=hi;
            if(head>=q_heads) {
                const int64_t slot=static_cast<const IdType*>(p.cache_loc)[token];
                assert(slot>=0 && slot<p.cache_slots);
                if(slot!=0) {
                    dst=static_cast<uint8_t*>(head<q_heads+kv_heads ? p.k_cache : p.v_cache)+(slot*kv_heads+kv_head)*128;
                    *reinterpret_cast<uint2*>(dst+col)=lo;
                    *reinterpret_cast<uint2*>(dst+64+col)=hi;
                }
            }
        }
    }
    device::PDLTriggerSecondary<kUsePDL>();
}

// Decode: four independent H128 heads per warp, vectorized 16-byte halves.
// Virtual old lanes (4*lane+j) retain the original XOR sum and both BF16
// rounding points. No shared memory or CTA barrier on the latency path.
template <bool kUsePDL, typename IdType>
__global__ void __launch_bounds__(128) qknorm_rope_decode_kernel(
    const __grid_constant__ QKNormRopeFP8Params p) {
  using namespace device;
  using Vec = AlignedVector<bf16x2_t, 4>;
  using Out = AlignedVector<uint16_t, 4>;
  const uint32_t work = blockIdx.x * 16 + threadIdx.x / 8;
  const uint32_t heads = p.num_qo_heads + p.num_kv_heads;
  const uint32_t token = work / heads;
  const uint32_t head = work % heads;
  const uint32_t lane = threadIdx.x % 8;
  const bool valid = token < p.num_tokens;
  PDLWaitPrimary<kUsePDL>();
  Vec x{}, y{};
  if (valid) {
    const void* src = head < p.num_qo_heads
        ? pointer::offset(p.q, static_cast<int64_t>(token) * p.q_stride_bytes + head * 256)
        : pointer::offset(p.k, static_cast<int64_t>(token) * p.k_stride_bytes + head * 256);
    x = load_as<Vec>(src, lane);
    y = load_as<Vec>(pointer::offset(src, 128), lane);
  }
  float partial[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const auto a = cast<fp32x2_t>(x[j]);
    const auto b = cast<fp32x2_t>(y[j]);
    float sum = 0.f;
    sum += a.x * a.x;
    sum += a.y * a.y;
    sum += b.x * b.x;
    sum += b.y * b.y;
    partial[j] = sum;
  }
#pragma unroll
  for (int mask = 4; mask >= 1; mask >>= 1) {
#pragma unroll
    for (int j = 0; j < 4; ++j) partial[j] += __shfl_xor_sync(0xffffffff, partial[j], mask, 8);
  }
  const float sum02 = partial[0] + partial[2];
  const float sum13 = partial[1] + partial[3];
  const float scale = math::rsqrt((sum02 + sum13) / 128.f + p.eps);
  if (valid) {
    const auto w = head < p.num_qo_heads ? p.q_weight : p.k_weight;
    const auto wx = load_as<Vec>(w, lane);
    const auto wy = load_as<Vec>(pointer::offset(w, 128), lane);
    const int64_t pos = static_cast<const IdType*>(p.positions)[token];
    const auto trig = static_cast<const float*>(p.cos_sin_cache) + pos * 128;
    Out ox, oy;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      auto a = cast<fp32x2_t>(x[j]);
      auto b = cast<fp32x2_t>(y[j]);
      const auto u = cast<fp32x2_t>(wx[j]);
      const auto v = cast<fp32x2_t>(wy[j]);
      a = cast<fp32x2_t>(cast<bf16x2_t, fp32x2_t>({a.x * scale * u.x, a.y * scale * u.y}));
      b = cast<fp32x2_t>(cast<bf16x2_t, fp32x2_t>({b.x * scale * v.x, b.y * scale * v.y}));
      const int d = lane * 8 + j * 2;
      const float c0 = trig[d], c1 = trig[d + 1], s0 = trig[64 + d], s1 = trig[65 + d];
      ox[j] = to_fp8x2(cast<bf16x2_t, fp32x2_t>({a.x * c0 - b.x * s0, a.y * c1 - b.y * s1}));
      oy[j] = to_fp8x2(cast<bf16x2_t, fp32x2_t>({a.x * s0 + b.x * c0, a.y * s1 + b.y * c1}));
    }
    if (head < p.num_qo_heads) {
      auto dst = pointer::offset(p.q_fp8, (static_cast<int64_t>(token) * p.num_qo_heads + head) * 128);
      store_as<Out>(dst, ox, lane);
      store_as<Out>(pointer::offset(dst, 64), oy, lane);
    } else {
      const int h = head - p.num_qo_heads;
      const int64_t slot = static_cast<const IdType*>(p.cache_loc)[token];
      assert(slot >= 0 && slot < p.cache_slots);
      if (slot != 0) {
        const int64_t offset = (slot * p.num_kv_heads + h) * 128;
        auto dst = pointer::offset(p.k_cache, offset);
        store_as<Out>(dst, ox, lane);
        store_as<Out>(pointer::offset(dst, 64), oy, lane);
        const auto src = pointer::offset(p.v, static_cast<int64_t>(token) * p.v_stride_bytes + h * 256);
        const auto vx = load_as<Vec>(src, lane);
        const auto vy = load_as<Vec>(pointer::offset(src, 128), lane);
        Out ovx, ovy;
#pragma unroll
        for (int j = 0; j < 4; ++j) { ovx[j] = to_fp8x2(vx[j]); ovy[j] = to_fp8x2(vy[j]); }
        dst = pointer::offset(p.v_cache, offset);
        store_as<Out>(dst, ovx, lane);
        store_as<Out>(pointer::offset(dst, 64), ovy, lane);
      }
    }
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct QKNormRopeFP8Kernel {
  template <typename IdType, bool kStoreKV>
  static constexpr auto _kernel = qknorm_rope_fp8_kernel<kUsePDL, IdType, kStoreKV>;

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      double eps,
      const tvm::ffi::TensorView q_fp8) {
    _run<false>(q, k, q_weight, k_weight, cos_sin_cache, positions, eps, q_fp8, nullptr, nullptr, nullptr, nullptr);
  }

  static void run_store(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      double eps,
      const tvm::ffi::TensorView q_fp8,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView cache_loc) {
    _run<true>(q, k, q_weight, k_weight, cos_sin_cache, positions, eps, q_fp8, &v, &k_cache, &v_cache, &cache_loc);
  }

  static void run_prefill(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      double eps,
      const tvm::ffi::TensorView q_fp8,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView cache_loc,
      const tvm::ffi::TensorView contiguous_k,
      const tvm::ffi::TensorView contiguous_v) {
    _run<true>(q, k, q_weight, k_weight, cos_sin_cache, positions, eps, q_fp8, &v, &k_cache, &v_cache, &cache_loc, &contiguous_k, &contiguous_v);
  }

  static void run_prefill_image(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      double eps,
      const tvm::ffi::TensorView q_fp8,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView cache_loc,
      const tvm::ffi::TensorView contiguous_k,
      const tvm::ffi::TensorView contiguous_v,
      const tvm::ffi::TensorView v_image, int64_t image_rows) {
    _run<true>(q, k, q_weight, k_weight, cos_sin_cache, positions, eps, q_fp8, &v, &k_cache, &v_cache, &cache_loc, &contiguous_k, &contiguous_v, &v_image, image_rows);
  }

  template <bool kStoreKV>
  static void _run(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      double eps,
      const tvm::ffi::TensorView q_fp8,
      const tvm::ffi::TensorView* v,
      const tvm::ffi::TensorView* k_cache,
      const tvm::ffi::TensorView* v_cache,
      const tvm::ffi::TensorView* cache_loc,
      const tvm::ffi::TensorView* contiguous_k = nullptr,
      const tvm::ffi::TensorView* contiguous_v = nullptr,
      const tvm::ffi::TensorView* v_image = nullptr, int64_t image_rows = 0) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto Q = SymbolicSize{"num_qo_heads"};
    auto K = SymbolicSize{"num_kv_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto S = SymbolicSize{"cache_slots"};
    auto Dq = SymbolicSize{"q_stride"};
    auto Dk = SymbolicSize{"k_stride"};
    auto Dv = SymbolicSize{"v_stride"};
    auto device = SymbolicDevice{};
    auto id_type = SymbolicDType{};
    D.set_value(128);
    device.set_options<kDLCUDA>();
    TensorMatcher({N, Q, D}).with_strides({Dq, D, 1}).with_dtype<bf16_t>().with_device(device).verify(q);
    TensorMatcher({N, K, D}).with_strides({Dk, D, 1}).with_dtype<bf16_t>().with_device(device).verify(k);
    TensorMatcher({D}).with_dtype<bf16_t>().with_device(device).verify(q_weight).verify(k_weight);
    TensorMatcher({-1, D}).with_dtype<float>().with_device(device).verify(cos_sin_cache);
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(id_type).with_device(device).verify(positions);
    TensorMatcher({N, Q, D}).with_dtype<fp8_e4m3_t>().with_device(device).verify(q_fp8);
    if constexpr (kStoreKV) {
      TensorMatcher({N, K, D}).with_strides({Dv, D, 1}).with_dtype<bf16_t>().with_device(device).verify(*v);
      TensorMatcher({S, K, D}).with_dtype<fp8_e4m3_t>().with_device(device).verify(*k_cache).verify(*v_cache);
      TensorMatcher({N}).with_dtype<int32_t, int64_t>(id_type).with_device(device).verify(*cache_loc);
    }
    if (contiguous_k) {
      TensorMatcher({N, K, D}).with_dtype<fp8_e4m3_t>().with_device(device).verify(*contiguous_k).verify(*contiguous_v);
    }
    if (v_image) {
      RuntimeCheck(contiguous_k != nullptr, "Prefill image requires contiguous KV");
      TensorMatcher({4, static_cast<int64_t>((image_rows+223)/224), 224*128}).with_dtype<uint8_t>().with_device(device).verify(*v_image);
      RuntimeCheck(image_rows>=3584 && image_rows<=4096 && image_rows<=N.unwrap() && K.unwrap()==4 && Q.unwrap()==64, "Fixed Prefill image shape");
    }
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;
    RuntimeCheck(q_fp8.data_ptr() != q.data_ptr() && q_fp8.data_ptr() != k.data_ptr(), "FP8 Q output must not alias inputs");
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());
    const auto num_heads = num_qo_heads + num_kv_heads;
    constexpr int64_t kHeadBytes = 128 * sizeof(bf16_t);
    // Large (Prefill) batches: four warps per token, one CTA per token.
    // Small (Decode) batches: one chunk of heads per warp for latency.
    const uint32_t head_groups = num_tokens >= 1024 ? kQKNormRopeWarps : div_ceil(num_heads, kQKNormRopeChunk);
    const auto params = QKNormRopeFP8Params{
        .q = q.data_ptr(),
        .k = pointer::offset(k.data_ptr(), -static_cast<int64_t>(num_qo_heads) * kHeadBytes),
        .v = kStoreKV ? v->data_ptr() : nullptr,
        .cos_sin_cache = cos_sin_cache.data_ptr(),
        .positions = positions.data_ptr(),
        .q_weight = q_weight.data_ptr(),
        .k_weight = k_weight.data_ptr(),
        .q_fp8 = q_fp8.data_ptr(),
        .k_cache = kStoreKV ? k_cache->data_ptr() : nullptr,
        .v_cache = kStoreKV ? v_cache->data_ptr() : nullptr,
        .cache_loc = kStoreKV ? cache_loc->data_ptr() : nullptr,
        .q_stride_bytes = static_cast<int64_t>(Dq.unwrap() * sizeof(bf16_t)),
        .k_stride_bytes = static_cast<int64_t>(Dk.unwrap() * sizeof(bf16_t)),
        .v_stride_bytes = kStoreKV ? static_cast<int64_t>(Dv.unwrap() * sizeof(bf16_t)) : 0,
        .cache_slots = kStoreKV ? static_cast<int64_t>(S.unwrap()) : 0,
        .num_qo_heads = num_qo_heads,
        .num_kv_heads = num_kv_heads,
        .num_tokens = num_tokens,
        .head_groups = head_groups,
        .eps = static_cast<float>(eps),
    };
    if (contiguous_k) {
      const auto prefill = v_image
          ? (id_type.is_type<int32_t>() ? qknorm_rope_prefill_fast_kernel<kUsePDL, int32_t, true> : qknorm_rope_prefill_fast_kernel<kUsePDL, int64_t, true>)
          : (id_type.is_type<int32_t>() ? qknorm_rope_prefill_fast_kernel<kUsePDL, int32_t> : qknorm_rope_prefill_fast_kernel<kUsePDL, int64_t>);
      const auto tiles = v_image ? std::max(div_ceil(num_tokens, 32u), static_cast<uint32_t>((image_rows+223)/224*7)) : div_ceil(num_tokens, 32u);
      LaunchKernel(dim3(tiles, num_qo_heads+2*num_kv_heads), 256, device.unwrap()).enable_pdl(kUsePDL)(prefill, params, contiguous_k->data_ptr(), contiguous_v->data_ptr(), v_image ? v_image->data_ptr() : nullptr, image_rows);
      return;
    }
    if constexpr (kStoreKV) {
      if (num_tokens <= 192 && num_qo_heads == 64 && num_kv_heads == 4) {
        const auto decode = id_type.is_type<int32_t>() ? qknorm_rope_decode_kernel<kUsePDL, int32_t> : qknorm_rope_decode_kernel<kUsePDL, int64_t>;
        LaunchKernel(div_ceil(static_cast<uint64_t>(num_tokens) * num_heads, uint64_t{16}), 128, device.unwrap()).enable_pdl(kUsePDL)(decode, params);
        return;
      }
    }
    const auto kernel = id_type.is_type<int32_t>() ? _kernel<int32_t, kStoreKV> : _kernel<int64_t, kStoreKV>;
    const auto num_blocks = div_ceil(static_cast<uint64_t>(num_tokens) * head_groups, kQKNormRopeWarps);
    LaunchKernel(static_cast<uint32_t>(num_blocks), kQKNormRopeWarps * device::kWarpThreads, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

// QDType (default = DType) is q's dtype; k/v/caches use DType. Same-size
// dtypes only (fp16 q with bf16 k for the fp8-KV fp16-Q feed).
template <bool kIsNeox, int64_t kRopeDim, bool kUsePDL, typename DType, typename QDType = DType>
struct FusedRopeKernel {
  static constexpr uint32_t kDimPerThread = std::gcd(16 / sizeof(DType), kRopeDim);
  static constexpr uint32_t kWorkThreads = next_pow2(kRopeDim, kDimPerThread);
  static constexpr bool kSupportFused = kWorkThreads * kDimPerThread == kRopeDim;
  static_assert(kRopeDim % kDimPerThread == 0);
  static_assert(kBlockSize % kWorkThreads == 0);
  static_assert(sizeof(QDType) == sizeof(DType), "q/k must be same-size dtypes");

  template <typename IdType, bool kFP8Output = false>
  static constexpr auto _kernel_0 = fused_rope_kernel<kIsNeox, kRopeDim, kUsePDL, DType, QDType, IdType, kWorkThreads, kFP8Output>;
  template <typename IdType>
  static constexpr auto _kernel_1 =
      fused_rope_store_kernel<kIsNeox, kRopeDim, kUsePDL, DType, QDType, IdType, kWorkThreads>;

  static auto get_num_sm(DLDevice device) {
    static const auto kNumSM = host::runtime::get_sm_count(device.device_id);
    return kNumSM;
  }

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions) {
    _run<false>(q, k, cos_sin_cache, positions, q);
  }

  static void run_fp8(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView q_fp8) {
    static_assert(kIsNeox && kRopeDim == 128 && std::is_same_v<QDType, bf16_t>);
    _run<true>(q, k, cos_sin_cache, positions, q_fp8);
  }

  // Entry points of qknorm-rope-fp8.patch / qknorm-rope-fp8-kv.patch, now served
  // by the token-warp kernel above (same per-head arithmetic).
  static void run_norm_fp8(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      double eps,
      const tvm::ffi::TensorView q_fp8) {
    static_assert(kIsNeox && kRopeDim == 128 && std::is_same_v<QDType, bf16_t> && std::is_same_v<DType, bf16_t>);
    QKNormRopeFP8Kernel<kUsePDL>::run(q, k, q_weight, k_weight, cos_sin_cache, positions, eps, q_fp8);
  }

  static void run_norm_fp8_store(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      double eps,
      const tvm::ffi::TensorView q_fp8,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView cache_loc) {
    static_assert(kIsNeox && kRopeDim == 128 && std::is_same_v<QDType, bf16_t> && std::is_same_v<DType, bf16_t>);
    QKNormRopeFP8Kernel<kUsePDL>::run_store(q, k, q_weight, k_weight, cos_sin_cache, positions, eps, q_fp8, v, k_cache, v_cache, cache_loc);
  }

  template <bool kFP8Output>
  static void _run(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView q_fp8) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto Q = SymbolicSize{"num_qo_heads"};
    auto K = SymbolicSize{"num_kv_heads"};
    auto D = SymbolicSize{"rope_dim"};
    auto Dq = SymbolicSize{"q_stride"};
    auto Dk = SymbolicSize{"k_stride"};
    auto Dd = SymbolicSize{"head_stride"};
    auto device = SymbolicDevice{};
    auto id_type = SymbolicDType{};
    D.set_value(kRopeDim);
    device.set_options<kDLCUDA>();
    TensorMatcher({N, Q, D})  // q input
        .with_strides({Dq, Dd, 1})
        .with_dtype<QDType>()
        .with_device(device)
        .verify(q);
    TensorMatcher({N, K, D})  // k input
        .with_strides({Dk, Dd, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k);
    TensorMatcher({-1, D})  // cos_sin_cache
        .with_dtype<float>()
        .with_device(device)
        .verify(cos_sin_cache);
    TensorMatcher({N})  // positions
        .with_dtype<int32_t, int64_t>(id_type)
        .with_device(device)
        .verify(positions);

    if constexpr (kFP8Output) {
      TensorMatcher({N, Q, D}).with_dtype<fp8_e4m3_t>().with_device(device).verify(q_fp8);
      if (N.unwrap() != 0)
        RuntimeCheck(q_fp8.data_ptr() != q.data_ptr() && q_fp8.data_ptr() != k.data_ptr(), "FP8 Q output must not alias inputs");
    }
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());
    const auto q_stride_bytes = static_cast<int64_t>(Dq.unwrap() * sizeof(DType));
    const auto k_stride_bytes = static_cast<int64_t>(Dk.unwrap() * sizeof(DType));
    const auto head_stride_bytes = static_cast<int64_t>(Dd.unwrap() * sizeof(DType));

    // NOTE: we offset the k here to reduce computation cost in the kernel
    const int64_t k_offset = static_cast<int64_t>(num_qo_heads) * head_stride_bytes;
    const auto params = FusedRopeParams{
        .q_ptr = q.data_ptr(),
        .k_ptr = pointer::offset(k.data_ptr(), -k_offset),
        .cos_sin_cache_ptr = cos_sin_cache.data_ptr(),
        .positions = positions.data_ptr(),
        .q_stride_bytes = q_stride_bytes,
        .k_stride_bytes = k_stride_bytes,
        .head_stride_bytes = head_stride_bytes,
        .num_qo_heads = num_qo_heads,
        .num_kv_heads = num_kv_heads,
        .num_tokens = num_tokens,
        .q_fp8 = kFP8Output ? q_fp8.data_ptr() : nullptr,
    };

    const auto is_int32 = id_type.is_type<int32_t>();
    const auto kernel = is_int32 ? _kernel_0<int32_t, kFP8Output> : _kernel_0<int64_t, kFP8Output>;
    const uint32_t kNumSM = get_num_sm(device.unwrap());
    static const uint32_t kOccupancyTable[2] = {
        runtime::get_blocks_per_sm(_kernel_0<int32_t, kFP8Output>, kBlockSize),
        runtime::get_blocks_per_sm(_kernel_0<int64_t, kFP8Output>, kBlockSize),
    };
    const auto max_blocks = kOccupancyTable[is_int32 ? 0 : 1] * kNumSM;
    const auto num_works = (num_qo_heads + num_kv_heads) * num_tokens;
    const auto needed_blocks = div_ceil(num_works, (kBlockSize / kWorkThreads));
    const auto num_blocks = std::min(max_blocks, needed_blocks);
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }

  static void run_fused(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView out_loc) {
    if constexpr (kSupportFused) {
      return _run_fused_impl(q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc);
    } else {
      host::Panic("Fused rope + store is not supported for rope_dim ", kRopeDim);
    }
  }

  static void _run_fused_impl(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView out_loc) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto Q = SymbolicSize{"num_qo_heads"};
    auto K = SymbolicSize{"num_kv_heads"};
    auto D = SymbolicSize{"rope_dim"};
    auto R = SymbolicSize{"row_size"};
    auto Dq = SymbolicSize{"q_stride"};
    auto Dk = SymbolicSize{"k_stride"};
    auto Dv = SymbolicSize{"v_stride"};
    auto Dd = SymbolicSize{"head_stride"};
    auto Dc = SymbolicSize{"cache_stride"};
    auto device = SymbolicDevice{};
    auto id_type = SymbolicDType{};
    D.set_value(kRopeDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, Q, D})  // q input
        .with_strides({Dq, Dd, 1})
        .with_dtype<QDType>()
        .with_device(device)
        .verify(q);
    TensorMatcher({N, K, D})  // k input
        .with_strides({Dk, Dd, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k);
    TensorMatcher({N, K, D})  // v input
        .with_strides({Dv, Dd, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(v);
    TensorMatcher({-1, D})  // cos_sin_cache
        .with_dtype<float>()
        .with_device(device)
        .verify(cos_sin_cache);
    TensorMatcher({N})  // positions, out_loc
        .with_dtype<int32_t, int64_t>(id_type)
        .with_device(device)
        .verify(positions)
        .verify(out_loc);
    TensorMatcher({-1, R})  // k_cache
        .with_strides({Dc, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k_cache)
        .verify(v_cache);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());
    const auto q_stride_bytes = static_cast<int64_t>(Dq.unwrap() * sizeof(DType));
    const auto k_stride_bytes = static_cast<int64_t>(Dk.unwrap() * sizeof(DType));
    const auto head_stride = Dd.unwrap();
    const auto row_dim = R.unwrap();
    const auto head_stride_bytes = static_cast<int64_t>(Dd.unwrap() * sizeof(DType));

    RuntimeCheck(kRopeDim == head_stride, "rope_dim ", kRopeDim, " should = head_stride ", head_stride);
    RuntimeCheck(num_kv_heads * kRopeDim == row_dim, "invalid kvcache");

    // NOTE: we offset the k here to reduce computation cost in the kernel
    const int64_t k_offset = static_cast<int64_t>(num_qo_heads) * head_stride_bytes;
    const auto params = FusedRopeParams{
        .q_ptr = q.data_ptr(),
        .k_ptr = pointer::offset(k.data_ptr(), -k_offset),
        .cos_sin_cache_ptr = cos_sin_cache.data_ptr(),
        .positions = positions.data_ptr(),
        .q_stride_bytes = q_stride_bytes,
        .k_stride_bytes = k_stride_bytes,
        .head_stride_bytes = head_stride_bytes,
        .num_qo_heads = num_qo_heads,
        .num_kv_heads = num_kv_heads,
        .num_tokens = num_tokens,
    };

    const auto v_stride_bytes = static_cast<int64_t>(Dv.unwrap() * sizeof(DType));
    const auto cache_stride_bytes = static_cast<int64_t>(Dc.unwrap() * sizeof(DType));
    const auto store_params = FusedRopeStoreParams{
        .base_params = params,
        .v_ptr = v.data_ptr(),
        .k_cache = pointer::offset(k_cache.data_ptr(), -k_offset),
        .v_cache = v_cache.data_ptr(),
        .out_loc = out_loc.data_ptr(),
        .v_stride_bytes = v_stride_bytes,
        .cache_stride_bytes = cache_stride_bytes,
    };

    const auto is_int32 = id_type.is_type<int32_t>();
    const auto kernel = is_int32 ? _kernel_1<int32_t> : _kernel_1<int64_t>;
    const uint32_t kNumSM = get_num_sm(device.unwrap());
    static const uint32_t kOccupancyTable[2] = {
        runtime::get_blocks_per_sm(_kernel_1<int32_t>, kBlockSize),
        runtime::get_blocks_per_sm(_kernel_1<int64_t>, kBlockSize),
    };
    const auto max_blocks = kOccupancyTable[is_int32 ? 0 : 1] * kNumSM;
    // rope works for q+k heads, plus v store works for kv heads
    const auto num_total_works = (num_qo_heads + 2 * num_kv_heads) * num_tokens;
    const auto needed_blocks = div_ceil(num_total_works, (kBlockSize / kWorkThreads));
    const auto num_blocks = std::min(max_blocks, needed_blocks);
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, store_params);
  }
};


}  // namespace
