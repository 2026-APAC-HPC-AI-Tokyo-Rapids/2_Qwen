// Derived read-only from SGLang fc275674 SM90 launcher/helper; CUTLASS57e3cfb.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/arch/arch.h>
#include <torch/all.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAException.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/tensor_view_io.h"

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

template <
    typename ElementAB,
    typename ElementC,
    typename ElementAccumulator,
    typename LayoutSFA,
    typename LayoutSFB,
    typename ScaleConfig>
__global__ void get_group_gemm_starts(
    int32_t* expert_offsets,
    ElementAB** a_offsets,
    ElementAB** b_offsets,
    ElementC** out_offsets,
    ElementAccumulator** a_scales_offsets,
    ElementAccumulator** b_scales_offsets,
    ElementAB* a_base_as_int,
    ElementAB* b_base_as_int,
    ElementC* out_base_as_int,
    ElementAccumulator* a_scales_base_as_int,
    ElementAccumulator* b_scales_base_as_int,
    LayoutSFA* layout_sfa_base_as_int,
    LayoutSFB* layout_sfb_base_as_int,
    int* problem_sizes,
    int* problem_sizes_transpose,
    bool transpose = false) {
  int64_t expert_id = static_cast<int64_t>(threadIdx.x);

  if (expert_id >= gridDim.x * blockDim.x) {
    return;
  }

  int m = problem_sizes[expert_id * 3];
  int n = problem_sizes[expert_id * 3 + 1];
  int k = problem_sizes[expert_id * 3 + 2];
  if (transpose) {
    problem_sizes_transpose[expert_id * 3] = n;
    problem_sizes_transpose[expert_id * 3 + 1] = m;
    problem_sizes_transpose[expert_id * 3 + 2] = k;
  }

  int64_t expert_offset = static_cast<int64_t>(expert_offsets[expert_id]);
  int64_t a_stride = 0;
  int64_t b_stride = 0;
  int64_t a_scale_stride = 0;
  int64_t b_scale_stride = 0;
  if (!transpose) {
    a_stride = expert_offset * k;
    b_stride = expert_id * k * n;
    a_scale_stride = expert_offset * k / 128;
    b_scale_stride = expert_id * k * n / 128 / 128;
  } else {
    a_stride = expert_id * k * n;
    b_stride = expert_offset * k;
    a_scale_stride = expert_id * k * n / 128 / 128;
    b_scale_stride = expert_offset * k / 128;
  }
  a_offsets[expert_id] = a_base_as_int + a_stride;
  b_offsets[expert_id] = b_base_as_int + b_stride;
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  a_scales_offsets[expert_id] = a_scales_base_as_int + a_scale_stride;
  b_scales_offsets[expert_id] = b_scales_base_as_int + b_scale_stride;

  LayoutSFA* layout_sfa_ptr = layout_sfa_base_as_int + expert_id;
  LayoutSFB* layout_sfb_ptr = layout_sfb_base_as_int + expert_id;

  if (!transpose) {
    *layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    *layout_sfb_ptr = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  } else {
    *layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(n, m, k, 1));
    *layout_sfb_ptr = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(n, m, k, 1));
  }
}

template <typename OutType, typename ScheduleConfig, typename LayoutD>
void launch_prefill_tile(
    torch::Tensor& out_ptrs,
    const torch::Tensor& a_ptrs,
    const torch::Tensor& b_ptrs,
    const torch::Tensor& a_scales_ptrs,
    const torch::Tensor& b_scales_ptrs,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = void;
  using ElementD = OutType;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = LayoutD;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using CustomEVTIdentity =  // acc
      cutlass::epilogue::fusion::Sm90EVT<
          cutlass::epilogue::fusion::
              Sm90Compute<cutlass::epilogue::thread::Identity, ElementD, ElementAccumulator, RoundStyle>,
          cutlass::epilogue::fusion::Sm90AccFetch>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      typename ScheduleConfig::MmaTileShape,
      typename ScheduleConfig::ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,  // Use void to avoid load Matrix C
      LayoutC*,
      AlignmentC,
      ElementD,
      LayoutC*,
      AlignmentC,
      typename ScheduleConfig::EpilogueSchedule,
      CustomEVTIdentity>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutA*, typename ScheduleConfig::LayoutSFA*>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutB*, typename ScheduleConfig::LayoutSFB*>,
      AlignmentB,
      ElementAccumulator,
      typename ScheduleConfig::MmaTileShape,
      typename ScheduleConfig::ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename ScheduleConfig::KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  static_assert(sizeof(StrideA) == sizeof(int64_t) && sizeof(StrideB) == sizeof(int64_t) && sizeof(StrideD) == sizeof(int64_t), "packed stride contract");
  static_assert(sizeof(typename ScheduleConfig::LayoutSFA) <= 5 * sizeof(int32_t) && sizeof(typename ScheduleConfig::LayoutSFB) <= 5 * sizeof(int32_t), "scale layout storage contract");
  int num_experts = (int)expert_offsets.size(0);
  Gemm gemm_op;

  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementA**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(stride_a.data_ptr()),
      static_cast<const ElementB**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(stride_b.data_ptr()),
      static_cast<const ElementAccumulator**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<typename ScheduleConfig::LayoutSFA*>(layout_sfa.data_ptr()),
      static_cast<const ElementAccumulator**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<typename ScheduleConfig::LayoutSFB*>(layout_sfb.data_ptr())};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = c10::cuda::current_device();
  hw_info.sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  typename GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr,
      static_cast<StrideC*>(stride_c.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(stride_c.data_ptr())};

  UnderlyingProblemShape* problem_sizes_as_shapes = static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info};

  at::cuda::CUDAGuard device_guard{(char)a_ptrs.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a_ptrs.get_device());

  auto can_implement_status = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess, "Failed to implement GEMM");

  TORCH_CHECK(workspace.nbytes() >= gemm_op.get_workspace_size(args), "prefill tile workspace too small");
  auto status = gemm_op.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm_op.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}


struct Stock {
  using MmaTileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _2, _1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8Blockwise;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128, cute::GMMA::Major::K, cute::GMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};
struct SingleCluster : Stock { using ClusterShape = Shape<_1, _1, _1>; };
struct Pingpong : Stock {
  using MmaTileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8Blockwise;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
};

template<class Config>
void run_tile(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace) {
  const auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  const int groups = expert_offsets.size(0);
  get_group_gemm_starts<cutlass::float_e4m3_t, cutlass::bfloat16_t, float,
      typename Config::LayoutSFA, typename Config::LayoutSFB, typename Config::ScaleConfig><<<1, groups, 0, stream>>>(
      static_cast<int32_t*>(expert_offsets.data_ptr()), static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr()),
      static_cast<cutlass::float_e4m3_t**>(b_ptrs.data_ptr()), static_cast<cutlass::bfloat16_t**>(out_ptrs.data_ptr()),
      static_cast<float**>(a_scales_ptrs.data_ptr()), static_cast<float**>(b_scales_ptrs.data_ptr()),
      static_cast<cutlass::float_e4m3_t*>(a.data_ptr()), static_cast<cutlass::float_e4m3_t*>(b.data_ptr()),
      static_cast<cutlass::bfloat16_t*>(output.data_ptr()), static_cast<float*>(scales_a.data_ptr()), static_cast<float*>(scales_b.data_ptr()),
      reinterpret_cast<typename Config::LayoutSFA*>(layout_sfa.data_ptr()), reinterpret_cast<typename Config::LayoutSFB*>(layout_sfb.data_ptr()),
      static_cast<int*>(problem_sizes.data_ptr()), static_cast<int*>(problem_sizes.data_ptr()), false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  launch_prefill_tile<cutlass::bfloat16_t, Config, cutlass::layout::RowMajor>(out_ptrs, a_ptrs, b_ptrs, a_scales_ptrs, b_scales_ptrs,
      stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace);
}

void grouped_mm(torch::Tensor output, torch::Tensor a_ptrs, torch::Tensor b_ptrs, torch::Tensor out_ptrs,
                torch::Tensor a_scales_ptrs, torch::Tensor b_scales_ptrs, torch::Tensor a, torch::Tensor b,
                torch::Tensor scales_a, torch::Tensor scales_b, torch::Tensor stride_a, torch::Tensor stride_b, torch::Tensor stride_c,
                torch::Tensor layout_sfa, torch::Tensor layout_sfb, torch::Tensor problem_sizes, torch::Tensor expert_offsets,
                torch::Tensor workspace, int mode) {
  TORCH_CHECK(a.is_cuda() && a.get_device() == output.get_device(), "CUDA same-device tensors required");
  for (const auto& tensor : {output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.get_device() == a.get_device(), "all arguments must be on A device");
  }
  for (const auto& ptr : {a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs}) { TORCH_CHECK(ptr.scalar_type() == torch::kInt64 && ptr.numel() >= 128 && ptr.is_contiguous(), "pointer metadata mismatch"); }
  for (const auto& stride : {stride_a, stride_b, stride_c}) { TORCH_CHECK(stride.scalar_type() == torch::kInt64 && stride.numel() == 128 && stride.is_contiguous(), "stride metadata mismatch"); }
  TORCH_CHECK(problem_sizes.dim() == 2 && problem_sizes.size(0) == 128 && problem_sizes.size(1) == 3 && problem_sizes.is_contiguous(), "problem metadata mismatch");
  TORCH_CHECK(layout_sfa.scalar_type() == torch::kInt32 && layout_sfb.scalar_type() == torch::kInt32 && layout_sfa.numel() >= 640 && layout_sfb.numel() >= 640 && layout_sfa.is_contiguous() && layout_sfb.is_contiguous(), "scale layout metadata mismatch");
  c10::cuda::CUDAGuard guard(a.device());
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 9 && prop->minor == 0, "SM90 required");
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn && b.scalar_type() == torch::kFloat8_e4m3fn, "FP8 e4m3 inputs required");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16 && scales_a.scalar_type() == torch::kFloat32 && scales_b.scalar_type() == torch::kFloat32, "BF16 output / FP32 scales required");
  TORCH_CHECK(a.dim() == 2 && output.dim() == 2 && b.dim() == 3 && a.is_contiguous() && output.is_contiguous(), "dense A/D required");
  const int k = a.size(1), n = output.size(1), groups = expert_offsets.numel();
  TORCH_CHECK(groups == 128 && b.size(0) == groups && b.size(1) == k && b.size(2) == n && b.stride(1) == 1 && b.stride(2) == k && b.stride(0) == k * n, "E128 column-major B required");
  TORCH_CHECK(scales_a.dim() == 2 && scales_a.size(0) == a.size(0) && scales_a.size(1) == k / 128 && scales_a.is_contiguous(), "row-major A scales required");
  TORCH_CHECK(scales_b.dim() == 3 && scales_b.size(0) == groups && scales_b.size(1) == k / 128 && scales_b.size(2) == n / 128 && scales_b.stride(1) == 1 && scales_b.stride(2) == k / 128 && scales_b.stride(0) == k * n / 16384, "column-major B scales required");
  TORCH_CHECK(a.size(0) > 0 && output.size(0) == a.size(0) && ((k == 384 && n == 4096) || (k == 4096 && n == 768)), "large-row Qwen TP4 up/down only");
  TORCH_CHECK(problem_sizes.scalar_type() == torch::kInt32 && expert_offsets.scalar_type() == torch::kInt32 && workspace.scalar_type() == torch::kUInt8 && workspace.is_contiguous() && expert_offsets.is_contiguous(), "metadata dtype mismatch");
#define RUN_CONFIG(Config) run_tile<Config>(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace)
  switch(mode) {
    case 0: RUN_CONFIG(Stock); break;
    case 1: TORCH_CHECK(k == 384, "down only"); RUN_CONFIG(SingleCluster); break;
    case 3: TORCH_CHECK(k == 384, "down only"); RUN_CONFIG(Pingpong); break;
    case 4: TORCH_CHECK(k == 4096, "up only"); RUN_CONFIG(Pingpong); break;
    default: TORCH_CHECK(false, "unknown finite tile mode");
  }
#undef RUN_CONFIG
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("grouped_mm", &grouped_mm); }
