#pragma once
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
struct TileGeometry {
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

  static constexpr int KernelReserve = sizeof(cute::TmaDescriptor) * 2 + sizeof(cutlass::PipelineDetail::PipelineAsyncSharedStorage<8>);
  static constexpr int Capacity = (cutlass::gemm::collective::detail::sm90_smem_capacity_bytes - KernelReserve) / 128 * 128;
  static constexpr int StageBytes = cutlass::round_up(128 * 128 * 2 + 4 * 128 + 4, 128) + sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  static constexpr int EpilogueBytes = sizeof(typename CollectiveEpilogue::SharedStorage);
  static constexpr int RequestedStages = ScheduleConfig::RequestedStages;
  static constexpr int Reserve = RequestedStages == 0 ? EpilogueBytes : (Capacity - RequestedStages * StageBytes) / 128 * 128;
  static_assert(Reserve >= EpilogueBytes, "requested stages exceed available shared memory");
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
      cutlass::gemm::collective::StageCountAutoCarveout<Reserve>,
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
  static constexpr int ResolvedStages = CollectiveMainloop::DispatchPolicy::Stages;
  static_assert(RequestedStages == 0 || ResolvedStages == RequestedStages, "fixed stage reserve mismatch");
  static_assert(sizeof(typename GemmKernel::SharedStorage) <= cutlass::gemm::collective::detail::sm90_smem_capacity_bytes, "shared storage exceeds SM90 capacity");
};

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
  using Geometry = TileGeometry<OutType, ScheduleConfig, LayoutD>;
  using GemmKernel = typename Geometry::GemmKernel;
  using Gemm = typename Geometry::Gemm;
  using StrideA = typename Geometry::StrideA;
  using StrideB = typename Geometry::StrideB;
  using StrideC = typename Geometry::StrideC;
  using ElementA = typename Geometry::ElementA;
  using ElementB = typename Geometry::ElementB;
  using ElementD = typename Geometry::ElementD;
  using ElementAccumulator = typename Geometry::ElementAccumulator;
  using UnderlyingProblemShape = typename Geometry::UnderlyingProblemShape;
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


template<int CM, int CN, int Stages>
struct GridConfig {
  static constexpr int RequestedStages = Stages;
  using MmaTileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<Int<CM>, Int<CN>, _1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8Blockwise;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128, cute::GMMA::Major::K, cute::GMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};
using Config0 = GridConfig<1, 2, 0>;
using Config1 = GridConfig<1, 2, 2>;
using Config2 = GridConfig<1, 2, 3>;
using Config3 = GridConfig<1, 2, 4>;
using Config4 = GridConfig<1, 1, 0>;
using Config5 = GridConfig<1, 1, 2>;
using Config6 = GridConfig<1, 1, 3>;
using Config7 = GridConfig<1, 1, 4>;
using Config8 = GridConfig<2, 1, 0>;
using Config9 = GridConfig<2, 1, 2>;
using Config10 = GridConfig<2, 1, 3>;
using Config11 = GridConfig<2, 1, 4>;
using Config12 = GridConfig<2, 2, 0>;
using Config13 = GridConfig<2, 2, 2>;
using Config14 = GridConfig<2, 2, 3>;
using Config15 = GridConfig<2, 2, 4>;

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

template<class Config>
pybind11::dict geometry_metadata(int mode) {
  using Geometry = TileGeometry<cutlass::bfloat16_t, Config, cutlass::layout::RowMajor>;
  pybind11::dict result;
  result["mode"] = mode;
  result["resolved_stages"] = Geometry::ResolvedStages;
  result["shared_storage_bytes"] = sizeof(typename Geometry::GemmKernel::SharedStorage);
  result["mainloop_bytes"] = sizeof(typename Geometry::CollectiveMainloop::SharedStorage);
  result["epilogue_bytes"] = Geometry::EpilogueBytes;
  result["stage_bytes"] = Geometry::StageBytes;
  result["carveout_bytes"] = Geometry::Reserve;
  return result;
}
void run_config0(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
pybind11::dict metadata_config0();
void run_config1(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
pybind11::dict metadata_config1();
void run_config2(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
pybind11::dict metadata_config2();
void run_config3(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
pybind11::dict metadata_config3();
void run_config4(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
pybind11::dict metadata_config4();
void run_config5(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
pybind11::dict metadata_config5();
void run_config6(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
pybind11::dict metadata_config6();
void run_config7(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
pybind11::dict metadata_config7();
void run_config8(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
pybind11::dict metadata_config8();
void run_config9(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
pybind11::dict metadata_config9();
void run_config10(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
pybind11::dict metadata_config10();
void run_config11(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
pybind11::dict metadata_config11();
void run_config12(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
pybind11::dict metadata_config12();
void run_config13(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
pybind11::dict metadata_config13();
void run_config14(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
pybind11::dict metadata_config14();
void run_config15(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace);
pybind11::dict metadata_config15();
