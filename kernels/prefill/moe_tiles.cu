#include "moe_grid.cuh"
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
  switch(mode) {
    case 0: run_config0(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); break;
    case 1: run_config1(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); break;
    case 2: run_config2(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); break;
    case 3: run_config3(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); break;
    case 4: run_config4(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); break;
    case 5: run_config5(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); break;
    case 6: run_config6(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); break;
    case 7: run_config7(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); break;
    case 8: run_config8(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); break;
    case 9: run_config9(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); break;
    case 10: run_config10(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); break;
    case 11: run_config11(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); break;
    case 12: run_config12(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); break;
    case 13: run_config13(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); break;
    case 14: run_config14(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); break;
    case 15: run_config15(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); break;
    default: TORCH_CHECK(false, "unknown finite grid mode");
  }
}
pybind11::list configurations() {
  pybind11::list result;
  result.append(metadata_config0());
  result.append(metadata_config1());
  result.append(metadata_config2());
  result.append(metadata_config3());
  result.append(metadata_config4());
  result.append(metadata_config5());
  result.append(metadata_config6());
  result.append(metadata_config7());
  result.append(metadata_config8());
  result.append(metadata_config9());
  result.append(metadata_config10());
  result.append(metadata_config11());
  result.append(metadata_config12());
  result.append(metadata_config13());
  result.append(metadata_config14());
  result.append(metadata_config15());
  return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("grouped_mm", &grouped_mm); m.def("configurations", &configurations); }
