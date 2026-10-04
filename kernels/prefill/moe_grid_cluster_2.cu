#include "moe_grid.cuh"
void run_config8(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace) { run_tile<Config8>(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); }
pybind11::dict metadata_config8() { return geometry_metadata<Config8>(8); }
void run_config9(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace) { run_tile<Config9>(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); }
pybind11::dict metadata_config9() { return geometry_metadata<Config9>(9); }
void run_config10(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace) { run_tile<Config10>(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); }
pybind11::dict metadata_config10() { return geometry_metadata<Config10>(10); }
void run_config11(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace) { run_tile<Config11>(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); }
pybind11::dict metadata_config11() { return geometry_metadata<Config11>(11); }
