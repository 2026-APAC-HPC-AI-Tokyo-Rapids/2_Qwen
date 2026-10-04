#include "moe_grid.cuh"
void run_config0(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace) { run_tile<Config0>(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); }
pybind11::dict metadata_config0() { return geometry_metadata<Config0>(0); }
void run_config1(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace) { run_tile<Config1>(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); }
pybind11::dict metadata_config1() { return geometry_metadata<Config1>(1); }
void run_config2(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace) { run_tile<Config2>(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); }
pybind11::dict metadata_config2() { return geometry_metadata<Config2>(2); }
void run_config3(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace) { run_tile<Config3>(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); }
pybind11::dict metadata_config3() { return geometry_metadata<Config3>(3); }
