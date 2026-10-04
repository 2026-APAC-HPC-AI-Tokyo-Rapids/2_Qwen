#include "moe_grid.cuh"
void run_config12(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace) { run_tile<Config12>(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); }
pybind11::dict metadata_config12() { return geometry_metadata<Config12>(12); }
void run_config13(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace) { run_tile<Config13>(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); }
pybind11::dict metadata_config13() { return geometry_metadata<Config13>(13); }
void run_config14(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace) { run_tile<Config14>(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); }
pybind11::dict metadata_config14() { return geometry_metadata<Config14>(14); }
void run_config15(torch::Tensor& output, torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
              torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs, const torch::Tensor& a,
              const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b,
              const torch::Tensor& stride_a, const torch::Tensor& stride_b, const torch::Tensor& stride_c,
              const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb, const torch::Tensor& problem_sizes,
              const torch::Tensor& expert_offsets, const torch::Tensor& workspace) { run_tile<Config15>(output, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a, b, scales_a, scales_b, stride_a, stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes, expert_offsets, workspace); }
pybind11::dict metadata_config15() { return geometry_metadata<Config15>(15); }
