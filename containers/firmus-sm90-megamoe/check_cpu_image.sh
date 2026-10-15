#!/usr/bin/env bash
set -euo pipefail

sm90_stub_dir="$(mktemp -d /tmp/firmus-sm90-import.XXXXXXXX)"
trap 'rm -f -- "${sm90_stub_dir}/libcuda.so.1"; rmdir -- "${sm90_stub_dir}"' EXIT
ln -s /usr/local/cuda/lib64/stubs/libcuda.so "${sm90_stub_dir}/libcuda.so.1"
LD_LIBRARY_PATH="${sm90_stub_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" python3 /opt/apac/firmus-sm90-megamoe/build_native.py /tmp/apac-sm90-native-source /opt/apac/sm90-provider/deep_gemm
LD_LIBRARY_PATH="${sm90_stub_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" python3 /opt/apac/firmus-sm90-megamoe/check_cpu_image.py /opt/apac/sm90-provider /opt/apac/sm90-megamoe-cpu-gate.json

# Full native factory/loader and TP1/DP4/EP4 groups; CPU tensors and Gloo only.
LD_LIBRARY_PATH="${sm90_stub_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 VLLM_USE_DEEP_GEMM_E8M0=0 torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29567 /opt/apac/firmus-sm90-megamoe/check_native_cpu.py --model /opt/apac/firmus-sm90-megamoe/qwen3-config --output /opt/apac/sm90-megamoe-native-cpu-gate.json
