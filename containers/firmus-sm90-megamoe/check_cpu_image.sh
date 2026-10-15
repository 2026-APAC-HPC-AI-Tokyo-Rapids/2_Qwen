#!/usr/bin/env bash
set -euo pipefail

sm90_stub_dir="$(mktemp -d /tmp/firmus-sm90-import.XXXXXXXX)"
trap 'rm -f -- "${sm90_stub_dir}/libcuda.so.1"; rmdir -- "${sm90_stub_dir}"' EXIT
ln -s /usr/local/cuda/lib64/stubs/libcuda.so "${sm90_stub_dir}/libcuda.so.1"
LD_LIBRARY_PATH="${sm90_stub_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" python3 /opt/apac/firmus-sm90-megamoe/build_native.py /tmp/apac-sm90-native-source /opt/apac/sm90-provider/deep_gemm
LD_LIBRARY_PATH="${sm90_stub_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" python3 /opt/apac/firmus-sm90-megamoe/check_cpu_image.py /opt/apac/sm90-provider /opt/apac/sm90-megamoe-cpu-gate.json
