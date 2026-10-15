#!/usr/bin/env bash
# Actions-only CPU revalidation of the pinned image; no scheduler submission.
set -euo pipefail
: "${GITHUB_WORKSPACE:?}"
: "${GITHUB_RUN_ID:?}"
: "${GITHUB_RUN_ATTEMPT:?}"
: "${GITHUB_OUTPUT:?}"
config="${GITHUB_WORKSPACE}/configs/firmus-vllm-megamoe-gate.json"
command -v bwrap >/dev/null
image=$(jq -er .image.path "${config}")
expected_sha=$(jq -er .image.sha256 "${config}")
[[ "$(stat -c '%s' "${image}")" == "$(jq -er .image.size_bytes "${config}")" ]]
runtime=$(mktemp -d "/scratch/home/ACCOUNT/qwen/runs/megamoe-cpu-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.XXXXXXXX")
printf 'runtime_dir=%s\n' "${runtime}" >>"${GITHUB_OUTPUT}"
mkdir -p "${runtime}/source" "${runtime}/results" "${runtime}/logs" "${runtime}/enroot-data" "${runtime}/enroot-runtime"
cp "${config}" "${GITHUB_WORKSPACE}/research/firmus-vllm-megamoe/check_native_cpu.py" "${runtime}/source/"
sha256sum "${image}" >"${runtime}/source/image.sha256"
[[ "$(cut -d' ' -f1 "${runtime}/source/image.sha256")" == "${expected_sha}" ]]
export ENROOT_DATA_PATH="${runtime}/enroot-data" ENROOT_RUNTIME_PATH="${runtime}/enroot-runtime"
export NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES=
trap 'rm -rf -- "${runtime}/enroot-data" "${runtime}/enroot-runtime"' EXIT
enroot create --name cpu-image "${image}" >"${runtime}/logs/extract.log" 2>&1
cat >"${runtime}/source/run-cpu.sh" <<'CPU'
#!/usr/bin/env bash
set -euo pipefail
stub=$(mktemp -d /tmp/megamoe-cpu-stub.XXXXXXXX)
trap 'rm -rf -- "${stub}"' EXIT
ln -s /usr/local/cuda/lib64/stubs/libcuda.so "${stub}/libcuda.so.1"
export LD_LIBRARY_PATH="${stub}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" CUDA_VISIBLE_DEVICES= VLLM_USE_DEEP_GEMM_E8M0=0 HF_HUB_OFFLINE=1
python3 /opt/apac/firmus-sm90-megamoe/check_cpu_image.py /opt/apac/sm90-provider /audit/results/installed-image-cpu.json
torchrun --standalone --nproc_per_node=4 /audit/source/check_native_cpu.py --model /model --output /audit/results/native-factory-cpu.json
CPU
# Login Enroot's systemd cgroup mount points at an absent session scope.
# Use a private CPU filesystem namespace with a fresh /dev (no GPU devices).
# The compute-node GPU launcher continues to use scheduler-managed Enroot.
bwrap --unshare-user --uid 0 --gid 0 --unshare-pid --die-with-parent --bind "${runtime}/enroot-data/cpu-image" / --proc /proc --dev /dev --ro-bind /sys /sys --bind "${runtime}/source" /audit/source --bind "${runtime}/results" /audit/results --ro-bind /scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8 /model --tmpfs /tmp --setenv PATH /usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin --setenv XDG_CACHE_HOME /tmp/cpu-cache --chdir /audit bash /audit/source/run-cpu.sh >"${runtime}/logs/cpu.log" 2>&1
touch "${runtime}/run-success"
