#!/usr/bin/env bash
# Stage the same pinned, userspace Nsight distribution used by Firmus.
set -euo pipefail
runtime_dir=${1:?runtime directory required}
repository=${2:?repository directory required}
install -m 0755 "${repository}/scripts/run-aspire-profile.sh" "${repository}/scripts/capture-firmus-profile.sh" "${runtime_dir}/scripts/"
install -m 0644 "${repository}/scripts/validate-firmus-profile.py" "${repository}/scripts/run-aiperf.py" "${runtime_dir}/scripts/"
install -m 0755 "${repository}/platforms/firmus/framework-comparison/official-aiperf.sh" "${runtime_dir}/scripts/"
command -v ar >/dev/null
package="${runtime_dir}/nsight.deb"
curl --fail --location --show-error --max-time 600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nsight-systems-2025.3.2_2025.3.2.474-1_amd64.deb -o "${package}"
printf 'c7cfe27e2250eb91e1a67e7feb5f2c490c7f598e3b3a3d047aff000bc49f9d6b  %s\n' "${package}" | sha256sum --check >"${runtime_dir}/source/nsys-package.sha256"
install -d -m 0755 "${runtime_dir}/nsys-extract"
ar p "${package}" data.tar.gz | tar -xz -C "${runtime_dir}/nsys-extract" --wildcards './opt/nvidia/nsight-systems/2025.3.2/bin/*' './opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/*' './opt/nvidia/nsight-systems/2025.3.2/host-linux-x64/*' './opt/nvidia/nsight-systems/2025.3.2/EULA.txt'
mv "${runtime_dir}/nsys-extract/opt/nvidia/nsight-systems/2025.3.2" "${runtime_dir}/nsys"
rm -rf -- "${package}" "${runtime_dir}/nsys-extract"
test -x "${runtime_dir}/nsys/bin/nsys"
cp "${runtime_dir}/scripts/run-aspire-profile.sh" "${runtime_dir}/scripts/capture-firmus-profile.sh" "${runtime_dir}/scripts/validate-firmus-profile.py" "${runtime_dir}/scripts/official-aiperf.sh" "${runtime_dir}/scripts/run-aiperf.py" "${runtime_dir}/scripts/exec-profiler.sh" "${runtime_dir}/source/"
