#!/usr/bin/env bash
set -euo pipefail

export BUILDAH_ISOLATION=chroot
export STORAGE_DRIVER=overlay

: "${BUILD_DIR:?build directory is required}"
: "${BUILD_CACHE_REPOSITORY:?Buildah cache repository is required}"
: "${CONTAINER_ARCHIVE_DIR:?container archive directory is required}"
: "${GITHUB_RUN_ATTEMPT:?workflow run attempt is required}"
: "${GITHUB_RUN_ID:?workflow run ID is required}"
: "${GITHUB_SHA:?workflow commit is required}"
: "${GITHUB_WORKSPACE:?workflow workspace is required}"
: "${GIT_MIRROR_DIR:?source mirror directory is required}"

[[ "${GITHUB_RUN_ID}" =~ ^[0-9]+$ ]]
[[ "${GITHUB_RUN_ATTEMPT}" =~ ^[0-9]+$ ]]
[[ "$(id -u)" -eq 0 ]]

archive="${CONTAINER_ARCHIVE_DIR}/${GITHUB_RUN_ID}.sqsh"
sidecar_archive="${CONTAINER_ARCHIVE_DIR}/${GITHUB_RUN_ID}.container-sidecar.json"
archive_temporary="${CONTAINER_ARCHIVE_DIR}/.${GITHUB_RUN_ID}.sqsh.${GITHUB_RUN_ATTEMPT}.tmp"
sidecar_temporary="${CONTAINER_ARCHIVE_DIR}/.${GITHUB_RUN_ID}.container-sidecar.json.${GITHUB_RUN_ATTEMPT}.tmp"
storage_root="${BUILD_STORAGE_ROOT:-/var/lib/containers/storage}"
squashfs_work="${BUILD_DIR}/sglang.sqsh"
run_root="${BUILD_DIR}/runroot"
context="${BUILD_DIR}/context"
source_dir="${BUILD_DIR}/sglang-source"
oci_build="${BUILD_DIR}/oci-build"
oci_runtime="${BUILD_DIR}/oci-runtime"
image="localhost/apac-qwen-sglang:${GITHUB_SHA}"
sglang_revision=fc275674a97d78553b22597ee72c8fd48f5420f4
mounted_container=

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "${mounted_container}" ]]; then buildah --root "${storage_root}" --runroot "${run_root}" unmount "${mounted_container}" >/dev/null 2>&1 || true; fi
  buildah --root "${storage_root}" --runroot "${run_root}" rm --all >/dev/null 2>&1 || true
  rm -f -- "${archive_temporary}" "${sidecar_temporary}" "${squashfs_work}"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

rm -rf -- "${BUILD_DIR:?}"
rm -f -- "${squashfs_work}"
install -d -m 2775 "${BUILD_DIR}" "${BUILD_DIR}/logs" "${context}/containers" "${context}/patches" "${context}/sglang-project-metadata/python" "${CONTAINER_ARCHIVE_DIR}" "${storage_root}"
storage_fstype="$(stat -f -c '%T' "${storage_root}")"
case "${storage_fstype}" in ext2/ext3|xfs) :;; *) printf 'Unsupported Buildah storage filesystem %s: %s\n' "${storage_fstype}" "${storage_root}" >&2; exit 1;; esac

if [[ -d "${GIT_MIRROR_DIR}/5_SGLang_Src.git" ]]; then git clone --reference-if-able "${GIT_MIRROR_DIR}/5_SGLang_Src.git" --no-checkout "${GIT_MIRROR_DIR}/5_SGLang_Src.git" "${source_dir}"; else gh repo clone 2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src "${source_dir}" -- --no-checkout; fi
git -C "${source_dir}" checkout --detach "${sglang_revision}"
sglang_commit="$(git -C "${source_dir}" rev-parse HEAD)"
[[ "${sglang_commit}" = "${sglang_revision}" ]]
git -C "${source_dir}" diff --quiet
git -C "${source_dir}" diff --cached --quiet

# Apply the measured precision-preserving bundle plus the routing helper to the private clone.
sglang_patch="patches/sglang/prefill-cutlass-silu-quant.patch"
draft_head_patch="patches/sglang/draft-extend-selected-head-rows.patch"
fused_shuffle_patch="patches/sglang/prefill-fp8-scale-row-shuffle.patch"
routing_patch="patches/sglang/prefill-tiled-routing.patch"
staging_patch="patches/sglang/firmus-staged-eagle-kv.patch"
mega_patch="patches/sglang/qwen3-sm90-megamoe.patch"
completion_patch="patches/sglang/firmus-staging-completion.patch"
page_release_patch="patches/sglang/chunk-cache-page-release.patch"
decode_quant_patch="patches/sglang/decode-silu-quant.patch"
expired_chunk_patch="patches/sglang/nixl-expired-chunks.patch"
dp_rank_query_patch="patches/sglang/firmus-dp-rank-query.patch"
prefill_handoff_patch="patches/sglang/prefill-ready-handoff.patch"
deepgemm_activation_patch="patches/sglang/deepgemm-activation-layout.patch"
rope_fp8_patch="patches/sglang/rope-fp8-q.patch"
qknorm_rope_fp8_patch="patches/sglang/qknorm-rope-fp8.patch"
aligned_chunks_patch="patches/sglang/prefill-request-aligned-chunks.patch"
qknorm_rope_fp8_kv_patch="patches/sglang/qknorm-rope-fp8-kv.patch"
topk_padding_patch="patches/sglang/qwen-topk-padding.patch"
qknorm_rope_fp8_token_warps_patch="patches/sglang/qknorm-rope-fp8-token-warps.patch"
async_admission_patch="patches/sglang/decode-async-admission.patch"
rotate_ties_patch="patches/sglang/dp-total-requests-rotate-ties.patch"
idle_geometry_patch="patches/sglang/megamoe-idle-prefill-geometry.patch"
gigatoken_patch="patches/sglang/gigatoken-encode.patch"
# Application order. The token-warps and group-quant patches build on the FP8 KV patch, and the Tier 2 patches come last.
sglang_patches=("${sglang_patch}" "${draft_head_patch}" "${fused_shuffle_patch}" "${routing_patch}" "${staging_patch}" "${mega_patch}" "${completion_patch}" "${page_release_patch}" "${decode_quant_patch}" "${expired_chunk_patch}" "${dp_rank_query_patch}" "${prefill_handoff_patch}" "${deepgemm_activation_patch}" "${rope_fp8_patch}" "${qknorm_rope_fp8_patch}" "${aligned_chunks_patch}" "${qknorm_rope_fp8_kv_patch}" "${topk_padding_patch}" "${qknorm_rope_fp8_token_warps_patch}" "${async_admission_patch}" "${rotate_ties_patch}" "${idle_geometry_patch}" "${gigatoken_patch}")
# Later patches depend on additions made by earlier patches; check each updated private tree.
for patch in "${sglang_patches[@]}"; do
  git -C "${source_dir}" apply --check "${GITHUB_WORKSPACE}/${patch}"
  git -C "${source_dir}" apply "${GITHUB_WORKSPACE}/${patch}"
done
python3 "${GITHUB_WORKSPACE}/scripts/check-firmus-staging.py" "${source_dir}"
python3 "${GITHUB_WORKSPACE}/scripts/check-firmus-megamoe-source.py" "${source_dir}"
python3 "${GITHUB_WORKSPACE}/scripts/check-firmus-latency-source.py" "${source_dir}"
python3 "${GITHUB_WORKSPACE}/scripts/check-firmus-prefill-chunks.py" "${source_dir}"
python3 "${GITHUB_WORKSPACE}/scripts/check-firmus-tier2-source.py" "${source_dir}" admission dp-ties idle-geometry

cp "${GITHUB_WORKSPACE}/containers/Dockerfile" "${GITHUB_WORKSPACE}/containers/build-nccl-tests.sh" "${GITHUB_WORKSPACE}/containers/install-accelerator-dependencies.sh" "${GITHUB_WORKSPACE}/containers/install-python-dependencies.sh" "${GITHUB_WORKSPACE}/containers/install-sglang-source.sh" "${GITHUB_WORKSPACE}/containers/install-system-toolchain.sh" "${context}/containers/"
cp --no-preserve=ownership,xattr "${source_dir}/LICENSE" "${source_dir}/README.md" "${source_dir}/python/pyproject.toml" "${context}/sglang-project-metadata/python/"
cp -a --no-preserve=ownership,xattr "${source_dir}" "${context}/sglang-source"
cp "${GITHUB_WORKSPACE}/containers/build-prefill-moe-routing.sh" "${GITHUB_WORKSPACE}/containers/build-prefill-moe-routing.py" "${context}/containers/"
install -d "${context}/prefill-moe-routing"
cp "${GITHUB_WORKSPACE}/kernels/prefill/moe_routing.cu" "${context}/prefill-moe-routing/"
cp "${GITHUB_WORKSPACE}/containers/build-sm90-megamoe.py" "${context}/containers/"
cp "${GITHUB_WORKSPACE}/containers/check-decode-kernel-build.py" "${context}/containers/"
cp "${GITHUB_WORKSPACE}/containers/check-rope-kernel-build.py" "${context}/containers/"
cp "${GITHUB_WORKSPACE}/scripts/check-firmus-latency-source.py" "${context}/containers/"
cp "${GITHUB_WORKSPACE}/containers/build-sglang-router.sh" "${context}/containers/"
cp "${GITHUB_WORKSPACE}/patches/sglang-router/prefill-load-lifetime.patch" "${context}/prefill-load-lifetime.patch"
cp "${GITHUB_WORKSPACE}/patches/sglang-router/no-delay-acceptor.patch" "${context}/no-delay-acceptor.patch"
cp "${GITHUB_WORKSPACE}/patches/sglang-router/Cargo.lock" "${context}/sglang-router.Cargo.lock"
# Ordered patch series and the expected final header SHA-256 travel with the patches.
cp -r "${GITHUB_WORKSPACE}/patches/deep-gemm" "${context}/deep-gemm"
rm -rf -- "${context}/sglang-source/.git"
printf '%s\n' "${sglang_commit}" >"${context}/sglang-commit.txt"

base_reference="$(sed -n 's/^FROM //p' "${GITHUB_WORKSPACE}/containers/Dockerfile")"
base_manifest_sha256="$(skopeo inspect --raw "docker://${base_reference}" | tee "${BUILD_DIR}/logs/base-oci-manifest.json" | sha256sum | awk '{print $1}')"
qwen_object_format="$(git -C "${GITHUB_WORKSPACE}" rev-parse --show-object-format)"
sglang_object_format="$(git -C "${source_dir}" rev-parse --show-object-format)"
dockerfile_blob_oid="$(git -C "${GITHUB_WORKSPACE}" hash-object containers/Dockerfile)"
build_steps_json="$(sed -n 's%^COPY \(containers/[^ ]*\.sh\) .*%\1%p' "${GITHUB_WORKSPACE}/containers/Dockerfile" | while read -r build_step; do build_step="${GITHUB_WORKSPACE}/${build_step}"; jq -cn --arg path "${build_step#"${GITHUB_WORKSPACE}/"}" --arg oid "$(git -C "${GITHUB_WORKSPACE}" hash-object "${build_step}")" '{path:$path,blob_oid:$oid}'; done | jq -s '.')"
patches_json='[]'
for patch in "${sglang_patches[@]}"; do
  case "${patch}" in
    "${gigatoken_patch}") targets=(python/sglang/srt/server_args.py python/sglang/srt/utils/hf_transformers/tokenizer.py python/sglang/srt/utils/hf_transformers/gigatoken_encoder.py) ;;
    "${rope_fp8_patch}"|"${qknorm_rope_fp8_patch}"|"${qknorm_rope_fp8_kv_patch}"|"${qknorm_rope_fp8_token_warps_patch}") targets=(python/sglang/kernels/jit/csrc/elementwise/rope.cuh python/sglang/kernels/ops/attention/rope.py python/sglang/srt/models/qwen3_moe.py) ;;
    "${topk_padding_patch}") targets=(python/sglang/kernels/jit/csrc/trtllm_lora_temp/topk_softmax_pack.cuh python/sglang/kernels/ops/moe/trtllm_lora_temp/topk_softmax_pack.py python/sglang/srt/layers/moe/topk.py) ;;
    "${aligned_chunks_patch}") targets=(python/sglang/srt/managers/schedule_policy.py python/sglang/srt/managers/scheduler.py) ;;
    "${async_admission_patch}") targets=(python/sglang/srt/disaggregation/async_admission.py python/sglang/srt/disaggregation/decode.py python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py python/sglang/srt/sampling/penaltylib/min_new_tokens.py python/sglang/srt/speculative/eagle_disaggregation.py) ;;
    "${rotate_ties_patch}") targets=(python/sglang/srt/managers/data_parallel_controller.py) ;;
    "${idle_geometry_patch}") targets=(python/sglang/srt/layers/moe/mega_moe_sm90.py) ;;
    "${deepgemm_activation_patch}") targets=(python/sglang/srt/layers/moe/moe_runner/deep_gemm.py python/sglang/kernels/ops/attention/dsv4/moe.py) ;;
    "${prefill_handoff_patch}") targets=(python/sglang/srt/disaggregation/prefill.py) ;;
    "${dp_rank_query_patch}") targets=(python/sglang/srt/disaggregation/decode.py) ;;
    "${expired_chunk_patch}") targets=(python/sglang/srt/disaggregation/nixl/conn.py) ;;
    "${decode_quant_patch}") targets=(python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh python/sglang/kernels/ops/moe/decode_silu_quant.py) ;;
    "${page_release_patch}") targets=(python/sglang/srt/mem_cache/chunk_cache.py) ;;
    "${completion_patch}") targets=(python/sglang/srt/disaggregation/common/staging_handler.py) ;;
    "${mega_patch}") targets=(python/sglang/srt/models/qwen3_moe.py) ;;
    "${staging_patch}") targets=(python/sglang/srt/disaggregation/common/staging_buffer.py python/sglang/srt/disaggregation/prefill.py python/sglang/srt/disaggregation/decode.py python/sglang/srt/disaggregation/nixl/conn.py) ;;
    "${draft_head_patch}") targets=(python/sglang/srt/speculative/eagle_worker_v2.py python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py) ;;
    "${fused_shuffle_patch}") targets=(python/sglang/srt/layers/moe/cutlass_moe.py python/sglang/kernels/ops/moe/prefill_fp8_row_shuffle.py) ;;
    "${routing_patch}") targets=(python/sglang/srt/layers/moe/cutlass_moe.py python/sglang/kernels/ops/moe/prefill_routing.py) ;;
    *) targets=(python/sglang/srt/layers/moe/cutlass_moe.py) ;;
  esac
  targets_json='[]'
  for target in "${targets[@]}"; do
    targets_json="$(jq --arg path "${target}" --arg blob_oid "$(git -C "${source_dir}" hash-object "${target}")" --arg sha256 "$(sha256sum "${source_dir}/${target}" | awk '{print $1}')" '. + [{path:$path,blob_oid:$blob_oid,sha256:$sha256}]' <<<"${targets_json}")"
  done
  patches_json="$(jq --arg path "${patch}" --arg blob_oid "$(git -C "${GITHUB_WORKSPACE}" hash-object "${patch}")" --arg sha256 "$(sha256sum "${GITHUB_WORKSPACE}/${patch}" | awk '{print $1}')" --argjson targets "${targets_json}" '. + [{path:$path,blob_oid:$blob_oid,sha256:$sha256,targets:$targets}]' <<<"${patches_json}")"
done
for router_patch in patches/sglang-router/prefill-load-lifetime.patch patches/sglang-router/no-delay-acceptor.patch; do
  patches_json="$(jq --arg path "${router_patch}" --arg blob_oid "$(git -C "${GITHUB_WORKSPACE}" hash-object "${router_patch}")" --arg sha256 "$(sha256sum "${GITHUB_WORKSPACE}/${router_patch}" | awk '{print $1}')" '. + [{path:$path,blob_oid:$blob_oid,sha256:$sha256,targets:[]}]' <<<"${patches_json}")"
done
jq -n --arg qwen_object_format "${qwen_object_format}" --arg qwen_commit "${GITHUB_SHA}" --arg sglang_object_format "${sglang_object_format}" --arg sglang_commit "${sglang_commit}" --arg dockerfile_blob_oid "${dockerfile_blob_oid}" --arg base_reference "${base_reference}" --arg base_manifest_sha256 "${base_manifest_sha256}" --arg routing_source_sha256 "$(sha256sum "${GITHUB_WORKSPACE}/kernels/prefill/moe_routing.cu" | awk '{print $1}')" --arg routing_build_sha256 "$(sha256sum "${GITHUB_WORKSPACE}/containers/build-prefill-moe-routing.py" | awk '{print $1}')" --argjson build_steps "${build_steps_json}" --argjson patches "${patches_json}" '{schema_version:"oci-embedded-manifest/v1",source_commits:[{repository_id:"apac2026-qwen",commit_role:"container_build_definition",object_hash_algorithm:$qwen_object_format,oid:$qwen_commit},{repository_id:"apac2026-sglang",commit_role:"sglang_source",object_hash_algorithm:$sglang_object_format,oid:$sglang_commit}],recipe:{path:"containers/Dockerfile",blob_oid:$dockerfile_blob_oid,build_steps:$build_steps},applied_patches:$patches,local_kernels:{path:"kernels/prefill/moe_routing.cu",sha256:$routing_source_sha256,build_sha256:$routing_build_sha256,architecture:"sm_90",translation_units:1,gemm_templates:0},base_oci:{reference:$base_reference,manifest_sha256:$base_manifest_sha256}}' >"${context}/apac-manifest.json"

buildah --root "${storage_root}" --runroot "${run_root}" bud --isolation chroot --network host --layers --cache-from "${BUILD_CACHE_REPOSITORY}" --cache-to "${BUILD_CACHE_REPOSITORY}" --cache-ttl 336h --file "${context}/containers/Dockerfile" --tag "${image}" "${context}" 2>&1 | tee "${BUILD_DIR}/logs/buildah.log"
test_container="$(buildah --root "${storage_root}" --runroot "${run_root}" from "${image}")"
buildah --root "${storage_root}" --runroot "${run_root}" run --isolation chroot "${test_container}" /bin/bash -c 'set -euo pipefail; export XDG_CACHE_HOME=/var/tmp/sglang-recipe-test-cache; test -x /sbin/ldconfig; /opt/sglang/venv/bin/python -c "import sglang, sglang_router, torch; print(sglang.__file__); print(sglang_router.__file__); print(torch.__version__)"; /opt/sglang/venv/bin/python -c "from importlib.metadata import version; expected = {\"aiperf\": \"0.12.0\", \"sglang-router\": \"0.3.2\"}; actual = {package: version(package) for package in expected}; assert actual == expected, actual; print(actual)"; /opt/sglang/venv/bin/python -c "from importlib.metadata import version; packages = (\"flashinfer-python\", \"flashinfer-cubin\", \"flashinfer-jit-cache\"); versions = {package: version(package).split(\"+\", 1)[0] for package in packages}; assert len(set(versions.values())) == 1, versions; print(versions)"; /opt/sglang/venv/bin/python -c "from importlib.metadata import version; print({p: version(p) for p in (\"deep-ep\", \"mooncake-transfer-engine-cuda13\", \"mscclpp\", \"nixl\", \"nixl-cu13\")})"; /opt/sglang/venv/bin/python -m sglang_router.launch_router --help >/dev/null; mpirun --version; test "$(find /opt/nccl-tests/bin -maxdepth 1 -type f -name "*_perf" | wc -l)" -eq 11; for binary in all_reduce_perf all_gather_perf reduce_scatter_perf; do /opt/nccl-tests/bin/${binary} -h 2>&1 | grep -q -- "-J"; done; test "$(cat /opt/sglang/nccl-tests-commit.txt)" = 717b68318278e93f371d8ffb46b076069d7c7851; test -s /opt/sglang/server-args-source-blob-oid.txt; test -s /opt/apac/manifest.json; ! ldd /opt/nccl-tests/bin/all_reduce_perf | grep -q "not found"; ucx_info -v; uv --version; rm -rf "${XDG_CACHE_HOME}"' 2>&1 | tee "${BUILD_DIR}/logs/buildah-test.log"
buildah --root "${storage_root}" --runroot "${run_root}" run --isolation chroot "${test_container}" /opt/sglang/venv/bin/python -m sglang.launch_server --help >"${BUILD_DIR}/logs/server-cli-help.log" 2>&1
grep -q -- "--moe-a2a-backend" "${BUILD_DIR}/logs/server-cli-help.log"
grep -q -- "megamoe" "${BUILD_DIR}/logs/server-cli-help.log"
buildah --root "${storage_root}" --runroot "${run_root}" push "${image}" "oci:${oci_build}:sglang" 2>&1 | tee "${BUILD_DIR}/logs/buildah-push.log"
skopeo copy "oci:${oci_build}:sglang" "oci:${oci_runtime}:sglang" 2>&1 | tee "${BUILD_DIR}/logs/skopeo-copy.log"
skopeo inspect --raw "oci:${oci_runtime}:sglang" >"${BUILD_DIR}/logs/oci-manifest.json"
umoci stat --json --image "${oci_runtime}:sglang" | tee "${BUILD_DIR}/logs/umoci-stat.json"
jq -e '.history | length > 0' "${BUILD_DIR}/logs/umoci-stat.json" >/dev/null
rootfs="$(buildah --root "${storage_root}" --runroot "${run_root}" mount "${test_container}")"
mounted_container="${test_container}"
cmp "${rootfs}/opt/apac/manifest.json" "${context}/apac-manifest.json"
mksquashfs "${rootfs}" "${squashfs_work}" -noappend -all-root -no-xattrs -no-duplicates -no-fragments -comp gzip -Xcompression-level 2 -processors 8 -mem 8G 2>&1 | tee "${BUILD_DIR}/logs/mksquashfs.log"
buildah --root "${storage_root}" --runroot "${run_root}" unmount "${test_container}"
mounted_container=
buildah --root "${storage_root}" --runroot "${run_root}" rm "${test_container}"

container_sha256="$(sha256sum "${squashfs_work}" | awk '{print $1}')"
container_size="$(stat -c '%s' "${squashfs_work}")"
oci_manifest_sha256="$(sha256sum "${BUILD_DIR}/logs/oci-manifest.json" | awk '{print $1}')"
jq -n --arg artifact_path "sglang.sqsh" --argjson size_bytes "${container_size}" --arg sha256 "${container_sha256}" --arg oci_manifest_sha256 "${oci_manifest_sha256}" --arg qwen_commit "${GITHUB_SHA}" --arg sglang_commit "${sglang_commit}" --arg buildah_version "$(buildah --version)" --arg skopeo_version "$(skopeo --version)" --arg umoci_version "$(umoci --version | head -n 1)" '{schema_version:"container-sidecar/v1",container_artifact:{artifact_relative_path:$artifact_path,size_bytes:$size_bytes,sha256:$sha256},oci_manifest:{sha256:$oci_manifest_sha256},source_commits:{qwen:$qwen_commit,sglang:$sglang_commit},toolchain:{buildah:$buildah_version,skopeo:$skopeo_version,umoci:$umoci_version}}' >"${BUILD_DIR}/container-sidecar.json"

cp --reflink=auto --sparse=always "${squashfs_work}" "${archive_temporary}"
cp "${BUILD_DIR}/container-sidecar.json" "${sidecar_temporary}"
test "$(stat -c '%s' "${archive_temporary}")" = "${container_size}"
test "$(sha256sum "${archive_temporary}" | awk '{print $1}')" = "${container_sha256}"
mv -- "${archive_temporary}" "${archive}"
mv -- "${sidecar_temporary}" "${sidecar_archive}"
printf 'Published %s\n' "${archive}"
