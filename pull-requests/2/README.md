# Pull request 2: [require-build] Qwenランタイムバックエンドおよびビルド時間計測の追加

Author: yayoimizuha
Created: 2026-07-18T09:54:45Z
Closed: 2026-07-18T18:47:31Z
Merged: 2026-07-18T18:47:31Z

The original pull request is retained in a private archive for maintainers. This public copy preserves the sanitized conversation and its original authors and timestamps.

## Description



## Source snapshots

- Base: recorded `56251ae0256d56c770d24a8d75f45586ba056dd8`; [sanitized snapshot](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/commit/60ee06d8ccdb42a1dd3457f1ac8cfd43bf0f8fe5)
- Head: recorded `1ee97ddf96304f7f673624379c7e9cf0edc3f6a7`; [sanitized snapshot](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/commit/bb3bd1dcda0f39f4151adea6d89184ec3a8e06aa)

## Conversation


## Reviews

### Review 4729088234

copilot-pull-request-reviewer[bot] · 2026-07-18T18:49:17Z · COMMENTED

## Pull request overview

This PR updates the Genkai-based Qwen 3.5 benchmark workflow to (1) build a richer SGLang runtime image with additional optional backends and (2) make build/run execution more measurable and self-contained via a provision step, environment logging, and a profiler wrapper.

**Changes:**
- Add `provision-qwen35.sh` to validate/warm non-GPU initialization before the measured GPU job.
- Update `run-qwen35.sh` to run directly on Genkai nodes, record environment/command logs, and route execution through `exec-profiler.sh`.
- Expand the Singularity recipe (`containers/sglang.def`) with extra comms/backends and symbol-preserving build settings; adjust the workflow to pack the runtime via a SquashFS handoff.

### Reviewed changes

Copilot reviewed 6 out of 6 changed files in this pull request and generated 2 comments.

<details>
<summary>Show a summary per file</summary>

| File | Description |
| ---- | ----------- |
| scripts/run-qwen35.sh | Moves benchmark execution responsibilities into a node-local runtime runner with structured logs and profiler hook. |
| scripts/provision-qwen35.sh | Adds a login-node provision/validation step for model+dataset+SGLang readiness (non-GPU). |
| scripts/exec-profiler.sh | Introduces a transparent wrapper to enable future profiling without changing workflow/benchmark code. |
| README.md | Updates documentation to match the new build/provision/run split and logging layout. |
| containers/sglang.def | Adds optional backends (DeepEP/Mooncake/NIXL/MSCCL++) and preserves symbols/metadata for debugging. |
| .github/workflows/genkai.yml | Reworks build reuse, provisioning, and SIF packing flow; wires in new scripts and dataset path env. |
</details>








---

💡 <a href="/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/new/main?filename=.github/instructions/*.instructions.md" class="Link--inTextBlock" target="_blank" rel="noopener noreferrer">Add Copilot custom instructions</a> for smarter, more guided reviews. <a href="https://docs.github.com/en/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot" class="Link--inTextBlock" target="_blank" rel="noopener noreferrer">Learn how to get started</a>.


## Inline review comments

### Review comment 3608967010

Copilot · 2026-07-18T18:49:17Z
Recorded file: `.github/workflows/genkai.yml`

The SIF reuse loop no longer skips empty run IDs. If `gh run list` ever emits an empty line (or parsing yields an empty `run_id`), `gh run view ""` will fail under `-euo pipefail` and the build job will abort instead of continuing the search.

### Review comment 3608967018

Copilot · 2026-07-18T18:49:17Z
Recorded file: `.github/workflows/genkai.yml`

The workflow builds and installs `mksquashfs` but names it `${BUILD_DIR}/sqfstar`, then later invokes it with stdin tar input. `sqfstar` is a different binary intended for tar-on-stdin; using `mksquashfs` here is likely to fail or produce an invalid SquashFS.

