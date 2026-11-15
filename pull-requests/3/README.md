# Pull request 3: SIFパッキングにおけるインストール済み sqfstar の使用

Author: yayoimizuha
Created: 2026-07-19T04:18:09Z
Closed: 2026-07-19T04:20:48Z
Merged: 2026-07-19T04:20:48Z

The original pull request is retained in a private archive for maintainers. This public copy preserves the sanitized conversation and its original authors and timestamps.

## Description



## Source snapshots

- Base: recorded `7e0d5edd9e3d0eeee238f01339e6dd64b14e7f57`; [sanitized snapshot](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/commit/08202a10a1bb2dadd7d5feab03d442371ad5b5ec)
- Head: recorded `f0d6910693a50e70a1d1aec9b2dd98bcf5a95025`; [sanitized snapshot](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/commit/d5f8c809c0a25bccdcf60ca972f5c2839d55434f)

## Conversation


## Reviews

### Review 4729989386

copilot-pull-request-reviewer[bot] · 2026-07-19T04:21:26Z · COMMENTED

## Pull request overview

Updates the Genkai GitHub Actions workflow to stop downloading/building SquashFS tools during the build job and instead use an already-installed `sqfstar` binary when packing the rootfs into a SquashFS image for SIF creation.

**Changes:**
- Adds a `SQFSTAR` environment variable to point at the `sqfstar` binary.
- Removes the workflow step that downloaded and compiled SquashFS Tools 4.7.5.
- Switches the compute-node packing command to invoke `${SQFSTAR}` instead of a locally-built `${BUILD_DIR}/sqfstar`.







---

💡 <a href="/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/new/main?filename=.github/instructions/*.instructions.md" class="Link--inTextBlock" target="_blank" rel="noopener noreferrer">Add Copilot custom instructions</a> for smarter, more guided reviews. <a href="https://docs.github.com/en/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot" class="Link--inTextBlock" target="_blank" rel="noopener noreferrer">Learn how to get started</a>.


## Inline review comments

### Review comment 3609825995

Copilot · 2026-07-19T04:21:25Z
Recorded file: `.github/workflows/genkai.yml`

`SQFSTAR` is hard-coded to a specific user’s home directory path, which makes the workflow brittle (breaks if the account/path changes or if another runner/user executes the workflow). Prefer sourcing this from a repo/runner-level variable (or PATH) so the workflow stays portable across Genkai runners.

### Review comment 3609826003

Copilot · 2026-07-19T04:21:25Z
Recorded file: `.github/workflows/genkai.yml`

`SQFSTAR` is required by the batch job, but the workflow doesn’t validate that it exists/is executable before calling `pjsub`. If it’s missing (or empty), the workflow will waste scheduler/queue time before failing in the compute job. Add a quick preflight check on the login node before generating/submitting the job script.

