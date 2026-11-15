# Pull request 1: SGLang Qwen実験ハーネスの追加

Author: komoriyuta
Created: 2026-07-11T17:31:58Z
Closed: 2026-07-12T00:29:48Z
Merged: 2026-07-12T00:29:48Z

The original pull request is retained in a private archive for maintainers. This public copy preserves the sanitized conversation and its original authors and timestamps.

## Description

## 概要

- Qwen 3.5 向けの SGLang オフラインスループット実験ハーネスを追加
- push ごとに Genkai セルフホストランナー上で検証を実行
- ベンチマーク準備の実行条件をコミットマーカー `[run-qwen]` で制御
- 安全ロックとして実際の `pjsub` コマンドはコメントアウトのまま維持

## 検証

- `tests/test_scripts.sh`
- Genkai ランナー上でワークフロー実行 29161761234 が成功し、ジョブ投入ステップはスキップされたことを確認


## Source snapshots

- Base: recorded `28d0df0796403130b28f56f34ddba255a906e4d2`; [sanitized snapshot](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/commit/6b79712b9098bb5880b4c203654d116387bebc17)
- Head: recorded `6cf7012cbdce22e1e272eaab2ae3b7085f48cbf7`; [sanitized snapshot](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/commit/e88c071561c1f6e077597d5c59aded962e1bf595)

## Conversation


## Reviews

### Review 4679063158

copilot-pull-request-reviewer[bot] · 2026-07-12T00:32:31Z · COMMENTED

## Pull request overview

This PR introduces an experiment harness to run and validate SGLang offline-throughput benchmarks for Qwen 3.5 on the Genkai environment, including job rendering/submission helpers, result summarization/comparison utilities, and a CI workflow that gates benchmark preparation on a commit marker while keeping submission locked.

**Changes:**
- Add Genkai job templates and scripts to render, run, and (optionally) submit Qwen 3.5 experiments safely.
- Add result processing utilities (`summarize.py`, `compare.py`) plus fixtures and a shell validation script.
- Add a Genkai self-hosted runner workflow that validates scripts on each push and gates benchmark preparation on `[run-qwen]`.

### Reviewed changes

Copilot reviewed 17 out of 18 changed files in this pull request and generated 4 comments.

<details>
<summary>Show a summary per file</summary>

| File | Description |
| ---- | ----------- |
| `tests/test_scripts.sh` | Adds a regression/validation script to sanity-check scripts, workflow gating, rendering, and CSV utilities. |
| `tests/fixtures/results.jsonl` | Provides deterministic fixture throughput samples for summarize/compare tests. |
| `scripts/summarize.py` | Summarizes `results.jsonl` into a stable CSV (median metrics per run). |
| `scripts/compare.py` | Compares two summarized runs and prints percent deltas per metric. |
| `scripts/render-job.py` | Renders immutable PJM job scripts from `experiments/*.env` configs via a template. |
| `scripts/submit.py` | Renders a job and optionally submits it with an explicit confirmation prompt. |
| `scripts/run-experiment.sh` | Runs SGLang offline-throughput modes while capturing metadata and enforcing Qwen 3.5 model compatibility. |
| `jobs/templates/genkai.pjm.in` | Introduces a parameterized PJM template for Genkai runs inside Singularity OCI. |
| `jobs/genkai-qwen.pjm` | Adds a simple Genkai PJM entrypoint for running `run-experiment.sh` (containerized or local). |
| `jobs/genkai-qwen35-offline-oci.pjm` | Adds a standalone reference PJM script for offline throughput in OCI mode. |
| `experiments/qwen35-fp8-baseline.env` | Adds a baseline experiment definition for Qwen 3.5 FP8 throughput runs. |
| `experiments/qwen35-fp8-dummy-smoke.env` | Adds a lightweight smoke experiment definition using dummy load format. |
| `configs/default.env` | Adds shared default env values consumed by `run-experiment.sh`. |
| `docs/experiment-policy.md` | Documents experiment comparison and acceptance policy (JP). |
| `README.md` | Replaces placeholder README with usage docs and workflow description (JP). |
| `results/.gitkeep` | Ensures `results/` exists in the repo for local outputs. |
| `.gitignore` | Ignores local results, build outputs, and common artifacts. |
| `.github/workflows/genkai.yml` | Adds a Genkai self-hosted workflow that validates scripts and gates benchmark preparation via `[run-qwen]`. |
</details>







---

💡 <a href="/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/new/main?filename=.github/instructions/*.instructions.md" class="Link--inTextBlock" target="_blank" rel="noopener noreferrer">Add Copilot custom instructions</a> for smarter, more guided reviews. <a href="https://docs.github.com/en/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot" class="Link--inTextBlock" target="_blank" rel="noopener noreferrer">Learn how to get started</a>.


## Inline review comments

### Review comment 3565326870

Copilot · 2026-07-12T00:32:30Z
Recorded file: `scripts/render-job.py`

`os` is imported but never used. Keeping unused imports makes linting noisier and can confuse readers about dependencies.

### Review comment 3565326874

Copilot · 2026-07-12T00:32:30Z
Recorded file: `scripts/render-job.py`

The unresolved-token check `if "@" in text` will also reject valid values that contain `@` (e.g., OCI/Docker image digests like `image@sha256:...`). This can break rendering even when all template tokens are replaced.

### Review comment 3565326879

Copilot · 2026-07-12T00:32:31Z
Recorded file: `scripts/render-job.py`

If `--output` is passed as a relative path, `render-job.py` will write/print it relative to the caller's working directory, which can make `submit.py` (and `pjsub`) fail to locate the rendered job. Resolving `--output` to an absolute path avoids this class of errors.

### Review comment 3565326883

Copilot · 2026-07-12T00:32:31Z
Recorded file: `scripts/compare.py`

Row selection uses substring matching (`name in r["run"]`), which can pick the wrong row when run paths share substrings (e.g., `baseline` vs `baseline2`, or a parent directory containing the name). Matching against the run directory basename is more reliable.

