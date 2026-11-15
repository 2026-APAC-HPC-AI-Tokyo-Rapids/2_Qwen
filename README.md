# Qwen3 distributed inference

This repository contains the Tokyo Rapids with RIKEN Qwen3 competition submission, its SGLang optimizations, and the experiment records for Firmus, Genkai and ASPIRE 2A+.

The target is `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`, served with Prefill–Decode separation and evaluated with the official AIPerf workload. Start with the [submission guide](submission/SUBMISSION.md) for configurations, image and model pins, validation scope and reproduction instructions.

## Recorded results

| Configuration | Good output tok/s | Evidence |
| --- | ---: | --- |
| Basic: unmodified SGLang 0.5.23, c160 | 6321.12 / 6339.31 | [Basic reproduction and both runs](submission/basic/README.md) |
| Submitted Bonus: modified SGLang, PP4 Prefill, c232 | **10054.49** | [Submitted run 41767886175](submission/results/41767886175/RESULT.md) |
| Bonus package verification | 10038.11 | [Standalone run 41781374312](submission/results/41781374312/RESULT.md) |

All three entries use two Firmus nodes with 16 H200 GPUs. Basic's two-run mean is **6330.21**. The submitted Bonus result is **1.59×** that mean; the complete configurations differ in code, Prefill parallelism, concurrency and draft precision. The Bonus result is one measurement, and the standalone package check does not replace it.

The score is `goodput.avg × 200`. At least 90% of requests must meet the latency limits. Throughput eligibility and the local numerical/integration checks are separate from the organisers' full-set model accuracy assessment. See the [validation scope](submission/SUBMISSION.md#validation-scope).

## Reproduce a submission

- **Basic:** follow [submission/basic/README.md](submission/basic/README.md). The official SGLang release, kernels and gateway remain unmodified; deployment settings are tuned.
- **Bonus:** follow [submission/README.md](submission/README.md). The package includes its build inputs, source reconstruction bundle, runtime overlays, pinned kernel choices, preparation script and Slurm job.
- **Historical experiments:** [submission/EXPERIMENTS.md](submission/EXPERIMENTS.md) maps the presented experiments to scripts, conditions, metrics and logs. [submission/patches/README.md](submission/patches/README.md) inventories the code changes and their numerical behavior.

Preparation runs on a login node before GPU allocation. A scheduler job starts serving, runs the official benchmark and local parser, and leaves logs, results and source provenance in its shared run directory. Adapt site storage paths to the account used for reproduction while preserving the declared model, tokenizer and image identities.

The recorded images were built with Buildah `chroot` isolation, validated with Skopeo and Umoci, and converted to SquashFS for Enroot execution. Image build instructions are included in each submission README. A fresh build may have a different archive hash; use its sidecar integrity fields in a deployment copy of the configuration.

## Repository layout

| Path | Contents |
| --- | --- |
| [submission/](submission/SUBMISSION.md) | Basic and Bonus reproduction packages, measured results and experiment evidence |
| [patches/](submission/patches/README.md) | SGLang, DeepGEMM and router patches, including rejected research candidates |
| [kernels/](kernels/probes/README.md) | Custom kernels, numerical checks and microbenchmarks |
| [configs/](configs/) | Tracked runtime conditions and image identities |
| [platforms/](platforms/) | Slurm, PJM and PBS execution scripts |
| [scripts/](scripts/) | Build, provisioning, validation, result collection and analysis helpers |
| [containers/](containers/) and [aws/](aws/) | Image recipes, dependencies and build infrastructure |
| [qwen-experiment-metrics/](qwen-experiment-metrics/README.md) | Experiment metrics collection and publication |
| [research/](research/) | Design decisions, experiment ledgers, profiles and rejected hypotheses |
| [TASK_2026_10_03.md](TASK_2026_10_03.md) | Historical development checklist |

Before further kernel work, read the [kernel theory tracker](research/runtime-performance/kernel-theory-tracker.md) and its [input model](research/runtime-performance/kernel-theory-model.json). ASPIRE's H100 results and Firmus's H200 results use different hardware and must be interpreted separately. The [ASPIRE migration record](research/runtime-performance/2026-10-26-aspire-firmus9120-port.json) documents the later 5925.07 / 6081.54 serving observations and their differing selected kernels.

## GitHub Actions and site setup

Repository Actions remains enabled for retained run metadata. Automatic triggers are removed from the current YAML, every job has a literal false execution gate, and file-backed workflows are disabled individually to prevent replaying historical definitions. The original routing conditions remain for reference.

[Sanitized execution logs, artifact evidence and archived pull requests](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115) retain original IDs, conclusions and timestamps. Use these records for runs whose hosted logs or metadata were removed to avoid exposing site details. Complete originals are retained privately.

Site account components in saved paths and configuration defaults are placeholders. Supply the paths for your reproduction account before using the archived execution scripts.

The archived [Qwen workflow](.github/workflows/qwen.yml) records the build and scheduler path. [Basic](.github/workflows/qwen-basic.yml) records its separate Firmus path. Site access requires the configured runner labels and custom shells, cluster accounts, shared storage and the AWS repository variables referenced by the workflows. The automated Bonus builder also requires access to its configured SGLang source mirror; the standalone package reconstructs that source from public upstream and the supplied bundle.

Set the repository secret `QWEN_ARTIFACT_ARCHIVE_ROOT` to the persistent archive directory visible to the archive runner. It is used by the regular and Basic archive jobs and the CPU-only Basic metrics publisher. Measurement subdirectories are `<Actions run>/<attempt>`. Archive summaries identify the measurement without publishing the server hostname or storage path; using a secret also masks the destination in Actions logs.

Use [.env.example](.env.example) as a reference for local SSH aliases. Keep actual connection settings in the untracked `.env` and private SSH configuration.

The tags below describe the recorded campaign; execution is disabled in the current archived workflows.

| Commit message tag | Recorded behavior |
| --- | --- |
| `[require-build]` at the start | Build a new image; omit it for runtime-only reuse |
| `[cluster:genkai]`, `[cluster:aspire2ap]`, `[cluster:firmus]` | Submit to the named clusters; multiple tags run in parallel |
| No cluster tag | Validate without a scheduler submission |
| `[nodes:1]`, `[nodes:2]`, `[nodes:4]` | Select the supported node count; default is 2 |
| `[profile]` | Collect a separate instrumented profile |
| `[basic] [cluster:firmus]` | Select the Basic workflow |
| `[skip ci]` at the start | Skip push-triggered Actions |

Build changes include source, toolchain, dependencies and image contents. Runtime, analysis and documentation changes reuse existing images. Keep each experimental condition in its own traceable commit. The [official protocol record](research/official-qwen3-pd.md), [image build record](research/aws-container-build.md) and [runtime migration record](research/container-runtime-migration.md) explain the execution path.

## Publication copies

Local operational instructions, connection settings and working notes are excluded from Git tracking. [Publication redactions](submission/PUBLICATION.md) identify the changes to logs and private infrastructure identifiers. Performance evidence and fixed build inputs remain available in the submission package.

## Task and research notes

[The task and research index](research/README.md) links the current task checklist, earlier plans, kernel optimization candidates and dated capacity estimates. Historical proposals retain their date and are separate from measured serving results.
