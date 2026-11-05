# Presented Qwen experiments and evidence

**Submitted result: 10054 good output tok/s (measured: 10054.49), Actions 41767886175 / Slurm 28660.** [SUBMISSION.md](SUBMISSION.md) identifies both reproduction entrypoints.

This index follows the numbered pages of `../presentation/Tokyo Rapids with RIKEN Presentation.pptx` (Qwen main slides 38–67). The raw exports, scores, source/configuration and scheduler logs under `experiments/<run>/` were copied without changes after verification against the canonical archive's SHA-256 manifest. Historical PBS scripts missing from those archives were extracted from the recorded Git commit and are identified below. They document the original experiment and require its original preparation and build inputs. Use the current READMEs for the delivered deployment.

## Comparison and validation scope

- The selected Basic run scored **6339.31**, run 41093781934 / Slurm 27709. Its repeat is 6321.12; the two-run mean **6330.21** is the comparison reference used in the talk and below. A historical milestone below this later Basic is not called a speedup.
- ASPIRE H100 and Firmus H200 are separate platforms. No stock ASPIRE Basic was submitted, so H100 runs have no ratio to the Firmus Basic.
- Most milestones bundle changes in code, topology, load or selected kernels. Adjacent scores describe the real history, not an isolated kernel effect. Separate runs and short kernel tests do not establish a repeatable serving improvement.
- Historical `profile: true` jobs are labelled below. Their presented fixed-duration client measurements are retained, but profiler attachment and other configuration differences limit comparison with clean serving runs. The submitted and standalone package runs have profiling disabled.
- Numerical arguments and gates have named controls and tested scopes. The organisers' full-set accuracy check remains outstanding; the evidence here covers SLA eligibility, kernel checks and short prompt probes.

## First full-length Firmus benchmark

[Initial execution record](experiments/initial-firmus/README.md): Slurm 17003 completed the fixed-duration client at **902.36**, then failed the old local length validator. The first run that completed successfully and passed local validation is **909.68**, Actions 34732532321 / Slurm 17021, with [raw evidence](experiments/34732532321/score.json). Neither is the organisers' 1832 reference.

## Presented milestones

| Experiment | Slide | Platform | Good output tok/s | Good fraction | Job | Ratio to Basic mean |
| --- | --- | --- | ---: | ---: | --- | --- |
| [Serving frameworks](#framework) | 42 | ASPIRE H100 | 1827.79 | 0.98602620 | 213369.pbs111 | not comparable |
| [ASPIRE layout](#layout) | 44 | ASPIRE H100 | 972.04 | 1.00000000 | 213000.pbs111 | not comparable |
| [EAGLE3](#eagle) | 45 | ASPIRE H100 | 2346.71 | 0.98502383 | 213029.pbs111 | not comparable |
| [ASPIRE options and Prefill patches](#options) | 46 | ASPIRE H100 | 2865.54 | 0.99209486 | 217588.pbs111 | not comparable |
| [ASPIRE P/D rebalance](#aspire-pd) | 50 | ASPIRE H100 | 3313.04 | 0.97611085 | 218108.pbs111 | not comparable |
| [Port to Firmus](#port) | 48–49 | Firmus H200 | 3170.21 | 0.99487179 | 21951 | 0.50x |
| [Firmus P/D rebalance](#firmus-pd) | 50 | Firmus H200 | 4369.38 | 0.93574158 | 22269 | 0.69x |
| [64-token KV pages](#pages) | 50 | Firmus H200 | 4429.95 | 0.93333333 | 22324 | 0.70x |
| [MegaMoE and DP attention](#megamoe) | 52 | Firmus H200 | 6344.96 | 0.95974782 | 23571 | 1.00x |
| [vLLM side track](#vllm) | 53 | Firmus H200 | 5274.86 | 0.97832631 | 23234 | 0.83x |
| [Iteration 1: Decode MoE](#iteration-1) | 55 | Firmus H200 | 7251.58 | 0.97594800 | 24414 | 1.15x |
| [Iteration 2: Decode admission](#iteration-2) | 56 | Firmus H200 | 7999.37 | 0.98336963 | 24543 | 1.26x |
| [Iteration 3: Prefill KV and MoE](#iteration-3) | 57 | Firmus H200 | 8356.18 | 0.98574144 | 25062 | 1.32x |
| [Iteration 4: Decode batches](#iteration-4) | 58 | Firmus H200 | 9036.12 | 0.97669160 | 26169 | 1.43x |
| [Iteration 5: Prefill host waits](#iteration-5) | 59–60 | Firmus H200 | 9235.28 | 0.98296335 | 26889 | 1.46x |
| [Iteration 6: PP4 Prefill](#iteration-6) | 61 | Firmus H200 | 9668.27 | 0.97826087 | 27422 | 1.53x |
| [Iteration 7: Prefill MoE rescale](#iteration-7) | 62 | Firmus H200 | 9832.74 | 0.97939472 | 27502 | 1.55x |
| [Iteration 8 and submitted final](#iteration-8) | 63–64 | Firmus H200 | 10054.49 | 0.98215696 | 28660 | 1.59x |

<a id="framework"></a>

### Serving frameworks (slide 42)

We compared SGLang, vLLM and TensorRT-LLM and selected SGLang.

**Computation argument:** Same target model and workload; engine implementations differ.

**Interpretation:** SGLang 1827.79 is c40; vLLM 1527.97 (36310351009, PBS 213388.pbs111) is c32. The archived settings are c40 and c32, so this is a comparison of the two deployments.

Evidence: [Actions 36305644422](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/36305644422), [script](experiments/36305644422/source/submit.pbs), [score](experiments/36305644422/score.json), [raw export](experiments/36305644422/profile_export_aiperf.json), [scheduler log](experiments/36305644422/logs/pbs-accounting.txt), [provenance](experiments/36305644422/source/provenance.json).

Source `683b3742fa28c64f1413376f9c7fe51b63cb67fe`; image SHA-256 `6b4d81eb6400fa33400e5725a2842f63565b700d579f9be812a7c229a80da3b5`; recorded profile flag `false`.
Historical script: `Git 683b3742fa28c64f1413376f9c7fe51b63cb67fe:platforms/aspire2ap/run-qwen-pd.pbs; historical source, requires its original preparation`.

<a id="layout"></a>

### ASPIRE layout (slide 44)

We used P2 TP4 / D2 TP4 without speculation at c16 after simulator-guided screening.

**Computation argument:** All model layers and experts remain; only placement and parallelism change.

**Interpretation:** 972.04 is H100 serving; simulator predictions are estimates, and there is no stock ASPIRE Basic in this submission.

Evidence: [Actions 36228974704](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/36228974704), [script](experiments/36228974704/source/submit.pbs), [score](experiments/36228974704/score.json), [raw export](experiments/36228974704/profile_export_aiperf.json), [scheduler log](experiments/36228974704/logs/pbs-accounting.txt), [provenance](experiments/36228974704/source/provenance.json).

Source `24a84d3171793cd48177accce59dfa566fedd41d`; image SHA-256 `6b4d81eb6400fa33400e5725a2842f63565b700d579f9be812a7c229a80da3b5`; recorded profile flag `false`.
Historical script: `Git 24a84d3171793cd48177accce59dfa566fedd41d:platforms/aspire2ap/run-qwen-pd.pbs; historical source, requires its original preparation`.

<a id="eagle"></a>

### EAGLE3 (slide 45)

We used the public Nebius draft, 3 steps / 4 draft tokens, at c40.

**Computation argument:** The target verifies the proposed tokens; target weights are unchanged.

**Interpretation:** Two EAGLE runs 36234985587 / 36235158063 average 2330.95; target-only c40 controls 36230331081 / 36232258544 average 1839.42 (+26.7%). FP8 KV at -0.88% is a separate comparison.

Evidence: [Actions 36234985587](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/36234985587), [script](experiments/36234985587/source/submit.pbs), [score](experiments/36234985587/score.json), [raw export](experiments/36234985587/profile_export_aiperf.json), [scheduler log](experiments/36234985587/logs/pbs-accounting.txt), [provenance](experiments/36234985587/source/provenance.json).

Source `66f3ca60486066b9a3685ef004082d8a81093061`; image SHA-256 `6b4d81eb6400fa33400e5725a2842f63565b700d579f9be812a7c229a80da3b5`; recorded profile flag `false`.
Historical script: `Git 66f3ca60486066b9a3685ef004082d8a81093061:platforms/aspire2ap/run-qwen-pd.pbs; historical source, requires its original preparation`.

<a id="options"></a>

### ASPIRE options and Prefill patches (slide 46)

We combined CUTLASS Prefill, routing/shuffle improvements and symmetric-memory tuning.

**Computation argument:** The model operations remain; layout changes and kernel rounding effects are documented separately in the patch inventory.

**Interpretation:** 2865.54 is a later milestone with different node pair and concurrency; its +23% over the EAGLE average is not an isolated patch effect.

Evidence: [Actions 37741311522](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37741311522), [script](experiments/37741311522/source/submit.pbs), [score](experiments/37741311522/score.json), [raw export](experiments/37741311522/profile_export_aiperf.json), [scheduler log](experiments/37741311522/logs/pbs-accounting.txt), [provenance](experiments/37741311522/source/provenance.json).

Source `7b5478ae4633e51a3c399bb615f3cedfdd314698`; image SHA-256 `182fbb6df7f941aa688a393dfc181bb269f3112a6d34c33f6007ddc5d09f0a42`; recorded profile flag `false`.
Historical script: `Git 7b5478ae4633e51a3c399bb615f3cedfdd314698:platforms/aspire2ap/run-qwen-pd.pbs; historical source, requires its original preparation`.

<a id="aspire-pd"></a>

### ASPIRE P/D rebalance (slide 50)

We assigned 12 GPUs to Prefill and 4 to Decode, enlarged the Decode KV pool and used c72.

**Computation argument:** Placement and capacity change while the target computation and FP8 KV format are retained.

**Interpretation:** 3313.04 uses H100 and different load from the earlier P2/D2 run; do not compare it with the H200 Basic.

Evidence: [Actions 37915507286](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37915507286), [script](experiments/37915507286/source/submit.pbs), [score](experiments/37915507286/score.json), [raw export](experiments/37915507286/profile_export_aiperf.json), [scheduler log](experiments/37915507286/logs/pbs-accounting.txt), [provenance](experiments/37915507286/source/provenance.json).

Source `ed8c5342c400f82681c724f7c1a2075187386df5`; image SHA-256 `182fbb6df7f941aa688a393dfc181bb269f3112a6d34c33f6007ddc5d09f0a42`; recorded profile flag `false`.
Historical script: `Git ed8c5342c400f82681c724f7c1a2075187386df5:platforms/aspire2ap/run-qwen-pd.pbs; historical source, requires its original preparation`.

<a id="port"></a>

### Port to Firmus (slide 48–49)

We ran the ASPIRE-derived P2/D2 recipe at c48 on H200.

**Computation argument:** The official checkpoint is retained; EAGLE3, FP8 KV and patches belong to the complete ported configuration.

**Interpretation:** 3170.21 versus the organiser reference 1832 is a whole-configuration comparison; it does not isolate the port or convert H100 performance to H200.

Evidence: [Actions 37868180304](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37868180304), [script](experiments/37868180304/source/submit.sbatch), [score](experiments/37868180304/score.json), [raw export](experiments/37868180304/profile_export_aiperf.json), [scheduler log](experiments/37868180304/logs/slurm-accounting.txt), [provenance](experiments/37868180304/source/provenance.json).

Source `e481de9ec084a124039e47c1b3c57fdfdad39822`; image SHA-256 `182fbb6df7f941aa688a393dfc181bb269f3112a6d34c33f6007ddc5d09f0a42`; recorded profile flag `false`.

<a id="firmus-pd"></a>

### Firmus P/D rebalance (slide 50)

We used six TP2 Prefill workers and one TP4 Decode worker at c104 on the same 16 H200 GPUs.

**Computation argument:** Only placement, capacity and load change; model layers and experts are retained.

**Interpretation:** 4369.38 follows the real TP2 history. No matched Firmus P3×TP4 control was run before MegaMoE.

Evidence: [Actions 38020231725](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38020231725), [script](experiments/38020231725/source/submit.sbatch), [score](experiments/38020231725/score.json), [raw export](experiments/38020231725/profile_export_aiperf.json), [scheduler log](experiments/38020231725/logs/slurm-accounting.txt), [provenance](experiments/38020231725/source/provenance.json).

Source `ff58ed80d6515cf261dfa95047f3fd9e89320c8d`; image SHA-256 `4b919a7465f1647f4eddc677393cef00f2f390a8544b52a5f6b2456dce940d62`; recorded profile flag `false`.

<a id="pages"></a>

### 64-token KV pages (slide 50)

We changed the P6/D1 workers from page size 1 to 64.

**Computation argument:** Addressing changes while stored KV values and dtype remain the same.

**Interpretation:** 4429.95 versus 4369.38 is +1.39% in separate single runs; H100 did not show the same gain.

Evidence: [Actions 38034080121](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38034080121), [script](experiments/38034080121/source/submit.sbatch), [score](experiments/38034080121/score.json), [raw export](experiments/38034080121/profile_export_aiperf.json), [scheduler log](experiments/38034080121/logs/slurm-accounting.txt), [provenance](experiments/38034080121/source/provenance.json).

Source `3519fc0641064ef42a2f3794bb239a91a07e8aa4`; image SHA-256 `4b919a7465f1647f4eddc677393cef00f2f390a8544b52a5f6b2456dce940d62`; recorded profile flag `false`.

<a id="megamoe"></a>

### MegaMoE and DP attention (slide 52)

We regrouped the 12/4 GPUs into P3/D1 DP4/EP4 workers with MegaMoE, routing and handoff changes.

**Computation argument:** The existing top-k IDs and FP32 weights feed the fused expert GEMMs, SiLU and combine; rounding changes are disclosed in the inventory.

**Interpretation:** 6344.96 bundles parallelism, kernels, routing, capacity and load. It does not isolate MegaMoE or establish that four-GPU groups are globally optimal.

Evidence: [Actions 38664373784](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38664373784), [script](experiments/38664373784/source/submit.sbatch), [score](experiments/38664373784/score.json), [raw export](experiments/38664373784/profile_export_aiperf.json), [scheduler log](experiments/38664373784/logs/slurm-accounting.txt), [provenance](experiments/38664373784/source/provenance.json).

Source `e6e295bd98ea16cb393c672f0b1b0af7b300cccf`; image SHA-256 `8df47c11d7ede66fd19f12c7012a9ab5c97b13387a2655fcbce9f13e1c334110`; recorded profile flag `true`.

<a id="vllm"></a>

### vLLM side track (slide 53)

We used native DeepGEMM for Prefill MoE with E8M0 weight-scale re-rounding disabled.

**Computation argument:** The published FP8 weights and FP32 scales are retained; kernel execution and summation order differ.

**Interpretation:** 5274.86 used P6/D1, c112 and two proposals. None of this side-track code enters the submitted SGLang package.

Evidence: [Actions 38485732174](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38485732174), [script](experiments/38485732174/source/submit.sbatch), [score](experiments/38485732174/score.json), [raw export](experiments/38485732174/profile_export_aiperf.json), [scheduler log](experiments/38485732174/logs/slurm-accounting.txt), [provenance](experiments/38485732174/source/provenance.json).

Source `83a1dbcb1895f6d62719d694c278a83e3ab32bbc`; image SHA-256 `2012b52332deaecdab9a7a397a35412b7e098c40308f93560c7a419519ebfb0e`; recorded profile flag `false`.

<a id="iteration-1"></a>

### Iteration 1: Decode MoE (slide 55)

We parallelised token conversion across four GPU threads, with Decode kernel and Prefill expert-placement changes.

**Computation argument:** FP8 operands/scales and selected expert operations are retained; each patch has a named numerical control.

**Interpretation:** 7251.58 is a bundle endpoint, with another run at 7185.18. The 21→12.5 microsecond local conversion timing does not explain the whole score gain.

Evidence: [Actions 39070273796](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39070273796), [script](experiments/39070273796/source/submit.sbatch), [score](experiments/39070273796/score.json), [raw export](experiments/39070273796/profile_export_aiperf.json), [scheduler log](experiments/39070273796/logs/slurm-accounting.txt), [provenance](experiments/39070273796/source/provenance.json).

Source `0374c551d212f27f05127a68102897fb582029a9`; image SHA-256 `7c6286e5dcfcd8f5fba37d03d0c1ac0b4f6e93562a24a147afa1063f8214bc82`; recorded profile flag `true`.

<a id="iteration-2"></a>

### Iteration 2: Decode admission (slide 56)

We prepared new request inputs while the GPU verified tokens, together with kernel fusions and cap 128.

**Computation argument:** Input dependencies are respected before consumption; target verification is retained and fusion rounding is disclosed.

**Interpretation:** 7999.37 combines several changes. Cap 136 on the same image scored 7082.82 with a different kernel geometry; this is not a scheduling-only gain.

Evidence: [Actions 39114949128](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39114949128), [script](experiments/39114949128/source/submit.sbatch), [score](experiments/39114949128/score.json), [raw export](experiments/39114949128/profile_export_aiperf.json), [scheduler log](experiments/39114949128/logs/slurm-accounting.txt), [provenance](experiments/39114949128/source/provenance.json).

Source `5066eaffdbdc7496a26552bb043ba8b1ab117d9b`; image SHA-256 `a8ed45928e94b70ea592feb9fbe606d9cff574b8f6c330c1bc0e9186c6fab07e`; recorded profile flag `true`.

<a id="iteration-3"></a>

### Iteration 3: Prefill KV and MoE (slide 57)

We read token-ordered KV for Prefill attention while preserving paged KV for Decode, alongside MoE/input changes.

**Computation argument:** The same KV values are used; attention reduction, FP8 rounding and per-128 FC2 scaling are disclosed in the inventory.

**Interpretation:** 8356.18 is cumulative, including two math warpgroups. Local attention and MoE timings are not additive estimates of serving speedup.

Evidence: [Actions 39447708936](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39447708936), [script](experiments/39447708936/source/submit.sbatch), [score](experiments/39447708936/score.json), [raw export](experiments/39447708936/profile_export_aiperf.json), [scheduler log](experiments/39447708936/logs/slurm-accounting.txt), [provenance](experiments/39447708936/source/provenance.json).

Source `39793c14e88ce3a19ec9737a7893a33432683a52`; image SHA-256 `652aaeef5486140efaa9acfe9c90a05db870dd3a1ad65c70228cdbef6e0c6f01`; recorded profile flag `true`.

<a id="iteration-4"></a>

### Iteration 4: Decode batches (slide 58)

We extended fast-kernel coverage beyond 128 verification rows and tuned attention/GEMM while raising Decode capacity.

**Computation argument:** The target verifies the same candidates; tiling changes execution order without pruning model work.

**Interpretation:** 9036.12 used cap 176, or 44 requests per rank. The slide illustration of 48 requests describes later coverage/cap.

Evidence: [Actions 40201493911](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40201493911), [script](experiments/40201493911/source/submit.sbatch), [score](experiments/40201493911/score.json), [raw export](experiments/40201493911/profile_export_aiperf.json), [scheduler log](experiments/40201493911/logs/slurm-accounting.txt), [provenance](experiments/40201493911/source/provenance.json).

Source `3287a43a620f95bf829bfaf8ae522bc7d8d947dd`; image SHA-256 `1f247668933c7252d3dfbe8f6c23897d354f095d56842788d4415a85d7dc5e55`; recorded profile flag `false`.

<a id="iteration-5"></a>

### Iteration 5: Prefill host waits (slide 59–60)

We removed host waits, loaded the draft as FP8, aligned Prefill starts and selected c215.

**Computation argument:** The target checkpoint is unchanged and verifies every draft token; FP8 draft loading does not requantise target weights.

**Interpretation:** 9235.28 is a bundle endpoint. Idle-gap, clock and power observations belong to their own profiles, not a synthetic timeline for this run.

Evidence: [Actions 40630160781](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40630160781), [script](experiments/40630160781/source/submit.sbatch), [score](experiments/40630160781/score.json), [raw export](experiments/40630160781/profile_export_aiperf.json), [scheduler log](experiments/40630160781/logs/slurm-accounting.txt), [provenance](experiments/40630160781/source/provenance.json).

Source `840b40f7ea24c83a474c1578a29af86d6bfb0d43`; image SHA-256 `1f247668933c7252d3dfbe8f6c23897d354f095d56842788d4415a85d7dc5e55`; recorded profile flag `false`.

<a id="iteration-6"></a>

### Iteration 6: PP4 Prefill (slide 61)

We moved Prefill to four layer stages with copy-engine transport and runtime changes.

**Computation argument:** All 94 layers remain, split 24/24/24/22; same-dtype peer copies replace proxy transfers with ordering and content checks.

**Interpretation:** 9668.27 completed the fixed-duration client, but the job later TIMEOUT during an accidentally enabled profile pass. This is not a clean terminal-success example.

Evidence: [Actions 40964714831](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40964714831), [script](experiments/40964714831/source/submit.sbatch), [score](experiments/40964714831/score.json), [raw export](experiments/40964714831/profile_export_aiperf.json), [scheduler log](experiments/40964714831/logs/slurm-accounting.txt), [provenance](experiments/40964714831/source/provenance.json).

Source `5937a5a17b8353fca212d6c16de6a0533d4f2625`; image SHA-256 `0a02c4ab526787663fece90c472b5d794f9a804ad1b0889fdb9d62a393b71de4`; recorded profile flag `true`.

<a id="iteration-7"></a>

### Iteration 7: Prefill MoE rescale (slide 62)

We overlapped scale correction with matrix multiplication using split accumulators, alongside other selected kernels.

**Computation argument:** In-job endurance checks required final BF16 outputs to match their control on the tested cases.

**Interpretation:** 9832.74 is the H200 serving result. The slide H100 cycle experiment is kernel-only; its ratio is not converted into H200 throughput.

Evidence: [Actions 41005872470](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41005872470), [script](experiments/41005872470/source/submit.sbatch), [score](experiments/41005872470/score.json), [raw export](experiments/41005872470/profile_export_aiperf.json), [scheduler log](experiments/41005872470/logs/slurm-accounting.txt), [provenance](experiments/41005872470/source/provenance.json).

Source `6abfef9202de8af50821011884ce74be8c382d5e`; image SHA-256 `db6eeb49303429ae4b3c0dc08f74fbea5101784d231c5e66dc8d7e6fabd59fe5`; recorded profile flag `false`.

<a id="iteration-8"></a>

### Iteration 8 and submitted final (slide 63–64)

We reused prepacked V in Prefill FA3 and retained later PP4, greedy-verify and FC1 updates in round 34.

**Computation argument:** Producer-byte and FP32-reference checks cover prepacking; FC1 first-K passed byte equality on 12 GPUs. Other rounding changes are disclosed separately.

**Interpretation:** 10054.49 is the submitted value. The progression from 9832.74 includes more than V prepacking. The added logprob relay was checked at 10038.11.

Evidence: [Actions 41767886175](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41767886175), [script](experiments/41767886175/source/submit.sbatch), [score](results/41767886175/score.json), [raw export](results/41767886175/profile_export_aiperf.json), [scheduler log](results/41767886175/logs/slurm-accounting.txt), [provenance](experiments/41767886175/source/provenance.json).

Source `50773368654b5a36497c1d69a1bb0e544eaee66e`; image SHA-256 `3a47b757bb88f6bc14ad5f151e8405f645007f25f381f631d88046d9bd1b7b04`; recorded profile flag `false`.

## Companion controls and repeats

The EAGLE3 mean uses [run 36235158063](experiments/36235158063/profile_export_aiperf.json) with [its configuration](experiments/36235158063/source/aspire-serving.json), alongside the primary run above. Its no-speculation controls are [36230331081](experiments/36230331081/profile_export_aiperf.json) and [36232258544](experiments/36232258544/profile_export_aiperf.json), with their source configurations and PBS accounting beside the exports. The framework comparison includes [vLLM run 36310351009](experiments/36310351009/profile_export_aiperf.json) and [its c32 configuration](experiments/36310351009/source/aspire-serving.json).

## Material clarifications to the presentation

- **Slide 42:** SGLang 1827.79 is c40; vLLM 1527.97 is c32 in the retained configuration. These archived settings differ from the slide's same-c40 label.
- **Slide 58:** the 9036.12 checkpoint uses cap 176 (44 per rank); 48 requests describes later capacity/coverage.
- **Slide 63:** 10054.49 also includes later kernel and scheduler updates. The prepack gate measures local savings on its tested shapes, not the whole score progression.
- **Slide 64:** 33 SGLang + 14 DeepGEMM runtime overlays describes the submitted tuning run. The delivered package has **34 + 14**, adding logprob relay. Build-time patches are additional inputs in the inventory.
- **Slides 70 and 81:** the 9120.51/9132.03 package and 9235.28 best-run labels are historical. The current submission is **10054.49**, with **10038.11** as package verification.
- **Slide 80:** local numerical gates do not certify the full final stack as bit-identical or establish end-to-end accuracy.
- **Slides 96–145:** the deck labels this section Archive. Superseded source pins, counts and scores there do not select the current submission.

The delivered scripts were simplified after the recorded GPU runs. CPU fixtures check launch arguments, environment, patch order and orchestration against archived references. There is no full GPU run of the final readability rewrite, and fresh image archives have not been shown bit-identical.
