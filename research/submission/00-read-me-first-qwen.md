Qwen Basic (the run to re-run)  
Platform: Firmus, 2 nodes x 8 H200 (16 GPUs)  
Framework: unmodified SGLang 0.5.23  
Speculative decoding: yes, nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507; BF16 draft, 3 steps / 4 draft tokens  
Good output tok/s (official command, goodput x 200): 6339.31; the same configuration's two-run mean is 6330.21 (6321.12 / 6339.31)  
good_request_fraction: 0.9791252485089463  
Concurrency N: 160  
Our first full-length run on this platform, tok/s: 902.36 (Slurm 17003; full client completed, old local length validator failed); first retained clean terminal and validated run: 909.68 (Slurm 17021). Evidence: qwen/experiments/initial-firmus/README.md  
Bottleneck we found, one sentence: Prefill supply limited throughput; on stock SGLang, routing and concurrency tuning raised goodput, while c192 failed the TTFT SLA gate.  
Job id: Slurm 27709; Actions 41093781934 (repeat: Slurm 27682 / Actions 41078642203)  
Script (path in this shared folder): qwen/basic/submit.sbatch; prepare with qwen/basic/prepare.sh and qwen/basic/config/firmus-basic.json  
Log (path in this shared folder): qwen/basic/results/41093781934/logs/slurm-27709.log; raw metrics: qwen/basic/results/41093781934/profile_export_aiperf.json

Qwen submitted experimental result: **10054 good output tok/s (10054.49 before rounding)**, Bonus; Actions 41767886175 / Slurm 28660. This is 1.5883x the Basic mean (1.5861x the named Basic run). The package verification value 10038.11 does not replace it. Main Qwen slides: 38–67; evidence and slide clarifications: qwen/EXPERIMENTS.md. The team has not run the organisers' full-set accuracy check.

Qwen experiment: Serving frameworks (slide 42)  
What changed, one sentence: Compared SGLang, vLLM and TensorRT-LLM; retained SGLang.  
Why the computation is the same, and where that evidence is: Same target model and workload; engine implementations differ. Argument and limits: qwen/EXPERIMENTS.md#framework; configuration and source: qwen/experiments/36305644422/source/.  
Speed-up over our Basic: Not comparable: ASPIRE H100; the submitted Basic is Firmus H200.  
Job id: 213369.pbs111; Actions 36305644422  
Script (path in this shared folder): qwen/experiments/36305644422/source/submit.pbs  
Log (path in this shared folder): qwen/experiments/36305644422/logs/pbs-accounting.txt; raw metrics: qwen/experiments/36305644422/profile_export_aiperf.json

Qwen experiment: ASPIRE layout (slide 44)  
What changed, one sentence: Used P2 TP4 / D2 TP4 without speculation at c16 after simulator-guided screening.  
Why the computation is the same, and where that evidence is: All model layers and experts remain; only placement and parallelism change. Argument and limits: qwen/EXPERIMENTS.md#layout; configuration and source: qwen/experiments/36228974704/source/.  
Speed-up over our Basic: Not comparable: ASPIRE H100; the submitted Basic is Firmus H200.  
Job id: 213000.pbs111; Actions 36228974704  
Script (path in this shared folder): qwen/experiments/36228974704/source/submit.pbs  
Log (path in this shared folder): qwen/experiments/36228974704/logs/pbs-accounting.txt; raw metrics: qwen/experiments/36228974704/profile_export_aiperf.json

Qwen experiment: EAGLE3 (slide 45)  
What changed, one sentence: Used the public Nebius draft, 3 steps / 4 draft tokens, at c40.  
Why the computation is the same, and where that evidence is: The target verifies the proposed tokens; target weights are unchanged. Argument and limits: qwen/EXPERIMENTS.md#eagle; configuration and source: qwen/experiments/36234985587/source/.  
Speed-up over our Basic: Not comparable: ASPIRE H100; the submitted Basic is Firmus H200.  
Job id: 213029.pbs111; Actions 36234985587  
Script (path in this shared folder): qwen/experiments/36234985587/source/submit.pbs  
Log (path in this shared folder): qwen/experiments/36234985587/logs/pbs-accounting.txt; raw metrics: qwen/experiments/36234985587/profile_export_aiperf.json

Qwen experiment: ASPIRE options and Prefill patches (slide 46)  
What changed, one sentence: Combined CUTLASS Prefill, routing/shuffle improvements and symmetric-memory tuning.  
Why the computation is the same, and where that evidence is: The model operations remain; layout changes and kernel rounding effects are documented separately in the patch inventory. Argument and limits: qwen/EXPERIMENTS.md#options; configuration and source: qwen/experiments/37741311522/source/.  
Speed-up over our Basic: Not comparable: ASPIRE H100; the submitted Basic is Firmus H200.  
Job id: 217588.pbs111; Actions 37741311522  
Script (path in this shared folder): qwen/experiments/37741311522/source/submit.pbs  
Log (path in this shared folder): qwen/experiments/37741311522/logs/pbs-accounting.txt; raw metrics: qwen/experiments/37741311522/profile_export_aiperf.json

Qwen experiment: ASPIRE P/D rebalance (slide 50)  
What changed, one sentence: Assigned 12 GPUs to Prefill and 4 to Decode, enlarged the Decode KV pool and used c72.  
Why the computation is the same, and where that evidence is: Placement and capacity change while the target computation and FP8 KV format are retained. Argument and limits: qwen/EXPERIMENTS.md#aspire-pd; configuration and source: qwen/experiments/37915507286/source/.  
Speed-up over our Basic: Not comparable: ASPIRE H100; the submitted Basic is Firmus H200.  
Job id: 218108.pbs111; Actions 37915507286  
Script (path in this shared folder): qwen/experiments/37915507286/source/submit.pbs  
Log (path in this shared folder): qwen/experiments/37915507286/logs/pbs-accounting.txt; raw metrics: qwen/experiments/37915507286/profile_export_aiperf.json

Qwen experiment: Port to Firmus (slide 48–49)  
What changed, one sentence: Ran the ASPIRE-derived P2/D2 recipe at c48 on H200.  
Why the computation is the same, and where that evidence is: The official checkpoint is retained; EAGLE3, FP8 KV and patches belong to the complete ported configuration. Argument and limits: qwen/EXPERIMENTS.md#port; configuration and source: qwen/experiments/37868180304/source/.  
Speed-up over our Basic: 0.5008x (3170.21 / 6330.21); historical whole-configuration ratio.  
Job id: 21951; Actions 37868180304  
Script (path in this shared folder): qwen/experiments/37868180304/source/submit.sbatch  
Log (path in this shared folder): qwen/experiments/37868180304/logs/slurm-accounting.txt; raw metrics: qwen/experiments/37868180304/profile_export_aiperf.json

Qwen experiment: Firmus P/D rebalance (slide 50)  
What changed, one sentence: Used six TP2 Prefill workers and one TP4 Decode worker at c104 on the same 16 H200 GPUs.  
Why the computation is the same, and where that evidence is: Only placement, capacity and load change; model layers and experts are retained. Argument and limits: qwen/EXPERIMENTS.md#firmus-pd; configuration and source: qwen/experiments/38020231725/source/.  
Speed-up over our Basic: 0.6902x (4369.38 / 6330.21); historical whole-configuration ratio.  
Job id: 22269; Actions 38020231725  
Script (path in this shared folder): qwen/experiments/38020231725/source/submit.sbatch  
Log (path in this shared folder): qwen/experiments/38020231725/logs/slurm-accounting.txt; raw metrics: qwen/experiments/38020231725/profile_export_aiperf.json

Qwen experiment: 64-token KV pages (slide 50)  
What changed, one sentence: Changed the P6/D1 workers from page size 1 to 64.  
Why the computation is the same, and where that evidence is: Addressing changes while stored KV values and dtype remain the same. Argument and limits: qwen/EXPERIMENTS.md#pages; configuration and source: qwen/experiments/38034080121/source/.  
Speed-up over our Basic: 0.6998x (4429.95 / 6330.21); historical whole-configuration ratio.  
Job id: 22324; Actions 38034080121  
Script (path in this shared folder): qwen/experiments/38034080121/source/submit.sbatch  
Log (path in this shared folder): qwen/experiments/38034080121/logs/slurm-accounting.txt; raw metrics: qwen/experiments/38034080121/profile_export_aiperf.json

Qwen experiment: MegaMoE and DP attention (slide 52)  
What changed, one sentence: Regrouped the 12/4 GPUs into P3/D1 DP4/EP4 workers with MegaMoE, routing and handoff changes.  
Why the computation is the same, and where that evidence is: The existing top-k IDs and FP32 weights feed the fused expert GEMMs, SiLU and combine; rounding changes are disclosed in the inventory. Argument and limits: qwen/EXPERIMENTS.md#megamoe; configuration and source: qwen/experiments/38664373784/source/.  
Speed-up over our Basic: 1.0023x (6344.96 / 6330.21); historical whole-configuration ratio.  
Job id: 23571; Actions 38664373784  
Script (path in this shared folder): qwen/experiments/38664373784/source/submit.sbatch  
Log (path in this shared folder): qwen/experiments/38664373784/logs/slurm-accounting.txt; raw metrics: qwen/experiments/38664373784/profile_export_aiperf.json

Qwen experiment: vLLM side track (slide 53)  
What changed, one sentence: Used native DeepGEMM for Prefill MoE with E8M0 weight-scale re-rounding disabled.  
Why the computation is the same, and where that evidence is: The published FP8 weights and FP32 scales are retained; kernel execution and summation order differ. Argument and limits: qwen/EXPERIMENTS.md#vllm; configuration and source: qwen/experiments/38485732174/source/.  
Speed-up over our Basic: 0.8333x (5274.86 / 6330.21); historical whole-configuration ratio.  
Job id: 23234; Actions 38485732174  
Script (path in this shared folder): qwen/experiments/38485732174/source/submit.sbatch  
Log (path in this shared folder): qwen/experiments/38485732174/logs/slurm-accounting.txt; raw metrics: qwen/experiments/38485732174/profile_export_aiperf.json

Qwen experiment: Iteration 1: Decode MoE (slide 55)  
What changed, one sentence: Parallelised token conversion across four GPU threads, with Decode kernel and Prefill expert-placement changes.  
Why the computation is the same, and where that evidence is: FP8 operands/scales and selected expert operations are retained; each patch has a named numerical control. Argument and limits: qwen/EXPERIMENTS.md#iteration-1; configuration and source: qwen/experiments/39070273796/source/.  
Speed-up over our Basic: 1.1456x (7251.58 / 6330.21); historical whole-configuration ratio.  
Job id: 24414; Actions 39070273796  
Script (path in this shared folder): qwen/experiments/39070273796/source/submit.sbatch  
Log (path in this shared folder): qwen/experiments/39070273796/logs/slurm-accounting.txt; raw metrics: qwen/experiments/39070273796/profile_export_aiperf.json

Qwen experiment: Iteration 2: Decode admission (slide 56)  
What changed, one sentence: Prepared new request inputs while the GPU verified tokens, together with kernel fusions and cap 128.  
Why the computation is the same, and where that evidence is: Input dependencies are respected before consumption; target verification is retained and fusion rounding is disclosed. Argument and limits: qwen/EXPERIMENTS.md#iteration-2; configuration and source: qwen/experiments/39114949128/source/.  
Speed-up over our Basic: 1.2637x (7999.37 / 6330.21); historical whole-configuration ratio.  
Job id: 24543; Actions 39114949128  
Script (path in this shared folder): qwen/experiments/39114949128/source/submit.sbatch  
Log (path in this shared folder): qwen/experiments/39114949128/logs/slurm-accounting.txt; raw metrics: qwen/experiments/39114949128/profile_export_aiperf.json

Qwen experiment: Iteration 3: Prefill KV and MoE (slide 57)  
What changed, one sentence: Read token-ordered KV for Prefill attention while preserving paged KV for Decode, alongside MoE/input changes.  
Why the computation is the same, and where that evidence is: The same KV values are used; attention reduction, FP8 rounding and per-128 FC2 scaling are disclosed in the inventory. Argument and limits: qwen/EXPERIMENTS.md#iteration-3; configuration and source: qwen/experiments/39447708936/source/.  
Speed-up over our Basic: 1.3200x (8356.18 / 6330.21); historical whole-configuration ratio.  
Job id: 25062; Actions 39447708936  
Script (path in this shared folder): qwen/experiments/39447708936/source/submit.sbatch  
Log (path in this shared folder): qwen/experiments/39447708936/logs/slurm-accounting.txt; raw metrics: qwen/experiments/39447708936/profile_export_aiperf.json

Qwen experiment: Iteration 4: Decode batches (slide 58)  
What changed, one sentence: Extended fast-kernel coverage beyond 128 verification rows and tuned attention/GEMM while raising Decode capacity.  
Why the computation is the same, and where that evidence is: The target verifies the same candidates; tiling changes execution order without pruning model work. Argument and limits: qwen/EXPERIMENTS.md#iteration-4; configuration and source: qwen/experiments/40201493911/source/.  
Speed-up over our Basic: 1.4275x (9036.12 / 6330.21); historical whole-configuration ratio.  
Job id: 26169; Actions 40201493911  
Script (path in this shared folder): qwen/experiments/40201493911/source/submit.sbatch  
Log (path in this shared folder): qwen/experiments/40201493911/logs/slurm-accounting.txt; raw metrics: qwen/experiments/40201493911/profile_export_aiperf.json

Qwen experiment: Iteration 5: Prefill host waits (slide 59–60)  
What changed, one sentence: Removed host waits, loaded the draft as FP8, aligned Prefill starts and selected c215.  
Why the computation is the same, and where that evidence is: The target checkpoint is unchanged and verifies every draft token; FP8 draft loading does not requantise target weights. Argument and limits: qwen/EXPERIMENTS.md#iteration-5; configuration and source: qwen/experiments/40630160781/source/.  
Speed-up over our Basic: 1.4589x (9235.28 / 6330.21); historical whole-configuration ratio.  
Job id: 26889; Actions 40630160781  
Script (path in this shared folder): qwen/experiments/40630160781/source/submit.sbatch  
Log (path in this shared folder): qwen/experiments/40630160781/logs/slurm-accounting.txt; raw metrics: qwen/experiments/40630160781/profile_export_aiperf.json

Qwen experiment: Iteration 6: PP4 Prefill (slide 61)  
What changed, one sentence: Moved Prefill to four layer stages with copy-engine transport and runtime changes.  
Why the computation is the same, and where that evidence is: All 94 layers remain, split 24/24/24/22; same-dtype peer copies replace proxy transfers with ordering and content checks. Argument and limits: qwen/EXPERIMENTS.md#iteration-6; configuration and source: qwen/experiments/40964714831/source/.  
Speed-up over our Basic: 1.5273x (9668.27 / 6330.21); historical whole-configuration ratio.  
Job id: 27422; Actions 40964714831  
Script (path in this shared folder): qwen/experiments/40964714831/source/submit.sbatch  
Log (path in this shared folder): qwen/experiments/40964714831/logs/slurm-accounting.txt; raw metrics: qwen/experiments/40964714831/profile_export_aiperf.json

Qwen experiment: Iteration 7: Prefill MoE rescale (slide 62)  
What changed, one sentence: Overlapped scale correction with matrix multiplication using split accumulators, alongside other selected kernels.  
Why the computation is the same, and where that evidence is: In-job endurance checks required final BF16 outputs to match their control on the tested cases. Argument and limits: qwen/EXPERIMENTS.md#iteration-7; configuration and source: qwen/experiments/41005872470/source/.  
Speed-up over our Basic: 1.5533x (9832.74 / 6330.21); historical whole-configuration ratio.  
Job id: 27502; Actions 41005872470  
Script (path in this shared folder): qwen/experiments/41005872470/source/submit.sbatch  
Log (path in this shared folder): qwen/experiments/41005872470/logs/slurm-accounting.txt; raw metrics: qwen/experiments/41005872470/profile_export_aiperf.json

Qwen experiment: Iteration 8 and submitted final (slide 63–64)  
What changed, one sentence: Reused prepacked V in Prefill FA3 and retained later PP4, greedy-verify and FC1 updates in round 34.  
Why the computation is the same, and where that evidence is: Producer-byte and FP32-reference checks cover prepacking; FC1 first-K passed byte equality on 12 GPUs. Other rounding changes are disclosed separately. Argument and limits: qwen/EXPERIMENTS.md#iteration-8; configuration and source: qwen/experiments/41767886175/source/.  
Speed-up over our Basic: 1.5883x (10054.49 / 6330.21); historical whole-configuration ratio.  
Job id: 28660; Actions 41767886175  
Script (path in this shared folder): qwen/experiments/41767886175/source/submit.sbatch  
Log (path in this shared folder): qwen/results/41767886175/logs/slurm-accounting.txt; raw metrics: qwen/results/41767886175/profile_export_aiperf.json

Highlights (Qwen)
- Submitted experimental result: 10054.49 good output tok/s, 98.22% good requests, 1.59x the Basic mean on the same 16 H200s; slide 64; Slurm 28660, qwen/results/41767886175/.
- Unmodified Basic: repeated c160 runs averaged 6330.21; c192 failed the SLA; slide 66; Slurm 27682 / 27709, qwen/basic/results/.
- PP4 Prefill retains all 94 layers while reducing expert communication between GPUs; slide 61; qwen/EXPERIMENTS.md#iteration-6 and qwen/patches/README.md.
- V prepacking removes repeated layout work, with producer-byte and FP32-reference checks on 12 Prefill H200s; slide 63; qwen/results/41767886175/prefill-fa3-prepack/selection.json.
- Package integration was checked at 10038.11, with eight token-logprob probe responses; this is not a full-set accuracy claim; slide 81 has historical package labels, clarified in qwen/EXPERIMENTS.md; Slurm 28673, qwen/results/41781374312/.
