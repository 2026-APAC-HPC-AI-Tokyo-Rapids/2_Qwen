# First full-length Firmus benchmark

The earliest completed official-duration benchmark retained in the execution history is Actions [34725941535](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/34725941535), Slurm **17003**, Qwen commit `148c97c`, on 13 September 2026 JST. It completed the 60/120/30 s AIPerf workload at c16: 555 requests, zero errors, good request fraction 1.0, goodput 4.511811222461859 req/s, or **902.36 good output tok/s**.

The job ended FAILED 1:0 because the team's then-current length validator rejected the observed input average 3986.001801801802. The preserved Actions excerpt shows both the completed measurement and that rejection. This is a historical measurement, not a claim of a successful job or organiser acceptance. The original raw export is not included here. Earlier profile/exploratory runs used shorter windows or never finished warmup and are not used as the first full-length result.

The first subsequent run that completed successfully and passed local validation is **Actions 34732532321 / Slurm 17021**, c16, **909.68 good output tok/s**, COMPLETED 0:0 in 11:16. Its raw export, score, configuration, provenance, scheduler log and historical script are in [../34732532321/](../34732532321/). The organiser reference of 1832 is a separate reference configuration, not our first run.

`actions-excerpt.log` copies matching lines verbatim from the Actions log of 34725941535. The benchmark executable path was `/opt/sglang/venv/bin/aiperf`; its fixed lengths, seed, SLA and duration are visible in the command. The failed local validation is not relabelled as a successful E2E execution.
