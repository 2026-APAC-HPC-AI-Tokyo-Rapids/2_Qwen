# Firmus profile run 16928

This directory contains the telemetry recovered from Slurm job 16928 (GitHub
Actions run 34693236779, commit `f97cdc8`). The run used the exploratory
concurrency-16 configuration and the InfiniBand-only HCA selection.

The Slurm allocation reached its 17-minute partition limit while 14 of 16
warm-up requests had completed. It therefore produced no benchmark score.
These CSV files are retained as diagnostic data only and must not be compared
as a normal benchmark result.

The recovered samples cover approximately 2026-09-12 13:14:23--13:31:30 UTC.
GPU rows have no header and contain, in order, the external timestamp, GPU
index, UUID, `nvidia-smi` timestamp, GPU utilization (%), memory utilization
(%), memory used (MiB), total memory (MiB), power (W), temperature (C), SM
clock (MHz), and memory clock (MHz).

## Aggregate observations

| Metric | Node 0 | Node 1 |
|---|---:|---:|
| GPU rows (samples per GPU) | 5,808 (726) | 6,256 (782) |
| Mean / maximum GPU utilization | 5.681% / 100% | 5.025% / 100% |
| Maximum GPU memory used | 134,628 MiB | 131,236 MiB |
| Mean / maximum GPU power | 132.661 / 611.90 W | 175.764 / 560.09 W |
| Maximum GPU temperature | 64 C | 73 C |
| Mean CPU user / system / idle | 6.588% / 6.323% / 85.900% | 6.036% / 3.848% / 88.913% |
| Mean CPU I/O wait | 0.089% | 0.035% |

The InfiniBand counters changed by only tens of kilobytes per HCA during the
recovered interval. The values in the CSV are cumulative hardware counters;
the raw magnitudes are not per-run traffic. The very small deltas, low average
GPU utilization, and incomplete warm-up are consistent with the observed
NIXL transfer stall, while this single timed-out profile is not sufficient for
a comparative performance conclusion.
