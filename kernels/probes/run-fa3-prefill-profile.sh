#!/usr/bin/env bash
# The runtime supplies the cluster-specific container launcher.
set -euo pipefail
: "${RUNTIME_DIR:?}"
# Software opcode counters + hardware pipe counters allow F2FP accounting against MUFU.
metrics=sm__inst_executed_pipe_xu.sum,sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor_op_gmma.sum,sm__cycles_elapsed.avg.per_second,sass__inst_executed_per_opcode,sass__inst_executed_per_opcode_with_modifier_all,sass__inst_executed_per_opcode_category,smsp__sass_thread_inst_executed_op_conversion_pred_on.sum,smsp__thread_inst_executed_pipe_xu_pred_on.sum,smsp__thread_inst_executed_pipe_alu_pred_on.sum,smsp__thread_inst_executed_pipe_fma_pred_on.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed
# SourceCounters supplies per-PC stall samples and execution counts; memory tables supply shared throughput.
bash "${RUNTIME_DIR}/scripts/run-container.sh" /opt/fa3-ncu/ncu --kernel-name-base demangled --kernel-name 'regex:flash.*FlashAttnFwdSm90' --launch-skip 3 --launch-count 1 --cache-control none --clock-control none --section SourceCounters --section InstructionStats --section MemoryWorkloadAnalysis_Tables --metrics "${metrics}" --export "${RUNTIME_DIR}/results/fa3-prefill-m2" --page source --print-source sass --csv /opt/sglang/venv/bin/python "${RUNTIME_DIR}/source/bench.py" profile >"${RUNTIME_DIR}/results/fa3-prefill-m2-sass.csv" 2>&1 || { status=$?; sed '/^PASS kernel=fa3_prefill_m2 /d' "${RUNTIME_DIR}/results/fa3-prefill-m2-sass.csv" >&2; exit "${status}"; }
ncu --import "${RUNTIME_DIR}/results/fa3-prefill-m2.ncu-rep" --page raw --print-metric-instances details --csv >"${RUNTIME_DIR}/results/fa3-prefill-m2-raw.csv"
test -s "${RUNTIME_DIR}/results/fa3-prefill-m2.ncu-rep"
grep -q '^PASS kernel=fa3_prefill_m2 ' "${RUNTIME_DIR}/results/fa3-prefill-m2-sass.csv"
# The existing workflow deletes the run directory after checking it: retain all
# textual counter results in its Actions/PJM logs as well as results/.
sed '/^PASS kernel=fa3_prefill_m2 /d' "${RUNTIME_DIR}/results/fa3-prefill-m2-sass.csv"
cat "${RUNTIME_DIR}/results/fa3-prefill-m2-raw.csv"
printf 'PASS kernel=fa3_prefill_m2 report=fa3-prefill-m2.ncu-rep raw=fa3-prefill-m2-raw.csv sass=fa3-prefill-m2-sass.csv\n'
