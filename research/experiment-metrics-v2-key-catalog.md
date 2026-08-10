# Qwen / SGLang experiment metrics v2 キーカタログ

## この文書の読み方

この文書は、2026-08-10 時点の v2 plan、observation map、contract schemas にある
収集予定値を、v1 のカテゴリ別 dotted key 規則へ揃えた一覧である。v2 の JSON artifact、
raw relation、consumer column の名前が異なる箇所では、検索・比較用の「キー」を標準名とし、
元 field/table を「収集方法」に記す。

- `<phase>`、`<node>`、`<rank>`、`<gpu>`、`<name>` は実値へ置換する。
- `{a,b}` は同じ prefix の全 leaf を列挙した省略表記である。
- `[]` は順序を保つ long relation または配列要素であり、scalar へ潰さない。
- 「raw」は producer/collector が観測する値、「派生」は view/reducer で算出する値である。
- `schema_version`、ID、path、argv、hash などの lineage/control field は最後に分ける。
- v2 master docs を内容の正本とし、v1 文書はキー階層の命名だけに参照する。
- 実行設定をまとめるID/objectは作らない。`nodes`、`mpi_processes`、`gpus_per_node`、TP、resource、transport、
  port、timeoutは独立keyで保持し、スプレッドシートでも一key一列にする。

## 1. Run、source、artifact、cache

| キー | 意味 | 収集方法（コマンド／関数） | 適用・種別 |
|---|---|---|---|
| `run.identity.{run_id,run_purpose,repository_id}` | run ID、用途、repository | runner が Git 管理の run manifest を読み `identity_projection/v1` で投影 | bench/profile、raw |
| `run.identity.instrumentation_bundle_sha256` | 適用 instrumentation bundle identity | run manifest と bundle manifest を照合 | bench/profile、raw |
| `run.{started_at,finished_at}` | run の開始・終了 UTC 時刻 | supervisor wall clock | 全 variant、raw |
| `run.elapsed_sec` | supervisor 範囲の経過秒 | monotonic clock の開始・終了差 | 全 variant、raw |
| `run.exit_code` | run 終了コード | supervisor の pipeline outcome | 全 variant、raw |
| `run.outcome` | `succeeded` / `failed` | `run.exit_code == 0` から view 導出 | 全 variant、派生。raw 保存しない |
| `run.stage.<name>.launch.{argv,cwd,stderr_mode}` | stage の実起動条件 | `StageRunner` が shell を介さず spawn する直前に記録 | 対象 stage、raw |
| `run.stage.<name>.termination.{kind,code,signal}` | stage の `exit` / `signal` / `spawn_error` | `StageRunner` の spawn/wait closed union。非該当 code/signal は absent | 対象 stage、raw |
| `perf.stage.<name>.wall_time_sec` | stage 外側 wall time | spawn 直前から child wait 完了までを monotonic 計時 | 対象 stage、raw |
| `collection_complete` | 必要なproducer/fileの完全性 | finalizer が実装上のexpected setとobserved setを比較 | bench/profile、raw bool。理由別fieldは作らない |
| `artifact.sif_manifest_sha256` | SIF provenance manifest identity | SIF sidecar と参照 raw file を canonical hash | 全 variant、raw lineage |
| `artifact.model.manifest_sha256` | model member manifest identity | local model の exact member manifest を JCS SHA-256 | bench/profile、raw |
| `artifact.dataset.manifest_sha256` | dataset member manifest identity | workload source dataset manifest を JCS SHA-256 | bench/profile、raw |
| `artifact.instrumentation_bundle_sha256` | instrumentation bundle identity | packager が bundle manifest を SHA-256 | bench/profile、raw lineage |
| `cache.node.<node>.input_manifest_sha256` | staging 後 node cache 内容の semantic identity | node wrapper が relative path byte 順に file/symlink member manifest を作り、subject binding を除いて JCS SHA-256 | bench/profile、raw lineage |
| `source.track.{head_commit,score_reference_commit,profile_base_commit,manifest_sha256,source_sha256}` | source track の Git provenance、canonical record hash、包含 manifest lineage hash | subject manifest の role 付き commit を Git object と照合し projection | variant 依存、raw projection。2つの hash を統合しない |
| `source.sglang.commit` | SIF に含まれる SGLang source commit | SIF sidecar の role=`sglang_source` commit を Git object と照合して projection | 全 run、raw provenance |

`<name>` の stage 集合は少なくとも node launch、engine、benchmark、artifact finalize を含み、
profile では warmup/measured generation、GPU sampler、profile hardware、profiler capture を追加する。
generation成功後は`network_imb_mpi1`、`network_nccl_all_reduce`、`network_nccl_all_gather`、
`network_nccl_reduce_scatter`を追加し、それぞれ既存のlaunch/termination/wall-time leafを持つ。

## 2. Scheduler と environment

| キー | 意味 | 収集方法（コマンド／関数） | 適用・種別 |
|---|---|---|---|
| `scheduler.request.{nodes,mpi_processes,gpus_per_node,tensor_parallel_size}` | PJM/node/GPU/TP の個別要求値 | submission runner が実行設定の各fieldと実 `pjsub` argvを記録・照合 | bench/profile、raw。各leafは独立列 |
| `scheduler.request.{resource_group,transport_policy}` | resource group と通信policyの個別要求値 | 同 submission request | 同上、raw。各leafは独立列 |
| `scheduler.request.{advertise_source,sglang_port,node_agent_port,rendezvous_timeout_sec,ready_timeout_sec}` | address、port、timeoutの個別設定値 | 実行設定からrequestへsealして照合 | 同上、raw。各leafは独立列 |
| `scheduler.request.subject_manifest_sha256` | submit 対象 manifest | submit 前に file bytes を SHA-256 | 同上、raw lineage |
| `scheduler.request.pjm_script.{repository_path,sha256}` | 使用 PJM script と hash | repository path と file hash | 同上、raw lineage |
| `scheduler.request.argv` | 実際の submit argv | submission runner の argv 配列 | 同上、raw control field |
| `scheduler.submit_job_id` | `pjsub` が返した job ID | submission result parser | 同上、raw。欠損時 key absent |
| `scheduler.job_entry_job_id` | job entry で観測した job ID | `PJM_JOBID` / `PJM_JOB_ID` snapshot | 同上、raw。欠損時 key absent |
| `scheduler.accounting.{jid,st,sdt}` | PJM job ID、terminal state、開始時刻 raw 語彙 | `pjstat --history -S --choose jid,st,sdt,elp,ec,sn,pc,nnumu,mmszu,uctmut,sctmut --data --delimiter "\t" <job-id>` | 同上、raw |
| `scheduler.accounting.elp.{lexeme,seconds}` | elapsed raw/秒 | fixed-column parser | 同上、raw nullable numeric |
| `scheduler.accounting.{ec,sn,pc,nnumu}.{lexeme,value}` | exit、signal/status、PJM code、node 数 | 同 parser | 同上、raw nullable numeric |
| `scheduler.accounting.mmszu.{lexeme,bytes}` | 最大 RSS | 同 parser、括弧内 bytes を parse | 同上、raw nullable numeric |
| `scheduler.accounting.{uctmut,sctmut}.{lexeme,seconds}` | user/system CPU time | 同 parser、millisecond を秒へ変換 | 同上、raw nullable numeric |
| `env.job_entry.<escaped_name>` | job-entry 環境 | name/value を byte-exact snapshot、name byte 順、value は Base64 | bench/profile、raw dynamic family |
| `env.host.node.<node>.<escaped_name>` | host executor node 環境 | node wrapper の full snapshot | 全 run、raw dynamic family |
| `sglang.env.node.<node>.<escaped_name>` | SGLang launch 時 node 環境 | SGLang launch 直前の full snapshot | bench/profile、raw dynamic family |
| `sglang.worker.env.rank.<rank>.<escaped_name>` | worker global rank の実効環境 | `worker_ready` producer が rank scope snapshot | bench/profile、raw dynamic family |

## 3. SGLang configuration、worker placement、process lifecycle

| キー | 意味 | 収集方法（コマンド／関数） | 適用・種別 |
|---|---|---|---|
| `sglang.node.<node>.bench_args_effective.<field>` | node で実効になった `BenchArgs` の全 dataclass field | CLI parse 直後、`Engine()` 前の `bench_args_effective` domain event。JCS field 名 byte 順で lossless projection | bench/profile、raw dynamic family |
| `sglang.node.<node>.server_args_effective.<field>` | launch-effective `ServerArgs` の全 dataclass field | auto parser、`_set_envs_and_config`、`check_server_args` 後、scheduler/daemon launch 前の `server_args_effective` event | 同上、raw dynamic family |
| `parallel.worker.<rank>.{tp_rank,gpu_id,gpu_uuid}` | global TP rank と GPU binding | worker process の `worker_ready` event | bench/profile、raw |
| `parallel.worker.<rank>.{hostname,node_rank,local_rank}` | worker の node/rank placement | 同 event と node wrapper context | 同上、raw |
| `parallel.worker.<rank>.cpu_affinity` | worker CPU affinity | worker 内 `os.sched_getaffinity(0)` | 同上、raw ordered set |
| `parallel.worker.<rank>.numa_mems` | worker の NUMA memory policy | `/proc/self/status` の `Mems_allowed_list` | 同上、raw nullable |
| `parallel.worker.<rank>.runtime.{prefill_attention_backend,decode_attention_backend,torch_distributed_backend,transport_backend}` | worker が実際に使った attention/distributed/transport backend | `worker_ready` payload。ServerArgs から推測しない | 同上、raw |
| `parallel.node.<node>.{global_rank,local_rank,size,hostname}` | outer MPI node launch placement | `node-topology/v1` の outer MPI context | 全 variant、raw |
| `parallel.node.<node>.{node_wrapper_argv,singularity_argv,sglang_argv}` | node wrapper、container、SGLang の実 argv | node wrapper が launch 直前に配列で記録 | 全 variant、raw control/configuration |
| `parallel.node.<node>.{cpu_affinity,numa_mems}` | node wrapper の CPU/NUMA binding | `os.sched_getaffinity(0)` と `/proc/self/status` | 全 variant、raw |
| `parallel.node.<node>.process.clock_id` | lifecycle timestamp の clock identity | boot ID + `CLOCK_MONOTONIC` | 全 variant、raw |
| `parallel.node.<node>.process.{spawn_attempt_ns,child_started_ns,child_finished_ns}` | child spawn 試行、開始、終了時刻 | `node_agent.py` の Python supervisor | 全 variant、raw。spawn error 時の child 時刻は null |
| `parallel.node.<node>.process.termination.kind` | `waited` / `spawn_error` | Python supervisor の spawn と `os.waitpid(child_pid, 0)` | 全 variant、raw |
| `parallel.node.<node>.process.termination.raw_wait_status` | POSIX raw wait status | `os.waitpid()` | `kind=waited` のみ、raw |
| `lifecycle.rendezvous.{rank0_hostname,address,port,created_at}` | node-agent control rendezvous | rank 0 が `control/lifecycle/rendezvous.json` を生成 | bench/profile、raw control/configuration |
| `lifecycle.node.<node>.ready.{sequence,clock_id,received_ns,ack_ns}` | coordinator が ready を受信・ACK した順序/時刻 | node-agent rendezvous summary | 全 variant、raw |
| `lifecycle.node.<node>.shutdown.{sequence,invocation_id,clock_id,received_ns,ack_ns}` | shutdown 対象 generation、順序、時刻 | control stream と lifecycle summary | 全 variant、raw nullable |
| `lifecycle.node.<node>.termination.status` | `spawn_error` / `early_exit` / `early_signal` / `coordinated_exit` / `coordinated_signal` | raw wait status、child finish、shutdown receive を reducer で比較 | participant node long relation、派生 |

## 4. Workload、result、timing

| キー | 意味 | 収集方法（コマンド／関数） | 適用・種別 |
|---|---|---|---|
| `workload.<kind>.request.<ordinal>.source` | request の dataset row または generated source | selection 時に ordered workload manifest へ保存 | `kind=warmup|measured`、raw |
| `workload.<kind>.request.<ordinal>.prompt.text` | 実際に generate へ渡す text prompt | ordered workload writer | text prompt のみ、artifact raw。BigQuery scalar metricへ本文は出さない |
| `workload.<kind>.request.<ordinal>.prompt.token_ids` | 実際に渡す token ID 配列 | ordered workload writer | token-id prompt のみ、artifact raw |
| `workload.<kind>.request.<ordinal>.input_token_count` | text prompt の実測 input token 数 | request 選択時に同 tokenizer で encode | text prompt のみ、raw |
| `workload.<kind>.request.<ordinal>.requested_output_token_count` | 要求 output token 数 | generate input object | warmup/measured、raw |
| `workload.<kind>.request.<ordinal>.sampling_param.<name>` | 実 sampling parameter | generate へ渡す mapping を field 名 byte 順で保存 | warmup/measured、raw dynamic family |
| `result.request.returned_count` | validate 済み正常 response 数 | primary result writer が response array を decode/検証 | bench/profile、raw |
| `result.request.error_count` | parse/validation error response 数 | 同 writer。parse 不能も error に加算 | 全 result、raw |
| `result.token.input.total` | 実測 input token 総数 | result artifact の `observed_input_tokens` を projection | 確定可能時のみ、raw nullable |
| `result.token.output.total` | validate 済み completion token 総数 | response の `completion_tokens` を集計。token-ID identity の可否とは独立 | 確定可能時、raw nullable |
| `result.token.output.ordered_token_ids_sha256` | response 順を含む output token ID identity | ordered token ID array を JCS SHA-256 | 確定可能時のみ、raw lineage/validity |
| `perf.benchmark.generation.wall_time_sec` | measured generation wall time | generate call の start/end 境界を result writer が `generation_wall_sec` として保存 | bench/profile、raw |
| `perf.phase.{distributed_engine_init,input_prepare,warmup_generation,engine_shutdown}.wall_time_sec` | init、input preparation、warmup、shutdown の wall time | supervisor/rank 0 の monotonic phase boundary | 全 variant、raw scalar。profile は timestamp pair と equality 検証 |
| `perf.throughput.request_per_sec` | request throughput | `returned_count / generation_wall_time_sec` | query-time 派生、raw 保存しない |
| `perf.throughput.input_token_per_sec` | input token throughput | `result.token.input.total / generation_wall_time_sec` | query-time 派生、raw 保存しない |
| `perf.throughput.output_token_per_sec` | output token throughput | `result.token.output.total / generation_wall_time_sec` | query-time 派生、raw 保存しない |
| `perf.throughput.total_token_throughput` | total token throughput | `(input + output tokens) / generation_wall_time_sec` | query-time派生 |
| `profile.phase.<invocation_id>.rank0.{clock_id,start_broadcast_ns,all_start_acks_received_ns,generation_start_ns,generation_end_ns,end_broadcast_ns,all_end_acks_received_ns}` | rank 0 の phase control/generation 境界 | phase controller summary | profile-only、raw diagnostics |
| `profile.phase.<invocation_id>.node.<node>.{clock_id,start_sequence,end_sequence,start_received_ns,start_ack_ns,end_received_ns,end_ack_ns}` | node GPU window、sequence、ACK 境界 | node control stream と phase summary | profile-only、raw diagnostics |

## 5. MPI・NCCL network metrics

| キー | 意味 | 収集方法（コマンド／関数） | 適用・種別 |
|---|---|---|---|
| `network.imb_mpi1.process_count` | IMB MPI process数 | generationと同じ値をargv `-npmin`とsectionで照合 | bench/profile、raw scalar |
| `network.imb_mpi1.message_size.{min_bytes,max_bytes}` | IMB message範囲 | 0、4194304を独立leafへ保存 | bench/profile、raw scalar |
| `network.imb_mpi1.iteration.{max_per_sample,time_limit_sec,policy}` | IMB iteration設定 | argvの1000、1、`dynamic`を独立leafへ保存 | bench/profile、raw scalar |
| `software.imb_mpi1.version` | IMB-MPI1 version | stdout version marker | bench/profile、raw scalar nullable |
| `network.nccl_tests.mpi_process_count` | nccl-tests MPI process数 | generationと同じ値 | bench/profile、raw scalar |
| `network.nccl_tests.gpus_per_process` | process当たりGPU数 | argv `-g` | bench/profile、raw scalar |
| `network.nccl_tests.total_gpu_count` | communicator総GPU数 | MPI process数×GPU/processを実launchと照合 | bench/profile、raw scalar |
| `network.nccl_tests.message_size.{min_bytes,max_bytes,factor}` | NCCL message範囲/倍率 | `-b 8 -e 1G -f 2` | bench/profile、raw scalar |
| `network.nccl_tests.{warmup_iterations,timed_iterations,check_iterations,rank_average_mode}` | warmup/timed/check/rank average設定 | `-w 5 -n 20 -c 1 -a 1` | bench/profile、raw scalar |
| `software.nccl_tests.commit` | nccl-tests source commit | SIF sidecarの固定値`717b68318278e93f371d8ffb46b076069d7c7851` | bench/profile、raw provenance |
| `software.nccl.version` | 実NCCL library version | runtime version query | bench/profile、raw scalar |

`mpi_network_results_v2`は`(run_id,benchmark,process_count,message_bytes)` grainで
`iterations,time_min_us,time_avg_us,time_max_us,bandwidth_mbytes_per_sec,source_member`を持つ。
`nccl_network_results_v2`は`(run_id,collective,message_bytes,buffer_mode)` grainで
`element_count,data_type,reduction_op,mpi_process_count,gpu_count,time_us,algorithm_bandwidth_gbps,
bus_bandwidth_gbps,error_count,source_member`を持つ。実在rowだけを返し、summary keyは作らない。

## 6. CPU、memory、GPU topology

| キー | 意味 | 収集方法（コマンド／関数） | 適用・種別 |
|---|---|---|---|
| `hardware.node.<node>.hostname` | node hostname | minimal hardware collector の `gethostname()` | 全 variant、raw |
| `hardware.node.<node>.cpu.{architecture,model_name,vendor_id,cpu_family,model,stepping}` | CPU identity | `lscpu_json` を projection | 全 variant、派生 nullable |
| `hardware.node.<node>.cpu.{physical_cores,numa_nodes}` | physical core/NUMA node 数 | `lscpu_json` から導出 | 全 variant、派生 nullable |
| `hardware.node.<node>.kernel.release` | kernel release | `/proc/sys/kernel/osrelease` | 全 variant、raw nullable |
| `hardware.node.<node>.cpu.{online,isolated,nohz_full}` | CPU set | `/sys/devices/system/cpu/{online,isolated,nohz_full}` | 全 variant、raw nullable |
| `hardware.node.<node>.memory.{total_bytes,hugepage_size_bytes,hugepages_total}` | node memory/hugepage | `/proc/meminfo` | 全 variant、raw nullable |
| `hardware.node.<node>.numa_balancing_enabled` | automatic NUMA balancing | `/proc/sys/kernel/numa_balancing` | 全 variant、raw nullable |
| `hardware.node.<node>.cpu_boost.{source_path,raw,enabled}` | CPU boost | cpufreq `boost`、fallback `intel_pstate/no_turbo` | 全 variant、raw nullable |
| `hardware.node.<node>.cpufreq_policy.<id>.{related_cpus,driver,governor,min_hz,max_hz}` | cpufreq policy | `/sys/devices/system/cpu/cpufreq/policy*`、kHz→Hz | 全 variant、raw nullable |
| `hardware.node.<node>.gpu.<gpu>.{index,name,uuid,pci_bus_id}` | GPU identity | host/container の `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` | 全 variant、raw |
| `hardware.node.<node>.gpu.<gpu>.memory.total_bytes` | GPU memory capacity | 同 query の memory.total を bytes へ変換 | 全 variant、raw nullable |
| `hardware.node.<node>.gpu.<gpu>.driver_version` | NVIDIA driver version | 同 query | 全 variant、raw nullable |
| `hardware.node.<node>.gpu.<gpu>.{persistence_mode,compute_mode,power_limit_watt}` | persistence、compute mode、power limit | expanded `nvidia-smi` query | 全 variant、raw nullable |
| `hardware.node.<node>.gpu.<gpu>.clock.{sm_max_mhz,memory_max_mhz}` | 最大 SM/memory clock | expanded `nvidia-smi` query | 全 variant、raw nullable |
| `hardware.node.<node>.gpu.<gpu>.{mig_mode,ecc_mode}` | MIG/ECC mode | expanded `nvidia-smi` query | 全 variant、raw nullable |
| `hardware.node.<node>.gpu.<gpu>.scope` | host/container の観測 scope | topology producer context | 全 variant、raw enum |
| `hardware.node.<node>.gpu_topology` | GPU/NIC/link topology | host `nvidia-smi topo -m` stdout を parser で構造化 | 全 variant、raw topology relation |

## 7. Profile GPU samples と集約

| キー | 意味 | 収集方法（コマンド／関数） | 適用・種別 |
|---|---|---|---|
| `profile.gpu.node.<node>.sample.<sequence>.monotonic_ns` | sample monotonic 時刻 | node-local GPU sampler が complete frame ごとに JSONL append | profile-only、raw |
| `profile.gpu.node.<node>.sample.<sequence>.wall_time_utc` | sample UTC 時刻 | 同 sampler の wall clock | profile-only、raw |
| `profile.gpu.node.<node>.sample.<sequence>.clock_id` | sample の monotonic clock identity | Linux boot ID + `CLOCK_MONOTONIC` | profile-only、raw。phase node window と一致必須 |
| `profile.gpu.node.<node>.sample.<sequence>.gpu.<gpu>.{uuid,logical_index}` | sample と topology の GPU binding | 同 sampler、UUID で topology join | profile-only、raw |
| `profile.gpu.node.<node>.sample.<sequence>.gpu.<gpu>.utilization_percent` | GPU utilization | `nvidia-smi` polling | profile-only、raw nullable |
| `profile.gpu.node.<node>.sample.<sequence>.gpu.<gpu>.memory.used_bytes` | GPU memory 使用量 | 同 polling、bytes 変換 | profile-only、raw nullable |
| `profile.gpu.node.<node>.sample.<sequence>.gpu.<gpu>.power_watt` | GPU power | 同 polling | profile-only、raw nullable |
| `profile.gpu.node.<node>.sample.<sequence>.gpu.<gpu>.temperature_celsius` | GPU temperature | 同 polling | profile-only、raw nullable |
| `profile.gpu.node.<node>.sample.<sequence>.gpu.<gpu>.clock.sm_mhz` | SM clock | 同 polling | profile-only、raw nullable |
| `resource.phase.measured_generation.node.<node>.gpu.<gpu>.duration_sec` | measured generation window 長 | `gpu-phase/v1` reducer が phase summary と sample interval の交差を計算 | profile-only、派生 |
| `resource.phase.measured_generation.node.<node>.gpu.<gpu>.sample.valid_count` | valid complete frame 数 | dedupe 済み raw sample reducer | profile-only、派生 |
| `resource.phase.measured_generation.node.<node>.gpu.<gpu>.utilization.{valid_duration_sec,coverage_ratio,mean_percent,p95_percent}` | utilization の有効時間、coverage、duration-weighted mean、nearest-rank p95 | `[start,end)` overlap、2 interval gap cutoff で reducer | profile-only、派生 nullable |
| `resource.phase.measured_generation.node.<node>.gpu.<gpu>.power.{valid_duration_sec,coverage_ratio,mean_watt}` | power の有効時間、coverage、平均 | overlap interval の power 積分 | profile-only、派生 nullable |
| `resource.phase.measured_generation.node.<node>.gpu.<gpu>.energy_joule` | measured generation 中の GPU energy | power sample を interval 積分。欠損区間を外挿しない | profile-only、派生 nullable |
| `resource.phase.measured_generation.node.<node>.gpu.<gpu>.memory.{valid_sample_count,max_used_bytes}` | memory 有効件数と最大値 | non-null sample の reducer | profile-only、派生 nullable |
| `resource.phase.measured_generation.node.<node>.gpu.<gpu>.temperature.{valid_sample_count,max_celsius}` | temperature 有効件数と最大 | non-null sample の reducer | profile-only、派生 nullable |
| `resource.phase.measured_generation.node.<node>.gpu.<gpu>.clock.sm.{valid_sample_count,min_mhz}` | SM clock 有効件数と最小 | non-null sample の reducer | profile-only、派生 nullable |
| `resource.phase.measured_generation.gpu.{aggregate_present_count,utilization_valid_count,power_valid_count,memory_valid_count,temperature_valid_count,clock_valid_count,energy_included_count}` | expected GPU grid 中の aggregate/各 metric 有効 GPU 数 | expected `(node,local_gpu_slot)` grid に per-GPU aggregate を left join して count | profile-only、fleet 派生 |
| `resource.phase.measured_generation.gpu.{utilization_coverage_min,power_coverage_min,valid_sample_count_min}` | fleet 最小 coverage/sample 数 | per-GPU aggregate の minimum | profile-only、fleet 派生 nullable |
| `resource.phase.measured_generation.gpu.{utilization_valid_duration_sec_min,power_valid_duration_sec_min}` | fleet 最小有効観測時間 | per-GPU valid duration の minimum | profile-only、fleet 派生 nullable |
| `resource.phase.measured_generation.gpu.fleet_utilization_mean` | fleet duration-weighted utilization 平均 | `sum(utilization_integral) / sum(valid_duration)`。GPU 平均の単純平均は禁止 | profile-only、fleet 派生 nullable |
| `resource.phase.measured_generation.gpu.utilization_mean.{min,median,max}` | per-GPU mean utilization の分布 | per-GPU aggregate の reducer | profile-only、fleet 派生 nullable |
| `resource.phase.measured_generation.gpu.utilization_p95.{min,median,max}` | per-GPU p95 utilization の分布 | per-GPU aggregate の reducer | profile-only、fleet 派生 nullable |
| `resource.phase.measured_generation.gpu.energy_joule_sum` | 観測できた fleet energy 合計 | valid power interval がある GPU だけを合計 | profile-only、fleet 派生 nullable |
| `resource.phase.measured_generation.gpu.{memory_max_bytes,temperature_max_celsius,sm_clock_min_mhz}` | fleet の memory最大、temperature最大、SM clock最小 | per-GPU aggregate の max/max/min | profile-only、fleet 派生 nullable |

## 8. Profile CPU/NUMA と profiler events

| キー | 意味 | 収集方法（コマンド／関数） | 適用・種別 |
|---|---|---|---|
| `profile.hardware.node.<node>.memory.available_bytes.postrun` | benchmark 後の MemAvailable | profile hardware capture が `/proc/meminfo` を読む | profile-only、raw nullable |
| `profile.hardware.node.<node>.numa.<id>.cpulist` | NUMA node の CPU list | `/sys/devices/system/node/node<id>/cpulist` | profile-only、raw nullable |
| `profile.hardware.node.<node>.numa.<id>.memory.{total_bytes,free_bytes}` | NUMA node memory | node `<id>/meminfo` parser | profile-only、raw nullable |
| `profile.profiler.global_rank.<rank>.<activity>.<category>.<event_name>.call_count` | global TP rank 別 profiler event 呼出回数 | rank 別 gzip Chrome trace を `(global_rank,activity,category,event_name)` で group/count | profile-only、派生 long relation |
| `profile.profiler.global_rank.<rank>.<activity>.<category>.<event_name>.inclusive_duration_sum_ns` | global TP rank 別 inclusive duration 合計 | 同 trace parser で duration を合計 | profile-only、派生 long relation |

## 9. Consumer/view でのみ作る値

| キー | 意味 | 算出方法 |
|---|---|---|
| `validation.placement_valid` | 個別実行設定/request/accounting/topology/config/worker placement が整合するか | `qwen-placement/v1` reducer。source 欠損は NULL、不一致は false |
| `validation.benchmark_valid` | workload と primary result が整合するか | request count、token identity、invocation、result schema を reducer で照合 |
| `validation.affinity_valid` | node/worker affinity と個別 affinity policy が整合するか | node launch、worker-ready、planned affinity policy を reducer で照合 |
| `eligibility.score_comparison` | bench score 比較へ入れられるか | collection、placement、benchmark validity と protocol 条件を view で評価 |
| `run.termination.benchmark.{kind,exit_code,signal}` | benchmark stage の終了分類 | `run.stage.benchmark.termination` を consumer 列へ展開 |
| `run.termination.engine.{kind,exit_code,signal}` | node 0 engine の終了分類 | raw POSIX wait status を `WIFEXITED/WEXITSTATUS/WIFSIGNALED/WTERMSIG` で一度だけ導出 |
| `resource.job.memory.max_rss_bytes` | job 最大 RSS | `scheduler.accounting.mmszu.bytes` を projection |
| `resource.job.cpu.{user_time_sec,system_time_sec}` | job user/system CPU time | `uctmut.seconds` / `sctmut.seconds` を projection |
| `scheduler.terminal.{state,exit_code}` | scheduler terminal state と exit code | scheduler terminal/accounting source を projection |
| `artifact.{sif_artifact_sha256,runtime_launcher_identity}` | SIF bytes と実 launcher の consumer identity | SIF sidecar/runtime-launcher validation | consumer projection |
| `artifact.{model_manifest_identity,dataset_manifest_identity}` | model/dataset semantic identity | parent-owned manifest projection を regular/child runへ join | consumer projection |
| `workload.{measured_manifest_identity,warmup_manifest_identity}` | measured/warmup ordered workload identity | workload manifest projection | consumer projection |
| `cache.manifest_identity` | expected node 順の cache identity 集合 | `[{node_rank,manifest_sha256},...]` を `cache_nodes/v1` で reduce。再 hash しない | consumer projection nullable |
| `workload.measured.{selected_count,planned_input_tokens,requested_output_tokens}` | measured workload の件数、予定 input/output token 総数 | ordered workload request relation を count/sum | 派生。成功 request 数とは別 |
| `workload.warmup.mode` | warmup workload の構成 mode | ordered warmup manifest の kind/request shape から projection | 派生 |
| `result.request.missing_response_count` | 予定 request のうち response がない件数 | `measured_selected_count - returned_count - error_count` を validity reducer で算出 | 派生 nullable |
| `execution.{nodes,mpi_processes,gpus_per_node,tensor_parallel_size}` | 予定したnode/MPI/GPU/TP値 | 実行設定とscheduler requestの一致済みprojection | consumer projection。各leafを独立列にする |
| `execution.{resource_group,transport_policy,advertise_source}` | resource、transport、address source | 同上 | consumer projection。各leafを独立列にする |
| `execution.{sglang_port,node_agent_port,rendezvous_timeout_sec,ready_timeout_sec}` | portとtimeout | 同上 | consumer projection。各leafを独立列にする |
| `parallel.observed.{node_count,distinct_host_count,mpi_process_count,global_rank_count}` | node launch/worker の観測 cardinality | node-launch と worker-ready relation を count | consumer projection nullable |
| `hardware.gpu.{observed_record_count,unique_uuid_count}` | GPU topology record 数と unique UUID 数 | container topology relation を count | consumer projection nullable |
| `hardware.gpu.per_node.{min,max}` | node 当たり観測 GPU 数の最小/最大 | container topology を node 単位に count | consumer projection nullable |
| `parallel.effective.{tp,nnodes}` | SGLang 実効 tensor parallel size と node 数 | node config consensus の `tp_size` / `nnodes` | consumer projection nullable |
| `sglang.config.{benchmark_args_identity,server_args_identity}` | BenchArgs/ServerArgs 全 field の node-consensus identity | config snapshot JCS identity と node 間 consensus | consumer projection nullable |
| `sglang.runtime.{backend_consensus,transport_consensus}` | worker backend/transport の consensus | worker-ready の4 backend fieldを reducer | consumer projection nullable |
| `hardware.consensus.gpu.{model,memory_bytes,driver,power_limit_watt,max_sm_clock_mhz,compute_mode,mig_mode}` | GPU hardware/settings の node/GPU consensus | topology rowsを consensus reducer。欠損・不一致は NULL | consumer projection |
| `hardware.consensus.cpu.{model,kernel_release,numa_node_count,memory_bytes,governor,boost_enabled}` | CPU/kernel/memory/settings consensus | minimal hardware rowsを consensus reducer。欠損・不一致は NULL | consumer projection |
| `profile.numa.cpu_layout_identity` | profile NUMA CPU layout identity | profile NUMA cpulist relationを canonical JSON 化 | profile-only、派生 nullable |
| `profile.hardware.memory.available_bytes.postrun.min` | node 間 post-run MemAvailable 最小値 | profile hardware node summary の minimum | profile-only、派生 nullable |
| `hardware.gpu.binding_valid` | worker GPU binding と topology の整合 | worker UUID/index、node topology、protocolの個別rank/GPU数を exact 比較 |

## 10. Lineage/control field と非収集項目

| 区分 | field／キー | 扱い |
|---|---|---|
| lineage/control | `schema_version`、`run_id`、`invocation_id`、record OID/hash、source member/path/hash、argv、stdout/stderr、bundle manifest | artifact の検証・再現・join に保持するが、科学メトリクスの leaf key とは数えない |
| 非収集 | network用`lane`、`qualification*`、`block_id`、`block_order`、`replicate_index`、`pair_id`、contract/protocol ID・hash | 各runを独立に扱い、通信設定は独立scalarへ展開する |
| 非収集 | MPI/NCCL summary、`ib_*_bw`、`ib_*_lat` | message size別long relationからquery時に導出する |
| artifact-only | prompt text、prompt token ID 配列、ordered workload request 全体 | reproducibility artifact に保持するが BigQuery の scalar metric へ本文を展開しない |
| 非収集 | `result.benchmark.successful_requests` | v2 は validate 済み `result.request.returned_count` と `error_count` に置換 |
| 非収集 | raw throughput、`last_gen_throughput` | primitive count と generation time から view 導出するため raw 保存しない |
| 非収集 | benchmark backend の raw result field、予定 request 数を成功数として保存する field | backend は worker runtime/config、予定件数は workload-derived consumer が所有する |
| 非収集 | generated text、per-request output hash、raw prompt の BigQuery row | 本文を検索表へ複製しない。必要な ordered token identity だけ result artifact に保持 |
| 非収集 | field 別 missing/reason/status、placeholder、retry counter、coverage threshold/verdict | 欠損は raw key/row absent、固定幅 consumer は NULL、run-level は `collection_complete` のみ |
| 非収集 | bench の GPU sample/profile aggregate placeholder | GPU telemetry は profile-only。非適用runは row を作らず consumer を NULL |
| 非収集 | participant termination の array/aggregate/worst status | `(run,node_rank)` long relation だけを出す |
| 第2段階候補 | DCGM の engine/tensor/DRAM active、PCIe/NVLink bandwidth、XID count | Actions 上の capability probe で利用可能性を確認後に profile namespace へ追加。現 v2 必須集合には含めない |
| 動的 family | `sglang.node.<node>.{bench_args_effective,server_args_effective}.<field>`、`profile.profiler.global_rank.<rank>.<activity>.<category>.<event_name>.*` | leaf shape は列挙済み。args field 名は pinned SGLang inventory、profiler event/category名は実 trace の文字列を単射 escapeして使う |
| artifact-only | `hardware.node.<node>.lscpu_json` | minimal hardware source artifactには保持するが、queryable dotted metricは上表のCPU/settings leafだけ |

## 11. 実装状況

この表は「v2 で収集する予定」の契約であり、現行 `qwen-experiment-metrics` package の
実装済み一覧ではない。現行 18 test は通るが、実装は主に v1 の即時 scalar 送信である。
v2 に必要な runner manifest、scope 別 env、node/rank event、validated primary result、
raw profile GPU sample、phase reducer、profile hardware/profiler、node-agent lifecycle、
finalizer/packager、runner-only publisher は未実装である。

## 参照元

- `research/experiment-metrics/00-v2-plan.md`
- `research/experiment-metrics/01-v2-observation-map.md`
- `research/experiment-metrics/02-v2-contract-schemas.md`
- `research/experiment-metrics.md` と `research/experiment-metrics/01-*.md`〜`07-*.md`
  （キー階層規則の参照）
