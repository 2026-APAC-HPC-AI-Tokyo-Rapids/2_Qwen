# Qwen/SGLang v2 contract schema handoff

## 役割

この文書は[概要計画](./00-v2-plan.md)と[観測マップ](./01-v2-observation-map.md)を、
`qwen-v2.toml`とJSON Schemaへ転記するためのfield-level正本である。値のsource、field名、型、順序、
reducerを実装者が追加解釈しない。raw valueを別名で複製せず、source artifact、`run/key/value` fact、
query-time consumerの関係だけを定義する。

本projectのfailure modelは単純にする。取得できなかった値とrun variantで非適用の値はconsumerで`NULL`とし、
field別reason/status/欠損用placeholderを作らない。run-level statusは`collection_complete`だけである。

以下のclosed objectは、特記しないfieldがすべてrequired、field名末尾の`?`はoptionalかつ非該当時absent、
値型の`|null`だけがnullableである。optionalとnullableを同じ記号で表さない。
`additionalProperties=false`である。integerは64-bit範囲、`uint`はnonnegative、decimalはfinite、hashはlowercase
64-hex SHA-256、arrayは記載順を意味として保持する。

## canonical identity

| 項目 | v1定義 |
|---|---|
| repository ID | fixed string `apac2026-qwen`。runtime Git remote URLはidentityに使わない |
| run ID | `r-` + OS CSPRNG 16 byteのlowercase hex 32文字 |
| JSON canonical bytes | RFC 8785 JCS、UTF-8。duplicate key、NaN、Infinity禁止 |
| committed TOML identity | checkoutしたfileの実byte列のSHA-256 |
| Git object identity | `{object_hash_algorithm:enum "sha1"|"sha256",oid}` |
| artifact member path | archive rootからのrelative POSIX path。absolute、`..`、空componentは禁止 |
| member reference | closed `{path,size_bytes,sha256}` |
| environment snapshot | name byte順の`name\0value\0`反復。最後もNUL |

manifest自身のhashは本文へ入れず、参照側またはouter manifestに置く。

## source track v1

`control/run-manifest.json`へ、全field requiredかつ`additionalProperties=false`の
`source_track`を埋め込む。

```text
schema_version: const "source-track/v1"
object_hash_algorithm: enum "sha1"|"sha256"
base_role: enum "score_reference"|"profile_base"
base_commit: algorithm-constrained full OID
head_commit: algorithm-constrained full OID
```

benchはscore reference、profileはprofile baseを選ぶ。validatorは同じrepository/object algorithm、
clean checkout、full OIDの存在、baseがheadのancestorであることをjob投入前に確認する。branch名はidentityにしない。
projectionはrun ID、purpose、repository identity、base/head commit、manifest/member hashを一度だけ公開する。

## run manifestと実行設定

`control/run-manifest.json`は通常runだけを表し、次を持つ。

```text
schema_version: const "run-manifest/v1"
run_id: run ID
run_purpose: enum "bench"|"profile"
repository_id: const "apac2026-qwen"
source_track: source-track/v1
observed_control_plane_sha256
instrumentation_bundle_sha256
```

nodes、MPI processes、GPU/node、tensor parallel size、resource group、transport、address source、portなどの実行設定は
実行設定文書の独立fieldであり、束ねるID/objectを作らない。

実行設定文書の`run_purpose`は`bench|profile`だけである。共通部はsource reference、
個別実行設定、workload、artifact policy、bundle path/hashを持つ。profileはprofiler設定を持つ。
用途不一致のfieldはabsentにする。

実control-plane hashは、source track、bundle、および実行設定文書が列挙するcontrol fileの
実blob OIDから計算する。設定文書に書いた期待値を観測値として複写しない。

## instrumentation bundle

bundleはpurpose-neutralなsemantic definitionであり、hook、event、sample、wrapper、node-agent、
environment scope、source path allowlist、argv template、sink source IDをclosed schemaで持つ。
runnerは`score -> bench`、`profile -> profile`の固定写像で一つ選び、manifestへbundle SHA-256を保存する。
実argv/cwdはStageRunner fact、sink path templateはsource registry、実byteを含むfile identityはartifactが所有する。

experiment-metrics-v2の実装中に行う計測なし／あり比較は一時的な非回帰確認である。結果は確認後に破棄し、
専用purpose、arm、pair、allocation、manifest、artifact、metric key、relation、Sheet列、eligibility、
decision recordを定義しない。
## container sidecar v1

`source/container-sidecar.json`はBuildah、Skopeo、Umociによるbuild完了後に作るclosed objectである。

```text
schema_version: const "container-sidecar/v1"
container_artifact: {artifact_relative_path:const "sglang.sqsh",size_bytes:uint,sha256:sha256}
oci_manifest: {sha256:sha256}
source_commits: {qwen:full Git OID,sglang:full Git OID}
toolchain: {buildah:nonempty version string,skopeo:nonempty version string,umoci:nonempty version string}
```

provision時は実際にlaunchするSquashFS byteのsizeとSHA-256をsidecarへ照合する。検証済みの値は
`build.provenance`、`source.container_build.qwen_commit`、`source.sglang.commit`としてrun factへ記録する。
モデルとdatasetは、それぞれ`source/model-metadata.json`と`source/dataset-manifest.json`を正本とし、
重みなしのdummy loadとrandom datasetの実引数を閉じる。workflow run/attemptはoperational relationだけが所有する。

## runtime launcher validation v1

run bundleが`source/runtime-launchers.json`を一度所有する。closed sourceは
`{schema_version:const "runtime-launcher-validation/v1",
executables:exactly 2 [{role:enum "mpirun"|"enroot",requested_path:string,resolved_realpath:absolute POSIX path,
size_bytes:uint,sha256:sha256}],versions:exactly 2 [{role:enum "mpirun"|"enroot",
argv:exact [same-role requested path,"--version"|"version"],outcome:{exit_code:uint},stdout:member,stderr:member}]}`で、両arrayは記載role順、
各role exactly onceである。outer MPI stageとnode wrapperはこのresolved absolute pathを使い、PATH検索しない。

publisher projectionは全field requiredの
`{schema_version:const "runtime-launcher-projection/v1",run_id:string,executables:[{role,requested_path,resolved_realpath,size_bytes,sha256},...],
versions:[{role,argv,outcome,stdout_member:relative POSIX path,stdout_b64:base64,
stderr_member:relative POSIX path,stderr_b64:base64},...],source_member:relative POSIX path,source_sha256:sha256}`である。
`runtime-launcher-semantic/v1`はpath/member locatorを除いたordered
`{schema_version:const "runtime-launcher-semantic/v1",launchers:[{role,sha256,outcome,stdout_b64,stderr_b64},...]}`で、
consumer identityは`SHA-256(JCS(object))`とする。

## model、dataset、cache

### model manifest v1

`source/model-manifest.json`の`source`はclosed tagged unionである。

- `{kind="hub_revision", repository:string, revision:full immutable hex commit}`
- `{kind="prebuilt_manifest", shards:[{path:relative path,size_bytes:uint,sha256},...]}`。
  shardsはpath byte順で、必要なsemantic IDは`SHA-256(JCS(shards))`としてviewで導出し、同recordへ複写しない

run recordはsourceに加え、次のsmall metadata allowlistの実在memberだけをpath byte順で持つ。

```text
config.json
model.safetensors.index.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
generation_config.json
chat_template.json または tokenizer_config内chat template source record
```

top-levelは`schema_version="model-manifest/v1"/source/small_metadata`、各metadata memberは
`role=configuration|index|tokenizer|tokenizer_config|special_tokens|generation_config|chat_template`、
`path/size_bytes/sha256`を持つ。indexが参照するshard集合はindex fileをparseしてprebuilt manifestへ
照合するが、run manifestへ再掲しない。runごとに約400 GBのshard本文を再hashしない。

### dataset manifest v1

`source/dataset-manifest.json`は`schema_version="dataset-manifest/v1"/repository:string/
full_revision:string/relative_path:relative path/size_bytes:uint/sha256:sha256`を持つ。
download commandはこれらから完全導出できるため保存しない。moving `main`やshared filesystem pathをidentityにしない。

### cache manifest v1

`source/cache/node-<node-rank>-input-manifest.json`は`schema_version="cache-manifest/v1"/run_id/node_rank:uint/
members`を持ち、cache rootからのrelative path byte順でmemberを並べる。
regular fileは`{type="file",path,size_bytes:uint,sha256}`、symlinkは
`{type="symlink",path,target_b64:base64}`を持つ。
mtime、uid、gid、absolute staging pathはidentityに含めない。Fact identityはsubject bindingを除いた
`{schema_version:"cache-members/v1",members:<sorted members>}`のJCS SHA-256とし、同じcache内容ならrunを跨いで一致する。
source file byte hashはartifact lineageだけに使う。semantic identityだけを
`cache.node.<node_rank>.input_manifest_sha256`へ索引化し、file count/total size/cold-warm labelをrawへ複製しない。

## environment、topology、runtime config

全`.snapshot`はname byte順の`name\0value\0`反復である。parserはname/valueを変更せず、BigQueryではnameを
単射escape、valueをRFC 4648 Base64にする。scopeは`job_entry|host_exec_node|sglang_launch_node|
worker_global_rank`で、full snapshotを保持する。common/deltaの第二raw表現は作らない。

minimal CPU hardwareはnode wrapperが各nodeで一度だけ
`LC_ALL=C lscpu --json`を実行する。`source/hardware/node-<node-rank>.json`は`hardware-node/v1`で、
`schema_version:const "hardware-node/v1"/node_rank:uint/hostname:string`、`commands={lscpu:{argv:["lscpu","--json"],exit_code:int,
stderr_member:relative POSIX path}}`、`lscpu_json:object|null`、closed
`platform_settings={kernel_release:string|null,
online_cpus:uint[]|null,isolated_cpus:uint[]|null,nohz_full_cpus:uint[]|null,
memory:{total_bytes:uint|null,hugepage_size_bytes:uint|null,hugepages_total:uint|null},
numa_balancing_enabled:bool|null,boost:{source_path:string|null,raw:string|null,enabled:bool|null},
cpufreq_policies:[{policy_id:uint,related_cpus:uint[],driver:string|null,governor:string|null,
min_hz:uint|null,max_hz:uint|null},...]|null}`を持つ。同じsource recordへlscpuからderived CPU identityを複写しない。
kernelは`/proc/sys/kernel/osrelease`、CPU setは
`/sys/devices/system/cpu/{online,isolated,nohz_full}`、memory/hugepageは`/proc/meminfo`を使いkBを1024倍する。
NUMA balancingは`/proc/sys/kernel/numa_balancing`、boostはcpufreq `boost`を優先し、なければintel_pstate
`no_turbo`を反転する。cpufreqはnumeric policy ID順にrelated CPUs/driver/governor/min/maxを読み、kHzを1000倍する。
CPU-list fileは存在して空なら`[]`、interface不存在/read不能ならnull、cpufreq rootが
存在してpolicy 0件なら`[]`、root不存在/列挙不能ならnullにする。BigQueryはnode-rank順に投影し、consensusはviewで導出する。
cpufreq `related_cpus`はASCII whitespace区切りuintをstrictly increasing unique arrayへ変換し、`nohz_full`だけは
whitespace trim後のliteral `(null)`も`[]`へcanonicalizeする。

`minimal-hardware-projection/v1`は全field requiredの
`{schema_version:const "minimal-hardware-projection/v1",run_id:string,node_rank:uint,
hostname:string,cpu_identity:{architecture,model_name,vendor_id,cpu_family,model,stepping:string|null,
physical_cores:uint|null,numa_nodes:uint|null},platform_settings:<hardware-node/v1のclosed settings>,
source_member:relative POSIX path,source_sha256:sha256}`である。CPU identityはlscpu label
`Architecture/Model name/Vendor ID/CPU family/Model/Stepping/Socket(s)/Core(s) per socket/NUMA node(s)`を
`field`のASCII末尾colonをexactly oneつ除去して照合し、physical coresはsocket数×core/socketとする。

`source/topology/node-<node-rank>.json`は次のclosed source recordを持つ。

- `schema_version="node-topology/v1"/run_id/node_rank:uint`、outer MPI
  `global_rank:uint/local_rank:uint/size:uint/hostname:string`
- exact `nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version,persistence_mode,
  compute_mode,power.limit,clocks.max.sm,clocks.max.memory,mig.mode.current,ecc.mode.current
  --format=csv,noheader,nounits`
  のhost/container別`{argv:string[],exit_code:int,stdout_member,stdout_size_bytes:uint,stdout_sha256:sha256,
  stderr_member,stderr_size_bytes:uint,stderr_sha256:sha256}`
- host `nvidia-smi topo -m`の同じhash-sealed command record
- node別actual `node_wrapper_argv`、`container_argv`、`sglang_argv`のstring array、host CPU affinity uint array、NUMA mems string

GPU arrayはraw stdoutから`topology/v1` reducerが一度parseし、node-rank/GPU index順の
`{node_rank,index,name,uuid,pci_bus_id,memory_total_bytes,driver_version,persistence_mode,compute_mode,power_limit_w,
max_sm_clock_mhz,max_memory_clock_mhz,mig_mode,ecc_mode,scope=host|container}`としてBigQueryへ
投影する。source JSON内へparsed copyを持たず、node argvや`hardware.gpu.nodes`の第二raw factも作らない。
`node-launch-projection/v1`は`schema_version:const "node-launch-projection/v1"/run_id/node_rank/global_rank/local_rank/size/hostname/node_wrapper_argv/
container_argv/sglang_argv/cpu_affinity/numa_mems/source_member/source_sha256`を持つ。
`topology-projection/v1`は上記GPU recordに`schema_version:const "topology-projection/v1"/run_id/source_member/source_sha256`を加える。

BenchArgs/ServerArgs event payloadは`config-snapshot/v1`で、`config_type=BenchArgs|ServerArgs`、
`definition={source_path:repository-relative POSIX path,class_name:nonempty string,
source_blob_oid:algorithm-constrained full OID}`、field名byte順の
`fields=[{name,declared_type,value},...]`を持つ。`value`は実dataclass値のJCS valueで、全fieldを保存する。このclosed
meta-shapeによりSGLang commitごとのfield増減を失わず、手書きallowlistも作らない。wide Runs列はcontractが指定した
scalar leafをquery時に投影するだけである。event typeはそれぞれ`bench_args_effective`、
`server_args_effective`へ固定する。
blob OIDのalgorithmはSIF provenanceの`source_commits[commit_role="sglang_source"]`から継承し、そのcommitの
tree下で`source_path`が指すblobとexact一致させる。SGLang commitはbuild projectionだけが所有し、eventへ複写しない。

worker-ready envelopeは次のclosed fieldを持つ。

```text
schema_version:const "qwen-domain-event/v1", run_id, producer_id, node_rank, global_rank, sequence,
event_type="worker_ready", payload={
  tp_rank, gpu_id, gpu_uuid, cpu_affinity, numa_mems,
  prefill_attention_backend, decode_attention_backend,
  torch_distributed_backend, transport_backend
}
```

expected global rankは`0..protocol.tensor_parallel_size-1`、node rankは`0..protocol.nodes-1`である。値をServerArgsから推測しない。

`qwen-placement/v1` reducerは新しいrawやreason fieldを作らず、protocolの個別実行設定、scheduler request、全node launch、
container GPU topologyを一度だけ照合し、bench/profileの各nodeの
`server_args_effective`と全worker-readyを追加照合する。scheduler requestの
nodes/MPI processes/resource group/transport policyはprotocolの各fieldとexact一致する。node launchはnode rankと
outer MPI global rankがともにexact `0..protocol.nodes-1`、各recordで`global_rank=node_rank`、`local_rank=0`、
`size=protocol.mpi_processes`、hostnameが全nodeでdistinctでなければならない。common container topologyは各expected nodeで
exact `protocol.gpus_per_node`件とindex集合`0..protocol.gpus_per_node-1`を持ち、全nodeでexact
`protocol.tensor_parallel_size`件のglobal-unique UUIDを持たなければならない。各ServerArgsは`tp_size`、`nnodes`、
`node_rank`がそれぞれprotocolのTP、nodes、当該node rankと一致し、distributed addressはprotocolのrank 0 advertise/port
contractと一致する。worker envelopeはglobal rankと`payload.tp_rank`がともにexact
`0..protocol.tensor_parallel_size-1`で一対一に一致し、各nodeにexact `protocol.gpus_per_node`件を持つ。
各workerの`gpu_id/gpu_uuid`は同じnodeのcontainer topology `(index,uuid)`へbijectionし、全nodeのworker UUIDもuniqueとする。
必要sourceが欠ければ既存`placement_valid`はNULL、揃って一つでも不一致ならfalse、
それ以外だけtrueにする。`benchmark_valid`はresult整合だけを表す既存意味から変えない。benchの
`score_comparison_eligible`だけが`placement_valid=true`を要求する。profileは値を表示するが
score統計へ入れず、原因列やvalidation artifactは作らない。

## event、lifecycle、phase

全domain eventの共通envelopeは
`schema_version:const "qwen-domain-event/v1"/run_id/producer_id/node_rank/global_rank?/sequence/event_type/payload`である。
producer内sequenceは0から連続増加する。clock IDはLinux boot IDとclock名を組み合わせた
`<boot_id>:CLOCK_MONOTONIC`で、別nodeのmonotonic値を減算しない。

`control/lifecycle/rendezvous.json`は`schema_version:const "lifecycle-rendezvous/v1"/run_id/rank0_hostname/address/
port=protocol.node_agent_port/created_at`、summaryは`schema_version="lifecycle-summary/v1"/run_id/nodes`を持つ。node entryは
`node_rank:uint/ready:{sequence:uint,clock_id:string,received_monotonic_ns:uint,ack_monotonic_ns:uint}/
shutdown:{sequence:uint,invocation_id:string,clock_id:string,received_monotonic_ns:uint,ack_monotonic_ns:uint}|null`である。
`nodes`はnode-rank順のexact `protocol.nodes`件で、rank集合は`0..protocol.nodes-1`と一致する。
各entryで`ready.received_monotonic_ns <= ready.ack_monotonic_ns`を要求する。shutdownがnon-nullならreadyと同じ
clock IDで、`ready.ack_monotonic_ns <= shutdown.received_monotonic_ns <= shutdown.ack_monotonic_ns`を要求する。
node-agent summaryがshutdown受信の唯一のsource、wrapper lifecycleが
child exitの唯一のsourceである。固定port衝突/timeoutはrun失敗とし、自動再試行しない。
`lifecycle-summary-projection/v1`はこのclosed summaryへ
`source_member:relative POSIX path/source_sha256:sha256`を加えた全field required objectである。

wrapperがFact `parallel.node.<node_rank>.process_lifecycle`へ出す`node-lifecycle/v1`もclosed objectとし、
次に固定する。

```text
schema_version: const "node-lifecycle/v1"
clock_id: nonempty string
spawn_attempt_monotonic_ns: uint
child_started_monotonic_ns: uint|null
child_finished_monotonic_ns: uint|null
termination: one of
  {kind:const "waited", raw_wait_status:uint}
  {kind:const "spawn_error"}
```

spawn成功時だけ両child timestampを持ち、attempt <= started <= finishedとする。`spawn_error`では両方null、
`waited`では両方non-nullかつraw statusがPOSIX `WIFEXITED xor WIFSIGNALED`を満たす。exit/signal分類と
code/signalはraw statusからconsumerが一度だけ導出する。wrapperはそれらやshutdown record、
early/coordinated分類を複写しない。
外側Factの`run`とkey内`node_rank`をvalueへ再掲しない。node childは`node_agent.py`内の小さなPython supervisorが
`os.waitpid(child_pid, 0)`で一度回収し、そのraw statusを保存する。Bash `wait`の`$?`はこのsourceに使わない。

`control/phase/summary.json`は`phase-summary/v1`でprofileだけに存在する。invocationごとに
`invocation_id/kind=warmup|measured`、rank 0の
`{clock_id,start_broadcast_ns,all_start_acks_received_ns,generation_start_ns,generation_end_ns,
end_broadcast_ns,all_end_acks_received_ns}`、node-rank順の
`{node_rank,clock_id,start_sequence,end_sequence,start_received_ns,start_ack_ns,end_received_ns,end_ack_ns}`を持つ。
node arrayはnode-rank順のexact `protocol.nodes`件で、rank集合は`0..protocol.nodes-1`と一致する。
各entryは一つのclock ID上で
`start_received_ns <= start_ack_ns <= end_received_ns <= end_ack_ns`を要求する。
rank 0の6 timestampは記載順にnondecreasingで同じclock IDを使う。start/end sequenceは同node control stream上で
strict increasing、GPU sampleとnode windowのclock_idは一致必須である。
GPU phase windowは同nodeの`[start_received_ns,end_received_ns)`で、ACK時刻はlead/trail診断だけに使う。
warmup/measuredのrank 0 generation境界は通常result writerが使う同じmonotonic-ns値であり、uniformな
`perf.phase.warmup_generation.wall_time_sec`または`perf.benchmark.generation.wall_time_sec`は
`(generation_end_ns-generation_start_ns)/1_000_000_000`のexact decimalに固定する。profileでは同じinvocation kindで
equalityを検査する。scalarは全variant共通result schema、timestamp pairはprofile window containmentに必要なためだけ併存する。
GPU sample intervalの初期値は1.0秒。
coverage thresholdを根拠なく固定せず、observed coverageとvalid durationを表示する。

`phase-summary-projection/v1`は全field requiredの
`{schema_version:const "phase-summary-projection/v1",run_id,invocation_id,kind,rank0:{clock_id,start_broadcast_ns,
all_start_acks_received_ns,generation_start_ns,generation_end_ns,end_broadcast_ns,all_end_acks_received_ns},
nodes:[{node_rank,clock_id,start_sequence,end_sequence,start_received_ns,start_ack_ns,end_received_ns,end_ack_ns},...],
source_member:relative POSIX path,source_sha256:sha256}`で、上記型/順序制約をそのまま持つ。

GPU sampleは次のclosed schemaである。

```text
schema_version="gpu-sample/v1", run_id:string, node_rank:uint, sequence:uint,
clock_id:string, monotonic_ns:uint, wall_time_utc:RFC3339 UTC string,
gpu_uuid:string, logical_index:uint,
utilization_percent:decimal|null, memory_used_bytes:uint|null, power_w:decimal|null,
temperature_c:decimal|null, sm_clock_mhz:decimal|null
```

`sequence`はnode poll-frame ordinalで、そのpollから得た全expected GPU rowが同じ値を持つ。frameはexpected UUIDを
exactly one rowずつ持つ時だけcompleteで、途中frameはartifactに残してもprojectionへ入れない。complete frameのsequence
gapは許すが補間用の架空sampleを作らない。utilizationは0..100、他のnon-null measurementはnonnegativeである。取得できないfieldはnullとし、poll
status/reason fieldは作らない。publisherはprofile raw tarをparseし、同じsample fieldに
`source_member:relative POSIX path/source_sha256:sha256`を加えたprojectionへ転記する。topologyとprofile hardwareも同様にexact source recordを
`qwen_topology_raw_v2`、`qwen_profile_hardware_raw_v2`へ転記し、hashだけで値を隠さない。
この追加後のclosed objectを`gpu-sample-projection/v1`とし、schema versionをconst同名に置換する。

profile-only CPU/NUMA rawはnode-rankごとに
`profile/raw/hardware/node-<node-rank>/{proc-meminfo.txt,
numa/node-<id>-cpulist.txt,numa/node-<id>-meminfo.txt,capture.json}`へ固定する。`capture.json`は各commandの
全field required、`additionalProperties=false`のclosed
`profile-hardware-capture/v1={schema_version:const "profile-hardware-capture/v1",run_id:run ID,node_rank:uint,hostname:nonempty string,
members:[{role:enum "proc_meminfo"|"numa_cpulist"|"numa_meminfo",numa_node_id:uint|null,
path:relative POSIX path,size_bytes:uint,sha256:sha256}]}`を持つ。
membersはrole/node/path順unique、proc_meminfoはnode ID nullでexactly one、NUMA二roleは検出nodeごとに同じIDを一件ずつ持つ。
読取元はrole/node IDから`/proc/meminfo`、`/sys/devices/system/node/node<id>/cpulist`、同`meminfo`へ固定導出し、
同じ`source_path`をrecordへ複写しない。読取境界はprofile node-agentの既存lifecycleが一度所有し、node別captureへ
argv/outcomeを複写しない。
publisher rowは全field required、`additionalProperties=false`の
`profile-hardware-projection/v1={schema_version:const "profile-hardware-projection/v1",run_id,node_rank:uint,hostname:nonempty string,record_kind:enum
"node_summary"|"numa_node",numa_node_id:uint|null,postrun_memory_available_bytes:uint|null,
cpulist:string|null,mem_total_bytes:uint|null,mem_free_bytes:uint|null,
source_members:[{role:enum "capture"|"proc_meminfo"|"numa_cpulist"|"numa_meminfo",
path:relative POSIX path,sha256:sha256},...]}`である。全rowにexactly one `capture`を持たせ、node summaryではNUMA field、
NUMA rowではpostrun fieldをnullにし、NUMA rowのsource memberはcapture/cpulist/meminfoを別entryで保持する。
captureは`gethostname()`を一度記録し、同runのminimal hardwareとnode-rank集合・hostname集合および各
`(node_rank,hostname)`をexact一致させる。`/proc/meminfo`はASCII `Key: uint kB`の`MemAvailable`、NUMA meminfoは
`^Node <id> (MemTotal|MemFree):[ \t]+([0-9]+)[ \t]+kB$`を各exactly once選び、別node ID prefixをrejectして
kBを1024倍してbytesへ変換する。duplicate key、非uint、overflow、unit不一致では
該当captureを投影しない。cpulistはASCII whitespaceをtrimし、kernel list grammarからstrictly increasing unique CPU集合を
最小range表記へcanonicalizeする。空fileは空string、read不能/parse不能値だけNULLにする。minimal hardwareにある
CPU identity/settingsを再収録しない。

profileのTorch Profilerはglobal TP rankごとに
`profile/raw/torch/qwen-profile-TP-<global-rank>.trace.json.gz`と同stemの`.capture.json`を一件ずつ作る。
captureは全field required、`additionalProperties=false`のclosed
`profiler-capture/v1={schema_version:const "profiler-capture/v1",run_id:run ID,global_rank:uint,
invocation_id:nonempty string,source_member:relative POSIX path,size_bytes:uint,sha256:sha256}`である。
`global_rank`はfilename、trace metadata、protocolの`0..tensor_parallel_size-1`と一致させ、source memberは対応gzip traceを指す。
capture自身を別raw relationへ投影せず、traceとのlineageにだけ使う。全rank traceが揃わなければ既存
`collection_complete=false`とし、欠損rankのplaceholder、profiler専用status/reason、retryを作らない。

publisherは実在traceを一度parseし、complete duration eventを
`(run,global_rank,activity,event_category,event_name)`でgroup化したclosed
`profiler-event-projection/v1={schema_version:const "profiler-event-projection/v1",run_id:run ID,
global_rank:uint,activity:enum "CPU"|"GPU",event_category:string,event_name:string,
call_count:positive uint,inclusive_duration_sum_ns:nonnegative decimal,
source_member:relative POSIX path,source_sha256:sha256}`として`qwen_profiler_events_raw_v2`へ転記する。
平均、構成比、rank/run合計、top-N、分類summaryは保存せずquery時に導出する。benchはこのrelationへ行を作らず、
固定幅consumerはNULLだけにする。

GPU aggregateはraw factへ保存しない。`gpu-phase/v1` reducer/viewは`phase,node_rank,gpu_uuid` grainで次のclosed列を返す。

```text
phase_duration_sec:decimal,
valid_sample_count:uint,
valid_utilization_duration_sec:decimal, utilization_coverage:decimal,
utilization_mean:decimal|null, utilization_p95:decimal|null,
valid_power_duration_sec:decimal, power_coverage:decimal,
energy_j:decimal|null, power_mean:decimal|null,
valid_memory_sample_count:uint, memory_max_bytes:uint|null,
valid_temperature_sample_count:uint, temperature_max_c:decimal|null,
valid_sm_clock_sample_count:uint, sm_clock_min_mhz:decimal|null
```

`valid_sample_count`はschema-validなcomplete frameに属し、下記step intervalがphase windowと正durationで交差する
当該GPU row数で、全measurementがnullでも数える。metric別countとmemory/temperature/SM-clockのmax/minは、同じ
overlap条件を満たし当該measurementがnon-nullなrowだけを使う。したがってpre-start rowも有効区間がphaseへ入る時は
count/valueの双方へ入る。各sample値は次sample、
phase end、`sample_time + 2 * nominal_interval`の最小まで一定とする。duration系は
このstep intervalとphase windowのoverlapだけを足し、coverageはvalid duration / phase duration、energyは
valid power intervalをstep積分して欠損intervalへ外挿しない。utilization p95はoverlap durationで重み付けした
nearest-rank、`power_mean=energy_j/valid_power_duration_sec`とする。fleet projectionはexpected GPU gridへleft joinし、included/valid GPU count、coverage min、GPU統計の
min/median/max、energy sumをquery時に導出する。complete/status booleanは作らず、必要入力が足りないprojectionはNULLにする。

## workloadとresult

### ordered workload v1

`source/workload-{warmup,measured}.json`は
`schema_version:const "ordered-workload/v1"/dataset_manifest_sha256/kind/requests`を持つ。`kind`は`warmup|measured`、warmupだけ`mode` enum
`random|dataset`を追加する。run IDはsemantic workload bytesへ入れず、run bundle manifestとpublisher
projectionだけがbindingを所有する。このため同じactual workloadはrunを跨いでも同じsource SHA-256になる。
requestsは実generate順で、各要素は次を持つ。

```text
source:{kind="dataset_row", array_index:uint} | {kind="generated", generator_id:string, seed:uint, generator_ordinal:uint},
prompt_payload:{kind="text", text:string, input_token_count:uint} | {kind="token_ids", token_ids:[int,...]},
requested_output_token_count:uint,
sampling_params:[{name:string, declared_type:string, value:JCS value}, ...]
```

`prompt_payload`は現行text pathを勝手にtoken ID pathへ変えず、実generateへ渡す値をtagどおり保存する。textの
`input_token_count`はrequest選択時にtokenizerが実測した値で、token_idsではarray lengthから導出するため保存しない。
sampling paramsは実generateへ渡す全fieldをname byte順に持つ。manifest identityはrequest array順とexact
prompt payloadを含む。generated本文は保存しない。publisherはmanifest top-levelを
`qwen_workload_manifests_raw_v2`、同じrequest recordを`qwen_workload_requests_raw_v2`へ転記する。source member/hashは
manifest relationだけが所有し、request relationはmanifest identityでjoinする。

`workload-manifest-projection/v1`は全field required、`additionalProperties=false`の
`{schema_version:const "workload-manifest-projection/v1",run_id:string,
workload_kind:enum "warmup"|"measured",dataset_manifest_sha256:sha256,warmup_mode:enum "random"|"dataset"|null,
source_member:relative POSIX path,source_sha256:sha256}`を持つ。measuredのmodeはnullで、
`source_sha256`がsubject bindingを除いたcanonical workload fileのsemantic manifest identityを兼ねる。
`workload-request-projection/v1`は全field requiredの
`{schema_version:const "workload-request-projection/v1",run_id,workload_kind,ordinal:uint,request:<上記request closed object>}`を持ち、親manifest projectionを
`run_id/workload_kind`でjoinする。manifest hashとsource lineageをrequest rowへ複写しない。

`ordinal`はordered `requests` arrayの0-based offsetからpublisherがnormalization時に一度だけ付与し、source itemへは複写しない。

manifestだけがordered requestのraw正本である。selected count、planned input/output token total、warmup modeの
別raw keyを作らず、consumerがarrayから導出する。

### primary result v1

`source/primary-result.json`はrank 0がgenerate return直後に一度write/renameし、次を持つ。

```text
schema_version: const "primary-result/v1", run_id:nonempty string, invocation_id:nonempty string,
generation_wall_sec:positive finite decimal, returned_count:uint, error_count:uint,
observed_input_tokens:uint|null, observed_output_tokens:uint|null,
ordered_output_token_ids_sha256:sha256|null
```

parse不能responseは`error_count`へ数え、取れないtoken totalを0で補わない。generate例外時はprimary resultを
捏造しない。output identityはresponse array順を保持した`[[uint token_id,...],...]`のJCS SHA-256で、全responseの
`output_ids`がnonnegative integer arrayとして読め、各array長が同じresponseの`completion_tokens`、全長の和が
`observed_output_tokens`と一致する時だけnon-nullにする。生成本文、per-request hash、同じhashの別Factは作らない。
`benchmark_valid`はerror=0、returned=workload requests length、input observed=workload由来planned total、generation wallが正有限、
`ignore_eos`固定時はoutput observed=requestedであることだけから導出する。

view formulasは次で、throughputをrawへ保存しない。

```text
request_throughput = returned_count / generation_wall_sec
input_token_throughput = observed_input_tokens / generation_wall_sec
output_token_throughput = observed_output_tokens / generation_wall_sec
total_token_throughput = (observed_input_tokens + observed_output_tokens) / generation_wall_sec
```

## network captureとprojection v1

```text
mpi-network-result/v2 = {
  run_id:string, benchmark:enum "PingPong"|"Sendrecv"|"Allreduce",
  process_count:uint>0, message_bytes:uint, iterations:uint>0,
  time_min_us:finite number|null,time_avg_us:finite number|null,time_max_us:finite number|null,
  bandwidth_mbytes_per_sec:finite number>=0|null,source_member:relative POSIX path
}

nccl-network-result/v2 = {
  run_id:string,collective:enum "all_reduce"|"all_gather"|"reduce_scatter",
  message_bytes:uint>0,element_count:uint>0,data_type:const "float",
  reduction_op:enum "sum"|null,buffer_mode:enum "out_of_place"|"in_place",
  mpi_process_count:uint>0,gpu_count:uint>0,time_us:finite number>=0,
  algorithm_bandwidth_gbps:finite number>=0,bus_bandwidth_gbps:finite number>=0,
  error_count:uint,source_member:relative POSIX path
}
```

IMB source membershipは`source/network/imb-mpi1/{argv.json,stdout.txt,stderr.txt}`、NCCLはcollectiveごとの
`{argv.json,result.json,stdout.txt,stderr.txt}`に固定する。IMB parserはcompleteな全sectionだけをacceptし、NCCL parserは
`-J` JSONだけを正本とするpure functionである。truncated/parse不能sourceから部分rowを作らない。

`mpi_network_results_v2`のgrainは`(run_id,benchmark,process_count,message_bytes)`、
`nccl_network_results_v2`は`(run_id,collective,message_bytes,buffer_mode)`である。各rowはsource member pathだけを持ち、
artifact hashはmanifest/projection lineageだけが所有する。algbw/busbwを再計算せず、summary row/keyを追加しない。
`all_reduce`と`reduce_scatter`の`reduction_op`は`sum`、`all_gather`はNULLに固定する。JSON内のsize/count/type、
out-of-place/in-place、time、algbw、busbw、errorをlosslessに写し、8 byteから1 GiBまでfactor 2の実在sizeだけをacceptする。
container buildでは11個のNCCL test binary、JSON出力用`-J`、主要3 collectiveの起動可能性をBuildahの作業container内で確認し、成功しなければSquashFSを公開しない。

## control、merge、package

PJM entryは通常runだけを扱う。

```text
job-entry/v1:
  schema_version, run_id, pj_job_id, entered_at
bundle-manifest/v1:
  schema_version, run_id, members[]
```

runnerはsubmit前後のexact argv/stdout/stderr/response、PJM側job ID、terminal accountingを保存する。
manifest run IDとjob-entry run ID、submit/job-entry/accountingのjob IDが一致しなければpublishしない。
通常run bundleは`control/`、`source/`、`logs/`、`metrics.jsonl`、profile時だけ`profile/raw/`を持つ。
artifact filesはrole順uniqueとし、outer manifest自身をmemberへ含めない。

mergeは実装とpurposeからexpected producer/file集合を一度導出する。完成fragmentだけを読み、
同一ownerのbyte-identical factだけをdedupeし、異値はconflictとしてrunをincompleteにする。
取得不能と非適用はconsumerでNULLとし、欠損reason/status/placeholder keyを作らない。

## artifact projection registry v1

source instanceのowner scopeは`run`または`build`だけである。run ownerはrun ID、build ownerはbuild identityを
持ち、一つのsource instanceを一つのprojection parserへだけrouteする。

主なprojectionは次である。

| source/relation | grain | applicability |
|---|---|---|
| source track | run ID | bench/profile |
| runtime launcher | run ID、role | bench/profile |
| container provenance | SquashFS SHA-256 | bench/profile |
| model/dataset/cache manifest | run ID、kind | bench/profile |
| workload manifest/request | run ID、kind、ordinal | bench/profile |
| minimal hardware | run ID、node rank | bench/profile |
| topology/placement | run ID、node/global rank | bench/profile |
| environment | run ID、scope、name | bench/profile |
| lifecycle/result | run IDまたはinvocation ID | bench/profile |
| GPU samples/phases | run ID、node rank、sequence | profile only |
| artifact index | run ID、role | bench/profile |
| IMB network | run ID、benchmark、process count、message bytes | bench/profile、generation成功時 |
| NCCL network | run ID、collective、message bytes、buffer mode | bench/profile、generation成功時 |

metric familyやprojection instanceを専用の非擾乱確認用namespaceへ複製しない。

## factとconsumer routing

scalar/compound factは`run/key/value`だけへ出す。完全recordをartifact/projectionが所有する場合、
first/last/count/average、構成比、rank/run合計、top-Nなどのsubset/summary keyは保存せずquery時に導出する。
source artifactとfactの二重ownershipを禁止する。

consumerはkeyとpurposeによりprojectionを選ぶ。用途別applicabilityを評価してから
expected locator/gridを生成し、profile-only列はbenchで理由なしのNULLにする。benchの比較適格性は通常runの
source/workload/result/placement/completenessだけから導出し、開発時の非擾乱確認をjoinしない。

実装中の計測なし／あり比較は公開契約の外で行い、出力を破棄する。そのため、このcontractには専用schema、
source、writer、parser、projection、relation、artifact role、metric keyを追加しない。
