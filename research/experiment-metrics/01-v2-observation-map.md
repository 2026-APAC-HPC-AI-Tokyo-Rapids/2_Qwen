# Qwen/SGLang v2 観測マップ

## この文書の役割

この文書は、各factについて「どこにある値を、誰が、いつ取得するか」を固定する実装契約で
ある。[概要計画](./00-v2-plan.md)が情報モデルと比較規則、
[contract schema handoff](./02-v2-contract-schemas.md)がfield-level shapeを定め、本書が実コード・command・
artifactへの配線を定める。

key registryへ入れるには、次の六項目を全て埋めなければならない。

1. authoritative source: object、structured file、command、またはsource symbol
2. producer: そのsourceを一度だけ読むowner
3. capture moment: 値が確定し、まだ失われていない時点
4. raw destination: `run/key/value`または耐久artifact
5. validation: 誤読、scope混同、部分取得を検出する規則
6. applicability: どのrun variantで取得するか

sourceの優先順位は、native object/API、machine-readable command/file、専用structured marker、
stable log parser、汎用text regexの順とする。選んだsourceをfixtureで確認する。通常runで同じ値を
上位・下位sourceから二重取得する必要はない。CLI defaultや入力fileに書いた値を、実効値や実測結果として
扱ってはならない。

## 値の段階とscope

| 段階 | 意味 | Qwen/SGLangの例 |
|---|---|---|
| requested | Git管理script/protocolが指定した値 | `--tp-size 8`、32 prompts |
| launch-effective | default、environment、auto resolutionを反映し、worker launchへ渡す直前の値 | resolved `BenchArgs`、resolved `ServerArgs` |
| runtime-observed | worker/backend初期化後に実際に成立した値 | rank、GPU UUID、communication backend、worker-ready state |
| result-observed | 実responseを検査して得た値 | returned count、completion tokens、generation duration |

node scopeも分ける。rank 0の値を各nonzero node rankへ複製せず、まずnode別full snapshotを保存する。merge時に
node/rank-scoped full factを変換せず保持し、viewだけが全expected scopeで一致したfieldをcommon
表示する。common/deltaの第二raw表現は作らない。requestedとobserved、warmupとmeasured workload、
予定件数と返却件数は別factである。

## producerと時系列

```text
Actions runner
  create regular-run manifest or qualification-allocation parent plus two child manifests
  verify source/artifact sidecars
  seal scheduler request from lane, PJM script identity, and exact pjsub argv
  record subject ID and exact pjsub argv
  submit PJM without BigQuery credentials
  save stdout/stderr, outcome, and the job ID when returned

PJM supervisor
  snapshot job-entry environment
  write subject kind/ID, PJM_JOBID, and qualification ordered child IDs
  regular: let StageRunner launch one mpiexec/node-wrapper set
  qualification: run ordered child 1 then child 2, each with a fresh StageRunner/mpiexec/node-wrapper set

node wrapper / node-agent
  all variants: rank 0 writes common lifecycle rendezvous; every nonzero node rank validates and connects; record topology
  wait for expected-node readiness before starting containers
  verify cache staging and write input manifest
  all variants: snapshot exact outer/node argv and common host launch state
  bench/profile/qualification ON: additionally snapshot score config/container state
  qualification OFF: consume the parent actual workload; do not start score config/primary-result producers
  in profile only, append raw GPU samples and local phase markers
  flush/close all node source files before the PJM script exits

SGLang score hook (bench/profile/qualification ON, each node)
  emit launch-effective ServerArgs before scheduler launch
  emit runtime worker-ready facts after initialization

rank-0 score result hook (bench/profile/qualification ON)
  bench/profile: persist warmup/measured workload manifests
  qualification ON: consume the parent-owned manifests without copying them
  write the sole authoritative primary-result artifact through temporary file and rename
  in profile only, emit distributed GPU phase boundaries

rank-0 neutral qualification writer (qualification OFF/ON)
  validate the actual response array and write the common D-001 estimand result

Actions runner
  after PJM terminal state, parse/reduce completed saved source files
  verify expected node and global-worker scoped facts without scope reduction
  regular: merge -> validate expected files -> GitHub Artifact upload -> publish metrics and artifact projections
  qualification: package allocation source and each non-null child source/log/metrics in one artifact set
```

`run-manifest.json`で生成したrun IDを全producerが読む。各processで再計算しない。node間で共有
Unix domain socketをtransportにしない。socketを使うならendpointとJSONL writerは各node localに
置き、node rankをevent envelopeで必須にする。target hookはlocal structured eventを書くだけで、
metrics key、BigQuery、retryを知らない。

## fact別の取得契約

表の「欠損」はraw valueへ空文字を入れる意味ではない。keyを出さず、consumerは`NULL`を返す。
required sourceならrun-level `collection_complete=false`とするだけで、原因、family status、欠損用placeholderを
作らない。非適用のprofile-only列も同じく`NULL`だけだが、`collection_complete`の判定対象にはしない。

| fact family | authoritative source | producer / capture moment | raw keyまたはartifact | validation | 欠損時の扱い |
|---|---|---|---|---|---|
| run/subject identity | runnerが一度生成したsubject/child manifests | `runner.manifest` / provision前 | raw tableの`run`列、regularは`control/run-manifest.json`、qualificationはparent `control/qualification-allocation-manifest.json`とchild manifest | manifestに汎用subject IDは保存せず、projection時にregular/child subject ID=run ID、allocation subject ID=allocation IDと導出。parent ordered childとchild armを照合し、block/order/replicateはallocation/protocol/array offsetから導出。PJM job IDを含めない | manifestなしでは投入しない |
| metrics contract | Git管理`qwen-v2.toml`とlane/run variantが選んだexpected producer graph | `runner.manifest` / provision前 | `run.metrics_contract`、`control/contract.toml`、resolved laneと元lane path/SHA-256を含む`control/protocol.toml` | contract ID/content hash、exact producer version、view compatibility、mixed contract禁止、family/列の`applicable_run_variants` | required family欠損なら`collection_complete=false`。非適用列はNULL |
| source track / control plane | Gitのscore reference/profile base、実checkout head、purpose/qualification arm、protocol列挙control fileの実blob | `runner.manifest` / provision前、publisher / archive後 | compact source-track projection。実control-plane JCS hashはregular manifestまたはqualification parent manifest/allocation projectionが一度所有。qualification ONだけ`control/source-track-diff.json` | base/head規則、ON diff path集合=score bundle paths。control-plane filesは実checkoutからprotocol path順にhashし、protocol/contextの配列hashへ照合。childへ配列/hashを複写せずparent join | jobを投入しない |
| instrumentation bundle | Git管理purpose-neutral semantic bundle、SIF sidecar | `runner.manifest` / submit前 | `control/instrumentation-bundle.json`とsubject manifestのbundle SHA-256。actual argv/sinkはstage/artifact source | hook/event/sample設定、wrapper/node-agent hash、contract、env scope、source path allowlist、relative argv template、sink source ID。sink path templateはcontract source registryだけが所有。v1 hookはbuild-definition repositoryに限定し、hook pathをbundle source pathsとSIF score treeへ含める。referenceはempty。D-001合格後のbenchは通常contractからscore bundleを選ぶ | required source欠損なら`collection_complete=false`。profile-only bundleはbenchでproducerを持たない |
| instrumentation qualification decision | Git管理D-001 decision record | collector実装時に一度だけ、qualification publish後 | 一回限りのD-001 report。run manifestや通常benchへdecision ID/hashを複写しない | 固定5 pairのevidenceとverdictを確認し、最初のadopt後は固定。変更検知、lane/環境/commit/bundle別の再qualificationなし | adopt前はbenchをaudit-only。adopt後は通常run eligibilityへjoinしない |
| scheduler join | `pjsub` argv/stdout/stderr/responseとPJM側job-entry file | `runner.submission` / pjsub前後、`pjm.supervisor` / entry直後、finalizer / terminal後 | regularは`scheduler.submit_job_id`と`scheduler.job_entry_job_id`、qualificationはallocation projection | normalは二scalar、qualificationはallocation ID+ordered child IDsをquery時照合。完全導出できるbinding fileは作らずchildへ共通値を複写しない | reject、応答不明、片方欠損、不一致はattempt/comparison不適格。自動復旧しない |
| producer completion | producerのnormal close、node-agent summary、StageRunnerのchild outcome | supervisor/node-agent/StageRunner / return後 | 完成producer fragment、stage termination | immutable contract/bundleからexpected fragment/fileを一回導出し`collection_complete`を決める。merge後fragmentは削除 | 欠損値はNULL。partial/tmpは手元に残り得るがartifact/mergeへ入れない |
| protocol assignment | Git管理protocolとrunnerの割当record | `runner.manifest` / job投入前 | regularはplanned-treatmentのblock ID/order、replicate index、planned group/set ID、dimensions。qualification childはtop-level pair/arm、planned set/dimensionsだけ | closed namespace+purpose+dimensionsからgroup/set IDを再計算。qualificationのblock ID/order/replicateはallocation ID、parent/set child offset、protocol pair offsetから導出しchildへ複写しない | purpose別comparison不適格。attempted groupやpairを観測値・順序から推測しない |
| qualification allocation | Git管理paired scheduleとrunnerが一度作るparent manifest | `runner.manifest` / pair submit前、`pjm.supervisor` / entry、`runner.postrun` / terminal後 | namespace、allocation ID、submission/job-entryの二job ID、scheduler request/accounting/terminal、observed control-plane hash、nullable job-entry env、minimal hardware、ordered workload、pair/ordered child IDs、OFF/ON order | parent/set/projectionのordered IDs、job-entryは実在時だけ照合。protocol pair arm order/offset、child arm/parent hash、parent↔child namespace/lane、OFF/ON bundle、request↔parentを一回照合。hardware/workloadはparent relationが一度所有 | arm間crash/未起動/未完成はsetのchild artifactをNULLにしてpair不適格。片armを別allocationへ再pairしない |
| Qwen/SGLang source | SIF build時の二checkoutで各一回実行する`git rev-parse --show-object-format`と`git rev-parse HEAD` | `build.packager` / source checkout直後 | `source/sif-sidecar.json`のordered source commits、`qwen_sif_provenance_raw_v2` | fixed repository ID、`sha1|sha256`、algorithm別40/64-hex full OID、embedded valueとの一致 | SIFを公開しない |
| SIF/toolchain | digest-pinned `From:`が消費したOCI manifest/config/layer、完成SIF、score source tree、embedded manifest、role付き実参照runtime file | `build.packager` / SIF完成・test後、publisher / archive後 | `source/sif-sidecar.json`、`qwen_sif_provenance_raw_v2`、artifact identity/URI索引 | 完成SIF hash、score bundle pathごとのabsent/mode/blob state、embedded identity、python/SGLang/CUDA/NCCL role別realpath/hashをseal。ON diff new stateとも照合 | 欠損値はNULL。正常build/reuse pathは続行しない |
| runtime MPI/container launcher | 実run entryの`mpirun`/`singularity` resolved realpath、byte hash、absolute-path `--version` | `pjm.supervisor` / launch前、publisher / archive後 | regular `source/runtime-launchers.json`、qualification parent同member、`qwen_runtime_launchers_raw_v2` | role順exact 2、全outer MPI/container launchは検査済みabsolute path。D-001 pair内ではsemantic hashを照合し、通常benchは各run自身の値を保存 | 欠損値はNULL。正常launchは続行しない |
| model/tokenizer/template | full Hub revisionまたは一度だけ作るper-shard hash付きprebuilt record、tokenizer/template allowlist | `provision.verifier` / node copy前 | regularは`source/model-manifest.json`と`artifact.model.manifest_sha256`、qualificationはparent manifest roleからchildへjoin | schemaのexact allowlist。prebuilt shard path/size/hashは同じsource recordだけが所有し、導出manifest IDを複写しない。400 GB本文はrunごとに再hashしない | 欠損値はNULL。正常provision pathは続行しない |
| dataset | tracked protocolがpinしたfull repository revisionとdataset file | `provision.verifier` / `hf download --revision <full-revision>`後 | regularは`source/dataset-manifest.json`と`artifact.dataset.manifest_sha256`、qualificationはparent manifest roleからchildへjoin | repository/full revision、relative path、size、SHA-256をshared pathと独立に照合。moving `main`禁止 | provision失敗 |
| full job-entry environment | `os.environb`相当のbyte environment | `pjm.supervisor` / script entry直後、module/unset前 | regularは`env.job_entry.*`と`source/env/job-entry.snapshot`、qualificationはparent snapshotとallocation env projection | byte-exact name escapeと全value Base64。child rawへ複写せずEnvironmentがblock/allocation join | 欠損値はNULL、requiredなら`collection_complete=false` |
| full host exec environment | node wrapperのbyte environment | `node.wrapper` / module/cache staging後、container exec直前 | `env.host.node.<node_rank>.*`と`source/env/host-node-<node-rank>.snapshot` | byte-exact、expected node全件、raw scope保持 | 欠損値はNULL |
| full SGLang launch environment（score bundle） | container内のbyte environment | bench/profile/qualification ONの`sglang.engine` / `_set_envs_and_config`等の解決後、scheduler launch直前 | `sglang.env.node.<node_rank>.*`と`source/env/sglang-node-<node-rank>.snapshot` | `--cleanenv`後かつSGLang mutation後の全件、raw scope保持 | 欠損値とqualification OFF consumerはNULL |
| full worker environment（score bundle） | scheduler processのbyte environment | bench/profile/qualification ONの`sglang.worker_ready` / ready通知直前 | `sglang.worker.env.rank.<global_rank>.*`と`source/env/worker-rank-<rank>.snapshot` | byte-exact、global rank `0..lane.tensor_parallel_size-1`、device event一致 | 欠損値とqualification OFF consumerはNULL。requiredなら`collection_complete=false` |
| scheduler request | Git管理lane、PJM script identity、runnerが実行するexact `pjsub` argv | `runner.submission` / submit直前 | regularは`run`の`scheduler.request`、qualificationはallocation artifact/raw projection | positive nodes/MPI processes、resource group、override照合。v1は一node一MPI process。child rawへ複写しない | 欠損値はNULL。invalidなら投入しない |
| scheduler accounting | `pjstat --history -S --choose jid,st,sdt,elp,ec,sn,pc,nnumu,mmszu,uctmut,sctmut --data --delimiter "\t" <job>` | `runner.postrun` / terminal state後 | regularは`scheduler.accounting`と`source/scheduler/pjstat.{stdout,stderr}`/`pjstat-version.txt`、qualificationはparent raw/allocation projection | field/unit/job ID固定、requestとの一致はview。childはallocation IDでjoin | 取得できないfieldはNULL。host/rankを捏造しない |
| minimal CPU hardware/settings | allocated nodeごとの`LC_ALL=C lscpu --json`とkernel/cpufreq/boost/NUMA/memory設定 | node wrapper。regularはcontainer前、qualificationはparentがarm 1前に一度。publisher / archive後 | `source/hardware/node-<node-rank>.json`とsubject-keyed `qwen_minimal_hardware_raw_v2` | CPU identityに加えkernel、online/isolated CPU、memory/hugepage、NUMA balancing、boost、policy別governor/driver/min/max。regular runまたはallocationが一度所有しchild rawへ複写しない | 欠損値はNULL、requiredなら`collection_complete=false` |
| observed node topology/launch | 同じMPI launch内のOMPI rank/size/local rank、hostname、node wrapper/container argv、host affinity | `node.wrapper` / container exec直前、publisher / archive後 | `source/topology/node-<node-rank>.json`と`qwen_node_launch_raw_v2` | rank `0..lane.nodes-1`、`lane.nodes` distinct hosts、一rank一node。StageRunnerはouter MPI argv、node launch relationはnode wrapper/Singularity/SGLang argvを所有 | 欠損値はNULL |
| lifecycle control | lane contractのtransport/address source/port strategy、rank 0 node-agent endpoint、startup ready、shutdown start | `node-agent coordinator` / container開始前とgeneration後、publisher / archive後 | `control/lifecycle/rendezvous.json`、最終`summary.json`、`qwen_lifecycle_summaries_raw_v2` | run/version/freshness、全nonzero node rankのexplicit endpoint接続、expected node identity ACK、shutdown一意性。env/DNS/任意探索禁止 | 欠損値はNULL。ready不成立ならcontainerを開始せずnonzero終了 |
| GPU topology/settings | host/containerのfixed `nvidia-smi --query-gpu`とhost `nvidia-smi topo -m` | `node-agent` / model load前に一度、publisher / archive後 | node別host/container query stdout/stderr、host topology matrix stdout/stderr、node topology JSON、`qwen_topology_raw_v2` | UUID/PCI/memory/driverに加えpersistence、compute mode、power limit、max clocks、MIG/ECC。各node `lane.gpus_per_node`、全体 `lane.tensor_parallel_size` unique UUID。parsed copyをsource JSONへ作らない | 欠損値はNULL |
| node CPU/NUMA detail（profile only） | minimal hardwareに含まれない各NUMA nodeの`cpulist`/`meminfo`とpost-run `/proc/meminfo` | profile `node.auditor` / benchmark後、各node、publisher / archive後 | `profile/raw/hardware/node-<node-rank>/{proc-meminfo.txt,numa/...,capture.json}`と`qwen_profile_hardware_raw_v2` | hostname付きcaptureをminimal hardwareの`(node_rank,hostname)`と照合。meminfo key/kB→bytes、cpulist canonicalization、exact membership、source member/hashを固定 | 欠損値はNULL。bench consumerもNULL |
| Torch profiler trace（profile only） | SGLang built-in legacy PyTorch ProfilerのCPU/GPU Chrome trace | scheduler / warmup後、measured generation先頭5 forward。publisher / archive後 | global TP rank別`profile/raw/torch/qwen-profile-TP-<rank>.trace.json.gz`とcapture、`qwen_profiler_events_raw_v2` | lane TP rank集合、固定profile ID/path、stack/shape/merge無効。complete duration eventをcategory/nameでcount・duration sumへ一度集約 | trace欠損はNULL/long rowなしと既存`collection_complete=false`だけ。bench/qualificationは非適用 |
| outer actual launch | stage wrapperが`subprocess.Popen`へ渡すargv/cwd/stderr mode | `StageRunner` / spawn直前 | `run.stage.benchmark.launch` | shell表示ではなくclosed object。profiler wrapperを含む実spawn値 | stage未開始ならNULL |
| run timing/outcome | supervisorのUTC clock、monotonic clock、最終exit | `pjm.supervisor` / regularはPJM entry→compute回収終了、qualificationは各child orchestration直前→そのchild回収終了 | `run.started_at`、`run.finished_at`、`run.elapsed_sec`、`run.exit_code` | wall/monotonicを別clockで保持。`run_outcome`はexit codeからview導出。allocation共通のscheduler request/accounting/terminalは親projだけが所有し、別allocation lifecycleは作らない | 保存できたterminationとstage outcomeから中断を表す |
| stage outcome | supervisorの同じmonotonic clockとraw child wait status | `StageRunner` / spawn直前とwait直後 | `perf.stage.benchmark.wall_time_sec`、`run.stage.benchmark.termination` | exit/signal/spawn_errorを区別し、publish/GPU集計をtimer外へ | termination objectで中断を表現 |
| cache input | shared cacheのnode-local rsync完了直後のcanonical member manifest | `node.wrapper` / monitor/container開始前 | `cache.node.<node_rank>.input_manifest_sha256`、`source/cache/node-<node-rank>-input-manifest.json` | relative path byte順。fileはtype/path/size/SHA-256、symlinkはtype/path/target bytes。mtime/uidは除外 | 欠損値はNULL、requiredなら`collection_complete=false` |
| BenchArgs（score bundle） | `BenchArgs.from_cli_args(args)`が返すdataclass | bench/profile/qualification ONの各node `sglang.cli` / CLI parse直後、`Engine()`前 | node eventの`config-snapshot/v1` full typed field array、consumer `qwen_node_configs_v2` | event type `bench_args_effective`、definition source blob、default込み全field、expected node全件の一致をview検証 | 欠損値とqualification OFF consumerはNULL。requiredなら`collection_complete=false` |
| launch-effective ServerArgs（score bundle） | `_set_envs_and_config`、`check_server_args`、auto parser解決後の`ServerArgs` | bench/profile/qualification ONの各node `sglang.engine` / scheduler/daemon launch直前 | node eventの`config-snapshot/v1` full typed field array、consumer `qwen_node_configs_v2` | event type `server_args_effective`、definition source blob、TP/nnodes/node_rank/dist address invariant | 欠損値とqualification OFF consumerはNULL。requiredなら`collection_complete=false` |
| runtime backend/transport（score bundle） | `run_scheduler_process()`で`Scheduler(...)`構築後、`pipe_writer.send(scheduler.get_init_info())`直前のactual state | bench/profile/qualification ONの`sglang.worker_ready` / global workerごと | `parallel.worker.<global_rank>.runtime` | global rank `0..lane.tensor_parallel_size-1`、node rank、GPU ID/UUID、affinity、actual attention/communication backend | 欠損値とqualification OFF consumerはNULL。requiredなら`collection_complete=false` |
| qualification common workload | ShareGPT selection後のmeasured/warmup ordered requestsとactual sampling params | allocation parent neutral harness / arm 1前、publisher / archive後 | parent `source/workload-{measured,warmup}.json`、subject=allocationのworkload manifest/request relations | OFF/ONが同じmanifest byteを順番通り消費し、neutral qualification resultと照合。child/allocation raw rowへpayloadを複写しない | 欠損値はNULL、pair不適格 |
| measured workload | ShareGPT shuffle/filter/chat-template後のordered `DatasetRow`とsampling params | regular bench/profileの`rank0.offline` / generate前、qualificationは上記parent、publisher / archive後 | `source/workload-measured.json`、manifest hash、`qwen_workload_manifests_raw_v2`、`qwen_workload_requests_raw_v2` | dataset manifest+元array index、実prompt text+observed input token countまたはtoken-ID payload、requested output/sampling/order。source member/hashはmanifest relationだけが所有 | 欠損値はNULL。requiredなら`collection_complete=false` |
| warmup workload | random/dataset warmup request作成後のordered requestsとsampling params | regular bench/profileの`rank0.offline` / warmup generate前、qualificationは上記parent、publisher / archive後 | `source/workload-warmup.json`、`qwen_workload_manifests_raw_v2`、`qwen_workload_requests_raw_v2` | top-level modeはmanifest relationだけ、request payloadはrequest relationだけが所有。measured manifestへ混ぜない | 欠損値はNULL。requiredなら`collection_complete=false` |
| input prepare phase | tokenizer取得、dataset load/sample/tokenizeのmonotonic boundaries | `rank0.offline` / request object完成まで | `perf.phase.input_prepare.wall_time_sec` | 同一clock。manifest hash/writeはtimer終了後、generate前 | 欠損値はNULL |
| distributed init phase | rank 0 processの`Engine(...)`呼出し | `rank0.offline` / constructor直前からreturn直後 | `perf.phase.distributed_engine_init.wall_time_sec` | rank 0同一monotonic clockだけ。worker-readyはcardinality検証専用 | 欠損値はNULL |
| warmup generation | warmupの`backend.generate`直前・return直後 | `rank0.offline`、profileだけ追加control forwarder / warmup中 | `perf.phase.warmup_generation.wall_time_sec`、profileだけphase summary | rank0同一clock。profileでは別invocationとexpected node start/end/ACKも検証 | 欠損値とbenchのprofile summary列はNULL |
| measured generation | measured `backend.generate`直前・return直後 | `rank0.offline`がlocal timer値をauthoritative resultへ、runner parserがterminal後にFact化 | `perf.benchmark.generation.wall_time_sec`、profileだけphase summary | rank0 `time.perf_counter`、正/有限。profileでは同じinvocation IDとforward lead/trailも検証 | 欠損値とbenchのprofile summary列はNULL |
| observed responses | decode済み`gen_out` | bench/profile/qualification ONの`PrimaryResultWriter` / generate return直後、runner parser / terminal後 | `source/primary-result.json`からreturned/error count、prompt/completion totals、ordered output token-ID SHA-256 | schema固定。response順の`output_ids` array全体を一度hashし、各lengthをcompletion tokens、length sumをobserved output totalへ照合する。本文/per-request hashを作らない。malformed responseはerrorへ数え、取れないtokenを0で補わない。temporary write/close後にrename | 欠損値はNULL。qualification OFFもNULL |
| neutral qualification result | 共通benchmark harnessが検証した実response array | qualification OFF/ONのrank0 `QualificationResultWriter` / generate return直後 | `source/qualification-result.json`とpair estimand point | `qualification-result/v1`のactual returned/error/token totals、ordered output token-ID SHA-256、wall、parent workload request/token整合。ON primaryとはinvocation/result identity/wallを照合。予定件数・throughput・verdictは保存しない | 欠損値はNULL。normal bench/profileもNULL |
| primary outcome | purpose別authoritative resultとgeneration duration | BigQuery view / query時 | throughput派生値はrawへ重複保存しない | bench/profileはprimary artifact、qualification OFF/ONは共通neutral resultからthroughputを導出。ON primaryとの重複はwriter検証として明示 | source不足ならNULL |
| node process lifecycle | wrapper child start/end、raw wait statusまたはspawn error | `node_agent.py`のchild supervisor / child境界と`os.waitpid` | `parallel.node.<node_rank>.process_lifecycle`、nonzero nodeはconsumer `qwen_participant_terminations_v2` | exit/signal/kindやshutdown受信をrawへ複写しない。同じ`<boot_id>:CLOCK_MONOTONIC`内だけshutdown前後をnode別long viewで導出 | source不足なら当該consumer rowなし |
| shutdown phase | shutdown開始・全child終了のrank0 monotonic boundaries | `rank0.offline` / shutdown前後 | `perf.phase.engine_shutdown.wall_time_sec`、shutdown invocation ID | node lifecycleと照合 | 欠損値はNULL |
| raw GPU telemetry（profile only） | node-local periodic `nvidia-smi` sample | profile `node-agent` / process実行中にnode別JSONLへbuffered append、publisher / archive後 | `telemetry/node-<node-rank>/gpu-samples.jsonl`と`qwen_gpu_samples_raw_v2` | 1.0秒間隔、monotonic/wall anchor/UUID/index/nullable values。sequenceはnode poll-frame ordinalで、expected UUIDが一件ずつ揃うcomplete frameだけをprojection。publisherがsource member/hash付きで転記し、poll status/reason fieldなし | 読めるcomplete frameだけ保存。欠損値とbench consumerはNULL |
| GPU phase markers（profile only） | persistent coordinatorからのcontrol message | profile `node-agent` / warmup・generation message受信時、publisher / archive後 | `control/phase/summary.json`と`qwen_phase_summaries_raw_v2` | expected node `0..lane.nodes-1`、node-local clock ID、start/end sequence、local `[start,end)`に加え、rank 0同一clockのbroadcast/全ACK受領/generation境界6時刻の順序。別markerと混在禁止 | 欠損値とbench consumerはNULL |
| GPU aggregate（profile only） | deduped raw sampleとphase summary projection | BigQuery `gpu-phase/v1` reducer/view / query時 | raw keyへ保存せず、phase/node/GPUとfleet consumerだけ | same-node clock ID、`[start,end)`交差、2 interval gap cutoff、duration-weighted mean/nearest-rank p95/step energy。complete booleanなし | 欠損値とbench consumerはNULL |
| artifact identity | localで作ったrun bundle、qualification allocation archive/set manifest、profileだけのraw telemetry/trace tar、SIF sidecar | `runner.packager` / GitHub Artifact upload前 | bundle/artifact/set manifests | OpenFOAM基準の外側pathでallocation/childを分離。profile rawは同じlogical artifact setの別outer file。同じbyteを重複しない | uploadできなければworkflow failure。consumer値はNULL |
| archive/publish result | GitHub Artifact upload結果、SCP結果、BigQuery response | workflow archive job/`Publisher` / upload・転送・送信後 | workflow logとData Status | SCP成功後だけGitHub Artifactを削除。失敗はArtifactを残してjob failure | scientific familyへ理由を伝播せず、run-level状態だけにする |

## SGLang hookの正確な位置

### configuration

- CLI層では`python/sglang/benchmark/offline_throughput.py`の`ServerArgs.from_cli_args`と
  `BenchArgs.from_cli_args`がrequested/default込みobjectを作る。各nodeが`Engine()`前にnode-scoped
  full BenchArgsを出す。
- launch-effective `ServerArgs`はそれより後である。`python/sglang/srt/entrypoints/engine.py`の
  `_launch_subprocesses`で`_set_envs_and_config`、`check_server_args`、auto parser resolutionを終え、
  daemon/scheduler processをlaunchする直前にemitする。
- non-zero nodeは`Engine()` constructor後に戻るとは限らず、worker-ready待ちへ入る。したがって
  `throughput_test()`のconstructor後を全node configuration producerにしない。
- actual communication backendやdevice/rank stateはlaunch-effective argsから推測せず、scheduler/
  worker初期化後のready eventとして別にemitする。現commitでは
  `python/sglang/srt/managers/scheduler.py::run_scheduler_process`で`Scheduler(...)`構築後、
  `pipe_writer.send(scheduler.get_init_info())`直前を境界にし、global scheduler rank
  `0..lane.tensor_parallel_size-1`のeventを要求する。

対象SGLang versionごとにsymbolとfull commitをregistry source欄へ固定する。line番号だけを契約に
しない。

### workloadとresult

- ShareGPTは`benchmark/datasets/sharegpt.py`でshuffle後、chat template適用、tokenize、length filterを
  行って`DatasetRow`を作る。現状recordにsource row IDとtoken IDがないため、この地点でsource IDと
  exact prompt payloadを保持できるようにする。現行はtextであり、token-ID入力へ勝手に変更しない。source IDは
  dataset manifest+元JSON array indexとする。
- sampling paramsは`offline_throughput.py`の`throughput_test_once`で`DatasetRow`から構成される。
  manifestはこの構成後、`backend.generate`前にsealする。
- generation timerは同関数の`time.perf_counter()`から`backend.generate` returnまでを使う。
- 現在の`successful_requests = len(reqs)`は予定件数であって成功実測ではない。実際の返却配列を
  decodeし、returned count、malformed/error count、response meta_infoのprompt/completion tokensを作る。
- normal bench/profileのprimary resultは唯一のartifactとしてtemporary write/close後にrenameする。
  後続の`get_server_info()`、profile finalize、shutdown失敗で失ってはならない。eventとartifactを
  二重正本にしない。qualificationは同じresponse validationを持つcommon harnessがneutral resultを作り、
  ON primaryとの一致でbench採用予定writerを検証する。

## 現実装からの変更点

| 現在の箇所 | 問題 | v2での境界 |
|---|---|---|
| `qwen-experiment-metrics/.../core.py::make_run` | 各呼出しでrun IDを再構成する | runnerがmanifestを一度作り全producerは読むだけ |
| `core.py::send` | compute側packageがBigQuery credential/retryを所有する | `publish.py`へ隔離しPJM bundle/環境からcredentialを除去 |
| `stage.py::EventReceiver` | 共有socketを全nodeへbindし、node rankなしでrun-scoped keyを直接sendする | node-local endpoint -> node-local Recorder、node/rank scopeを保持 |
| `stage.py::run_stage` | receiverのBigQuery完了待ちと集計をstage elapsedへ含める | child waitでtimerを閉じ、merge/publishはrunnerへ移す |
| `stage.py::fallback_result` | missing/Noneを空文字化し、予定`successful_requests`も受け入れる | primary-result artifactのpure parser。欠損keyは出さずconsumerをNULLにする |
| `stage.py::GpuSampler` | memoryへ蓄積し終了時集計だけを送る | raw node JSONLへbuffered append/periodic flushし、後処理で集計 |
| `scripts/run-qwen35-node.sh` | 全nodeが同じ共有socket pathをbindし、cache manifestがない | node-local endpointとstaging直後cache manifest |
| 同node script | node別実argvをartifact/metricへ残さない | Singularity exec直前の配列をnode producerが記録 |
| `scripts/run-qwen35.pjm` | credentialをPJMへ持ち込み、outer commandはshell表示だけ | runner-only credential、spawn配列を`StageRunner`が取得 |
| `.github/workflows/genkai.yml` | credentialを`pjsub -x`で渡し、必要fileのupload前にruntimeをcleanupし得る | variant別expected fileをGitHub Artifactへuploadできた後にcleanup |
| `scripts/build-sglang-sif.sh` | base image tag、部分的provenance、完成SIF sidecarなし | digest-pinned OCI、dependency manifest、SIF rename後sidecar commit marker |
| `offline_throughput.py` | 予定件数をsuccessfulと命名し、primary result作成が後続処理より遅い | return decode/検証後すぐprimary resultをwrite/renameし、throughputはview派生 |
| `sharegpt.py`の`DatasetRow` | source ID/token identityを失う | selection時にordered workload manifest用identityを保持 |

## laneとcontrol-forwarder契約

収集器とcontractは任意のpositive `nodes`、`mpi_processes`、`gpus_per_node`、`tensor_parallel_size`をlaneから
展開する。v1は一node一MPI wrapperと全GPU tensor parallelを対象にするため、
`mpi_processes = nodes`かつ`tensor_parallel_size = nodes * gpus_per_node`を要求する。expected node rankは
`0..nodes-1`、expected global scheduler rankは`0..tensor_parallel_size-1`である。現在の実機lane
`experiments/lanes/qwen-genkai-2node-tp8.toml`は2 PJM nodes、2 MPI node processes、4 GPU/node、TP=8、
rank 0 `hostname`をadvertise source、SGLang port `25000`、control port `25001`、rendezvous timeout 60秒、
ready timeout 120秒を固定する。required producerはcontractのvariant applicability、instrumentation modeは
protocol/bundle selectionだけが所有する。将来の1/4-node laneはモデル適合性を
確認したうえでlane追加だけで表現し、collectorを分岐しない。port衝突/timeoutはそのrunを失敗とし、自動retry、
動的allocator、fallback discoveryを作らない。runnerはlane、PJM script identity、exact `pjsub` argvを
`scheduler.request`へsealし、job内のOMPI state、worker-ready、GPU UUID、accounting `nnumu`と照合する。

全variantで各node wrapperがcommon lifecycle node-agentを起動する。bench/profile/qualification ONだけが
score config producerを有効にし、qualification OFFはwrapper topologyとcommon lifecycleだけを持つ。rank 0 coordinatorはlane規定sourceから
得たaddressへbindして`control/lifecycle`のrun-bound rendezvousをsealし、全nonzero node rankのparticipantがその明示endpointへ
接続する。coordinatorは全expected nodeのidentity付きready ACKを確認してからcontainerを始め、
generation後は`lifecycle.shutdown.start`を一度だけ送る。profile runだけがさらにGPU sampler/phase receiverを
起動し、benchではGPU sample/phase messageを作らない。
container内rank 0 offline hookはbind済みlocal endpointへphase messageを送り、coordinatorが同じ
messageを全expected node receiverへbroadcastする。messageはrun、invocation ID、warmup/measured、start/end、sequenceを
持ち、sender、node受信時刻、ACK時刻を最終profile summaryへ残す。start/end間では新規接続、DNS、filesystem pollを
行わない。nonzero node rankのoffline processはphase messageを送らない。別nodeのmonotonic timestampは差し引かず、
GPU aggregateは各node自身のobservation windowを使う。

profileの各generateは`start broadcast -> all expected node ACK -> rank0 timer start -> backend.generate ->
timer end -> end broadcast -> all expected node ACK`の順にする。rank0同一clockでACK-to-generate leadと
generate-to-ACK trailを記録する。shutdownはphase forwarderでなく共通lifecycle controlで送り、
全expected nodeの通常ACK後に`backend.shutdown()`を呼ぶ。lifecycle reducerが最終summary内のshutdown
invocationの一意性を検証し、各nonzero node rankのearly/coordinated判定は同じclock IDのnode-local受信時刻と
child exitをviewで比較する。

## validation fixture

最低限、次のfixtureをGit管理する。

- CLI指定、default、environment override、auto-resolutionが異なるServerArgs
- synthetic 1/2/4-node laneのcardinality展開、expected node full configの一致/不一致、expected global worker rankのmissing/duplicate/wrong-rank event
- wrong MPI node/size/local rank、worker node/TP rank重複、worker GPU ID/UUIDとcontainer topologyの不一致
- 1-node participant relation 0行、4-node rank 1/2/3のearly/coordinated/spawn error、任意participant source欠損時に他nodeの有効rowを保持
- token総数は同じだがrequest順序またはlength分布が違うworkload
- dataset row/chat template/tokenizerが1 byteだけ違うworkload
- returned count不足/過多、malformed response、prompt/completion token欠損、output IDs欠損/順序差、generate例外、NaN/zero duration
- warmup成功/measured失敗、generate成功/get_server_info失敗、nonzero node rankのcoordinated kill/early death
- cache count/sizeは同じでcontentが違う二manifest
- variant applicability grid：profile-only collectorのprofile失敗、bench非適用、qualification OFF非適用はいずれもconsumer値がNULL。required source欠損時だけrun-level `collection_complete=false`
- compact source trackのbench/profile/qualification OFF/ON、base非ancestor、score bundle source path外差分、control-plane blob差、checkout tree不一致、同bundleでbase/workload/runtimeだけ変更、moving branch名が同じでもcommitが異なるfixture
- qualification allocationの一pair二child、二pair混入を全pair reject、child順/arm不一致、別allocationへの再pair、arm間crash、片child未起動/metrics未完成fixture
- bench lifecycle ready/shutdownだけ、profile lifecycle+phase、benchへphase summaryを誤生成するfixture
- lifecycle/phase ACKのnode-local timestamp逆転とclock不一致
- instrumentation bundleのhook/sample interval一つだけを変えたhash差fixture
- D-004の1/2/4-node synthetic laneでTP rank trace集合、固定path/profile ID、stack/shape/merge無効、trace一件欠損時のNULL/0-row fixture
- 一回限りのD-001でinconclusive union、対象pair不一致、共通Eとmetric別Eの差、取得済みarm/result false、片arm orderだけ残る欠測、0.99/1.01境界、evaluable=4の連続median、semantic identity conflict、output identity不一致、adopt完全一致、adopt後の通常benchがdecision joinなしでeligibleになるfixture
- GPU sampleのsequence gap、unfinished frame、poll error、marker欠損、pre-start sample、既知p95/energy系列
- 正常close、program failure、完成node fragment欠損、異値conflict。途中fileはmerge/artifactへ入れない
- contract欠損legacy、expected producer version混在、exact v2 contract
- OpenFOAM方式を移植したhappy path、GitHub upload失敗、SCP失敗

全fixtureで、node/rank full snapshotのscope保持、raw row集合、`collection_complete`、comparison
eligibility、result artifactとの代数的parityを検査する。

## 確定事項

bench/profile分離と、取得できなかった値およびprofile-only metricは値をNULLにするだけで原因/status/欠損用recordを
作らないこと、極稀な失敗を救う多段safeguardは作らないことはユーザー確定済みである。
OpenFOAMとQwenのD-003は2026-08-09にlane-driven multi-node収集で確定した。Qwenの現在の実機conformanceは
モデルが適合する2-node TP=8 laneで行い、synthetic 1/2/4-node fixtureでcardinality非固定を検証する。
D-004はSGLang legacy PyTorch ProfilerのCPU+GPU、warmup外の先頭5 scheduler forward、stack/shape/mergeなし、
global TP rank別traceで確定した。D-001は固定5 pair、最低3評価可能pair、throughput/stage wall 1%以内の簡素screenを
collector実装時に一度だけ行うことで確定した。GPU polling/aggregate、phase control marker、profiler、verbose backend trace、
full placement診断はprofile-onlyとする。最初のadoptまではinline eventを含むbenchをaudit-onlyにし、adopt後は結果を
決め打ちして通常benchへdecision ID/hashを持たせず再qualificationしない。
