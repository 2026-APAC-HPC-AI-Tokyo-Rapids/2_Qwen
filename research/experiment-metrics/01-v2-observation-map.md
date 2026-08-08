# Qwen/SGLang v2 観測マップ

## この文書の役割

この文書は、各factについて「どこにある値を、誰が、いつ取得するか」を固定する実装契約で
ある。[概要計画](./00-v2-plan.md)が情報モデルと比較規則を定め、本書が実コード・command・
artifactへの配線を定める。

key registryへ入れるには、次の六項目を全て埋めなければならない。

1. authoritative source: object、structured file、command、またはsource symbol
2. producer: そのsourceを一度だけ読むowner
3. capture moment: 値が確定し、まだ失われていない時点
4. raw destination: `run/key/value`または耐久artifact
5. validation: 誤読、scope混同、部分取得を検出する規則
6. absence semantics: 非該当、未到達、取得失敗のどれか

sourceの優先順位は、native object/API、machine-readable command/file、専用structured marker、
stable log parser、汎用text regexの順とする。下位sourceを使う場合はfixtureと上位sourceの照合を
必須にする。CLI defaultや入力fileに書いた値を、実効値や実測結果として扱ってはならない。

## 値の段階とscope

| 段階 | 意味 | Qwen/SGLangの例 |
|---|---|---|
| requested | Git管理script/protocolが指定した値 | `--tp-size 8`、32 prompts |
| launch-effective | default、environment、auto resolutionを反映し、worker launchへ渡す直前の値 | resolved `BenchArgs`、resolved `ServerArgs` |
| runtime-observed | worker/backend初期化後に実際に成立した値 | rank、GPU UUID、communication backend、worker-ready state |
| result-observed | 実responseを検査して得た値 | returned count、completion tokens、generation duration |

node scopeも分ける。rank 0の値をrank 1へ複製せず、まずnode別full snapshotを保存する。merge時に
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
  all variants: rank 0 writes common lifecycle rendezvous; rank 1 validates and connects; record topology
  wait for expected-node readiness before starting containers
  verify cache staging and write input manifest
  all variants: snapshot exact outer/node argv and common host launch state
  bench/profile/qualification ON: additionally snapshot score config/container state
  qualification OFF: do not start score config/result producers; retain common lifecycle/topology/native result only
  in profile only, append raw GPU samples and local phase markers
  flush/close all node evidence before the PJM script exits

SGLang score hook (bench/profile/qualification ON, each node)
  emit launch-effective ServerArgs before scheduler launch
  emit runtime worker-ready facts after initialization

rank-0 score result hook (bench/profile/qualification ON)
  persist warmup/measured workload manifests
  write the sole authoritative primary-result artifact through temporary file and rename
  in profile only, emit distributed GPU phase boundaries

rank-0 native qualification writer (qualification OFF/ON)
  write the common native result used only by the D-001 estimand

Actions runner
  after PJM terminal state, parse/reduce saved source best effort after timeout/failure
  verify expected node and global-worker scoped facts without scope reduction
  regular: merge -> validate expected files -> GitHub Artifact upload -> publish
  qualification: package parent evidence and both child source/log/metrics in one artifact set
```

`run-manifest.json`で生成したrun IDを全producerが読む。各processで再計算しない。node間で共有
Unix domain socketをtransportにしない。socketを使うならendpointとJSONL writerは各node localに
置き、node rankをevent envelopeで必須にする。target hookはlocal structured eventを書くだけで、
metrics key、BigQuery、retryを知らない。

## fact別の取得契約

表の「欠損」はraw valueへ空文字を入れる意味ではない。keyを出さず、post-runのexpected-file検証と
`collection-status.json`のcompleteness reasonで説明する。
これはapplicableかつexpectedなfamilyだけの規則である。bench/qualificationでprofile-onlyとなる列は
consumerでNULLを返すだけで、raw key、family status、Evidence、reasonを作らない。

| fact family | authoritative source | producer / capture moment | raw keyまたはartifact | validation | 欠損時の扱い |
|---|---|---|---|---|---|
| run/subject identity | runnerが一度生成したsubject/child manifests | `runner.manifest` / provision前 | raw tableの`run`列、run/allocation manifest artifact | regular subject ID=run ID、qualification subject ID=allocation ID、ordered child/arm、全fragmentのrun一致。run/allocation IDはglobal unique/safe path。PJM job IDを含めない | manifestなしでは投入しない |
| metrics contract | Git管理`qwen-v2.toml`とlane/run variantが選んだexpected producer graph | `runner.manifest` / provision前 | `run.metrics_contract`、contract/protocol artifact | contract ID/content hash、exact producer version、view compatibility、mixed contract禁止、family/列の`applicable_run_variants`、applicable missing policyとinapplicable projection policyの分離 | required familyならcomparison不適格。非適用列はconsumerでNULLだけを返す |
| source track | Gitのscore reference/profile base、実checkout head、purpose/qualification arm | `runner.manifest` / provision前 | tagged-union canonical source-track artifact、content hash、`control/source-track-validation.json` | benchはreference/head、profileはprofile base/head、qualification OFFはhead=score reference、ONはreference ancestorかつcanonical tree diff=SIF/bundleのinstrumentation patch identity。diff algorithm/version/base/head/expected patch/passをsubmit前にsealしquery時Git参照なし。clean HEAD/tree、reachability/source archive、submodule/mode/symlink/LFSも検証。moving branch名をidentityにせずdiff/hashをrawへ保存しない | jobを投入しない |
| instrumentation bundle | Git管理bundle manifest、SIF sidecar、実launch設定 | `runner.manifest` / submit前、`runner.postrun` / run後照合 | canonical manifest artifact。consumer hashはartifact contentから導出 | hook/event/sample設定、wrapper/node-agent executable hash、contract version、env capture scope、SIF patchの列挙と実activation一致。labelだけで同一視しない | benchのrequired bundleならcomparison不適格。profile-only bundleはbenchでproducerを持たない |
| instrumentation qualification decision | Git管理D-001 decision record | `runner.manifest` / bench submit前 | manifest fields、`control/instrumentation-qualification-decision.json`、Runsのdecision hash/status | canonical hash、adopt/reject、qualification protocol、reference/candidate tuple、bundle/SIF/workload/runtime、response canary result、evidence一致。通常bench tupleを再構成して照合 | 未存在/未回答はpendingでaudit-only、reject/異値はrejected/invalidでscore不適格。無計測normal v2 benchは作らない |
| scheduler binding | `pjsub` argv/stdout/stderr/responseとPJM側job-entry file | `runner.submission` / pjsub前後、および`pjm.supervisor` / entry直後 | normal runまたはqualification allocationのsubmission result/binding、`control/job-entry.json` | normalはrun ID、qualificationはallocation ID+ordered child IDsをrunner responseと`PJM_JOBID`へ照合。childへ共通値を複写せずparent join | reject、応答不明、片方欠損、不一致はattempt/comparison不適格。自動復旧しない |
| producer completion | producerのnormal close、node-agent summary、StageRunnerのchild outcome | supervisor/node-agent/StageRunner / return後 | producer fragment、stage termination、post-run collection status | variant別expected producer/file/keyを一度検査 | 欠損はcollection incomplete。失敗runのpartialはbest effort回収 |
| protocol assignment | Git管理protocolとrunnerの割当record | `runner.manifest` / job投入前 | `experiment.block_id`、`experiment.block_order`、`experiment.replicate_index`、`experiment.pair_id`、`experiment.planned_replicate_group_id`、`experiment.planned_comparison_set_id`、qualification armを含むplanned treatment tuple | contract/protocol/policy hash込みcanonical tupleからgroup/set IDを再計算、run purpose、予定/max attempt、stopping rule、lane membership、pair IDの一意性 | purpose別comparison不適格。attempted groupやpairを観測値・順序から推測しない |
| qualification allocation | Git管理paired scheduleとrunnerが一度作るparent manifest | `runner.manifest` / pair submit前、`pjm.supervisor` / entry、`runner.postrun` / terminal後 | allocation ID、scheduler binding/accounting、job-entry env、pair/ordered child IDs、OFF/ON order、common lane/node/GPU binding | 一allocation一pair、二childの独立manifest/log/metrics、child parent ref、block ID=allocation ID、同じartifact set、pair間は別allocation | arm間crash/未起動/未完成は回収できたfileだけ保存しpair不適格。片armを別allocationへ再pairしない |
| SGLang source | SIF build時のcheckout済み`5_SGLang_Src` full commit | `build.packager` / source checkout直後 | SIF build sidecarとsource commit索引 | branch名ではなくfull commit、embedded valueとの一致 | SIFを公開しない |
| SIF/toolchain | digest-pinned `From:`が消費したOCI manifest/config/layer、完成SIF、package/toolchain、実参照binary | `build.packager` / SIF完成・test後 | `source/sif-sidecar.json`、artifact identity/URI索引 | SIF hashは完成後sidecarへ書き、reuse時に実体と一致確認 | reuse禁止 |
| model/tokenizer/template | pinned model manifest、index参照shard、tokenizer files、chat template | `provision.verifier` / node copy前 | `source/model-manifest.json`と`artifact.model.manifest_sha256` | repository/revision、config/index hash、sorted shard path/size、small metadata file hash。400 GB shard本文はrunごとにhashしない | provision失敗 |
| dataset | pinned repository revisionとdataset file | `provision.verifier` / node copy前 | `source/dataset-manifest.json`と`artifact.dataset.manifest_sha256` | repository/revision、relative path、size、SHA-256をshared pathと独立に照合 | provision失敗 |
| full job-entry environment | `os.environb`相当のbyte environment | `pjm.supervisor` / script entry直後、module/unset前 | regularは`env.job_entry.*`と`source/env/job-entry.snapshot`、qualificationはparent snapshotとallocation env projection | byte-exact name escapeと全value Base64。child rawへ複写せずEnvironmentがblock/allocation join | applicable snapshot collector failed |
| full host exec environment | node wrapperのbyte environment | `node.wrapper` / module/cache staging後、container exec直前 | `env.host.node.<node_rank>.*`と`source/env/host-node-<node>.snapshot` | byte-exact、expected node全件、raw scope保持 | configuration incomplete |
| full SGLang launch environment（score bundle） | container内のbyte environment | bench/profile/qualification ONの`sglang.engine` / `_set_envs_and_config`等の解決後、scheduler launch直前 | `sglang.env.node.<node_rank>.*`と`source/env/sglang-node-<node>.snapshot` | `--cleanenv`後かつSGLang mutation後の全件、raw scope保持 | applicable runではconfiguration incomplete。qualification OFF consumerはNULLだけ |
| full worker environment（score bundle） | scheduler processのbyte environment | bench/profile/qualification ONの`sglang.worker_ready` / ready通知直前 | `sglang.worker.env.rank.<global_rank>.*`と`source/env/worker-rank-<rank>.snapshot` | byte-exact、global rank `{0..7}`、device event一致 | applicable runではruntime incomplete。qualification OFF consumerはNULLだけ |
| scheduler request | Git管理lane、PJM script identity、runnerが実行するexact `pjsub` argv | `runner.submission` / submit直前 | regularは`run`の`scheduler.request`、qualificationはparent evidence artifact/allocation projection | 2 nodes、2 MPI processes、resource group、override照合。child rawへ複写しない | invalid protocol assignment |
| scheduler accounting | `pjstat --history -S --choose jid,st,sdt,elp,ec,sn,pc,nnumu,mmszu,uctmut,sctmut --data --delimiter "\t" <job>` | `runner.postrun` / terminal state後 | regularは`scheduler.accounting`と`source/scheduler/pjstat.{stdout,stderr}`/`pjstat-version.txt`、qualificationはparent raw/allocation projection | field/unit/job ID固定、requestとの一致はview。childはallocation IDでjoin | 取得fieldだけ欠損扱い。host/rankを捏造しない |
| observed node topology | 同じMPI launch内のOMPI rank/size/local rank、hostname | `node.wrapper` / container exec直前 | node/rank placement record | rank `{0,1}`、2 distinct host、一rank一node | topology incomplete |
| lifecycle control | lane contractのtransport/address source/port strategy、rank 0 node-agent endpoint、startup ready、shutdown start | `node-agent coordinator` / container開始前とgeneration後 | `control/lifecycle/rendezvous.json`と最終`summary.json` | run/version/freshness、rank 1 explicit endpoint、expected node identity ACK、shutdown一意性。env/DNS/任意探索禁止 | containerを開始せず、またはtermination分類不能でconfiguration/lifecycle failure |
| GPU topology | `nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version --format=csv,noheader,nounits`とcontainer-visible device query | `node-agent` / model load前に一度 | `hardware.gpu.nodes`、raw outputs | 4 GPU/node、8 unique UUID、PCI/logical index mapping | topology incomplete |
| node CPU/NUMA hardware（profile only） | `LC_ALL=C lscpu --json`、`/sys/devices/system/cpu/online`、`/sys/devices/system/node/online`、各nodeの`cpulist`/`meminfo`、`/proc/meminfo` | profile `node.auditor` / benchmark後、各node | `hardware.cpu_numa.nodes`、raw files | observed hostname集合、exact membership/parser fixture | profileではhardware incomplete。bench consumerはNULLだけを返す |
| outer actual argv | stage wrapperが`subprocess.Popen`へ渡すargv配列 | `StageRunner` / spawn直前 | `run.launch_argv` | shell表示ではなく配列、profiler wrapperを含む実spawn値 | stage未開始またはconfiguration incomplete |
| node actual argv | node wrapperの`command`配列 | `node.wrapper` / Singularity exec直前 | `parallel.node.<node_rank>.benchmark_argv` | bind/env/benchmark argvを順序保持、node rankとの整合 | node configuration incomplete |
| run timing/outcome | supervisorのUTC clock、monotonic clock、最終status | `pjm.supervisor` / entry時とcompute回収終了時 | `run.started_at`、`run.finished_at`、`run.elapsed_sec`、outcome/exit | runner archive時刻を混ぜず、wallとmonotonicを別clockとして保持 | 保存できたterminationとstage outcomeから中断を表す |
| stage outcome | supervisorの同じmonotonic clockとraw child wait status | `StageRunner` / spawn直前とwait直後 | `perf.stage.benchmark.wall_time_sec`、`run.stage.benchmark.termination` | exit/signal/spawn_errorを区別し、publish/GPU集計をtimer外へ | termination objectで中断を表現 |
| cache input | shared cacheのnode-local rsync完了直後のcanonical member manifest | `node.wrapper` / monitor/container開始前 | `cache.node.<node_rank>.input_manifest_sha256`、`source/cache/node-<node>-input-manifest.json` | path/type/size/content/order、SIF/model/backend互換dimension | 欠損ならprovenance incomplete |
| BenchArgs（score bundle） | `BenchArgs.from_cli_args(args)`が返すdataclass | bench/profile/qualification ONの各node `sglang.cli` / CLI parse直後、`Engine()`前 | `parallel.node.<node_rank>.sglang.benchmark_arg.*` | default込み、node `{0,1}`のfull object一致をview検証 | applicable runではconfiguration incomplete。qualification OFF consumerはNULLだけ |
| launch-effective ServerArgs（score bundle） | `_set_envs_and_config`、`check_server_args`、auto parser解決後の`ServerArgs` | bench/profile/qualification ONの各node `sglang.engine` / scheduler/daemon launch直前 | `parallel.node.<node_rank>.sglang.server_arg.*` | raw full object、TP/nnodes/node_rank/dist address invariant | applicable runではconfiguration incomplete。qualification OFF consumerはNULLだけ |
| runtime backend/transport（score bundle） | `run_scheduler_process()`で`Scheduler(...)`構築後、`pipe_writer.send(scheduler.get_init_info())`直前のactual state | bench/profile/qualification ONの`sglang.worker_ready` / global workerごと | `parallel.worker.<global_rank>.runtime` | global rank `{0..7}`、node rank、GPU ID/UUID、affinity、actual attention/communication backend | applicable runではtopology/runtime incomplete。qualification OFF consumerはNULLだけ |
| planned qualification workload | Git管理protocolのdataset/filter/seed/argv/sampling policy | `runner.manifest` / qualification submit前 | protocol artifactとplanned workload identity | OFF/ON同一、source-diff gateがselection codeを変更しない。native totalsとも照合 | qualification arm不適格 |
| measured workload（score bundle） | ShareGPT shuffle/filter/chat-template/tokenize後のordered `DatasetRow`とsampling params | bench/profile/qualification ONの`rank0.offline` / generate前 | measured manifest、hash/count/planned totals索引 | dataset manifest+元array index、canonical token-ID array SHA-256、length/output/sampling/order | applicable variantではprovenance/benchmark incomplete。qualification OFFは非適用でNULLだけ |
| warmup workload（score bundle） | random warmup request作成後のordered requestsとsampling params | bench/profile/qualification ONの`rank0.offline` / warmup generate前 | warmup manifest hash/count/planned totals、`workload.warmup.mode` | measured manifestへ混ぜずcomparison fingerprint必須 | applicable variantではwarmup incomplete。qualification OFFは非適用でNULLだけ |
| input prepare phase | tokenizer取得、dataset load/sample/tokenizeのmonotonic boundaries | `rank0.offline` / request object完成まで | `perf.phase.input_prepare.wall_time_sec` | 同一clock。manifest hash/writeはtimer終了後、generate前 | phase incomplete |
| distributed init phase | rank 0 processの`Engine(...)`呼出し | `rank0.offline` / constructor直前からreturn直後 | `perf.phase.distributed_engine_init.wall_time_sec` | rank 0同一monotonic clockだけ。worker-readyはcardinality検証専用 | phase incomplete |
| warmup generation | warmupの`backend.generate`直前・return直後 | `rank0.offline`、profileだけ追加control forwarder / warmup中 | `perf.phase.warmup_generation.wall_time_sec`、profileだけphase summary | rank0同一clock。profileでは別invocationとexpected node start/end/ACKも検証 | rank0 timer欠損はphase incomplete。benchはsummary列NULLだけ |
| measured generation | measured `backend.generate`直前・return直後 | `rank0.offline`がlocal timer値をauthoritative resultへ、runner parserがterminal後にFact化 | `perf.benchmark.generation.wall_time_sec`、profileだけphase summary | rank0 `time.perf_counter`、正/有限。profileでは同じinvocation IDとforward lead/trailも検証 | benchmark invalid。benchはsummary列NULLだけ |
| observed responses | decode/validate済み`gen_out` | bench/profile/qualification ONの`PrimaryResultWriter` / generate return直後、runner parser / terminal後 | primary artifactからreturned/error count、prompt/completion totals | schema固定、`len(gen_out)`、meta tokens、malformed/over-return reason。temporary write/close後にrename。raw key ownerはparserだけ | applicable runではstage nonzero。qualification OFFの列はNULLだけ |
| native qualification result | SGLang標準`result_filename`のnative JSON | qualification OFF/ONのrank0 native writer / generation後、qualification parser / terminal後 | `source/native-qualification-result.json`とpair estimand point | OFF/ON同じschema/parser、正有限native throughput/time、workload identity一致。normal bench/profileでは生成しない | arm eligibility false。normal bench/profile consumerは非適用 |
| primary outcome | purpose別authoritative resultとgeneration duration | BigQuery view / query時 | throughput派生値はrawへ重複保存しない | bench/profileはprimary artifact、qualification OFF/ONは共通native resultからthroughputを導出。ON primaryはreliability照合だけ | applicable source不足ならNULLと理由 |
| qualification response canary | timed pairとは別`analysis_role=response_canary` allocationの両SIF共通canary result | runner-side canary validator / 両child result検証後 | qualification artifact内`response-canary-result.json`、decisionのartifact hash | subject manifest、両child result hash、returned/error、ordered request ID、per-request token、generation token-ID digest、両SIF/harness/workload/result identity。timed pairはprotocolの同一result hashをjoin | tupleをadoptしない。timed pair内で再実行しない |
| internal last-generation stat（profile only） | `get_server_info()["internal_states"]` | profile rank0 result artifact / primary結果完成後 | full result artifactのみ | 一次estimandに使わず、API failureでprimary resultを失わない | bench/qualification consumerはNULLだけ |
| node process lifecycle | wrapper child start/end、raw wait status/signal、termination reason | `node.wrapper` / child境界 | `parallel.node.<node_rank>.process_lifecycle` | shutdown受信を複写しない。lifecycle reducerがsummaryのinvocation/sequence/node一意性を検証後、同一`CLOCK_MONOTONIC` clock IDの時だけshutdown前exit=early、後=coordinatedを導出 | summary欠損、invocation/clock不一致は分類NULLとreason。topology/run incomplete |
| shutdown phase | shutdown開始・全child終了のrank0 monotonic boundaries | `rank0.offline` / shutdown前後 | `perf.phase.engine_shutdown.wall_time_sec`、shutdown invocation ID | node lifecycleと照合 | phase incomplete |
| raw GPU telemetry（profile only） | node-local periodic `nvidia-smi` sampleとpoll error | profile `node-agent` / process実行中にnode別JSONLへbuffered append | node別raw JSONL | 一定間隔と正常終了時にflush、monotonic/wall anchor/UUID/index/value/error/sequence gap | profileでは読めるsampleだけ保存してtelemetry incomplete。bench consumerはNULLだけを返す |
| GPU phase markers（profile only） | persistent coordinatorからのcontrol message | profile `node-agent` / warmup・generation message受信時 | node/invocation/sequence別phase summary | expected node `{0,1}`、local `[start,end)`、順序、latency。native markerと混在禁止 | profileでは対象aggregateを作らない。bench consumerはNULLだけを返す |
| GPU aggregate（profile only） | raw sampleとphase marker | `runner.postrun` / profile run後 | phase/node/GPU scoped count、coverage duration、util mean/p95、memory max、energy/power-valid duration、temp max、clock min | `[start,end)`交差、pre-start sample可、`2*interval` gap cutoff、duration-weighted mean/nearest-rank p95/energy | profileではtelemetry incomplete。bench consumerはNULLだけを返す |
| artifact identity | localで作ったrun bundle、profileだけのraw telemetry/trace tar、SIF/build sidecar | `runner.packager` / GitHub Artifact upload前 | artifact URI、bundle/profile hash、SIF source/hash、`artifact-manifest.json` | SIF/model/dataset/env/cache/topology/PJM raw+version/`logs/pjm-job.log`/benchmark stdout/stderr/result/workloadを列挙。profileはGPU telemetryを回収し、D-004でprofiler familyがapplicableなprotocolだけ固定出力先のtrace/tool metadata/exact argvもrequiredにする。benchではprofile artifact/placeholder/absence recordを作らない | uploadできなければrun-level archive failure。benchのprofile consumerは理由なしNULLだけ |
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
  `pipe_writer.send(scheduler.get_init_info())`直前を境界にし、global scheduler rank `{0..7}`の
  8 eventを要求する。

対象SGLang versionごとにsymbolとfull commitをregistry source欄へ固定する。line番号だけを契約に
しない。

### workloadとresult

- ShareGPTは`benchmark/datasets/sharegpt.py`でshuffle後、chat template適用、tokenize、length filterを
  行って`DatasetRow`を作る。現状recordにsource row IDとtoken IDがないため、この地点でsource IDと
  prompt-token digestを保持できるようにする。source IDはdataset manifest+元JSON array index、digestは
  canonical token-ID arrayのSHA-256とする。
- sampling paramsは`offline_throughput.py`の`throughput_test_once`で`DatasetRow`から構成される。
  manifestはこの構成後、`backend.generate`前にsealする。
- generation timerは同関数の`time.perf_counter()`から`backend.generate` returnまでを使う。
- 現在の`successful_requests = len(reqs)`は予定件数であって成功実測ではない。実際の返却配列を
  decodeし、returned count、malformed/error count、response meta_infoのprompt/completion tokensを作る。
- primary resultは唯一のartifactとしてtemporary write/close後にrenameする。
  後続の`get_server_info()`、profile finalize、shutdown失敗で失ってはならない。eventとartifactを
  二重正本にせず、通常の`result_filename`は診断用とする。

## 現実装からの変更点

| 現在の箇所 | 問題 | v2での境界 |
|---|---|---|
| `qwen-experiment-metrics/.../core.py::make_run` | 各呼出しでrun IDを再構成する | runnerがmanifestを一度作り全producerは読むだけ |
| `core.py::send` | compute側packageがBigQuery credential/retryを所有する | `publish.py`へ隔離しPJM bundle/環境からcredentialを除去 |
| `stage.py::EventReceiver` | 共有socketを両nodeへbindし、node rankなしでrun-scoped keyを直接sendする | node-local endpoint -> node-local Recorder、node/rank scopeを保持 |
| `stage.py::run_stage` | receiverのBigQuery完了待ちと集計をstage elapsedへ含める | child waitでtimerを閉じ、merge/publishはrunnerへ移す |
| `stage.py::fallback_result` | missing/Noneを空文字化し、予定`successful_requests`も受け入れる | primary-result artifactのpure parser。欠損keyは出さずreasonを残す |
| `stage.py::GpuSampler` | memoryへ蓄積し終了時集計だけを送る | raw node JSONLへbuffered append/periodic flushし、後処理で集計 |
| `scripts/run-qwen35-node.sh` | 全nodeが同じ共有socket pathをbindし、cache manifestがない | node-local endpointとstaging直後cache manifest |
| 同node script | node別実argvをartifact/metricへ残さない | Singularity exec直前の配列をnode producerが記録 |
| `scripts/run-qwen35.pjm` | credentialをPJMへ持ち込み、outer commandはshell表示だけ | runner-only credential、spawn配列を`StageRunner`が取得 |
| `.github/workflows/genkai.yml` | credentialを`pjsub -x`で渡し、必要fileのupload前にruntimeをcleanupし得る | variant別expected fileをGitHub Artifactへuploadできた後にcleanup |
| `scripts/build-sglang-sif.sh` | base image tag、部分的provenance、完成SIF sidecarなし | digest-pinned OCI、dependency manifest、SIF rename後sidecar commit marker |
| `offline_throughput.py` | 予定件数をsuccessfulと命名し、primary result作成が後続処理より遅い | return decode/検証後すぐprimary resultをwrite/renameし、throughputはview派生 |
| `sharegpt.py`の`DatasetRow` | source ID/token identityを失う | selection時にordered workload manifest用identityを保持 |

## laneとcontrol-forwarder契約

`experiments/lanes/qwen-genkai-2node-tp8.toml`は2 PJM nodes、2 MPI node processes、4 GPU/node、
TP=8、expected node rank `{0,1}`、expected global scheduler rank `{0..7}`、required producer、
instrumentation modeに加え、rank 0 `hostname`をadvertise source、SGLang port `25000`、control port
`25001`、rendezvous timeout 60秒、ready timeout 120秒を固定する。port衝突/timeoutはrun failureとして
再実行し、動的allocatorやfallback discoveryは作らない。runnerはlane、PJM script identity、exact `pjsub` argvを
`scheduler.request`へsealし、job内のOMPI state、worker-ready、GPU UUID、accounting `nnumu`と照合する。

全variantで各node wrapperがcommon lifecycle node-agentを起動する。bench/profile/qualification ONだけが
score config producerを有効にし、qualification OFFはwrapper topologyとcommon lifecycleだけを持つ。rank 0 coordinatorはlane規定sourceから
得たaddressへbindして`control/lifecycle`のrun-bound rendezvousをsealし、rank 1 participantがその明示endpointへ
接続する。coordinatorは両nodeのidentity付きready ACKを確認してからcontainerを始め、
generation後は`lifecycle.shutdown.start`を一度だけ送る。profile runだけがさらにGPU sampler/phase receiverを
起動し、benchではGPU sample/phase messageを作らない。
container内rank 0 offline hookはbind済みlocal endpointへphase messageを送り、coordinatorが同じ
messageを両receiverへbroadcastする。messageはrun、invocation ID、warmup/measured、start/end、sequenceを
持ち、sender、node受信時刻、ACK時刻を最終profile summaryへ残す。start/end間では新規接続、DNS、filesystem pollを
行わない。rank 1 offline processはphase messageを送らない。別nodeのmonotonic timestampは差し引かず、
GPU aggregateは各node自身のobservation windowを使う。

profileの各generateは`start broadcast -> node 0/1 ACK -> rank0 timer start -> backend.generate ->
timer end -> end broadcast -> node 0/1 ACK`の順にする。rank0同一clockでACK-to-generate leadと
generate-to-ACK trailを記録する。shutdownはphase forwarderでなく共通lifecycle controlで送り、
両nodeの通常ACK後に`backend.shutdown()`を呼ぶ。lifecycle reducerが最終summary内のshutdown
invocationの一意性を検証し、rank 1のearly/coordinated判定は同じclock IDのnode-local受信時刻とchild
exitをviewで比較する。

## validation fixture

最低限、次のfixtureをGit管理する。

- CLI指定、default、environment override、auto-resolutionが異なるServerArgs
- node 0/1 full configの一致/不一致、global worker rank 0..7のmissing/duplicate/wrong-rank event
- token総数は同じだがrequest順序またはlength分布が違うworkload
- dataset row/chat template/tokenizerが1 byteだけ違うworkload
- returned count不足/過多、malformed response、prompt/completion token欠損、generate例外、NaN/zero duration
- warmup成功/measured失敗、generate成功/get_server_info失敗、rank1 coordinated kill/early death
- cache count/sizeは同じでcontentが違う二manifest
- variant applicability grid：profile-only collectorのprofile失敗はreasonあり、benchは理由/status/EvidenceなしNULL、qualification OFFのON専用列は同じNULL、qualification ONの同collector失敗はreasonあり
- tagged-union source trackのbench/profile/qualification OFF/ON、base非ancestor、instrumentation以外の差分、checkout tree不一致、同bundleでbase/workload/runtimeだけ変更、moving branch名が同じでもcommitが異なるfixture
- qualification allocationの一pair二child、二pair混入を全pair reject、child順/arm不一致、別allocationへの再pair、arm間crash、片child未起動/metrics未完成fixture
- bench lifecycle ready/shutdownだけ、profile lifecycle+phase、benchへphase summaryを誤生成するfixture
- instrumentation bundleのhook/sample interval一つだけを変えたhash差とactual activation不一致fixture
- D-001 decisionの未存在、non-adopt、hash不一致、candidate/canary result不一致、adopt完全一致fixture
- GPU sampleのsequence gap、unfinished frame、poll error、marker欠損、pre-start sample、既知p95/energy系列
- 正常close、program failure、途中JSON line、missing node fragment、異値conflict
- contract欠損legacy、expected producer version混在、exact v2 contract
- OpenFOAM方式を移植したhappy path、GitHub upload失敗、SCP失敗

全fixtureで、node/rank full snapshotのscope保持、raw row集合、completeness reason、comparison
eligibility、result artifactとの代数的parityを検査する。

## 未決定事項

bench/profile分離と、benchのprofile-only metricは値をNULLにするだけで原因/status/Evidenceを作らないことは
ユーザー確定済みである。初版profileで必須にするprofiler/activity/step範囲/required raw file集合D-004と、
score runに許す最小bench bundleの統計上限D-001はユーザー確認待ちである。プロジェクト全体ではOpenFOAMの
lane scope D-003を先に決め、D-004、D-001の順に一問ずつ確認する。決まるまではGPU
polling/aggregate、phase control marker、profiler、verbose backend trace、full placement診断はprofile-only、
未qualificationのinline eventはaudit-onlyとする。profiler toolや閾値を文書へ仮置きして
profile/score/qualification eligibleへ混入させない。
