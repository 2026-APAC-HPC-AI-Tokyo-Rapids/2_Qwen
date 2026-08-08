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
  create and fsync run-manifest.json
  verify source/artifact sidecars
  seal scheduler request from lane, PJM script identity, and exact pjsub argv
  fsync submission intent with scheduler-searchable run nonce
  submit PJM without BigQuery credentials
  after definite response or nonce recovery, fsync immutable binding/outcome

PJM supervisor
  snapshot job-entry environment
  seal run ID and PJM_JOBID in job-entry-receipt without waiting for binding
  let StageRunner record the actual outer argv
  launch one node wrapper per allocated node

node wrapper / node-agent
  rank 0 seals run-bound control rendezvous; rank 1 validates and connects
  wait for expected-node durable protocol-ready before starting containers
  verify cache staging and write input manifest
  snapshot host/container launch state and actual argv
  append raw GPU samples and local phase markers
  flush/seal all node evidence before the PJM script exits

SGLang Engine hook (each node)
  emit launch-effective ServerArgs before scheduler launch
  emit runtime worker-ready facts after initialization

rank-0 offline benchmark hook
  persist warmup/measured workload manifests
  emit phase boundaries
  atomic-write and fsync the sole authoritative primary-result artifact

Actions runner
  verify expected node and global-worker scoped facts without scope reduction
  merge -> validate -> archive and verify -> publish/replay
```

`run-manifest.json`で生成したrun IDを全producerが読む。各processで再計算しない。node間で共有
Unix domain socketをtransportにしない。socketを使うならendpointとWAL writerは各node localに
置き、node rankをevent envelopeで必須にする。target hookはlocal structured eventを書くだけで、
metrics key、BigQuery、retryを知らない。

## fact別の取得契約

表の「欠損」はraw valueへ空文字を入れる意味ではない。keyを出さず、対応producerのlifecycleと
completeness reasonで説明する。

| fact family | authoritative source | producer / capture moment | raw keyまたはartifact | validation | 欠損時の扱い |
|---|---|---|---|---|---|
| run identity | runnerが一度生成してfsyncした`run-manifest.json` | `runner.manifest` / provision前 | raw tableの`run`列、manifest artifact | repository、Actions run/attempt、submissionごとの128-bit nonceと全fragmentのrun一致。PJM job IDを含めない | manifestなしでは投入しない |
| metrics contract | Git管理`qwen-v2.toml`とlaneが選んだexpected producer graph | `runner.manifest` / provision前 | `run.metrics_contract`、contract/protocol artifact | contract ID/content hash、exact producer version、view compatibility、mixed contract禁止 | comparison不適格 |
| scheduler binding | pre-submit intent、`pjsub`応答またはnonce検索、PJM側の独立job-entry receipt | `runner.submission` / pjsub前後、および`pjm.supervisor` / entry直後 | `scheduler.job_id`、submission intent/result/binding、`control/job-entry-receipt.json` artifact | jobはbindingを待たない。曖昧応答をblind retryせず検索し、terminal後finalizerがrun ID、submit response、entry receiptの`PJM_JOBID`一致を要求 | reject/unknownを別receipt化。片方欠損または不一致はartifactを保全して`invalid-quarantine` |
| producer lifecycle | supervisorのcollector/child境界 | `pjm.supervisor`、`StageRunner`、node agent / 呼出し前とreturn後、`LifecycleJournal`が永続化 | 独立`lifecycle/<producer>.jsonl`のstarted/final | complete line salvage、末尾partial破棄、data fragmentと別owner | startedのみを中断、journalなしを未到達 |
| protocol assignment | Git管理protocolとrunnerの割当record | `runner.manifest` / job投入前 | `experiment.block_id`、`block_order`、`replicate_index` | protocol commit、予定attempt、laneへのmembership | comparison不適格、推測しない |
| experiment source | checkout済み`2_Qwen`の`git rev-parse HEAD`とclean/reachability check | `runner.manifest` / provision前 | repository/full commit | dirty tree拒否、remote到達または耐久bundleを要求 | jobを投入しない |
| SGLang source | SIF build時のcheckout済み`5_SGLang_Src` full commit | `build.finalizer` / source checkout直後 | SIF build sidecarとsource commit索引 | branch名ではなくfull commit、embedded valueとの一致 | SIFを公開しない |
| SIF/toolchain | digest-pinned `From:`が消費したOCI manifest/config/layer、完成SIF、package/toolchain、実参照binary | `build.finalizer` / SIF完成・test後 | immutable SIF sidecar、artifact identity/URI索引 | SIF hashは完成後sidecarへ、SIF rename後sidecarをcommit markerにする | reuse禁止 |
| model/tokenizer/template | pinned model manifest、全shard identity、tokenizer files、chat template | `provision.verifier` / node copy前 | immutable manifest artifactとidentity索引 | pathではなく全member/order、実体との一致 | provision失敗 |
| dataset | pinned repository revisionまたはdataset file manifest | `provision.verifier` / node copy前 | dataset manifest artifactとidentity索引 | shared pathだけで同一視せずbyte identity検査 | provision失敗 |
| full job-entry environment | `os.environb`相当のbyte environment | `pjm.supervisor` / script entry直後、module/unset前 | `env.job_entry.*` | byte-exact name escapeと全value Base64 | snapshot collector failed |
| full host exec environment | node wrapperのbyte environment | `node.wrapper` / module/cache staging後、container exec直前 | `env.host.node.<node_rank>.*` | byte-exact、expected node全件、raw scope保持 | configuration incomplete |
| full SGLang launch environment | container内のbyte environment | `sglang.engine` / `_set_envs_and_config`等の解決後、scheduler launch直前 | `sglang.env.node.<node_rank>.*` | `--cleanenv`後かつSGLang mutation後の全件、raw scope保持 | configuration incomplete |
| full worker environment | scheduler processのbyte environment | `sglang.worker_ready` / ready通知直前 | `sglang.worker.env.rank.<global_rank>.*` | byte-exact、global rank `{0..7}`、device event一致 | runtime completeness incomplete |
| scheduler request | Git管理lane、PJM script identity、runnerが実行するexact `pjsub` argv | `runner.submission` / submit直前 | `scheduler.request` | 2 nodes、2 MPI processes、resource group、override照合 | invalid protocol assignment |
| scheduler accounting | `pjstat --history -S --choose jid,st,sdt,elp,ec,sn,pc,nnumu,mmszu,uctmut,sctmut --data --delimiter "\t" <job>` | `runner.finalizer` / terminal state後 | `scheduler.accounting`、stdout/stderr/PJM version artifact | field/unit/job ID固定、requestとの一致はview | 取得fieldだけ欠損扱い。host/rankを捏造しない |
| observed node topology | 同じMPI launch内のOMPI rank/size/local rank、hostname | `node.wrapper` / container exec直前 | node/rank placement record | rank `{0,1}`、2 distinct host、一rank一node | topology incomplete |
| control rendezvous | lane contractのtransport/address source/port strategyとrank 0 node-agentが実際にbindしたendpoint | `node-agent coordinator` / container開始前 | `control/phase/rendezvous.json`、`protocol-ready.json` artifact | run/nonce/version/freshness、rank 1 explicit endpoint、expected node identity ACK。env/DNS/任意探索禁止 | containerを開始せずconfiguration failure |
| GPU topology | `nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version --format=csv,noheader,nounits`とcontainer-visible device query | `node-agent` / benchmark前 | `hardware.gpu.nodes`、raw outputs | 4 GPU/node、8 unique UUID、PCI/logical index mapping | topology incomplete |
| node CPU/NUMA hardware | `LC_ALL=C lscpu --json`、`/sys/devices/system/cpu/online`、`/sys/devices/system/node/online`、各nodeの`cpulist`/`meminfo`、`/proc/meminfo` | `node.auditor` / benchmark後、各node | `hardware.cpu_numa.nodes`、raw files | observed hostname集合、exact membership/parser fixture | hardware completenessだけfalse |
| outer actual argv | stage wrapperが`subprocess.Popen`へ渡すargv配列 | `StageRunner` / spawn直前 | `run.launch_argv` | shell表示ではなく配列、profiler wrapperを含む実spawn値 | stage未開始またはconfiguration incomplete |
| node actual argv | node wrapperの`command`配列 | `node.wrapper` / Singularity exec直前 | `parallel.node.<node_rank>.benchmark_argv` | bind/env/benchmark argvを順序保持、node rankとの整合 | node configuration incomplete |
| run timing/outcome | supervisorのUTC clock、monotonic clock、最終status | `pjm.supervisor` / entry時とcompute回収終了時 | `run.started_at`、`run.finished_at`、`run.elapsed_sec`、outcome/exit | runner archive時刻を混ぜず、wallとmonotonicを別clockとして保持 | lifecycleから中断理由を出す |
| stage outcome | supervisorの同じmonotonic clockとraw child wait status | `StageRunner` / spawn直前とwait直後 | `perf.stage.benchmark.wall_time_sec`、`run.stage.benchmark.termination` | exit/signal/spawn_errorを区別し、publish/GPU集計をtimer外へ | lifecycleで中断を表現 |
| cache input | shared cacheのnode-local rsync完了直後のcanonical member manifest | `node.wrapper` / monitor/container開始前 | `cache.node.<node_rank>.input_manifest_sha256`、manifest artifact | path/type/size/content/order、SIF/model/backend互換dimension | 欠損ならprovenance incomplete |
| BenchArgs | `BenchArgs.from_cli_args(args)`が返すdataclass | 各node `sglang.cli` / CLI parse直後、`Engine()`前 | `parallel.node.<node_rank>.sglang.benchmark_arg.*` | default込み、node `{0,1}`のfull object一致をview検証 | configuration incomplete |
| launch-effective ServerArgs | `_set_envs_and_config`、`check_server_args`、auto parser解決後の`ServerArgs` | 各node `sglang.engine` / scheduler/daemon launch直前 | `parallel.node.<node_rank>.sglang.server_arg.*` | raw full object、TP/nnodes/node_rank/dist address invariant | configuration incomplete |
| runtime backend/transport | `run_scheduler_process()`で`Scheduler(...)`構築後、`pipe_writer.send(scheduler.get_init_info())`直前のactual state | `sglang.worker_ready` / global workerごと | `parallel.worker.<global_rank>.runtime` | global rank `{0..7}`、node rank、GPU ID/UUID、affinity、actual attention/communication backend | topology/runtime incomplete |
| measured workload | ShareGPT shuffle/filter/chat-template/tokenize後のordered `DatasetRow`とsampling params | `rank0.offline` / generate前 | measured manifest、hash/count/planned totals索引 | dataset manifest+元array index、canonical token-ID array SHA-256、length/output/sampling/order | provenance/benchmark incomplete |
| warmup workload | random warmup request作成後のordered requestsとsampling params | `rank0.offline` / warmup generate前 | warmup manifest hash/count/planned totals、`workload.warmup.mode` | measured manifestへ混ぜずcomparison fingerprint必須 | warmup incomplete |
| input prepare phase | tokenizer取得、dataset load/sample/tokenizeのmonotonic boundaries | `rank0.offline` / request object完成まで | `perf.phase.input_prepare.wall_time_sec` | 同一clock。manifest hash/fsyncはtimer終了後、generate前 | phase incomplete |
| distributed init phase | rank 0 processの`Engine(...)`呼出し | `rank0.offline` / constructor直前からreturn直後 | `perf.phase.distributed_engine_init.wall_time_sec` | rank 0同一monotonic clockだけ。worker-readyはcardinality検証専用 | phase incomplete |
| warmup generation | warmupの`backend.generate`直前・return直後 | `rank0.offline` + control forwarder / warmup中 | `perf.phase.warmup_generation.wall_time_sec`、node receipt artifact | measuredと別invocation、expected node start/end/ACK | benchmark valueは保持しphase incomplete |
| measured generation | measured `backend.generate`直前・return直後 | `rank0.offline`がtimer値をprimary artifactへ、runner parserがterminal後にFact化 | `perf.benchmark.generation.wall_time_sec`、node receipt artifact | rank0 `time.perf_counter`、正/有限、同じinvocation ID、forward lead/trail | benchmark invalid |
| observed responses | decode/validate済み`gen_out` | `PrimaryResultWriter` / generate return直後、runner parser / terminal後 | atomic primary artifactからreturned/error count、prompt/completion totals | schema固定、`len(gen_out)`、meta tokens、malformed/over-return reason。O_EXCL temporary、no-replace rename、既存同byteだけ冪等。raw key ownerはparserだけ | generate例外またはwrite/fsync/rename失敗ではcountを捏造せずkey不在、stage nonzero |
| primary outcome | observed count/tokensとgeneration duration | BigQuery view / query時 | throughput派生値はrawへ重複保存しない | request/input/output/total throughputをartifactの代数値と許容誤差比較 | source不足ならNULLと理由 |
| internal last-generation stat | `get_server_info()["internal_states"]` | rank0 result artifact / generate結果fsync後 | full result artifactのみ | 一次estimandに使わず、API failureでprimary resultを失わない | raw metric不要 |
| node process lifecycle | wrapper child start/end、raw wait status/signal、termination reason | `node.wrapper` / child境界 | `parallel.node.<node_rank>.process_lifecycle` | shutdown receiptを複写しない。phase reducerがinvocation/sequence/node一意性を検証後、viewがvalid receiptとrun/nodeでjoinし、同一`CLOCK_MONOTONIC` clock IDの時だけreceipt前exit=early、receipt後=coordinatedを導出 | receipt 0/複数、invocation/clock不一致は分類NULLとreason。topology/run incomplete |
| shutdown phase | shutdown開始・全child終了のrank0 monotonic boundaries | `rank0.offline` / shutdown前後 | `perf.phase.engine_shutdown.wall_time_sec`、shutdown invocation ID | node lifecycleと照合 | phase incomplete |
| raw GPU telemetry | node-local periodic `nvidia-smi` sampleとpoll error | `node-agent` / process実行中にshared durable runtimeへ逐次append+sync | node別raw JSONLまたは独立zstd-frame artifact | O_APPEND、per-sample/frame fdatasync、monotonic/wall anchor/UUID/index/value/error/sequence gap | 途中まで保存、telemetry incomplete |
| GPU phase markers | persistent coordinatorからのcontrol message | `node-agent` / warmup・generation receipt時 | node/invocation/sequence別sender・receipt・ACK artifact | expected node `{0,1}`、local `[start,end)`、順序、latency。native markerと混在禁止 | 対象phase aggregateを作らない |
| GPU aggregate | raw sampleとphase marker | `runner.finalizer` / run後 | phase/node/GPU scoped count、coverage duration、util mean/p95、memory max、energy/power-valid duration、temp max、clock min | `[start,end)`交差、pre-start sample可、`2*interval` gap cutoff、duration-weighted mean/nearest-rank p95/energy | telemetry completenessのみfalse |
| artifact identity | localでsealしたrun bundleとdeterministic archive URI | `runner.finalizer` / final merge前 | artifact URI、bundle hash、manifest identity | bundle対象固定、metrics sidecar自身を含めない | bundleをsealできなければ`archive-retry-pending` |
| delivery receipt | remote PUT/checksum verify、BigQuery publish response | `ArtifactStore`/`Publisher` / run WAL seal後 | run WAL外かつrunner workspace外のdurable append-only delivery store/outer manifest | run/bundle hash/attempt一致、put-if-absent、replay冪等 | retry pending。run WALへ書き戻さない |

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
- primary resultは唯一のatomic artifactとしてtemporary write+fsync、rename、directory fsyncする。
  後続の`get_server_info()`、profile finalize、shutdown失敗で失ってはならない。eventとartifactを
  二重正本にせず、通常の`result_filename`は診断用とする。

## 現実装からの変更点

| 現在の箇所 | 問題 | v2での境界 |
|---|---|---|
| `qwen-experiment-metrics/.../core.py::make_run` | 各呼出しでrun IDを再構成する | runnerがmanifestを一度作り全producerは読むだけ |
| `core.py::send` | compute側packageがBigQuery credential/retryを所有する | `publish.py`へ隔離しPJM bundle/環境からcredentialを除去 |
| `stage.py::EventReceiver` | 共有socketを両nodeへbindし、node rankなしでrun-scoped keyを直接sendする | node-local endpoint -> node-local Recorder、node/rank scopeを保持 |
| `stage.py::run_stage` | receiverのBigQuery完了待ちと集計をstage elapsedへ含める | child waitでtimerを閉じ、merge/publishはrunnerへ移す |
| `stage.py::fallback_result` | missing/Noneを空文字化し、予定`successful_requests`も受け入れる | atomic primary-result artifactのpure parser。欠損keyは出さずreasonを残す |
| `stage.py::GpuSampler` | memoryへ蓄積し終了時集計だけを送る | pollごとにraw node JSONLへappend+fdatasync、後処理で集計 |
| `scripts/run-qwen35-node.sh` | 全nodeが同じ共有socket pathをbindし、cache manifestがない | node-local endpointとstaging直後cache manifest |
| 同node script | node別実argvをartifact/metricへ残さない | Singularity exec直前の配列をnode producerが記録 |
| `scripts/run-qwen35.pjm` | credentialをPJMへ持ち込み、outer commandはshell表示だけ | runner-only credential、spawn配列を`StageRunner`が取得 |
| `.github/workflows/genkai.yml` | credentialを`pjsub -x`で渡し、runtimeを無条件cleanupし得る | archive/checksum verify後cleanup、失敗時`archive-retry-pending` |
| `scripts/build-sglang-sif.sh` | base image tag、部分的provenance、完成SIF sidecarなし | digest-pinned OCI、dependency manifest、SIF rename後sidecar commit marker |
| `offline_throughput.py` | 予定件数をsuccessfulと命名し、primary result durabilityが後続処理より遅い | return decode/検証後すぐatomic result artifactをfsyncし、throughputはview派生 |
| `sharegpt.py`の`DatasetRow` | source ID/token identityを失う | selection時にordered workload manifest用identityを保持 |

## laneとcontrol-forwarder契約

`experiments/lanes/qwen-genkai-2node-tp8.toml`は2 PJM nodes、2 MPI node processes、4 GPU/node、
TP=8、expected node rank `{0,1}`、expected global scheduler rank `{0..7}`、required producer、
instrumentation modeを固定する。runnerはlane、PJM script identity、exact `pjsub` argvを
`scheduler.request`へsealし、job内のOMPI state、worker-ready、GPU UUID、accounting `nnumu`と照合する。

各node wrapperはbenchmark前にnode-agentのGPU sampler/local control receiverを起動する。rank 0
coordinatorはlane規定sourceから得たaddressへbindしてrun-bound rendezvousをsealし、rank 1 participantが
その明示endpointへ接続する。coordinatorは両nodeのidentity付きready ACKとprotocol-ready receiptを
確認してからcontainerを始める。
container内rank 0 offline hookはbind済みlocal endpointへphase messageを送り、coordinatorが同じ
messageを両receiverへbroadcastする。messageはrun、invocation ID、warmup/measured、start/end、sequenceを
持ち、sender、node receipt、ACK時刻をartifactへ残す。start/end間では新規接続、DNS、filesystem pollを
行わない。rank 1 offline processはphase messageを送らない。別nodeのmonotonic timestampは差し引かず、
GPU aggregateは各node自身のreceipt windowを使う。

各generateは`start broadcast -> node 0/1 durable ACK -> rank0 timer start -> backend.generate ->
timer end -> end broadcast -> node 0/1 durable ACK`の順にする。rank0同一clockでACK-to-generate leadと
generate-to-ACK trailを記録する。shutdownは同じforwarderで`shutdown.start(invocation_id)`を送り、
両nodeのcontrol journalへのdurable receipt ACK後に`backend.shutdown()`を呼ぶ。phase reducerがshutdown
invocationの一意性を検証し、rank 1のearly/coordinated判定は同じclock IDのnode-local receiptとchild
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
- GPU sampleのsequence gap、unfinished frame、poll error、marker欠損、pre-start sample、既知p95/energy系列
- append直後crash、途中JSON line、missing node fragment、同値replay、異値conflict
- contract欠損legacy、expected producer version混在、exact v2 contract
- archive upload成功、checksum不一致、唯一copyしかない状態

全fixtureで、node/rank full snapshotのscope保持、raw row集合、completeness reason、comparison
eligibility、result artifactとの代数的parityを検査する。

## 未決定事項

score runに許す計測の上限はユーザー確認待ちである。決まるまでは、effective config、result、
phase boundaryの軽量local eventだけをscore候補へ入れ、GPU polling、profiler、verbose backend debug、
full placement診断は`instrumentation.mode=audit`として別groupにする。閾値を文書へ仮置きして
comparison eligibleへ混入させない。
