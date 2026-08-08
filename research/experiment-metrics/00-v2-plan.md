# 実験メトリクス再設計計画

## 目的

メトリクスは、Gitやlogを複製する保管庫ではない。BigQueryとスプレッドシートから
次の問いへ短く答えるための索引である。

1. どのGit管理下の条件で実行したか。
2. Git外で、実行時にどの条件が選ばれたか。
3. 実workloadと分散実行は妥当で、比較に必要な観測が揃っているか。
4. 何を一次指標として、どれだけ速かったか。
5. 比較不能または要調査と判断すべき欠損や競合がないか。

変更内容と変更理由の正本はGit、実行詳細の正本はrun artifact、raw factの索引は
BigQuery、比較規則の正本はGit管理されたSQLとtestとする。BigQuery viewはその
deploy済み実体であり、暗黙に手編集しない。

## 正本とidentity

| 情報 | 正本 | raw metricの役割 |
|---|---|---|
| script、設定、patch | `2_Qwen`のcommit | repositoryとfull commitを指す |
| 仮説、変更理由、期待する効果 | commit messageのbody | 保存しない |
| SGLangの変更 | `5_SGLang_Src`のcommit | SIF provenanceを索引化する |
| stdout、stderr、SGLang結果 | run artifactのlog/JSONL | 比較に必要な原始観測だけを索引化する |
| host/container環境 | scopeごとのbyte-exact snapshot | 全件保存する |
| SIF、model、dataset、cache | immutable artifact | revisionまたはmanifest identityを指す |
| throughput、speedup、median、順位 | Git管理されたview SQL | raw tableへ保存しない |

次の三つを混同しない。

- source identity: repository URLとresolved full commit
- artifact identity: immutable revision、manifest ID、またはbuild時のcontent hash
- reproducibility: base image digest、toolchain、dependency、全source commitを含むmanifest

Git管理されたrecipeやpatchごとにSHA-256 metricを作らない。`2_Qwen`のcommitとpathで
内容が一意になるためである。hashはGit外artifact、binary転送、順序付きworkload/cache
manifestの冪等性確認に限定し、巨大artifactをbenchmark runごとに再hashしない。

## Gitによる実験条件の管理

- 性能、workload、parallelism、cache policy、build、collectorの挙動を変える変更は
  job投入前にcommitする。
- Actions checkout直後、provision前にclean treeとfull commitを検証する。
- benchmarkに使うcommitはremoteから到達可能、または耐久archiveに保存済みでなければ
  ならない。provenanceのないmanual runは投入しない。
- 一つのcommitは一つの説明可能な変更単位にする。複数fileが同じ仮説を実現する場合は
  同じcommitでよい。
- subjectは変更の要約、空行後のbodyは変更理由、仮説、期待する影響、確認方法を書く。
- 同じ条件の反復ではcommitを増やさず、同じcommitから複数runを作る。
- model、dataset、SGLang引数、配置をworkflow dispatch inputへ逃がさず、tracked script
  または設定に置く。
- SIFを再利用したrunは、現在の実行commitとは別にoriginal SIF build commitと
  `5_SGLang_Src` commitを記録する。

commit messageの例を示す。

```text
Pin one SGLang process to each NUMA domain

The current process spans both sockets while each GPU pair is local to one NUMA
domain. Bind one process per domain because the hypothesis is that local CPU and
memory placement shortens engine initialization without reducing generation
throughput. Compare repeated phase times against the parent commit.
```

この文章を`experiment.hypothesis`や`run.note`へ複製しない。人間とエージェントは
runのcommitから読む。

## raw relationの契約

raw tableは現在の三列を維持する。

```text
run    STRING NOT NULL
key    STRING NOT NULL
value  STRING NOT NULL
```

`run`は全producerが共有するglobally uniqueでimmutableなIDである。Actions経路ではjob投入前に
repository、Actions run ID、attemptと各submissionで必須の128-bit random nonceから固定する。nonceは
OS CSPRNGの16 byteを32文字のlowercase hexで表す。manual経路では
clusterを含むUUIDをjob投入前に生成する。`pjsub`返却後にしか分からないPJM job IDはrun IDへ
含めず、immutableなsubmission bindingと`scheduler.job_id`で結ぶ。初期run manifestを書き換えない。
run IDはUTF-8で`^[a-z0-9][a-z0-9._-]{0,199}$`を満たすよう生成し、同じ値を安全なruntime path
componentとして使う。任意のrepository名やuser入力を未escapeで連結しない。
raw relationは次のfunctional dependencyを満たす。

```text
(run, key) -> value
```

- 完全一致する`(run,key,value)`の再送は、論理上の一rowとして扱う。
- 同じ`(run,key)`に複数のdistinct valueがあれば値を選ばず`conflicting`とし、runを
  比較から除外する。
- 既存runの値を訂正しない。訂正測定は新しいrunとして残す。
- row不在は未観測、`value=""`は実際に観測した空文字、`"0"`は真のゼロを表す。
- 取得不能や非該当をplaceholder空文字にしない。
- collector lifecycle keyはsupervisorだけが所有する。collector data fragmentとは独立した
  `lifecycle/<producer>.jsonl`へ、呼出し前に`collector.<name>.started`を耐久appendし、呼出しから
  戻った後にstatusとerror countを持つ`collector.<name>.final`を一度だけappendする。どちらのkeyも
  更新しない。lifecycle journalはatomic sealを要求せず、mergeはvalidなcomplete JSON lineを必ず
  回収し、末尾のpartial lineだけを破棄して`fragment_complete=false`をrunner fragmentへ記録する。
- startedなしは未到達、startedあり/finalなしは中断、finalのfailedは処理済み失敗とする。
  finalがあってもrequired keyがなければ別理由のincompleteとし、viewで混同しない。

`run.metrics_contract`は、emitterとviewの非互換schemaを識別するための小さな運用索引として
意図的に保存する。既存keyの意味、unit、JSON shapeは変更せず、非互換変更は新contract
または新keyにする。

各keyまたはkey patternは、owner、logical type、unit、cardinality、source、scope、
取得時点、required/optional、欠損規則を一つのGit管理registryで定義する。動的keyの
component escapeは単射にし、append/merge時にもownershipと衝突を検査する。

## scalarとcanonical JSON

自然に一つのscalarである値は、そのscalarをSTRINGへ変換する。順序や対応関係を持つ
複数値は、添字付きkeyへ分解せず一つのcanonical JSON valueにする。

canonical JSONはRFC 8785 JSON Canonicalization Schemeに固定する。array順序を保持し、
duplicate object key、NaN、Infinityを禁止する。RFC 8785の表現可能範囲を超える整数や精度を
必要とするdecimalはJSON numberにせず、registryで型を定めたcanonical decimal stringにする。

scalar encoderも次に固定する。

- boolは`true`/`false`、intは符号付き十進最短表現とし、leading `+`/zeroと`-0`を許さない。
- floatは有限IEEE 754値のRFC 8785 number serializationとし、NaN/Infinityを許さない。
- timestampはUTCへ正規化した`YYYY-MM-DDTHH:MM:SS[.fraction]Z`とし、小数秒の末尾zeroを除く。
- 時間のunitは秒、data量はbyteとする。stringはUTF-8で保持する。
- JSON `null`はregistryが定義する「観測済みcompound record内のfield欠損」にだけ使い、
  key全体の取得失敗や非該当には使わない。

これにより同じlogical valueはprocessやlanguageにかかわらず同じbyte列になる。

Pythonの`str(list)`は使用しない。logical typeとunitはvalueから推測せずkey registryを
正本にする。

## 重複を避ける規則

候補Aを削除できるのは、残存するraw factとimmutableなGit/artifact identityだけから、
同じsemantic、source、scope、取得時点、unit、精度で、Aのkey membershipとvalueを
常に一意かつ損失なく復元でき、comparison viewの利用性も失わない場合だけである。

単に値が等しい、名前が似る、またはlogに存在するだけでは削除しない。requestedと
effective、plannedとobserved、rank 0とrank 1、別phaseの時間は独立事実である。公式producer
との照合、schema routing、保存層をまたぐmanifest索引などの正当な重複は、registryの
`justification`へ理由と検証式を書く。

全環境変数の保存は、取り漏らしのないforensic比較という要件に基づく明示的な非最小例外
である。個別env値を別namespaceへ再掲せず、標準Sheetでは比較関連projectionだけを出す。

## write-ahead logと送達

`metrics.jsonl`はBigQueryと同じ`run/key/value`を持つwrite-ahead logである。別event
envelope、型別value列、ack queueは作らない。

target programの薄いhookが出すversion付きdomain recordは、local adapterへの入力または
diagnostic/source logであって第二の送達queueではない。単発factはadapterが直ちにregistryで検証して
WAL化する。反復eventを最終的に一arrayへするfamilyは、fsync済みappend-only domain journalを
artifactとして残し、terminal後のpure reducerが一度だけFactへ変換する。registryはどちらか一方を
familyごとに固定し、BigQueryへの耐久送達の正本は`run/key/value` WALだけにする。

```text
compute producer
  -> node別のmetrics JSONLへ耐久append
  -> node別fragmentをatomic completion
  -> runnerがexpected nodeとfragmentを検証し、log/sample/fragmentのrun bundleを作成
  -> bundle identityをfinalizer fragmentへappendして最終merge
  -> run bundleとmerged JSONL sidecarをarchiveしchecksum/manifestで検証
  -> runnerがBigQueryへ全行をat-least-once送信またはreplay待ちに登録
  -> archive検証後にruntimeを削除
```

- 一つの完全なJSON lineを`O_APPEND`またはlock下で書き、flush/fsync後に完了とする。
- expected producer集合はGit管理protocolとrun manifestからjob投入前に固定する。scheduler
  accountingとobserved placementは、その期待集合が実際に成立したかを後から検証するsourceである。
- mergeはrun ID、JSON、owner、expected node、一意性を検査し、欠損や異値競合を隠さない。
- BigQuery `insertId`はbest-effortの補助にすぎない。logical viewが完全一致rowをdeduplicate
  し、異値競合を隔離する。
- replayは常に同じrowを全件送ってよい。送信成功後もJSONLをtruncateしない。
- archive/upload失敗時はruntimeを`archive-retry-pending`とし、後続reconcilerが回収する。
- publish失敗時は検証済みarchiveをreplay pendingにし、raw JSONLから再送する。
- remote checksumとmanifest確認前に唯一のlocal copyやGitHub artifactを削除しない。

artifact identityの自己参照を避ける。まずlog、raw GPU sample、producer fragmentからrun bundleを
作ってhashし、そのdeterministic URI、bundle hash、manifest identityをrunner finalizer fragmentへ
appendする。その後に最終mergeした`metrics.jsonl`をbundleとは別のsidecarとして同じarchive
prefixへ保存する。bundleとsidecarの両方をremoteで検証してからpublish/cleanupする。metrics
sidecar自身のhashをsidecar内のmetricにはしない。

timed benchmark中はBigQueryへ接続しない。送信用credentialはrunnerだけが保持するため、
PJM processの性能条件へ混入しない。各snapshotはそのscopeに実在する環境をfilterなしで
保存し、機密値も例外扱いしない。

## 最小の実装境界

対象programへ計測システムの都合を持ち込まないため、責務を次の六境界へ固定する。

1. domain probe/parserは`Fact(key, logical_value)`だけを返すpure functionとする。run ID、文字列化、
   filesystem、BigQueryを知らない。
2. `journal.py`の`Recorder`と`LifecycleJournal`だけがpersist済みrun manifest、registry、canonical
   encoder、fact/lifecycleのappend/fsync、data fragmentのatomic completionを所有する。probeは
   `Recorder`、その外側のsupervisor/node-agentは`LifecycleJournal`だけを見る。
3. `StageRunner`だけがspawnするactual argv、monotonic timer、signal、child wait statusを所有し、
   観測したfactを`Recorder`へ渡す。
4. runner側`merge`がexpected producer、run ID、owner、一意性、required key、fragment完了を検証し、
   archive対象のmerged JSONLを作る。node-scoped full factをrun-scopedへ変換しない。
5. `ArtifactStore/RunFinalizer`がrun bundle/outer manifestのfsync、seal、put/remote verify、quarantine、reconcileを
   所有する。delivery receiptはrun WALへ戻さず外側のdelivery ledgerへappendする。
6. runner側`publish`だけがBigQuery credential、retry、replayを所有する。collector、SGLang、PJM
   processはBigQuery、artifact upload、HTTP/cloud client等の外部送達networkを使わない。唯一、
   `node-agent`だけがlane contractで許可した明示rendezvous endpointへのrun内control transportを使える。
   credential、DNS discovery、外部egressを許可しない。

SGLang本体へ必要なhookは、version付きlocal structured eventを定義済み境界でだけ書く薄いadapterとする。
metrics packageやBigQuery clientをimportせず、送達、再試行、key展開を知らない。実装上は
`domain.py`、`journal.py`、`submission.py`、`stage_runner.py`、`merge.py`、`artifact_store.py`、`publish.py`、domain別
probe/parserを深いmodule境界にし、CLIはそのcomposition rootに限定する。

QwenのSGLang patch内で唯一のatomic result writerとして、rank 0の`PrimaryResultWriter`だけは
`backend.generate` return後にprimary artifactをatomic write+fsync+renameする。これはWAL writerでも
bundle storeでもなく、metrics keyを知らないsource-artifact writerである。schemaはcontract、run、
invocation ID、generation duration、returned/error count、observed input/output totals、validation
reasonsに固定し、runnerのprimary-result parserだけが対応raw keyのownerになる。

Qwen用とOpenFOAM用を直ちに第三の共通packageへ統合しない。まず同じcontractとconformance
fixtureを両repositoryで満たし、interfaceが安定してから、canonical encoder、registry validator、
merge validatorだけを抽出する。domain collectorやpipeline固有の失敗処理は共通化しない。

実装開始前の設計正本として、各factの権威source、producer、取得時点、raw key/artifact、validation、
欠損時の扱いは[観測マップ](./01-v2-observation-map.md)に置く。最初の実装sliceでこれらを一つの
機械可読contractへ移し、以後は観測マップの表をcontractから生成する。sourceと取得時点をcontractへ
特定できない候補は追加しない。

## 環境snapshot

同じ名前でも時点、node、processが違えば意味が異なるため、次を分ける。

- `env.job_entry.*`: PJM entry直後、module操作やunsetより前の全環境
- `env.host.node.<node_rank>.*`: module設定後、各nodeのbenchmark exec直前の全環境
- `sglang.env.node.<node_rank>.*`: 各container内、`_set_envs_and_config`等の解決後、scheduler
  process launch直前の全環境
- `sglang.worker.env.rank.<global_rank>.*`: 各workerがruntime初期化を完了しreadyを通知する直前の
  全環境

名前と値はbyte-exactに取得する。nameの各byteは`[A-Za-z0-9_]`だけをそのまま使い、それ以外は
`~HH`の大文字hexでescapeするため単射になる。valueはUTF-8可否にかかわらず全件をRFC 4648の
padding付きBase64にし、registryで`bytes/base64`と宣言する。これにより通常文字列とBase64の
曖昧性を作らない。snapshot間で同じ値が重複してもscopeが異なる独立観測である。標準の二run
差分はGit管理された比較関連keyだけを表示し、全env差分は別viewでdecodeして提供する。

## 実験protocolと比較条件

一次estimandは、固定workloadをwarmup後に生成したtotal token throughputとする。二次
estimandはdistributed engine init、input prepare、warmup generation、generation、engine
shutdown、process lifecycle全体のwall timeとする。結果を見て指標を選び直さない。

実験protocol、予定試行数、停止規則、A/B順序はGitへcommitする。実行時に割り当てられた
次の値はGitだけから実行事実を復元できないためraw metricとして残す。

- `experiment.block_id`
- `experiment.block_order`
- `experiment.replicate_index`

既定は各条件5 attemptとし、同じ時間block内でA/Bを交互または事前乱数化順に投入する。
有効runが3未満なら環境差を確定しない。invalid/failed runを遅かったという理由で除外・
置換せず、最大attempt数と置換規則も事前にcommitする。

比較次元はGit管理SQLで次へ分類する。

- 一致必須: workload、SIF/model/dataset、effective args、cache input、resource shape、
  instrumentation mode
- 意図的に変えるtreatment: 対象commit、cluster、node/GPU type等
- nuisance: hostname、GPU UUID、開始日時、queue待ち、個別node identity
- outcome: throughput、phase時間、resource値

protocol commitはtreatmentにするkey集合と一致必須key集合を事前宣言する。viewは次の二つを
導出し、fingerprint自体はrawへ保存しない。

- `replicate_fingerprint`: 一致必須条件とtreatment値を含み、同一条件の反復統計に使う。
- `comparison_set_fingerprint`: 宣言済みtreatment次元だけを除き、A/B ratioの候補を束ねる。

commit、artifact、cluster等がtreatmentならcomparison set内では差を許すが、replicate groupは
必ず分離する。nuisanceとoutcomeはどちらにも含めない。競技提出用bestと、環境比較用の全点、
median、MAD、ratio、不確実性を分ける。

## 分散producer契約

現行の共有filesystem上Unix domain socketをnode間event transportとして使わない。Unix
socket endpointは各node localに作り、各nodeのeventはそのnodeのWALへ書く。

- rank 0だけがworkload manifest、benchmark result、run-scoped phaseのauthoritative
  producerになる。
- 各nodeはresolved args、environment、actual argv、affinity、GPU mapping、telemetryを
  node scopeで出す。
- rawはnode/rank-scoped full factを正本として保持する。viewだけが全expected scopeの一致を
  検証したfieldをcommon表示し、rawにcommon/deltaの第二表現を作らない。
- expected node rank `{0,1}`、global scheduler rank `{0..7}`、distinct hostname、GPU UUID集合を
  manifestとschedulerへ照合する。
- rank 1のcoordinated teardownと早期終了は、rank 1 node-local clock上のchild exitとdurable
  `shutdown.start` receiptの前後で区別する。

rank 1の`Engine()` constructorはdistributed completionまで内部でblockするため、constructor後を
configuration producerにしない。SGLang内部で最終runtime値を解決した後、rank 1がwaitへ入る
直前のhookからnode fragmentをemitする。GPU phase markerはSGLang worker内部へ広げず、benchmark
開始前に接続済みの小さなcontrol forwarderを使う。rank 0 offline hookが`invocation_id`、phase、
start/end、sequenceを各node monitorへbroadcastし、各monitorがnode-local receipt時刻をappendして
ACKする。sender/receipt/ACKをartifactへ残し、異なるnodeのmonotonic値同士を減算しない。このlane
ではnative-worker markerとcontrol-forwarder markerを同じmetricへ混在させない。

generation順序は`start broadcast -> expected 2 node durable ACK -> rank0 timer start -> generate ->
timer end -> end broadcast -> expected ACK`へ固定する。start ACK完了からgenerate開始までのleadと、
generate終了からend ACK完了までのtrailをrank 0 clockでartifactへ残すため、GPU receipt windowが
generationの保守的supersetであることを判定できる。shutdownも`shutdown.start(invocation_id)`を
両nodeへbroadcastし、durable ACK後に`backend.shutdown()`を呼ぶ。

metricは次の形にする。

- `run.launch_argv`: stageが起動したouter MPI argvのcanonical JSON配列
- `parallel.node.<node_rank>.benchmark_argv`: 各nodeのSingularity/SGLang argv
- `parallel.node.<node_rank>.sglang.benchmark_arg.*`: 各nodeのresolved BenchArgs
- `parallel.node.<node_rank>.sglang.server_arg.*`: 各nodeのlaunch-effective ServerArgs
- `parallel.node.<node_rank>.*`: node rank、dist init、MPI rank、affinity、GPU mapping、
  runtime-observed state
- `parallel.node.<node_rank>.process_lifecycle`: child start/end、raw wait status/signal、supervisor
  termination reason、`CLOCK_MONOTONIC` clock IDだけのJSON object

shell表示はviewで作る。同じnode固有fieldをrun-scoped keyへ送らない。
wrapperはshutdown receiptや分類結果を自身のfragmentへ複写しない。`shutdown.start` receiptはnode-agentの
control journalだけ、child lifecycleはwrapper fragmentだけを正本にする。control receiptはrun、node
rank、invocation ID、sequence、local monotonic timestamp、`CLOCK_MONOTONIC` clock IDを持つ。phase reducerが
sender/receipt/ACK間のshutdown invocation ID一致とnodeごとの一意性を先に検証し、viewはそのvalidな一件と
child lifecycleをrun/node rankでjoinする。clock IDも一致する時だけreceiptより前のchild exitをearly
failure、receipt後をcoordinated teardownと導出する。receipt 0/複数、invocation/clock不一致では分類を
作らずreasonを返す。rank 0側timestampを直接比較せず、rank 0 completion file到着だけでrank 1の早期死を
上書きしない。

## Qwen/SGLangで保存する原始事実

### run、stage、allocation

- `run.started_at`、`run.finished_at`、`run.elapsed_sec`
- `run.outcome`、`run.exit_code`
- `run.launch_argv`とnode別actual benchmark argv
- `perf.stage.benchmark.wall_time_sec`とexit/signal/spawn errorを区別するstage termination object
- `scheduler.request`: runner submit boundaryのlane、PJM script identity、exact `pjsub` argv
- `scheduler.accounting`: 固定した`pjstat --history -S --choose ... --data --delimiter`の結果
- `hardware.cpu_numa.nodes`: hostnameごとのhost-invariantなCPU、NUMA、memory構成
- `hardware.gpu.nodes`: hostnameごとのGPU UUID、PCI、memory、driver等の静的構成
- worker affinity/cpusetは`worker_ready`のglobal-rank-scoped runtime fact

started/finishedはUTC wall clock、elapsedは同じsupervisor境界のmonotonic clockで取得するため、
wall-clock jumpとtimestamp精度の影響を受けず厳密な派生関係ではない間だけ残す。同一scope、clock、
precisionへ統一した時点でelapsedまたはtimestamp pairの一方をview派生へ移す。

### SIF、model、dataset provenance

SIF内manifestにはbuild inputsとして次を固定する。

- buildを定義した`2_Qwen` commit
- `5_SGLang_Src`のresolved commit
- base OCI digest、resolved package/toolchain manifest
- original build workflow/run

SIF完成後に実際の参照先binaryをSHA-256で検証し、そのcontent identityとembedded manifestの
identityをatomic sidecarへ保存する。SIF全体hashをSIF自身へ埋め込む自己参照にはしない。
recipeの`From:`はdigest-pinned referenceにし、buildが消費したOCI manifest/config/layer digestを
sidecarへ記録する。archiveとsidecarは二file同時にatomicにはならないため、SIFをversion directoryへ
renameし、SIF hashを持つsidecarを最後のcommit markerとしてrenameする。consumerは両者一致時だけ
reuseする。

modelは全shardのimmutable manifest、datasetはpinned repository revisionまたはfile manifest、
tokenizerとchat templateはmodel artifact identityへ含める。共有pathだけをidentityにしない。
provision時にmanifestと実体の一致を検証する。

raw tableは比較groupとartifact到達に必要なcommit、manifest identity、artifact URIだけを
索引化する。SIFや全model shardを各benchmark runで再hashしない。

### 実workload identity

rank 0は、実際に選ばれたrequest順序を持つcanonical manifestをartifactへ残す。各recordは
少なくとも次を含む。

- dataset manifest identityと元JSON array indexから作るsource row ID
- canonical token-ID arrayのSHA-256であるprompt token content digest
- input token length
- requested output length
- sampling parameter
- request順序

prompt本文と生成本文はBigQueryへ保存しない。BigQueryには次だけを索引化する。

- `artifact.workload.measured.manifest_sha256`
- `workload.request.selected_count`
- `workload.token.input.planned_total`
- `workload.token.output.requested_total`
- `artifact.workload.warmup.manifest_sha256`
- `workload.warmup.request.selected_count`
- `workload.warmup.token.input.planned_total`
- `workload.warmup.token.output.requested_total`
- `workload.warmup.mode`

warmup manifest/modeもmeasured前のcache/stateを変える一致必須条件にする。これらのcountとtoken
totalsはresult completenessをartifact読取なしで判定するためのjustified index
である。総token数が等しくてもrequest順序や長さ分布が異なればmanifest identityが変わる。

### benchmark結果の原始観測値

- `result.request.returned_count`
- `result.request.error_count`
- `result.token.input.total`
- `result.token.output.total`
- `perf.benchmark.generation.wall_time_sec`

returned/error countは実際に返ったresponse配列と各objectを検証して作る。返らなかった件数は
viewで`selected_count - returned_count`として導出する。generation開始前の予定request数を
`successful_requests`と呼ばない。result fileは存在だけでなく、非空、valid JSON、期待row数、
有限で正のlatency、token整合を検査する。

runtime backendの文字列出力はgenerate return直後にdecodeし、各responseの
`meta_info.prompt_tokens`と`completion_tokens`からobserved totalsを作る。malformed/error responseは
`error_count`へ分類し、returned過多、token field欠損も別validation reasonにする。generate全体が
例外ならreturned/error/token countを0と捏造せず、generation lifecycle failedとkey不在で表す。

minimal primary resultの唯一のdurability sourceはrank 0がこの検証直後に作るatomic artifactとする。
temporary fileをwrite+fsyncし、rename後にdirectoryをfsyncしてからprofile finalize、
`get_server_info()`、shutdownへ進む。後段の通常`result_filename`は診断artifactであり、同じraw factの
第二正本にしない。runner parserがprimary artifactを`Fact`へ変換する。

benchmark validityは少なくとも`error_count = 0`、`returned_count = selected_count`、observed
input totalとplanned input totalの一致、有限かつ正のgeneration wall timeを要求する。現在の
`ignore_eos`固定protocolではobserved output totalとrequested output totalの一致も要求する。

request/input/output/total throughputは上記の原始値からviewで計算する。SGLangが出力する
代数的throughputはartifactに残し、fixtureでview値と照合する。`last_gen_throughput`はこれら
から導出不能だが一次比較に使わないため、raw metricへ保存せずartifactだけに残す。

### phase timing

外側から観測可能な境界に合わせ、同じmonotonic clock domainから次を記録する。

- `perf.phase.distributed_engine_init.wall_time_sec`
- `perf.phase.input_prepare.wall_time_sec`
- `perf.phase.warmup_generation.wall_time_sec`
- `perf.benchmark.generation.wall_time_sec`
- `perf.phase.engine_shutdown.wall_time_sec`

distributed engine initはrank 0 processの同じclockで`Engine(...)`呼出し直前からconstructor return
直後までとする。各worker-ready timestampを引き算へ混ぜず、ready eventはglobal rank cardinalityと
runtime completenessだけに使う。input prepareはmeasured request object完成時にtimerを閉じ、
workload manifestのhash/fsyncはtimer外かつgenerate前に行う。

純粋なmodel loadは全worker内の境界を観測できた場合だけ追加する。JITは複数phaseにまたがる
ため、根拠なく`jit_and_warmup`という一時間へまとめない。timed pathではWAL appendだけを
行い、BigQuery送信待ちをstage時間へ混入させない。

### cache identity

各nodeでstaging完了後、benchmark前のinput cacheをcanonical manifestとして保存し、
`cache.node.<node_rank>.input_manifest_sha256`だけを比較identityとして索引化する。
file count、size、cold/warmをraw metricへ重複保存しない。empty、compatible warm、unknownは
manifest dimensionとSIF/model/driver/backend互換規則からviewで導出する。

run後にcacheを共有領域へmergeしても、そのrunのinput identityは変更しない。同じfile count
とsizeでも内容が違えばmanifest identityが変わらなければならない。

### GPU telemetry

各nodeのsamplerはshared durable runtime上のnode-owned JSONLへ一行ずつ`O_APPEND`し、各sampleを
`fdatasync`して
SIGKILLでも直前まで残す。圧縮する場合は独立zstd frameごとにsyncし、sequence gapと未完成frameを
検出できる形に限る。sampleはnode-local monotonic timestamp、wall anchor、sequence、GPU UUID、
logical index、値、poll errorを持つ。control markerを各nodeの同じnode-local clockで受け、expected
nodeのstart/endが一つでも欠ければgeneration aggregateを作らない。node GPU windowは同じ
`invocation_id`のlocal receipt `[start,end)`とし、broadcast/ACK latencyを別artifactで監査する。

BigQueryにはphase/node/GPU scopeを明示して、次だけを集約する。

- valid sample countとvalid observed duration
- generation GPU utilization mean/p95
- memory max
- energyとpower-valid duration
- temperature max、SM clock min

gauge sampleは取得時刻から次sample、phase end、または`2 * nominal_interval`の最小時刻まで
有効なstep functionとして扱い、それより長いgapはcoverageへ入れない。meanはvalid duration
weighted mean、p95は値を昇順に並べたweighted nearest-rank、energyはpowerの同じstep-function
積分とする。最後のpre-start sampleもその有効区間がphase `[start,end)`と交差する部分には使う。
coverageはvalid observed duration / phase durationである。power meanは
energy/power-valid durationからviewで導出する。phase markerが欠けた場合はgeneration aggregateを
作らず、stage全体値で代用しない。raw sampleと集約の一致をfixtureで確認する。

## validity、completeness、comparison eligibility

viewは次を独立に導出する。

- benchmark validity: resultとtoken/count/timeが妥当か
- configuration completeness: required effective argsとphase eventがあるか
- provenance completeness: SIF/model/dataset/workload/cache identityがあるか
- topology completeness: expected rank/node/GPUが揃うか
- telemetry completeness: phase markerとsample coverageが十分か
- comparison eligibility: 一次throughput比較に必要な軸を満たすか
- run outcome: pipelineとschedulerの運用上の成否

telemetry欠損だけをbenchmark failureへ読み替えない。現在の2 node TP=8 laneでは、少なくとも
rank `{0,1}`、2 distinct hosts、4 GPU/node、8 unique GPU UUID、TP=8、nnodes=2、MPI process=2を
照合する。configuration eventやworkload manifestが欠けた成功runはthroughput値を残すが、
comparison eligibleにはしない。

## BigQuery viewとスプレッドシート

Git管理SQLは次の順でviewを作る。

1. `raw_distinct`: 完全一致rowをdeduplicate
2. `run_key_conflicts`: 同じrun/keyの異値を隔離
3. `run_completeness`: validityと各completenessを理由別に表示
4. `qwen_run_facts_v2`: 固定列のrun比較表
5. `qwen_run_diff_v2`: 二run間の比較関連差分
6. `qwen_env_diff_v2`: 全環境変数差分

Gitから生成するcommit dimensionをjoinし、subject、body、commit URLを表示できるようにする。
commit messageをrun metricへ複製しない。Connected Sheetsはraw tableではなくversioned consumer
viewだけを見る。

標準Sheetは少なくともrun、時刻、outcome、completeness、commit、artifact/log link、
workload/SIF/model/dataset/cache identity、主要effective args、topology、primary throughput、
phase時間、GPU集約、反復数、median、MAD、CV、baseline ratioを固定列で表示する。full env
差分は別tabにする。

## 決定状態

| 状態 | 決定 |
|---|---|
| 確定 | raw relationは`run/key/value`の三列、compound valueはcanonical JSON |
| 確定 | 定義した各process scopeの環境変数は機密値を含めfilterせずbyte-exactに全件保存 |
| 確定 | native object/structured sourceを優先し、log regexはfixture付きの最終手段 |
| 確定 | compute側はlocal WALだけ、archive/verify/publish/replayはrunner側 |
| 確定 | observed totalsから派生するthroughputやnode full factsのcommon copyをrawへ重複保存しない |
| 未決定 | score runへ許すinstrumentation overheadの統計的上限 |
| 未決定 | durable archiveの最終保存先、retention期間、quarantineの容量上限 |

## 実装ハンドオフ

### 公開面とfile境界

実装者が新しい抽象を増やさないよう、package内部の依存方向を次に固定する。

```text
SGLang hook -> version付きdomain record / primary artifact
                                      -> pure parser/reducer ---\
I/O probe ----> structured source value -> pure parser --------+-> Fact -> Recorder
StageRunner ---------------------------------------------------/

runner: merge -> ArtifactStore/RunFinalizer -> publish
```

hookはmetric keyや`Fact`を知らず、version付きsource recordまたはprimary artifactだけを書く。

公開するdomain型はimmutableな`Fact`と`Termination`だけにする。`Fact`はrun ID、owner、encoded
stringを持たず、`Termination`は`exit`、`signal`、`spawn_error`をunionとして表す。最低限のcall
surfaceは次で足りる。

```python
Fact(key: str, logical_value: LogicalValue)
init_run(protocol_path, contract_path, runtime_root, identity) -> ManifestPath
Recorder.open(manifest_path, registry_path, producer_id, runtime_dir)
Recorder.append_many(facts)
Recorder.seal()
LifecycleJournal.open(manifest_path, producer_id, runtime_dir)
LifecycleJournal.append_started()
LifecycleJournal.append_final(status, error_count)
collect_once(manifest_path, runtime_dir, producer_id, probe) -> CollectorOutcome
submit_run(manifest_path, runtime_dir, lane_adapter, pjsub_argv) -> SubmissionOutcome
run_stage(stage_name, argv, *, cwd, recorder, lifecycle) -> Termination
finalize_run(manifest_path, runtime_dir, archive_target) -> ArchiveReceipt
publish_run(archive_receipt_path, destination) -> PublishReceipt
reconcile_run(manifest_path, runtime_dir, archive_target, delivery_store) -> ReconcileOutcome
```

`Recorder.open`はpersist済みmanifestとregistryをread-onlyで検証し、runを生成・推測しない。
`LogicalValue`はstring、bytes、bool、int、finite float、Decimal、UTC timestamp、またはJSON treeに
限定する。top-level `None`は禁止し、bytes/Decimal/timestampとcompound JSONのencodingはregistryの
logical typeが決める。
`append_many`だけがlogical valueをcanonical stringへ変換し、一行ずつ`run/key/value`へfsyncする。
`seal`だけがdata fragmentをatomic renameして親directoryもfsyncする。`LifecycleJournal`はcanonical
append/fsyncだけを行い、semantic ownerはprobe外側のsupervisor、`StageRunner`、またはnode agentで
ある。data fragmentのappend/sealが失敗しても`append_final`を試せる構造にし、lifecycleはsealを
待たず回収する。registry loading、canonical encoding、manifest validationは
`journal.py`のprivate collaboratorとし、それぞれを薄いpublic moduleへ分割しない。

log pathはstage nameとruntimeから内部導出し、signalはchild process groupへforwardしてraw wait statusを
保存する。`finalize_run`は唯一のrunner orchestratorであり、privateなpre-bundle merge、bundle作成/
検証、finalizer fact、final merge、metrics sidecar検証を順に行う。別のpublic `merge_run`と二重に
orchestrationしない。

`ArchiveReceipt`はmemory上だけの値ではなく`receipts/archive-verified.json`へsealする。最低限
`schema/run/contract/bundle URI/hash/size/metrics URI/hash/size/target/attempt/started_at/finished_at/outcome/error`を持つ。`PublishReceipt`は
`schema/run/contract/metrics hash/destination/attempt/outcome/response digest`、内部`MergeReport`は
`schema/run/contract/input manifest hash/row counts/conflicts/completeness/output hash`を持ち、いずれも
process再起動後にfileから再構成できる。

`init_run`と`collect_once`は`journal.py`、`submit_run`は`submission.py`、`run_stage`は
`stage_runner.py`、`finalize_run`と`reconcile_run`は`artifact_store.py`、`publish_run`は`publish.py`が
所有する。`cli.py`はargument parseとhandler dispatchだけを行い、manifest生成、scheduler subprocess、
probe lifecycle、quarantine探索を実装しない。

use-case outcomeもclosed schemaにする。`CollectorOutcome`は`status/error_count/fragment_path/complete`、
`SubmissionOutcome`は`bound|failed|unknown|recovered`、receipt path、nullable job ID、`ReconcileOutcome`はreason class、
input receipt、実行action、output receiptを持つ。これらはraw metricではなく、対応するimmutable evidenceを
指すcontrol-plane return valueである。

現行fileからの移行先は次の通りとする。

| 現行 | 移行後の責務 |
|---|---|
| `core.py` | `domain.py`の`Fact`、`journal.py`、runner-only `publish.py`へ分割。空文字fallbackとcredential captureをcompute pathから廃止 |
| `collectors.py` | I/O probeとpure parserを分け、どちらも最後は`Fact`列を返す |
| `stage.py` | outer MPI processだけを待つ`StageRunner`とpure reducerへ分離。`EventReceiver`から`send()`を除く |
| `cli.py` | 下記use caseを配線するcomposition rootだけにする |
| 新規Qwen固有`node_agent.py` | 各node wrapperが一つ起動するGPU sampler、local Recorder、SGLang event endpoint、phase control receiver/ACKを所有 |
| SGLang patch | version付きlocal recordと`PrimaryResultWriter`だけを実装し、metric key、BigQuery、retryを知らない |
| `scripts/run-qwen35-node.sh` | node rank、manifest、node-local endpointを明示する薄いadapterにする |
| `scripts/run-qwen35.pjm` | manifest/runtime pathを明示してcompute collectorとstageだけを呼ぶ |
| `.github/workflows/genkai.yml` | pre-submit manifest、submission binding、terminal後finalize/archive/publishを所有 |

実装fileは計画済みの`domain.py`、`journal.py`、`submission.py`、`stage_runner.py`、`merge.py`、
`artifact_store.py`、`publish.py`とdomain別probe/parserに限定する。`manager`、`service`、汎用event
bus、DI frameworkは追加しない。`submission.py`はintent、scheduler adapter、曖昧結果照合を隠す一つの
深いmoduleであり、collectorやpublisherをimportしない。Qwen固有fileは`node_agent.py`、pureな`sglang_events.py`、
`source_artifacts.py`、`gpu_telemetry.py`だけを許可する。node control protocolを`node_agent.py`へ閉じ、
共通基盤へ漏らさない。

同じfamilyをsource journalとmetric fragmentの両方へ書かない。durability ownerとraw ownerを次へ固定する。

| family | durability source / single writer | Fact owner |
|---|---|---|
| job/host probe、stage termination、node child lifecycle | 対応producerの`Recorder`へ即時append | 各compute/wrapper producer |
| SGLang node config、worker-ready | node agentがnode/global-rank domain journalへappend | terminal `sglang_events` reducer |
| phase sender/receipt/ACK | coordinator/node agentが各control journalへappend | phase reducer |
| GPU sample | node agent内samplerがnode telemetry journalへappend | `gpu_telemetry` reducer |
| primary result | rank 0 `PrimaryResultWriter`のatomic artifact | `source_artifacts` parser |
| warmup/measured workload | rank 0 manifest writerのatomic artifact | `source_artifacts` parser |

`PrimaryResultWriter`はtargetと同じdirectoryへ`O_CREAT|O_EXCL`でtemporary fileを作り、write、file fsync、
no-replace rename、directory fsyncを行う。target既存時はcanonical byte完全一致だけをidempotent success、
異値をconflictとして上書きしない。いずれかが失敗したらbest-effort shutdown後にstageをnonzeroとし、
primary result由来のraw keyは作らない。workload manifest writerも同じno-replace規則を使う。

機械可読正本は`qwen-experiment-metrics/contracts/qwen-v2.toml`一fileにする。contract ID、producer
定義、key pattern、owner、logical type、unit、cardinality、source、scope、capture、required、
missing、validation、正当な重複のjustificationを同じentryに置く。lane/protocolはこのcontractに
定義済みのproducer IDからexpected graphを選ぶ。runtime manifestにはcontract IDとcontent hashを
sealし、raw tableにはroutingに必要なcontract IDだけを索引化する。観測マップの表はこのTOMLから
生成し、手書きproseだけを別に保つ。したがってregistryとMarkdown表の二つを正本にしない。

### CLIと設定境界

CLIは次の七orchestration use caseと一つのQwen固有internal entrypointだけを持つ。behaviorを環境変数で
切り替えず、manifest、runtime、archiveは
明示引数、比較条件はGit管理protocol、BigQuery宛先はrunner側のGit管理設定から読む。

```text
init-run       runnerがnonce、protocol、expected producerをrun-manifest.jsonへseal
submit         intentをsealしてexact pjsub argvを実行し、result/bindingを耐久化
collect        compute producerを一度実行しlifecycle/data fragmentへ記録
stage          target argvをspawnしterminationと時間を記録
finalize       terminal後にdurable sourceを回収/reduceし、merge、archive、remote verifyまで実行
publish        verify済みarchiveのmetrics sidecarだけをBigQueryへ送達
reconcile      invalid監査、archive-retry、replay-pendingをreason別に冪等処理

internal node-agent  各node wrapperが起動しnode-local event/control/telemetryを耐久記録
```

`init-run`以外は`--manifest`を必須にし、`EXPERIMENT_METRICS_RUN`、PID、PJM IDからrunを再構成しない。
`collect`、`stage`、`node-agent`はBigQuery設定やcredentialをimportしない。outer `StageRunner`は
`mpiexec`を一度だけspawnするためremote node agentを直接起動しない。各MPI rankで実行される
`run-qwen35-node.sh`が、そのnodeのcontainerより前に`node-agent --node-rank ...`を起動しdurable readyを
待つ。これによりrank 1にもsamplerとcontrol receiverが確実に存在する。`finalize`と`publish`を分け、
送信障害を理由にcomputeやarchiveを再実行しない。

`submit`は`pjsub`前にexact argvとrun nonceを`submission-intent.json`へfsyncし、stdout/stderrを耐久fileへ
直接redirectする。nonceはlane adapterが保証するscheduler検索可能なjob name/comment/request fieldにも
入れる。明示rejectは`submission-failed.json`、成功はresultとbindingをsealする。応答を確定できない
crashではblind retryせず、nonceでscheduler/accountingを検索する。exact 1 jobだけをbinding化し、0件で
検索完了または複数件、検索不能なら`submission-unknown.json`をsealして人手確認まで停止する。後に
exact 1 jobへ解決した場合もunknownを消さず、`submission-recovery.json`とbindingを追加sealする。この
検索fieldと照合commandを提供できないlaneはunattended v2 submitをreleaseしない。

node wrapperはnode-local filesystemに一意なendpoint directoryを作り、`node-agent --manifest ...
--runtime ... --node-rank ... --local-endpoint-dir ... --role coordinator|participant`を起動する。共有runtimeへ
Unix socketを作らない。containerにはlocal endpointだけを`/run/qwen-observation`へbindし、SGLang patchへ
次の明示引数を渡す。

```text
--qwen-observation-socket /run/qwen-observation/events.sock
--qwen-primary-result-path /run/qwen-source/primary-result.json
--qwen-run-manifest /run/qwen-control/run-manifest.json
```

node-agent間control transport、advertise addressのmachine-readable source、port allocation、connect timeoutは
lane contractへ固定する。rank 0 wrapperは`node-agent --control-listen <address>:<port-or-0>`でcoordinatorを
起動し、実際にbindしたaddress/port、run ID、nonce、protocol versionを
`control/phase/rendezvous.json`へatomic no-replace sealする。rank 1 wrapperはこのreceiptのrun/nonce/
versionとfreshnessを検証し、`--coordinator-address <address>:<port>`を明示してparticipantを起動する。
address/portを環境変数、DNS推測、任意file探索から導出しない。

coordinatorはexpected node `{0,1}`のidentity付きdurable ready ACKを集め、
`control/phase/protocol-ready.json`をsealする。両wrapperはこのreceiptを検証してからcontainerを開始する。
bind/connect/ready timeout、stale/wrong-run receiptはsolverを開始しないconfiguration failureである。

primary result pathはrank 0だけwrite可能にbindする。allocation内producerはnode agent停止時にdomain/
control/telemetryをshared durable runtimeへflush/sealしてからPJM scriptを終了する。runnerの`finalize`は
node/GPU/workerを再観測せず、terminal accountingと既存sourceのpure parser/reducerだけを実行する。

### runtimeとarchive layout

一runのruntimeは次の固定layoutにする。任意file探索や「最後に見つかったJSON」を禁止する。

```text
<runtime>/<run>/
  control/run-manifest.json                    # pre-submit immutable
  control/contract.toml                        # 実使用contractのimmutable copy
  control/protocol.toml                        # 実使用protocol/laneのimmutable copy
  control/submission-intent.json
  control/submission.stdout
  control/submission.stderr
  control/submission-result.json               # raw subprocess outcome when available
  control/submission-failed.json               # explicit rejectionの場合だけ
  control/submission-unknown.json              # ambiguous outcomeの場合だけ
  control/submission-recovery.json             # ambiguous outcomeを後で解決した場合だけ
  control/submission-binding.json              # post-submit immutable
  control/job-entry-receipt.json               # PJM側がbindingを待たずにseal
  control/phase/rendezvous.json
  control/phase/protocol-ready.json
  control/phase/sender.jsonl
  control/phase/node-<node>/receipts.jsonl
  control/phase/node-<node>/acks.jsonl
  metrics/lifecycle/<producer>.jsonl            # partial末尾も回収
  metrics/fragments/<producer>.partial          # append中
  metrics/fragments/<producer>.jsonl            # seal済みrename先
  metrics/fragments/runner-finalizer.partial
  metrics/fragments/runner-finalizer.jsonl
  domain/sglang/node-<node>.jsonl
  domain/worker/global-rank-<rank>.jsonl
  source/primary-result.json
  source/workload-warmup.json
  source/workload-measured.json
  telemetry/node-<node>/gpu-samples.jsonl
  logs/<stage>.stdout
  logs/<stage>.stderr
  receipts/scheduler-terminal.json
  receipts/archive-verified.json
  merge/metrics.pre-finalizer.jsonl
  merge/pre-bundle-report.json
  merge/metrics.jsonl
  archive-staging/bundle.tar.zst
  archive-staging/bundle.outer.json
  archive-staging/metrics.outer.json
```

manifestにexpected pathを列挙し、node/global rankをfilenameから推測せずrecord内部との一致も検査する。
node-local ephemeral endpointはshared layoutに含めず、node wrapperが作ったdirectory内の
`events.sock`と`control.sock`に限定する。job終了時にsocketをarchiveせず、対応するdurable receiptと
lifecycleをarchiveする。
archiveは次の四objectを同じrun prefixへ置く。

```text
bundle.tar.zst       # 下記exact allowlistの事故証拠とsource
bundle.outer.json    # bundleのhash、size、manifest identity
metrics.jsonl        # bundle identityを含む最終raw sidecar
metrics.outer.json   # metrics sidecarのhash、size、contract
```

`bundle.outer.json`はbundle外、`metrics.outer.json`はmetrics外に置くため自己参照しない。remoteで
四objectの存在、hash、sizeを再読取検証して初めてarchive済みとする。

bundle memberはmanifest駆動のexact allowlistとし、`control/`のmanifest、contract、protocol、submission
intent、raw stdout/stderr、実在するresult/failed/unknown/recovery outcome、binding、job-entry/phase receipt、
pre-bundle expected producerの`metrics/lifecycle/`、sealedおよびpartial producer
fragment、`domain/`、`source/`、`telemetry/`、`logs/`、scheduler-terminal receipt、
`merge/pre-bundle-report.json`、自己を除くmember path/hash/sizeを持つ`bundle.manifest.json`を含める。
最終`metrics.jsonl`、outer manifest、archive/delivery receipt、`runner-finalizer` fragment/lifecycle、一時
`metrics.pre-finalizer.jsonl`は含めない。producer fragmentとはpre-bundle mergeが消費したcompute/
runner source fragmentの固定集合を指す。これによりcleanup後もrun identity、expected graph、partial failure、replay根拠を失わず、
finalizer factとのhash cycleも作らない。

runner finalizerはcollector lifecycle keyを作らず、そのstarted/outcome/errorは外側の
`archive-verified`または`archive-retry-pending` receiptだけを正本にする。これによりbundle seal後に
bundle memberが増えない。

`bundle.tar.zst`はpathをbyte順に並べ、tar headerのuid/gid/owner/mtime/modeとzstd parameterを固定して
決定的に作る。保存が必要な元metadataはbundle manifestの値として記録する。同じ入力集合のretryが
同じbundle byte/hashになることをfixtureで保証し、compressor versionもouter manifestへ残す。
最終`metrics.jsonl`は完全一致rowを一件へし、異値conflictは全rowを残したまま`(run,key,value)`の
UTF-8 byte順にsortする。各lineはobject member順`run`,`key`,`value`、空白なし、UTF-8、LF終端へ固定する。

localのmanifest、report、outer、receiptはすべて同directoryの`.partial`へwrite/fsyncし、no-replace
rename後に親directoryをfsyncする。remote四objectは`put-if-absent`とし、既存objectはremote bytes/hash/
sizeがexact一致する時だけ成功扱いにする。異値objectをoverwriteせずcollision quarantineへ送る。

BigQuery attemptはrun archiveとは別namespaceのdurable append-only delivery storeへ
`archive-verified`、`publish-attempt`、`succeeded`、`replay-pending`をimmutable receiptとして保存する。
local `archive-verified.json`はこのreceiptのstaging copyであり、durable copyの確認前にcleanupしない。
runner workspaceはstagingにすぎず正本にしない。invalid inputは`invalid-quarantine`、通信・archive障害は
`archive-retry-pending`に分け、reason classをreceiptへ持たせる。durable delivery/quarantine backend、
retention、容量上限が確定していない間はproduction cutoverのrelease blockerとし、そこへ退避できない
runのruntimeをcleanupしたりworkflowを成功扱いしたりしない。

### 状態と失敗規則

`pjsub`成功応答とjob開始には順序保証がない。PJM supervisorはsubmission bindingを待たず、entry直後に
manifest run IDと自身の`PJM_JOBID`を`job-entry-receipt.json`へO_EXCL相当でsealしてcomputeを続ける。
runnerは`pjsub`返却後または曖昧応答の照合後にbindingをsealし、terminal後のmergeが両receiptのrun/job IDを照合する。
bindingをjob entryの前提条件にしない。

mutableな`status.json`は作らず、次のimmutable evidence DAGから状態を導出する。

```text
run-manifest -> submission-intent -> submit outcome -> submission-binding
                         |                    \-> failed/unknown -> optional recovery/binding
                         +-> job-entry -> compute lifecycle/source/fragments ---\
submission-binding -> scheduler-terminal ---------------------+-> post-terminal reducers
                                                              -> pre-bundle merge
                                                              -> bundle verify
                                                              -> finalizer fragment/final merge
                                                              -> metrics verify/archive receipt
                                                              -> durable delivery receipt
```

- lifecycle、metric `.partial`、SGLang domain journal、phase journal、GPU JSONLの各validatorはcomplete
  line/frameのvalid prefixを回収し、末尾の不完全line/frameだけを破棄する。partialからfactを回収しても
  fragment completenessはtrueに戻さない。required key欠損、expected node/rank欠損は、回収できたvalid factを保持して
  completeness reasonを付ける。空文字やzeroで補わない。
- 同一run/keyの異値は両方を保持し、conflictとしてpublishしてcomparisonから隔離する。
- foreign run、unknown key、owner違反、canonical encoding違反は入力をbundleへ保全するが、信頼できる
  metrics sidecarを作らず`invalid-quarantine`へ送る。
- process timeout、signal、spawn errorでもrunnerはscheduler terminal後にprimary result、domain event、
  GPU sampleをreduceしてfinalizeする。再実行は既存receiptを検証し、同じbyte列なら成功、異なれば
  競合として止める。
- terminal後にもsubmission bindingとjob-entry receiptの片方が欠ける、またはjob IDが異なるrunは、
  compute結果をbundleへ保全するがpublishせずtracking failureとして`invalid-quarantine`へ送る。
- cleanupはverified archive receiptの後だけ許し、publish失敗は`replay-pending`であってcompute失敗に
  戻さない。

### gateを証明するtest

| gate | 必須test/fixture | 証明すること |
|---|---|---|
| G0 依存境界 | `test_dependency_boundaries.py`、CLI subprocess import trace/egress guard | domain/probe/journal/collect/stageはpublisher/credential/networkをimportせず、node-agentはapproved control transport以外のnetwork client、credential、DNS、外部egressを使わない |
| G1 値契約 | `test_domain.py`、RFC 8785/decimal/env byte fixture | 空、zero、欠損、非有限拒否、単射escape、canonical byte |
| G2 submit/binding | `test_submission.py`、binding前job-entry/crash fixture | intent先行、曖昧成功でblind retryなし、0/1/複数job照合、entry/binding join |
| G3 local durability | `test_journal.py`、全journal/artifactの各byte crash matrix | lifecycle/metric/domain/phase/GPU valid prefix、primary no-replace、seal/directory fsyncの冪等性 |
| G4 ownership/merge | `test_merge.py`、全producer graph fixture | owner一意性、family routing、exact dedupe、異値conflict、required欠損 |
| G5 process観測 | `test_stage_runner.py` | actual argv/cwd、process-group signal forwarding、exit/signal/spawn error、monotonic境界 |
| G6 archive/送達 | `test_artifact_store.py`、`test_publish.py`、各PUT/verify/BQ accept直後crash matrix | exact bundle allowlist、deterministic bundle、put-if-absent、receipt再開、remote verify前cleanup禁止、二重replay同値 |
| G7 Qwen意味 | `test_observation_contract.py`、`test_primary_result.py`、`test_distributed_events.py` | contract source/時点/owner、primary crash/既存同異値、node/rank cardinality |
| G8 GPU phase | `test_gpu_reducer.py`、2-node `test_phase_protocol.py` | rank 0/1 node-agent、共有UDS不使用、wrong-run/stale rendezvous、bind/connect timeout、ready直後crash、ACK順、shutdown receipt単一owner、marker欠損、gap cutoff、weighted mean/p95/energy |
| G9 workflow移行 | legacy/canary workflow integration fixture | legacy CLIとv2 CLI、timeout後全source salvage、finalize/publish再開 |
| G10 非擾乱 | lane別hook on/off paired run | workload/result同値性と、未決定のoverhead採用基準を同時に満たすこと |

`test_observation_contract.py`はcontractのschemaとowner/source/capture/absenceの完全性を検査し、
生成した観測マップ表がcommit済み表とbyte一致することを検査する。これが「この値はどこから取るか」
を、一つの正本から人間向け表と実行時validationの双方へ機械的に保証する。

### release単位

移行はkeyごとに二重化せず、run contract単位で行う。

1. legacyのcollector event/result/GPU集約をgolden fixtureとして凍結する。
2. 専用canary protocolだけでpre-submit manifestとlocal WALを導入し、legacy runの送信経路は触らない。
3. canaryでrunner finalize/archive/publish/reconcileとfault injectionを通す。
4. node-local endpoint、worker-ready、workload manifest、atomic primary result、phase control、raw GPU
   sampleをv2へ追加する。
5. 2 node TP=8で意味・result同値性・非擾乱gateを通してから、run単位でv2を既定にする。
6. v2 viewとSheetを切り替え、legacy-only consumerがゼロになってから旧`send()`、fallback result、
   throughput等のsubset keyを削除する。

rollbackはlaneのprotocol contractを直前版へ戻して新しいrunを投入する。既存runのmanifestやrowを
書き換えず、一run内でlegacy producerとv2 producerを混在させない。

## 実装順序

1. 現行key、producer、consumer、artifact保存先をinventory化し、key registry、expected producer
   graph、`run.metrics_contract`を作る。contractなしはlegacy、run内のproducer version混在はinvalid、
   v2 viewはexact contractだけを読む。
2. clean checkout、remote到達性、SIF/model/dataset sidecar manifest、manual provenance
   gateを作る。
3. BigQuery credentialをrunnerだけへ移したうえで、現行metric集合のまま`Fact`、`Recorder`、
   JSONL WAL、runner merge/publish、archive/reconcilerを導入する。
4. `raw_distinct`、conflict、completeness、exact-contract v2 comparison viewを旧collectorと並行して作る。
5. node-local event endpoint、rank 0 result ownership、node-scoped full configurationを実装し、
   397B runのconfiguration欠落と競合を修正する。manifestが固定したexact contractだけをv2 viewへ
   入れ、contractなしはlegacy、異なるcontractのproducer混在はinvalidにする。
6. warmup/measured workload、cache manifest、atomic primary result validatorを追加し、2 node TP=8、
   global worker rank 0..7のcompletenessを確認する。
7. persistent control-forwarder phase eventとnode-local raw GPU sampleを追加し、timed pathから
   BigQueryを外す。
8. Connected Sheetsをv2 viewへ切り替え、旧key consumerがなくなってからthroughput、model名、
   GPU総数等の派生・部分集合keyの新規生成を止める。
9. retention期間後もarchive済みJSONLだけから空tableへ完全replayできることを確認して旧viewを
   廃止する。

冗長keyの停止をview作成より先に行わない。

## 完了条件

- 性能に影響する変更がすべてremote到達可能なcommitにあり、commit bodyから仮説が分かる。
- SIF reuseでもrun commit、original build commit、SGLang commit、content identityを区別できる。
- model shard、dataset、tokenizer、cache、workloadの1 byte/order変更をmanifestが検出する。
- 正当な空文字、未観測、collector失敗、collector未到達、真のゼロをfixtureで区別できる。
- canonical JSONがprocessやobject挿入順に依存せず同じbyte列になる。
- 各key patternのownerが一意で、distinct後のrun/key conflictがゼロである。
- 実2 node runでnode-scoped full configが届き、viewが一致fieldとnode差をlosslessに表示できる。
- expected node rank、global worker rank 0..7、host、GPU UUID、TP、nnodes、MPI processの不一致を
  理由別に検出できる。
- prompt順序だけが違う同token総数workloadをmanifest identityで分離できる。
- result欠損、不正JSON、returned count/token不一致、非有限時間をcomparison対象外にできる。
- viewの四つの代数的throughputがSGLang artifactと許容誤差内で一致する。
- phaseへ既知delayを入れたfixtureでinit、prepare、warmup、generation、shutdownを分離できる。
- generation marker欠損時にGPU aggregateを全stage値で代用しない。
- count/sizeが同じ別cacheをmanifest identityで区別できる。
- BigQuery停止、append直後crash、部分fragment、二重replay後もJSONLとlogical row集合が一致する。
- upload失敗時にruntimeが消えず、成功・失敗・timeout runのlog、GPU sample、JSONLを回収できる。
- instrumentation on/offの一次指標差が事前閾値内、またはscore runから分離される。
- 同一protocolの反復が実行順とblockを含め復元でき、全点、median、MADがSheetに表示される。
- treatment値を含む同一条件だけが同じreplicate groupへ入り、宣言したtreatment差だけが同じ
  comparison set内で許される。
- Sheets利用者がJSON展開なしでcommit、completeness、一次結果、反復統計、artifactへ到達できる。
