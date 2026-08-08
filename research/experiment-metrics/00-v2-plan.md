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

`run`は一childの全producerが共有するglobally uniqueでimmutableなIDである。各submission subjectにはjob投入前に
OS CSPRNGの16 byteを32文字のlowercase hexで表す128-bit `subject_nonce`を一つ生成する。regularでは
`submission_subject_id=run_id`、qualificationではglobally uniqueなallocation IDをsubject IDとし、二child run IDは
allocation ID、arm ordinal、各child用CSPRNG nonceから別々に固定する。manual経路でも同じsubject/child規則を使う。
`pjsub`返却後にしか分からないPJM job IDはrun/allocation IDへ
含めず、immutableなsubmission bindingと`scheduler.job_id`で結ぶ。初期run manifestを書き換えない。
run IDとallocation IDはUTF-8で`^[a-z0-9][a-z0-9._-]{0,199}$`を満たすよう生成し、同じ値を安全なruntime path
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
- supervisorは各collector callのstatus/error countを一つの`collection-status.json`へまとめる。個別の
  started/final耐久journalは作らない。post-run validatorはstatusとvariant別expected fragment/keyを照合する。
  status file自体がない中断runはincompleteであり、残ったsource/logだけをbest effortで回収する。

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

## raw metrics exportと送達

v2は任意のcrash境界から一件も失わず再開できることを目的にしない。release gateは、成功した各run
variantの正常pathでcontractが要求する設定、性能値、raw log、PJM accounting、profile runのprofile rawが
Artifactへ収録され、post-run validatorで確認できることである。失敗・中断runの部分回収はbest effortとし、
取得不足なら比較不適格として再実行する。

`metrics.jsonl`はBigQueryと同じ`run/key/value`を持つrun単位のraw exportである。別event
envelope、型別value列、ack queueは作らない。

target programの薄いhookが出すversion付きdomain recordは、local adapterへの入力または
diagnostic/source logであって第二の送達queueではない。単発factはadapterがregistryで検証してraw exportへ
変換する。反復eventを最終的に一arrayへするfamilyはline-buffered domain JSONLへ書き、stage終了時に
flushする。terminal後のpure reducerが一度だけFactへ変換する。稀なprocess/node failureで末尾が欠ける
ことは許容し、正常終了したのにexpected recordが毎回欠ける実装をreleaseしない。registryはsource
JSONL/artifactとmetric fragmentのどちらを正本にするかfamilyごとに固定する。

```text
compute producer
  -> node別のmetrics JSONLへappendしstage終了時にflush
  -> 完了fragmentをtemporary fileからrename
  -> runnerがexpected node、fragment、expected fileを一度検証してmetrics.jsonlへmerge
  -> log/sample/source/fragmentのrun bundleとartifact-manifest.jsonを作成
  -> run bundle、merged JSONL、artifact manifestをGitHub Artifactへupload
  -> upload成功後にGenkai runtimeを削除可能
  -> archive jobがhome RAIDへscpし、runnerがBigQueryへ送信
```

- 一つのJSON objectを一lineで書き、producer終了時にflush/closeする。各lineやdirectoryごとの`fsync`は
  要求せず、必要なら一定間隔のbuffer flushを使う。
- expected producer集合はGit管理protocolとrun manifestからjob投入前に固定する。scheduler
  accountingとobserved placementは、その期待集合が実際に成立したかを後から検証するsourceである。
- mergeはrun ID、JSON、owner、expected node、一意性を検査し、欠損や異値競合を隠さない。
- BigQuery `insertId`はbest-effortの補助にすぎない。logical viewが完全一致rowをdeduplicate
  し、異値競合を隔離する。
- replayは同じrowを全件送ってよい。送信成功後もarchiveのJSONLを変更しない。publish失敗時はhomeまたは
  GitHub ArtifactのJSONLからworkflowを手動再実行する。
- upload失敗はattempt failureとする。upload成功後はruntimeを削除してよく、home転送のSCPが
  失敗した時だけGitHub Artifactを残す。

artifact identityの自己参照を避ける。全fragmentを一度mergeして`metrics.jsonl`を確定した後、log、raw GPU sample、
source、producer fragmentからrun bundleを作り、bundle、metrics sidecar、profile raw tarのpath/size/hashを
`artifact-manifest.json`へ保存する。bundle hashやmetrics sidecar自身のhashをmetricへ戻して再mergeしない。publishはremote archive
verificationを前提にせず、同じattemptのlocal/GitHub artifactを入力にできる。

timed benchmark中はBigQueryへ接続しない。送信用credentialはrunnerだけが保持するため、
PJM processの性能条件へ混入しない。各snapshotはそのscopeに実在する環境をfilterなしで
保存し、機密値も例外扱いしない。

## 最小の実装境界

対象programへ計測システムの都合を持ち込まないため、責務を次の六境界へ固定する。

1. domain probe/parserは`Fact(key, logical_value)`だけを返すpure functionとする。run ID、文字列化、
   filesystem、BigQueryを知らない。
2. `recording.py`の`Recorder`だけがpersist済みrun manifest、registry、canonical encoder、factのbuffered
   append/flush、data fragmentのcompletionを所有する。supervisor/node-agentはcollector結果を一つの
   `collection-status.json`へまとめる。
3. `StageRunner`だけがspawnするactual argv、monotonic timer、signal、child wait statusを所有し、
   観測したfactを`Recorder`へ渡す。
4. runner側`merge`がexpected producer、run ID、owner、一意性、required key、fragment完了を検証し、
   archive対象のmerged JSONLを作る。node-scoped full factをrun-scopedへ変換しない。
5. `ArtifactStore/RunPackager`がrun bundle/manifestの作成と検証を所有する。GitHub Artifact uploadと
   home移管はworkflowが担当し、汎用reconciler、quarantine service、delivery ledgerは作らない。
6. runner側`publish`だけがBigQuery credential、retry、replayを所有する。collector、SGLang、PJM
   processはBigQuery、artifact upload、HTTP/cloud client等の外部送達networkを使わない。唯一、
   `node-agent`だけがlane contractで許可した明示rendezvous endpointへのrun内control transportを使える。
   credential、DNS discovery、外部egressを許可しない。

SGLang本体へ必要なhookは、version付きlocal structured eventを定義済み境界でだけ書く薄いadapterとする。
metrics packageやBigQuery clientをimportせず、送達、再試行、key展開を知らない。実装上は
`domain.py`、`recording.py`、`submission.py`、`stage_runner.py`、`merge.py`、`artifact_store.py`、`publish.py`、domain別
probe/parserを深いmodule境界にし、CLIはそのcomposition rootに限定する。

QwenのSGLang patch内で唯一のresult writerとして、rank 0の`PrimaryResultWriter`だけは
`backend.generate` return後にprimary artifactをtemporary fileからrenameして完成させる。これはmetrics writerでも
bundle storeでもなく、metrics keyを知らないsource-artifact writerである。schemaはcontract、run、
invocation ID、generation duration、returned/error count、observed input/output totals、validation
reasonsに固定し、runnerのprimary-result parserだけが対応raw keyのownerになる。

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
- `experiment.pair_id`: 欠損・置換後もA/B対応を推測しないprotocol割当ID
- `experiment.planned_replicate_group_id`: submit前に割り当てる非NULL arm/group ID
- `experiment.planned_comparison_set_id`: submit前に割り当てる非NULL比較集合ID

探索比較の既定は各条件5 attemptとし、記述統計だけに使う。同じ時間block内でA/Bを
事前乱数化し、順序を均衡させる。5 attemptや有効run 3件を、equivalence、non-inferiority、
release gateの根拠へ流用しない。意思決定を伴うprotocolは、estimand、margin、想定効果、
分散、alpha、powerからplanned attempt数を事前dimensioningする。invalid/failed runを遅かった
という理由で除外・置換せず、最大attempt数と置換規則も事前にcommitする。

比較次元はGit管理SQLで次へ分類する。

- 一致必須: workload、SIF/model/dataset、effective args、cache input、resource shape、
  instrumentation mode
- 意図的に変えるtreatment: 対象commit、cluster、node/GPU type等
- nuisance: hostname、GPU UUID、開始日時、queue待ち、個別node identity
- outcome: throughput、phase時間、resource値

protocol commitはtreatmentにするkey集合と一致必須key集合を事前宣言し、runner manifestがexact policyと
`dimension_policy_hash`をsealする。contractはdimensionごとのallowed classだけを制約し、protocolごとに
変わり得るclassを固定しない。viewは次の二つを導出し、fingerprint自体はrawへ保存しない。hash入力は
contract ID/hash、protocol commit/hash、dimension policy hash、dimension name、logical type、canonical
valueのsorted tupleである。required dimensionがmissing/conflictingならfingerprintを作らずreasonを返す。

- `observed_replicate_fingerprint`: 実観測した一致必須条件とtreatment値を含むnullable検証結果。
- `observed_comparison_set_fingerprint`: 実観測値から宣言済みtreatment次元だけを除いたnullable検証結果。

attempted countのgroupにはnullableなobserved fingerprintを使わない。pre-run protocol assignmentが必ず
非NULLのplanned group/set IDをsealし、failed/invalid runもそこへ残す。eligible countだけが、実観測
fingerprintとplanned policyの一致を要求する。
`planned_replicate_group_id`はcontract hash、protocol hash、dimension policy hash、planned must-matchと
planned treatment dimensionのcanonical tupleから生成する。`planned_comparison_set_id`は同じnamespaceで
宣言treatmentだけを除いたplanned tupleから生成する。`group-1`等の任意labelは表示属性にすぎない。

commit、artifact、cluster等がtreatmentならcomparison set内では差を許すが、replicate groupは
必ず分離する。nuisanceとoutcomeはどちらにも含めない。競技提出用bestと、環境比較用の全点、
median、MAD、ratio、不確実性を分ける。

上記は通常の性能比較policyである。計測擾乱専用protocolだけは、`instrumentation_mode`、
`instrumentation_bundle_hash`、およびinstrumentation patchだけが異なるSIF/build artifactをtreatmentへ
再分類する。これを行わずにon/offを同じcomparison setへ入れたり、通常比較でmode差を許したりしない。

## 分散producer契約

現行の共有filesystem上Unix domain socketをnode間event transportとして使わない。Unix
socket endpointは各node localに作り、各nodeのeventはそのnodeのJSONLへ書く。

- rank 0だけがworkload manifest、benchmark result、run-scoped phaseのauthoritative
  producerになる。
- 各nodeはresolved args、environment、actual argv、affinity、GPU mapping、telemetryを
  node scopeで出す。
- rawはnode/rank-scoped full factを正本として保持する。viewだけが全expected scopeの一致を
  検証したfieldをcommon表示し、rawにcommon/deltaの第二表現を作らない。
- expected node rank `{0,1}`、global scheduler rank `{0..7}`、distinct hostname、GPU UUID集合を
  manifestとschedulerへ照合する。
- rank 1のcoordinated teardownと早期終了は、rank 1 node-local clock上のchild exitと
  `lifecycle.shutdown.start`受信時刻の前後で区別する。

controlを二familyへ分ける。`lifecycle control`は2-node実行を正しく終了させる共通correctness harnessとして
bench/profileとqualification両armで共通にし、startup readyと`lifecycle.shutdown.start`の通常handshakeを
最終summary logへ記録する。`GPU phase control`はprofile-onlyで、warmup/measured start/endとnode受信時刻を
profile summaryへ記録する。両者のproducer/completenessを混ぜない。

rank 1の`Engine()` constructorはdistributed completionまで内部でblockするため、constructor後を
configuration producerにしない。SGLang内部で最終runtime値を解決した後、rank 1がwaitへ入る
直前のhookからnode fragmentをemitする。以下のGPU phase marker/control forwarderはprofile protocolだけで
起動し、benchではproducer graphから外してNULLにする。profileではmarkerをSGLang worker内部へ広げず、benchmark
開始前に接続済みの小さなcontrol forwarderを使う。rank 0 offline hookが`invocation_id`、phase、
start/end、sequenceを各node monitorへbroadcastし、各monitorがnode-local受信時刻を記録して
ACKする。最終summaryをartifactへ残し、異なるnodeのmonotonic値同士を減算しない。このlane
ではnative-worker markerとcontrol-forwarder markerを同じmetricへ混在させない。

profileのgeneration順序は`start broadcast -> expected 2 node ACK -> rank0 timer start -> generate ->
timer end -> end broadcast -> expected ACK`へ固定する。start ACK完了からgenerate開始までのleadと、
generate終了からend ACK完了までのtrailをrank 0 clockでartifactへ残すため、GPU observation windowが
generationの保守的supersetであることを判定できる。全purposeでgeneration timer終了後に
`lifecycle.shutdown.start(invocation_id)`を両nodeへ一度だけbroadcastし、通常ACK後に
`backend.shutdown()`を呼ぶ。この一往復のstage-wall影響はD-001 bundleに含める。

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
wrapperはshutdown受信や分類結果を自身のfragmentへ複写しない。node-agentの最終summaryをshutdown受信の正本、
wrapper fragmentをchild lifecycleの正本にする。summaryはrun、node rank、invocation ID、sequence、local
monotonic timestamp、`CLOCK_MONOTONIC` clock IDを持つ。reducerがshutdown invocation ID一致とnodeごとの一意性を検証し、viewはそのvalidな一件と
child lifecycleをrun/node rankでjoinする。clock IDも一致する時だけshutdown受信より前のchild exitをearly
failure、受信後をcoordinated teardownと導出する。受信record 0/複数、invocation/clock不一致では分類を
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
- `hardware.gpu.nodes`: benchでもmodel load前に一度だけ取るGPU UUID/model/memory/driverとlogical mapping
- `hardware.cpu_numa.nodes`: profileだけで取るfull CPU/NUMA/memory/placement診断。bench consumerは理由なしのNULL
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
identityをsidecarへ保存する。SIF全体hashをSIF自身へ埋め込む自己参照にはしない。
recipeの`From:`はdigest-pinned referenceにし、buildが消費したOCI manifest/config/layer digestを
sidecarへ記録する。build scriptがSIFとsidecarを書き終えて成功した後だけcacheへ公開し、reuse時に
両者のidentity一致を検査する。任意のcrash境界から二fileを復旧するtransactionは作らない。

modelは`source/model-manifest.json`へ、repository/revisionが分かればその値、`config.json`と
`model.safetensors.index.json`のSHA-256、indexが参照するsorted unique shardのrelative path/size、tokenizer/
chat-template fileのpath/size/SHA-256を保存する。400 GBのshard本文はrunごとに再hashしない。datasetはworkflowで
repository revisionをpinし、`source/dataset-manifest.json`へrepository/revision、relative file name、size、
SHA-256を保存する。共有pathだけをidentityにしない。provision時にmanifestと実体の一致を検証する。

raw索引は`artifact.model.manifest_sha256`、`artifact.dataset.manifest_sha256`と各artifact URIだけを持つ。
repository/revision/memberはmanifest内の完全recordが正本であり、個別raw keyへ再掲しない。

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

minimal primary resultの唯一のsourceはrank 0がこの検証直後に作るartifactとする。
temporary fileをwrite/closeしてrenameしてからprofile固有取得または
shutdownへ進む。`get_server_info()`は一次factを持たずprofileだけで呼ぶ。通常benchはprimary result完成後に
直ちにlifecycle shutdownへ進む。後段のnative `result_filename`はrollout後の通常bench/profileでは停止し、同じraw factの
第二正本にしない。D-001 qualificationではOFF/ON共通estimand sourceとしてだけ保存し、runner parserが
purposeに応じたauthoritative artifactを`Fact`へ変換する。

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
workload manifestのhash/writeはtimer外かつgenerate前に行う。

純粋なmodel loadは全worker内の境界を観測できた場合だけ追加する。JITは複数phaseにまたがる
ため、根拠なく`jit_and_warmup`という一時間へまとめない。timed pathではlocal source writeだけを
行い、BigQuery送信待ちをstage時間へ混入させない。

### cache identity

各nodeでstaging完了後、benchmark前のinput cacheをcanonical manifestとして保存し、
`cache.node.<node_rank>.input_manifest_sha256`だけを比較identityとして索引化する。
file count、size、cold/warmをraw metricへ重複保存しない。empty、compatible warm、unknownは
manifest dimensionとSIF/model/driver/backend互換規則からviewで導出する。

run後にcacheを共有領域へmergeしても、そのrunのinput identityは変更しない。同じfile count
とsizeでも内容が違えばmanifest identityが変わらなければならない。

### GPU telemetry（profile only）

各nodeのsamplerはnode-owned JSONLへbuffered appendし、一定間隔と正常終了時にflush/closeする。
稀なprocess killで末尾sampleを失うことは許容する。sampleはnode-local monotonic timestamp、wall anchor、sequence、GPU UUID、
logical index、値、poll errorを持つ。control markerを各nodeの同じnode-local clockで受け、expected
nodeのstart/endが一つでも欠ければgeneration aggregateを作らない。node GPU windowは同じ
`invocation_id`のlocal observation `[start,end)`とし、broadcast/ACK latencyをprofile summaryへ残す。

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
- qualification native-result validity: qualification両armの共通native estimand sourceがparse可能で正有限か
- configuration completeness: applicableなeffective args/config endpointとstartup-readyがあるか
- lifecycle completeness: purpose/armでrequiredなstartup-readyとshutdown summaryが揃うか
- provenance completeness: SIF/model/dataset/workload/cache identityがあるか
- topology completeness: expected rank/node/GPUが揃うか
- telemetry completeness: profile protocolでphase markerとsample coverageが十分か。benchではNULL
- score comparison eligibility: benchの一次throughput比較に必要な軸を満たすか
- qualification arm eligibility: D-001の一childとしてcommon sourceとarm固有required familyを満たすか。profileではNULL
- run outcome: pipelineとschedulerの運用上の成否

telemetry欠損だけをbenchmark failureへ読み替えない。現在の2 node TP=8 laneでは、少なくとも
rank `{0,1}`、2 distinct hosts、4 GPU/node、8 unique GPU UUID、TP=8、nnodes=2、MPI process=2を
照合する。configuration eventやworkload manifestが欠けた成功runはthroughput値を残すが、
score comparison eligibleにはしない。

`run_purpose`はplanned/observed双方の必須比較次元で、planned group/set IDのhash入力にも必ず含める。
`score_comparison_eligible`は`run_purpose=bench`かつexact D-001 qualification tupleがadopt済みの場合だけ
trueになり、`qualification_arm_eligible`は`run_purpose=qualification`の一runだけから導出する。
`qualification_pair_eligible`はrun列やraw factへ複写せず、両arm、parent allocation binding、common-source一致を
joinしたpair grainで一度だけ導出する。
profile/qualificationはthroughputを算出・表示してもscore stats、best、rankingへ入れない。

## bench/profile trackの分離

重いprofilerやGPU pollingを挟むrunと公式性能を測るrunは、protocolだけでなくsource trackも分離する。

- `bench`: score、環境比較、qualificationの基点。profiler、GPU polling、verbose traceを入れず、対象pathの
  観測はqualification済みscore bundleだけにする。
- `profile`: `bench`のexact commitへprofile専用overlayを載せる診断track。値は原因探索に使うが、
  score、replicate統計、環境ランキングへ混ぜない。
- `qualification`: correctnessに必要な共通lifecycle harnessだけを持つreference `bench`と最終score bundleを
  直接比較する専用protocol。profile trackの
  結果で非擾乱を代用しない。

ユーザー判断により、benchでprofile専用metricを取得できないことは許容する。contractの各family/consumer列は
`applicable_run_variants`（`bench`、`profile`、`qualification:off`、`qualification:on`の集合）、
`missing_policy_when_applicable=reasoned_missing`、
`projection_policy_when_inapplicable=silent_null`を別fieldとして持つ。expected producer graphは
run variantごとに作る。同じ列でもapplicable variantでcollectorが失敗した時は通常どおり
status/Evidence/reasonを生成し、非適用variantではRuns projectionを`NULL`にするだけでraw key、family status、
Evidence行、理由を生成しない。

| family | bench | profile | benchで取らない時のRuns値 |
|---|---|---|---|
| source/SIF/model/dataset/workload/cache、exact outer/node argv、job/node env、scheduler、termination、purpose-authoritative result/throughput、rank/host/GPU/TP topology、common lifecycle control | required | required | 欠損ならbench不適格 |
| container launch/worker env、effective-config event、PrimaryResultWriter | 全v2 benchでrequired。decision pending中はaudit-only、adoptedだけscore | required | 欠損ならapplicable bench/profile不適格 |
| GPU polling/aggregate、phase control marker、profiler、verbose backend trace、full placement dump | not expected | protocol指定時required | `NULL` |

benchで取得しないprofile専用familyはexpected producer graphから外し、raw key、family status、Evidence行、
NULL理由、reason codeを一切作らず、typed/Sheet値だけを`NULL`にする。benchの`score_comparison_eligible`や
completeness分母へ含めない。たとえばbenchでは`telemetry_complete`と全GPU projection列がNULLになり得る。
反対にbench/profileそれぞれでrequiredと宣言したfamilyのcollector failureはrun全体の通常の品質判定として
扱う。profile値を同じprofile baseのbench runへ移植しない。

qualificationは同じpurpose内でもarmごとにgraphが異なるため、expected graphを
`(run_purpose, qualification_arm)`で解決する。`qualification_arm`はplanned treatment tupleの一fieldである。

- `qualification/OFF`: 両arm共通のsource/SIF/model/dataset、Git管理planned workload policy/seed/argv、job/node env、
  scheduler、termination、native result、外側timer、common lifecycle controlだけをrequiredにする。実selected
  ordered workload manifestは非適用でNULLだけにする。
- `qualification/ON`: OFFのcommon familyに加え、actual workload manifestを含むD-001対象のexact minimal score
  bundleだけをrequiredにする。

両armともprofile-only列は非適用variantなので`silent_null`でNULLだけを返す。OFFでON専用列も同じくNULLだけとし、
理由/status/Evidenceを作らない。最終benchは、実bundle hashがD-001 decision内の
`instrumentation_bundle_hash`と一致し、現在runから
再構成したqualification tuple全体がdecision内の`qualified_candidate_tuple`と一致し、profile-only producer IDを
一つも含まない時だけそのbundleを選べる。decision identityはdecision record全体のcanonical hashであり、
bundle hashとは比較しない。

Git運用はlong-livedな相互mergeではなく、`bench -> profile`を基本方向にする。profile側はbenchの更新を
merge/rebaseして追従する。profileで見つけた最適化はprofile hookを含まない独立commitとしてbenchへ実装し、
その後benchからprofileへ取り込む。profile branch全体をbenchへmergeしない。必要なら二つのworktreeを使う。
branch名はmoving labelなのでidentityにせず、run manifestは次のtagged-union canonical source-track recordをsealする。

```text
bench:
  {run_purpose, repository_identity, score_reference_commit, bench_head_commit}
profile:
  {run_purpose, repository_identity, profile_base_commit, profile_head_commit}
qualification:
  {run_purpose, qualification_arm, repository_identity,
   score_reference_commit, source_head_commit}
```

`source_track_manifest_hash = SHA-256(canonical source-track record)`をindexにする。hashをrecord自身へ入れて
再帰させない。全v2 benchはscore bundleを持ち、score referenceがbench headのancestorであることを要求する。
decision pending中のbenchもaudit-onlyで同じbundle/source validationを行う。profileではprofile baseがprofile headの
ancestor、qualification OFFではhead=score referenceを要求する。
qualification ONではscore referenceがsource headのancestorであり、bench/qualification ONのbaseからheadへ
manifest/build validation時に計算したcanonical tree diffがproposed SIF/build manifestとinstrumentation bundleの
`instrumentation_patch_identity`へexact一致し、profile-only overlay、profiler flag、workload変更を含まないことを
required gateにする。algorithm/version、base/head、expected patch identity、pass/failをcontent-addressedな
`control/source-track-validation.json`へsealし、query時にGit treeを再取得しない。clean checkoutのHEAD/treeと
proposed buildのsource head/treeがrecordと一致することを検査し、
submodule pointer、file mode、symlink、LFS object identityもtree/diff検証対象にする。全variantのhead/base commitは
remote reachableまたは耐久source archive済みでなければ投入しない。base/headから決定的に計算できるdiff hashは
rawへ重複保存しない。

`instrumentation_bundle_hash`は、score/profile pathで有効なhook、event、handshake protocol、sample interval、sink、
buffer、timeout、wrapper/node-agent executable hash、observation contract version、env capture scope、
SIF/build patch identityを列挙したcanonical manifestのcontent hashである。`lightweight`等の
labelだけでは同一視しない。`source_track_head_commit`、`score_reference_commit`、`profile_base_commit`等のconsumer列はsource-track recordから
導出し、同じcommit tupleを別raw keyへ複写しない。profile runはprofile baseが同じでもbundle、SIF、argv、
environmentを独立に保存し、「profileで速かった」をbenchの性能値へ読み替えない。

## BigQuery viewとスプレッドシート

logical schemaを増やさず、raw tableはdaily ingestion-time partition、cluster順`run,key`で作り、
`require_partition_filter=false`とする。`_PARTITIONTIME`はUTC delivery partition bucketであってrun時刻にも
exact publish時刻にもならず、replayで変わり、streaming write-optimized storageでは一時的にNULLにも
なり得るためcomparison/freshness factへ使わない。project中はpartition expirationを設定しない。
recent publish確認だけ`(_PARTITIONDATE >= ... OR _PARTITIONTIME IS NULL)`を使い、run lookupは
cluster filterを使う。semanticなdedupe/conflict/comparisonは正しさのため全partitionを対象にする。

Git管理SQLは次の非二乗relationを作る。

| object | row grain | 役割 |
|---|---|---|
| `raw_distinct` | unique `(run,key,value)` | 完全一致retryだけをdeduplicate |
| `run_key_singletons` | `(run,key)` | distinct valueが一つだけのtyped pivot入力。異値keyを通さない |
| `run_key_conflicts` | `(run,key,value)` | 異値が二つ以上あるkeyの各distinct valueを一行ずつ保持し、値を選ばない |
| `qwen_run_reason_items_v2` | `(run,reason_item_id)` | class/code/subject/canonical detail JSONを持つ全理由の唯一のlong relation |
| `qwen_run_facts_v2` | one row per run | conflict-free factだけをtyped固定列へpivot |
| `qwen_run_commits_v2` | `(run,repository,commit_role,commit)` | source-trackとSIF manifestからrole付きcommitを導出し、Git subject/body/URLをjoin |
| `qwen_run_evidence_items_v2` | `(run,family,evidence_item_id)` | producer/capture/scope/raw key/artifact member/valueごとの決定的evidence |
| `qwen_run_family_status_v2` | `(run,family)` | evidence itemsからexpected/observed/status/reasonを集約 |
| `qwen_qualification_allocations_v2` | `(allocation_id)` | parent evidence artifactからordered child IDs、binding、scheduler request/accounting/terminal、job-entry env locator/statusを一度だけ投影 |
| `qwen_comparison_dimensions_v2` | `(run,dimension)` | `must_match/treatment/nuisance/outcome`とcanonical value |
| `qwen_estimand_points_v2` | `(analysis_purpose,run,estimand_id)` | bench/qualificationのpurpose-awareな基底point。profileは行を出さない |
| `qwen_replicate_points_v2` | `(run,estimand_id)` | `estimand_points`のscoreだけを取り、benchのprotocol/group/treatment、attempt/order/value、eligible/reasonsを全点表示 |
| `qwen_replicate_stats_v2` | `(contract_hash,protocol_hash,dimension_policy_hash,planned_replicate_group_id,estimand_id)` | bench attempted全件とobserved一致済みeligible n、median、MAD、CV。bestとは別 |
| `qwen_paired_points_v2` | `(analysis_purpose,contract_hash,protocol_hash,dimension_policy_hash,planned_comparison_set_id,pair_id,estimand_id,baseline_treatment_id,candidate_treatment_id)` | bench通常比較またはqualificationのprotocol指定pair。profileは行を出さない |
| `qwen_paired_comparisons_v2` | `(analysis_purpose,contract_hash,protocol_hash,dimension_policy_hash,planned_comparison_set_id,estimand_id,baseline_treatment_id,candidate_treatment_id,estimator_id)` | ratio方向、pair数、interval、decision、reason。未決定基準はNULL |
| `qwen_env_facts_v2` | `(run,scope_kind,scope_instance,escaped_name)` | 各scopeのraw full snapshotだけをdecodeしたbyte-exact name/value |
| `qwen_run_diff_v2(left_run,right_run)` | table function、一expected dimensionまたはsentinel一行 | 二runのvalue/equality/class/blocking reason |
| `qwen_env_diff_v2(left_run,right_run)` | table function、一scope/nameまたはsentinel一行 | full outer diffのadded/removed/changed/sameと左右値 |
| `qwen_archive_attempts_v2` | `(github_run_id,github_run_attempt)` | workflowが書くupload/SCP/publish outcomeと時刻、GitHub Artifact保持有無、home空き容量 |
| `qwen_sheet_status_v2` | one row | SQL/contract identity、query時刻、latest publish/archive、未移管Artifact数、home空き容量、issue件数 |

environmentの`scope_kind`はcontract enum `job_entry|host_exec_node|sglang_launch_node|worker_global_rank`、
`scope_instance`はnullable node/global rank等をfield名付きcanonical JSONにしたinjective IDで、Sheetにはtyped
scope列も出す。display hostnameだけをjoin keyにしない。`escaped_name`はbyte nameの単射表現であり、同名でも
scopeが違えば別rowである。

archive workflowは一attempt一rowのoperational recordだけを残す。hash chain、delivery FSM、receipt storeは
作らない。Data Statusはこのtableの`MAX`時刻と最新attempt、GitHub artifact inventory、home `df` snapshotを
表示する。scientific factやprofile missing reasonへdelivery状態を複写しない。
family statusはevidence itemのaggregateであって同じgrainのsubsetではない。別の保存表やproducerを作らず、
run facts/statusのmissing/incomplete countも同じaggregateを使う。
`reason_item_id`はreason class/code、`subject_kind=raw_key|consumer_column|family|contract|dimension|evaluator`、
subject ID、nullable raw key、`detail_sha256`から決定的に作る。`detail_json`はexpected/actual、evaluator、
source locator等を持つcanonical typed JSONで、hashだけで内容を捨てない。blocking booleanも持つ。

上表は独立した保存メトリクス集合ではなく、rawを唯一の科学的正本とするlogical query interfaceである。
相互に列が重なるprojectionも別のraw producerやscheduled copyを作らない。非NULL`expected_locator`は
contract field/key patternまたはartifact fieldとexpected scope instanceを一意に表す。`source_instance_id`は
contract hash、expected locator、producer、capture、expected scope instanceだけから作る観測非依存IDである。
`evidence_item_id`はsource instance ID、evidence kind、nullable concrete raw keyまたはartifact content
identity/member、nullable value hashまたはmissing sentinelから決定的に作る。mirror URIは表示locatorであり
IDへ入れない。manifestの
expected node/rank/source graphからmissing instanceも行として展開し、artifact-only証拠はraw keyをNULLに
する。同一source instanceの異値はvalue hashごとに別itemとしstatusをconflictingにするが、familyのexpected/
observed countはdistinct source instance IDを数えて水増ししない。全valueは`run_key_conflicts`へ戻せる。

`qwen-v2.toml`にはraw contractと同じcommitで`[[consumer.run_column]]`を置く。各固定列はexact
`name/type/unit/source family/raw pattern|artifact field/scope-cardinality/reducer/conflict rule/
transform_id/applicable_run_variants/missing_policy_when_applicable/
projection_policy_when_inapplicable/dimension name/allowed classes`を宣言する。
applicable variantの`reasoned_missing`列だけ`null-reason`を必須にする。`silent_null`はsource種別でなく
当該run variantで非適用と宣言されたRuns列へ作用し、RunsをNULLにするだけでraw key、family status、Evidence、
reason item/codeを生成せず、NULL reason検査からも外す。複数sourceの
派生列はordered source listを持つ。
生成SQL、Evidence lineage、Sheet headerだけがこれを読み、「主要field」の手動選択や手書きpivotを許さない。
contract testは各consumer列についてexactly one definition、variant applicability、applicable missing policy、
inapplicable projection policy、applicableなinput
evidence patternへの到達を検査する。

同じcontract/protocolに`[[consumer.estimand]]`を置き、`estimand_id`、source/transform、logical type、unit、
改善方向、eligibility rule ID、許すstatistic/estimator/interval、`applicable_run_variants`を固定する。
comparison dimensionとestimandはsource列のapplicabilityと二つのmissing/projection policyを継承する。当該variantで
silent NULLとなる列をdimension/estimand、observed fingerprint、completeness count、config diff expected gridへ入れない。
point viewはprofile runを展開せず、bench/qualification runとそのpurposeにapplicableなestimandの直積だけを
展開する。applicable source欠損は行を落とさずNULL value+reasonにする。protocolは一次・二次estimandと
decision対象を事前指定し、view実装者が結果を見て選ばない。

全run pairのdiffを通常viewとしてmaterializeしない。agent/SQL利用者とConnected Sheetsは二つのcell
parameterを持つcustom query `SELECT * FROM dataset.qwen_run_diff_v2(@LEFT_RUN,@RIGHT_RUN)`、同env diffを
直接呼ぶ。table functionはNULL、同一、存在しないrunを`selection_status`で拒否し、full outer joinで
片側欠損を落とさない。invalid selectionはzero rowにせず、`row_kind=selection_error`のsentinelを必ず一行
返す。valid config diffは左右contract/protocolのdeclared dimension gridのunionから始め、左右のdefinition
hash、logical type/unit/transform、policy classを表示する。definition/policy互換性を別判定し、互換な行だけ
equalityを計算する。非互換、片側undeclared、観測missingも固有`value_status`で残す。

env diffは左右manifestのexpected scope gridとscope-level family statusを先にjoinする。同scopeの両snapshotが
completeな場合だけvariableを`added/removed/changed/same`とし、片側collector/snapshot欠損は
`unknown_left|unknown_right|unknown_both`、片側でscope非該当なら`not_expected_left|not_expected_right`の
scope sentinelを返す。absence、未観測、非該当を混同しない。UI filter/
pivotだけで二run制約を実現しない。Sheetのfilterなしでfull environment extractを作らない。値はユーザー
指示どおり内部利用者へmask/redactせず表示するが、Sheetとdatasetの共有先を
同じ内部ACLへ限定する。

`qwen_run_facts_v2`は少なくとも次の固定列を持つ。ここで挙げる名前をcontractのconsumer projectionへ
個別登録し、概念名だけの列を作らない。aggregate統計をrun rowへ複写せず、replicate viewを別tabにする。

| group | columns |
|---|---|
| identity/protocol | `run_id`、`contract_id`、`contract_hash`、`protocol_id`、`protocol_hash`、`dimension_policy_hash`、`planned_replicate_group_id`、`planned_comparison_set_id`、`planned_attempts`、`max_attempts`、`stopping_rule_id`、`block_id`、`block_order`、`pair_id`、`replicate_index`、`lane_id`、`run_purpose`、`qualification_arm`、`instrumentation_mode`、`instrumentation_bundle_hash`、`instrumentation_qualification_decision_hash`、`instrumentation_qualification_status`、`started_at`、`finished_at`。qualificationでは`block_id=allocation_id`なので別列を作らない |
| source/artifact | `source_track_head_commit`、`score_reference_commit`、`profile_base_commit`、`source_track_manifest_hash`、`sglang_commit`、`sif_manifest_identity`、`model_manifest_identity`、`dataset_manifest_identity`、`measured_workload_manifest_identity`、`warmup_workload_manifest_identity`、`cache_manifest_identity`、`qualification_parent_archive_hash`、`bundle_uri`、`bundle_hash`、contract列挙log member。role別commit metadataは`qwen_run_commits_v2`だけに置く |
| quality | `benchmark_complete`、`benchmark_valid`、`qualification_native_result_valid`、`configuration_complete`、`lifecycle_complete`、`provenance_complete`、`topology_complete`、`telemetry_complete`、`missing_family_count`、`incomplete_family_count`、`conflicting_key_count`、`cast_failure_count`、sorted `reason_codes`、`score_comparison_eligible`、`qualification_arm_eligible`。qualification OFFでは`benchmark_complete/valid`は非適用のためNULL、normal bench/profileでは`qualification_native_result_valid`がNULL |
| termination | `run_outcome`、`run_exit_code`、`benchmark_termination_kind`、`benchmark_exit_code`、`benchmark_signal`、`engine_termination_kind`、`engine_exit_code`、`engine_signal`、`worker_termination_status`、`scheduler_terminal_state`、`scheduler_exit_code` |
| workload validity | `measured_selected_count`、`planned_input_tokens`、`requested_output_tokens`、`returned_count`、`error_count`、`missing_response_count`、`observed_input_tokens`、`observed_output_tokens`、`warmup_mode` |
| topology/hardware | bench coreの`expected_node_count`、`observed_node_count`、`distinct_host_count`、`expected_mpi_process_count`、`observed_mpi_process_count`、`expected_global_rank_count`、`observed_global_rank_count`、`expected_gpu_per_node`、`expected_gpu_total`、`observed_gpu_record_count`、`observed_gpu_per_node_min`、`observed_gpu_per_node_max`、`unique_gpu_uuid_count`、`effective_tp`、`effective_nnodes`、`placement_valid`、`affinity_valid`、`gpu_model_consensus`、`gpu_memory_bytes_consensus`、`gpu_driver_consensus`、`hardware_identity_min_reason_codes`と、profile-onlyの`cpu_model_consensus`、`cpu_numa_node_count_consensus`、`cpu_memory_bytes_consensus`、`hardware_topology_full_reason_codes`。profile-only列はbenchで理由なしのNULL |
| effective args | `bench_args_identity`、`server_args_identity`と、contractがcurrent dataclassの比較関連scalar leafをexact nameで列挙した列、`runtime_backend_consensus`、`runtime_transport_consensus`。`主要field`という自由選択は置かない |
| outcome | `request_throughput`、`input_token_throughput`、`output_token_throughput`、`total_token_throughput`、`run_elapsed_sec`、`stage_benchmark_wall_sec`、`init_wall_sec`、`prepare_wall_sec`、`warmup_wall_sec`、`measured_generation_wall_sec`、`shutdown_wall_sec` |
| GPU projection | `generation_aggregate_present_gpu_count`、`generation_utilization_valid_gpu_count`、`generation_power_valid_gpu_count`、`generation_memory_valid_gpu_count`、`generation_temperature_valid_gpu_count`、`generation_clock_valid_gpu_count`、`generation_energy_included_gpu_count`、`generation_coverage_min`、`generation_power_coverage_min`、`generation_valid_sample_count_min`、`generation_valid_duration_sec_min`、`generation_fleet_utilization_mean`、`generation_gpu_utilization_mean_min`、`generation_gpu_utilization_mean_median`、`generation_gpu_utilization_mean_max`、`generation_gpu_utilization_p95_min`、`generation_gpu_utilization_p95_median`、`generation_gpu_utilization_p95_max`、`observed_generation_energy_j_sum`、`generation_memory_bytes_max`、`generation_temperature_c_max`、`generation_sm_clock_mhz_min`、`generation_gpu_aggregate_complete`、`generation_gpu_aggregate_reason` |

current SGLang commitからBenchArgs/ServerArgs dataclass field inventoryを生成し、contractへcommitしたinventoryと
byte比較する。追加・削除・rename・type/default変更が未分類ならreleaseを止める。全scalar leafを
`comparison_dimensions`へ出し、各fieldをmust-match/treatment/nuisanceへ分類する。Runsのwide列はその中から
contractがexact nameで列挙したものだけであり、実装者判断で増減しない。

`benchmark_complete`はrequired result sourceの存在とparse completeness、`benchmark_valid`はerror count、
selected/returned count、planned/observed token、正有限generation timeの整合であり、別のNULL/reason ruleを持つ。
env factsはmapの各scope full snapshotだけから作り、common/delta/removedはraw sourceにせずenv diffの
query-time statusだけにする。

GPU projectionはapplicabilityをexpected locator/grid生成より先に評価する。bench/qualificationでは
`telemetry_complete`、全GPU projection、`generation_gpu_aggregate_complete`、
`generation_gpu_aggregate_reason`を直接`silent_null`にし、grid、family status、Evidence、reasonを作らない。
profileだけが`phase=measured_generation`とlane/manifestのexpected `(node_rank,local_gpu_slot)` Cartesian
gridを母集団にしてraw aggregateへleft joinする。observed topologyは各slotへexactly one unique GPU UUIDを
対応付ける。missing/duplicate mappingでもexpected slotを落とさずcoverage 0、projection incompleteにする。
expected/observed GPU countはtopology列だけが所有し、GPU groupへ複写しない。aggregate present countと
各metricのvalid GPU countを分け、全expected GPUがlane所定coverageを満たさない比較用fleet/per-GPU projectionはNULL+reasonにし、
observed-only診断値には`observed_` prefixとincluded GPU countを付ける。energyはmissing intervalへ外挿しない。
fleet utilization meanは`sum(utilization_integral) / sum(valid_duration)`で、per-GPU meanの単純平均にしない。
lane contractがnominal interval、minimum telemetry coverage、minimum power coverageを固定する。

GPU view projectionはper-node/GPU raw aggregateからquery時に導出し、raw keyを追加しない。applicableなtyped columnは
`run_key_singletons`だけを入力にしてcontract typeから`SAFE_CAST`/JSON validationし、cast失敗をNULLと
`run_reason_items`へ出す。複数値を`MAX`、`ANY_VALUE`等で隠さず、conflict keyはwide列をNULL、runを
purposeに対応するcomparison-ineligibleにする。applicable列のmissing/conflict/cast-invalidを同じNULLだけへ潰さない。
当該variantで`silent_null`となる列はこの処理へ入れない。reason itemsはfact
issue、family status、benchmark validity、comparison evaluator、protocol/dimension validatorをUNIONする。
Sheet用`reason_codes`はこのrelationだけをlexicographic sortして`;`区切りSTRINGへ投影し、code自体に`;`を
禁止する。applicableなeligibilityがfalseでblocking reasonが0件のrunをcontract testで拒否する。
commit metadataはGitから`qwen_run_commits_v2`へrole付きでjoinし、commit messageをrun metricやRunsの
曖昧な単数列へ複製しない。

qualification parentはprogram runではないので`run/key/value`へ擬似runとして入れない。runnerがparent
`allocation-evidence.json`を同じqualification artifact setへ書き、`qwen_qualification_allocations_v2`が
allocation ID一意性とchild referenceを検査して投影する。qualification childのRuns/
paired viewはqualification childの`block_id`でjoinし、同じscheduler request/accounting/terminalを二childのraw keyへ
複写しない。Environmentは同じjoinからparent job-entry snapshotをdecodeする。regular runのscheduler/env factは
従来どおり自身のrawへ置く。

`qwen_estimand_points_v2`はbenchとqualificationの基底pointを`analysis_purpose`付きで表し、profileには行を出さない。
replicate pointsはそのscore subsetとしてbenchのfailed/invalidも削除せずNULL value+reason付きでapplicable estimandを残し、protocol、planned/max
attempt、stopping rule、planned group/set ID、pair/block ID、nullable observed fingerprintを持つ。statsは
planned groupでattempted nを数え、事前protocolのeligible ruleとobserved fingerprint一致を満たす点だけを
eligible nへ数える。treatment IDは宣言した全treatment dimensionのname/type/canonical value sorted tupleの
hashで、canonical JSONを表示属性にする。paired pointsはestimand pointsから
`analysis_purpose=score|qualification`を持ち、protocolが割り当てたpair IDだけを対応付け、欠損・置換後に
block/orderから推測しない。scoreは`score_comparison_eligible`、qualificationは
両armの`qualification_arm_eligible`、parent binding、common-source equalityからpair grainで導出した
`qualification_pair_eligible`だけを使い、purposeを跨いでpairにしない。aggregateはbaseline/candidateを順序付きで保持しratioを
`candidate / baseline`と定義し、planned/attempted/eligible pair count、interval method/level、decision
criterion ID、equivalence bounds、alpha、target power、pilot evidence identity、estimate/CI、decision
class/reasonを併記する。これらはpoint rowとversioned protocolからquery時に導出し、rawへaggregateの
第二正本を作らない。estimator/intervalはprotocolが事前指定した方法だけを使い、
`instrumentation_overhead_decision`は現在未決定のcriterionを確定するまでNULLにする。暫定式を埋めない。
breakingなcolumn/type/meaning変更は
既存v2 viewを書き換えずv3 objectを作る。

Connected Sheetsは次のtabだけをversioned consumer objectへ接続する。

1. `Runs`: one run/rowの固定列。JSON展開なしでquality、config、一次結果、artifactへ到達する。
2. `Replicates`: point viewとstats/comparison viewを別tableとして全点とrobust statisticsを示す。競技bestを同じ列で置換しない。
3. `Evidence`: 選択runのfamily/evidence itemごとに、scope、raw key、値hash、source、artifact member、欠損理由を示す。
4. `Config Diff`: 左右run cell parameterからtable functionを直接呼ぶ。
5. `Environment`: 左右run cell parameterからscope-aware diffを直接呼び、full byte-exact envをmask/redactせず比較する。
6. `Data Status`: contract/SQL identity、各query時刻、latest publish/archive、未移管Artifact数、home空き容量、issue件数を表示する。

Connected Sheetsは自動同期されないため、deployment checklistにrefresh ownerとscheduleを記録する。
各Connected object自身のqueryで
`query_generated_at`を出し、Data StatusのSheet-local rangeが各tabのtimestamp cellを参照する。BigQueryの
status queryが他objectのrefresh時刻を知るとはみなさない。preview/extract/pivotは事前filter後の
row数とresult byteを検査し、保守的な内部budget（pivot 80,000 result/8 MB未満、extract 400,000 row/
8 MB未満かつ4 million cells以下）を超えたらscope単位に決定的分割する。`LIMIT`で先頭だけを
表示しない。
標準tabへraw tableやunbounded env relationを直接接続しない。

consumer relationはまずlogical viewとtable functionだけで実装し、同じprojectionのscheduled tableを
先回りして複製しない。実測したrefresh latency/bytesが運用budgetを超えた場合だけ、rawから完全再生成
できるcache tableを別contractで導入する。

この物理設計はBigQueryの[ingestion-time partitioning](https://docs.cloud.google.com/bigquery/docs/partitioned-tables)、
[STRING clustering](https://cloud.google.com/bigquery/docs/creating-clustered-tables)、consumer運用は
[Connected Sheetsのrefreshとresult制約](https://support.google.com/docs/answer/9703214)、
[cell parameter付きcustom query](https://support.google.com/docs/answer/9703000)を前提にする。

## D-001: bench側計測の非擾乱qualification

### 調査結果

profiler、GPU polling、phase control markerをprofile trackへ分離しても、benchに残すeffective-config eventや
primary-result writerにも負荷はある。したがって「profile branchを使った」こと自体を非擾乱の証明にはしない。
一方、profile専用familyはbenchでNULLを許し、このgateの対象をbenchで本当に有効にする最小bundleへ限定する。

現行の[`trial-record.md`](../runtime-performance/trial-record.md)はjob lifecycleの診断記録であり、end-to-end
成功や反復throughput比較を確立していない。従って2-node TP=8 laneのpaired分散を既存runから仮定できない。
paired-tの90% CIを5 pairでratioの約±1%へ収めるには、平均差が0でも観測paired log-ratio SDが約1.04%未満で
なければならない。GPU clock/thermal、worker startup、通信backend、warmup/JIT残差がある状態で、この安定性を
データなしに仮定しない。5 pairはscreen/pilotにはなってもformal qualificationにはならない。

### 推奨option R（ユーザー回答待ち）

判定単位は次のtupleとし、どれか一つでも変わればdecisionを再openする。branch名や最新revisionは使わない。

```text
(repo, lane_id, score_reference_commit, reference_sif_manifest,
 proposed_sif_manifest, protocol_hash, model/dataset/workload/cache identities,
 instrumentation_bundle_hash, SGLang/CUDA/NCCL/driver identities)
```

採用時のGit管理immutable decision recordは`decision_id`、`decision=adopt|reject`、`qualification_protocol_hash`、reference tuple、
`qualified_candidate_tuple`（repo/lane/source-track head/base、proposed SIF、model/dataset/workload/cache、bundle、
SGLang/CUDA/NCCL/driver）、`response_canary_result_hash`、`response_canary_artifact_member`と判定evidenceを持つ。通常bench manifestは
`instrumentation_qualification_decision_hash`をsealし、自身の通常protocol hashをqualification protocolへ一致させる
のではなく、実runから再構成したcandidate fieldsをdecision内の`qualified_candidate_tuple`へ一致させる。
source、SIF、model/dataset/workload/cache、runtime/driver、bundleのどれかが変わればdecisionを再openする。
同じbundle hashのまま他fieldだけ変えたrunを採用しない。runnerはsubmit前にdecision canonical bytes/hash、
`decision=adopt`、candidate/canary result一致を検証し、参照したrecordを
`control/instrumentation-qualification-decision.json`へimmutable copyする。Runsは
`instrumentation_qualification_decision_hash`と
`instrumentation_qualification_status=pending|adopted|rejected|invalid`を持つ。未回答中のbenchはhash NULLかつ
`pending`でaudit-only/score不適格、adoptedだけがscore候補である。無計測のnormal v2 bench variantは作らず、
reference source/SIFはqualification OFFでのみ実行する。

専用comparison policyでだけinstrumentation/SIF variantをtreatmentにする。primary armは次である。

- `OFF`: 正常な2-node teardownに必要な共通lifecycle harnessだけを持つreference bench SIF/codeと、
  両arm共通のnative result、外側timer。metrics event/PrimaryResultWriterは無効。
- `ON`: release予定SIF/codeと、bench score runで有効にするevent、writer、source fileをすべて含む最終bundle。

共通lifecycle harnessは性能計測familyではなく両armでbyte-identicalに固定し、その正しさを別canaryで検証する。
二つのSIFはinstrumentation patch以外のbase image、dependency、toolchain、modelを同じmanifestへ固定する。
同一SIFのruntime disabled/enabled比較はactive costの原因分解には使えるが、patch導入の総効果を消すため
release gateを代用しない。必要時は`unmodified / final-build-disabled / final-build-enabled`をprofile診断で
比較し、採用判定は常にreference OFF対final ONを直接見る。D-001のthroughput estimandは両arm共通のnative
resultだけ、outer wallは共通StageRunnerだけから取る。ONのPrimaryResultWriterはestimand sourceにせず、native
resultとの代数一致を含むON bundle固有reliability gateにする。

まずconfirmatoryへ混ぜない8 planned pilot pair slotを事前commitする。一pairの二armだけを同じallocationの
隣接slotで実行し、pair間は別allocationにしてallocationを統計上の独立単位にする。`allocation_id`と`pair_id`は
一対一であり、一allocationに0または2以上のpairが属した場合は全pairをprotocol-invalidとしてattempted pointだけを
保持し、estimate/SD/CIへ入れない。effective N、mean、SDは一つのeligible pairを持つdistinct allocationだけから
計算する。armごとにengine/processを完全再起動し、同じnode/GPU、
resource shape、TP/nnodes、affinity、backend、workload、cache preconditioning、warmupを使う。commit済みseedで
4 slotをOFF→ON、4 slotをON→OFFへ割り当てる。

submission subjectを`regular_run(run_id)`と
`qualification_allocation(allocation_id,pair_id,ordered_child_run_ids)`のtagged unionにする。
qualificationでは通常run manifestの親にimmutable qualification-allocation manifestを一つ置き、
`qualification_allocation_id/scheduler submission-binding identity/pair_id/ordered child run IDs/OFF-ON assignment/
common lane-node-GPU-binding policy`を最初のarm開始前にsealする。submission bindingとjob-entry fileは
allocation IDとordered child run IDsを所有し、各child runは独立したsource/SIF/argv/timer/log/metricsを
持つ。各childはparent manifest hash、job ID、arm ordinalをimmutable parent refで参照し、scheduler request/
accounting/terminalを複写しない。各childの`block_id=qualification_allocation_id`とする。同一allocationを
維持できないfailureは隠さない。

両SIF、cache、workloadと全common inputはarm 1前に準備し、qualificationではheavy auditとarchiveをarm 2終了後
だけに行う。child outer timer終了とmetrics fileのcloseを分け、arm間にはSIF/build/copy/cache refresh/full audit/
archiveを挟まない。scheduler terminal後、`finalize` commandはparent binding/entry/terminalと両childを一度検証し、
parent evidenceと二childのsource/log/metricsを同じworkflow artifact setへまとめる。arm間crash、
一方だけ未起動、一方だけmetrics未完成でも回収できたsourceをbest effortで残し、pairをineligibleにして
別allocationのrunと再pairしない。

全8 planned slot終了後、eligible subsetからpaired log差のSDの片側80%上限を求める。eligible=8になるまで
追加せず、推奨option Rの`minimum_eligible_allocations=8`を下回れば`pilot_insufficient`とする。不足時は結果を
見てreplacementを足さず、原因修正後に新しい固定pilot protocolを事前commitする。同じexact laneについて事前登録済みの過去paired SDがあれば
その大きい方を`sigma_plan`にする。次のunknown-variance paired TOSTでpower 80%以上となる最小の偶数Nを
求め、confirmatory protocolへ一度だけcommitする。planning上の真値は0.3% slowdown、最小Nは10 pair、
推奨campaign上限は32 pairとする。必要Nが上限を超えた時はmarginを緩めず`criterion_infeasible`である。
pilot、sample-size code/version、入力、出力、乱数seed identityをartifactとして保存する。

primary estimandは各eligible pairの正のtotal token throughputについて

```text
d_i = ln(throughput_ON_i / throughput_OFF_i)
estimate = exp(mean(d_i))
CI90 = exp(mean(d) +/- t(0.95, N - 1) * sd(d) / sqrt(N))
```

とする。候補marginはratio scaleのexact `[0.99, 1.01]`、各片側alphaは0.05である。
固定token workloadでは`measured_generation_wall_sec`はthroughputの代数的再表現なので、独立gateとして
重複させない。engine init、input preparation、warmup、shutdownを含む非冗長な`stage_benchmark_wall_sec`は
`wall_OFF / wall_ON`を同じ方法・marginでco-gateにする。両方が通ることを要求するintersection-union gateであり、
各CIを「同時90% CI」とは呼ばない。

各complete pairでは性能値より先に共通sourceの適格性を判定する。両armでnative result parse complete、
正有限native throughput、正有限outer wall、topology/provenance、Git管理planned workload/dataset/seed/argvと
SIF以外のplanned policy一致を要求する。OFFではactual ordered workloadを直接観測したとはclaimせず、
instrumentation-only source diff gateがworkload selection codeを変更していないことも要求する。
現行native resultはactual returned/error/per-request tokenを証明しないため、これらをD-001の
OFF arm eligibilityへ捏造しない。timed pairとは別allocationでqualification tupleごとに一度だけ、両arm共通canary
harnessによりactual returned/error、ordered request ID、per-request prompt/completion token、deterministic生成
token-ID digestを検査する。canary resultは両SIF、harness、workload/result identityとtupleを持ち、全timed pairが
同じresult hashを参照する。timed allocation内へcanaryを挟んでJIT/cache/thermal状態を変えない。これと同等の
対称sourceをOFFへ追加できるまで、厳密なresponse-level一致をpaired性能gateのclaimには含めない。

canaryは同じ`qualification_allocation`を`analysis_role=response_canary`で使う。runner側canary validatorが
subject manifest hash、両child result hash、両SIF/harness/workload/result identity、closed verdictを
`response-canary-result.json`へ書き、同じqualification artifact setへ含める。decision recordはそのartifact hashを
参照し、timed pair childへcanary内容を複写しない。

ONだけはPrimaryResultWriterのbenchmark complete/valid、returned=selected、error=0、planned/observed token、
required event/source/lifecycle completeness、native resultとの代数一致をbundle reliabilityとして要求する。
OFFのON専用consumer列は理由なしのNULLである。

ON側だけのresult mismatch、required writer/source欠損、crashは`qualification_pair_eligible=false`かつ計測bundleのreliability failureであり、pairを
捨てない。treatment activation前の外部scheduler/node failureだけ、事前確保したreplacement slotへ両armを
新しいpair IDで再投入できる。一方のengine開始後に片armを別runと再pairしない。固定Nの全planned slotが
終了するまでCIを判定に使わず、通常CIを見て追加・pass時停止・outlier削除をしない。

最終classは次のclosed enumとする。

- `adopt`: tuple-level canaryが通り、distinct eligible allocation数=N、全pairのcommon-source/reliability gate、throughputとstage-wall TOSTを通過。
- `perturbing`: output差、instrumentation-attributable failure、またはCIが許容域から明確に分離。
- `inconclusive`: CIが境界を跨ぐ、eligible pair不足、または必要Nがcampaign上限超過。

このdecisionは現在のexact 2-node TP=8 laneだけに有効であり、node/GPU/TP、model、request length/concurrency、
SGLang/CUDA/NCCL/driver、cache/warmup、bundle変更時に再openする。複数laneへ拡張するならpoolせず、各laneの
`adopt`を要求する。全attempt/pair pointはraw sourceから残すが、log-ratio、SD、ratio、CI、decisionは
protocolとpoint viewから導出し、rawへ保存しない。

これはTOSTでalpha 0.05と90% CI内包が対応するという[equivalence testingの原理](https://doi.org/10.1177/1948550617697177)、
software benchmarkではvariance levelをpilotでdimensioningすべきという
[Kalibera/Jones](https://doi.org/10.1145/2491894.2464160)、通常CIの反復peekを避けるべきという
[optional-stoppingの知見](https://www.kdd.org/kdd2017/papers/view/peeking-at-ab-tests-why-it-matters-and-what-to-do-about-it)
に基づく。

代替は、(A) 5 pairを一度だけ解析する低cost screen、(B) slowdownだけを禁じるratio下側95% bound > 0.99の
non-inferiority、(C) alpha-spendingを事前設計したsequential equivalenceである。非擾乱を目的とするため
推奨は双方向のoption Rである。ユーザー回答までは`instrumentation_overhead_decision=NULL`、benchの
instrumented runはaudit-only、profile runは常にaudit-onlyとする。

## 未決定事項と質問protocol

[Grill with Docs](https://github.com/mattpocock/skills/blob/main/docs/engineering/grill-with-docs.md)の進め方を
採用し、文書から調べられる事実を先に閉じた後、結果を変える未決定事項だけを一回に一問聞く。各回答は
`decision_id/question/recommended option/alternatives/user answer/decided_at/affected contract fields and
views/reopen trigger`をGit管理decision recordへ残す。未回答を推奨値で埋めず、依存するviewのdecisionはNULL、
release gateは未通過のままにする。

コード監査で、D-001とは別にprofile tool contract D-004が残っていることが分かった。現行
`scripts/exec-profiler.sh`はtransparent wrapperであり、SGLang側にはTorch Profilerと
`CUDA_PROFILER`/Nsight系の複数経路があるため、tool、activity、step範囲、required raw file集合はコードだけでは
一意に決まらない。D-004では初版profileでprofilerを必須にするか、必須ならどのtoolとstep範囲をcanonical
protocolにするかを一問で決める。未回答の間はtransparent wrapperをprofile protocolとしてreleaseしない。

instrumentation overhead D-001とする。D-003はD-001の対象laneを決めるため先行する。

## 決定状態

| 状態 | 決定 |
|---|---|
| 確定 | raw relationは`run/key/value`の三列、compound valueはcanonical JSON |
| 確定 | 定義した各process scopeの環境変数は機密値を含めfilterせずbyte-exactに全件保存 |
| 確定 | native object/structured sourceを優先し、log regexはfixture付きの最終手段 |
| 確定 | compute側はlocal raw fileだけ、finalize/publishはrunner、Artifact upload/home移管はworkflowが担当 |
| 確定 | observed totalsから派生するthroughputやnode full factsのcommon copyをrawへ重複保存しない |
| 確定 | bench/profileを分離し、profile専用familyはbenchで値だけNULL、原因/status/Evidenceなしとする。profile runはscore比較へ混ぜない |
| 確定 | home artifactは自動expireせず、容量が実際に問題になった時だけ手動整理する。専用quarantineは作らない |
| 未決定 | D-004: 初版profileで必須にするprofiler、activity、step範囲、required raw file集合 |
| 未決定 | score runへ許すinstrumentation overheadの統計的上限 |

## 実装ハンドオフ

### 公開面とfile境界

実装者が新しい抽象を増やさないよう、package内部の依存方向を次に固定する。

```text
SGLang hook -> version付きdomain record / primary artifact
                                      -> pure parser/reducer ---\
I/O probe ----> structured source value -> pure parser --------+-> Fact -> Recorder
StageRunner ---------------------------------------------------/

runner: single merge -> ArtifactStore/RunPackager -> publish
```

hookはmetric keyや`Fact`を知らず、version付きsource recordまたはprimary artifactだけを書く。

公開するdomain型はimmutableな`Fact`と`Termination`だけにする。`Fact`はrun ID、owner、encoded
stringを持たず、`Termination`は`exit`、`signal`、`spawn_error`をunionとして表す。最低限のcall
surfaceは次で足りる。

```python
Fact(key: str, logical_value: LogicalValue)
init_subject(kind, protocol_path, contract_path, runtime_root, identities) -> SubjectManifestPath
Recorder.open(manifest_path, registry_path, producer_id, runtime_dir)
Recorder.append_many(facts)
Recorder.seal()
collect_once(manifest_path, runtime_dir, producer_id, probe) -> CollectorOutcome
submit_subject(subject_manifest_path, runtime_dir, lane_adapter, pjsub_argv) -> SubmissionOutcome
run_stage(stage_name, argv, *, cwd, recorder) -> Termination
finalize_subject(subject_manifest_path, runtime_dir) -> tuple[FinalizeOutcome, ...]
publish_run(metrics_jsonl_path, destination) -> PublishOutcome
```

`Recorder.open`はpersist済みmanifestとregistryをread-onlyで検証し、runを生成・推測しない。
`LogicalValue`はstring、bytes、bool、int、finite float、Decimal、UTC timestamp、またはJSON treeに
限定する。top-level `None`は禁止し、bytes/Decimal/timestampとcompound JSONのencodingはregistryの
logical typeが決める。
`append_many`だけがlogical valueをcanonical stringへ変換し、一行ずつ`run/key/value`へbuffered writeする。
`seal`はflush/closeしてtemporary fragmentをrenameする。semantic ownerはprobe外側のsupervisor、
`StageRunner`、またはnode agentである。正常終了時にexpected fragmentが存在する
ことをpost-run validatorが検査し、途中crashから全eventを救うための逐次syncは行わない。registry loading、canonical encoding、manifest validationは
`recording.py`のprivate collaboratorとし、それぞれを薄いpublic moduleへ分割しない。

log pathはstage nameとruntimeから内部導出し、signalはchild process groupへforwardしてraw wait statusを
保存する。`finalize_subject`は唯一のrunner orchestratorであり、post-run validation、metrics merge、
bundle/artifact manifest作成を順に行う。別のpublic `merge_run`と二重にorchestrationしない。

`FinalizeOutcome`はbundle/metrics path、hash、sizeとcompletenessを返す。`PublishOutcome`はdestination、
attempt、outcomeを返し、workflow logへ残す。内部`MergeReport`はinput manifest hash、row counts、conflicts、
completeness、output hashを持つ。制御状態を再構成するためのreceipt DAGは作らない。

`init_subject`と`collect_once`は`recording.py`、`submit_subject`は`submission.py`、`run_stage`は
`stage_runner.py`、`finalize_subject`は`artifact_store.py`、`publish_run`は`publish.py`が
所有する。`cli.py`はargument parseとhandler dispatchだけを行い、manifest生成、scheduler subprocess、
probe lifecycleを実装しない。

use-case outcomeも小さなclosed schemaにする。`CollectorOutcome`は`status/error_count/fragment_path/complete`、
`SubmissionOutcome`は`submitted|failed`とnullable job IDを持つ。曖昧submitを自動探索・復旧するstate machineは
作らず、そのworkflow attemptを失敗として人が再実行する。

現行fileからの移行先は次の通りとする。

| 現行 | 移行後の責務 |
|---|---|
| `core.py` | `domain.py`の`Fact`、`recording.py`、runner-only `publish.py`へ分割。空文字fallbackとcredential captureをcompute pathから廃止 |
| `collectors.py` | I/O probeとpure parserを分け、どちらも最後は`Fact`列を返す |
| `stage.py` | outer MPI processだけを待つ`StageRunner`とpure reducerへ分離。`EventReceiver`から`send()`を除く |
| `cli.py` | 下記use caseを配線するcomposition rootだけにする |
| 新規Qwen固有`node_agent.py` | 各node wrapperが一つ起動し、全variantのcommon lifecycleとapplicable variantのconfig Recorderを所有。profileだけGPU samplerとphase receiver/ACKも有効化 |
| SGLang patch | reference baseのcommon lifecycle hookと、score bundleのversion付きlocal config record/`PrimaryResultWriter`だけを実装し、metric key、BigQuery、retryを知らない |
| `scripts/run-qwen35-node.sh` | node rank、manifest、node-local endpointを明示する薄いadapterにする |
| `scripts/run-qwen35.pjm` | manifest/runtime pathを明示してcompute collectorとstageだけを呼ぶ |
| `.github/workflows/genkai.yml` | pre-submit manifest、submission binding、terminal後finalize/archive/publishを所有 |

実装fileは計画済みの`domain.py`、`recording.py`、`submission.py`、`stage_runner.py`、`merge.py`、
`artifact_store.py`、`publish.py`とdomain別probe/parserに限定する。`manager`、`service`、汎用event
bus、DI frameworkは追加しない。`submission.py`はintent、scheduler adapter、曖昧結果照合を隠す一つの
深いmoduleであり、collectorやpublisherをimportしない。Qwen固有fileは`node_agent.py`、pureな`sglang_events.py`、
`source_artifacts.py`、`gpu_telemetry.py`だけを許可する。node control protocolを`node_agent.py`へ閉じ、
共通基盤へ漏らさない。

同じfamilyをsource fileとmetric fragmentの両方へ書かない。source ownerとraw ownerを次へ固定する。

| family | durability source / single writer | Fact owner |
|---|---|---|
| job/host probe、stage termination、node child lifecycle | 対応producerの`Recorder`へ即時append | 各compute/wrapper producer |
| SGLang node config、worker-ready | node agentがnode/global-rank JSONLへappend | terminal `sglang_events` reducer |
| lifecycle handshake | coordinator/node agentの最終`control/lifecycle/summary.json` | lifecycle reducer |
| phase handshake（profile） | coordinator/node agentの最終`control/phase/summary.json` | phase reducer |
| GPU sample（profile） | node agent内samplerがnode telemetry JSONLへbuffered append | `gpu_telemetry` reducer |
| primary result（bench/profile/qualification ON） | rank 0 `PrimaryResultWriter`のsource artifact | `source_artifacts` parser |
| native qualification result（OFF/ON） | rank 0 native result file | qualification parser。通常bench/profile raw ownerにはしない |
| warmup/measured workload | rank 0 manifest writerのsource artifact | `source_artifacts` parser |

`PrimaryResultWriter`はtargetと同じdirectoryへtemporary fileをwrite/closeし、`os.replace`で完成させる。
write/renameが失敗したらbest-effort shutdown後にstageをnonzeroとし、primary result由来のraw keyは作らない。
workload manifest writerも同じ小さなhelperを使う。retry競合やdirectory crashを扱う専用protocolは作らない。

機械可読正本は`qwen-experiment-metrics/contracts/qwen-v2.toml`一fileにする。contract ID、producer
定義、key pattern、owner、logical type、unit、cardinality、source、scope、capture、required、
missing、validation、正当な重複のjustificationを同じentryに置く。lane/protocolはこのcontractに
定義済みのproducer IDからexpected graphを選ぶ。runtime manifestにはcontract IDとcontent hashを
sealし、raw tableにはroutingに必要なcontract IDだけを索引化する。観測マップの表はこのTOMLから
生成し、手書きproseだけを別に保つ。したがってregistryとMarkdown表の二つを正本にしない。

### CLIと設定境界

CLIは次の六orchestration use caseと一つのQwen固有internal entrypointだけを持つ。behaviorを環境変数で
切り替えず、manifestとruntimeは
明示引数、比較条件はGit管理protocol、BigQuery宛先はrunner側のGit管理設定から読む。

```text
init-subject   regular runまたはqualification parent+二childのmanifestを作成
submit         exact pjsub argvを一度実行し、stdout/stderrと取得できたjob IDを保存
collect        compute producerを一度実行しdata fragmentとcollection statusへ記録
stage          target argvをspawnしterminationと時間を記録
finalize       terminal後にsubjectを検証し、expected artifactとmetrics.jsonlを作成
publish        metrics.jsonlをBigQueryへ送達

internal node-agent  各node wrapperが起動し、全variantのcommon lifecycle、applicable variantのconfig、profileだけのphase/telemetryをlocal fileへ記録
```

`init-subject`以外は`--subject-manifest`を必須にし、compute内部の`collect`/`stage`/`node-agent`だけはparentが指定した
`--run-manifest`を必須にする。`EXPERIMENT_METRICS_RUN`、PID、PJM IDからsubject/runを再構成しない。
`collect`、`stage`、`node-agent`はBigQuery設定やcredentialをimportしない。outer `StageRunner`は
`mpiexec`を一度だけspawnするためremote node agentを直接起動しない。各MPI rankで実行される
`run-qwen35-node.sh`が、そのnodeのcontainerより前に`node-agent --node-rank ...`を起動しready handshakeを
待つ。bench agentはlifecycle/config、profile agentはさらにsampler/phase、qualification OFFはcommon lifecycle/
topologyだけ、qualification ONはlifecycle/configを持つ。profile-only producerをbench/qualificationで起動しない。`finalize`と`publish`を分け、
送信障害を理由にcomputeやarchiveを再実行しない。

`submit`はexact argv、subject ID、stdout/stderr、subprocess outcome、取得できたjob IDを通常fileへ保存する。
応答を確定できないrunner crashはそのattemptを失敗とし、自動nonce検索、recovery receipt、blind resubmitは
実装しない。PJM側はrun/allocation IDと`PJM_JOBID`をjob-entry fileへ書き、terminal後にrunner側job IDと照合する。

node wrapperはnode-local filesystemに一意なendpoint directoryを作り、`node-agent --run-manifest ...
--runtime ... --node-rank ... --local-endpoint-dir ... --role coordinator|participant
--purpose bench|profile|qualification [--qualification-arm off|on]`を起動する。共有runtimeへUnix socketを作らない。
containerにはlocal endpointだけを`/run/qwen-observation`へbindする。共通correctness harnessのlifecycle flagは
全variantへ渡し、score observation/primary flagはbench/profile/qualification ONだけへ渡す。

```text
all variants:
  --qwen-lifecycle-socket /run/qwen-observation/lifecycle.sock
  --qwen-run-manifest /run/qwen-control/run-manifest.json
bench/profile/qualification ON:
--qwen-observation-socket /run/qwen-observation/events.sock
--qwen-primary-result-path /run/qwen-source/primary-result.json
qualification OFF/ON:
  --result-filename /run/qwen-source/native-qualification-result.json
```

qualification OFFへscore observation/primary flagを渡さず、qualification ONはnative estimandとprimary reliability
artifactの両方を書く。normal bench/profileへnative result pathを渡さない。argv allowlist testでvariant間の境界を固定する。

初版laneは現行実行へ合わせ、rank 0 MPI wrapperが取得した`hostname`をadvertise address、SGLang distributed
portを`25000`、node-agent control portを`25001`、rendezvous待ちを60秒、ready待ちを120秒に固定する。
rank 0はaddress、固定port、run ID、protocol versionを`control/lifecycle/rendezvous.json`へ一度writeし、rank 1は
その一fileだけを待って明示endpointへ接続する。port衝突やtimeoutはrun failureとして再実行し、動的port allocator、
複数discovery source、自動fallbackは作らない。

coordinatorはexpected node `{0,1}`のidentity付きready ACKをmemory上で集めてからcontainer開始を許可し、
最終summaryだけをfileへ書く。bind/connect/ready timeout、stale/wrong-run messageはsolverを開始しない
configuration failureである。

primary result pathはrank 0だけwrite可能にbindする。allocation内producerはnode agent停止時にpurpose上
applicableなdomain/control/telemetryをshared runtimeへflush/closeしてからPJM scriptを終了する。
runnerの`finalize`は
node/GPU/workerを再観測せず、terminal accountingと既存sourceのpure parser/reducerだけを実行する。

### runtimeとarchive layout

runとsubmission subjectを分け、run側は次の固定layoutにする。任意file探索や「最後に見つかったJSON」を禁止する。

```text
<runtime>/runs/<run>/
  control/run-manifest.json                    # pre-submit immutable
  control/contract.toml                        # 実使用contractのimmutable copy
  control/protocol.toml                        # 実使用protocol/laneのimmutable copy
  control/source-track-validation.json         # diff algorithm/base/head/patch/pass
  control/instrumentation-qualification-decision.json # decisionを参照するbenchだけ
  control/submission-request.json
  control/submission.stdout
  control/submission.stderr
  control/submission-result.json               # raw subprocess outcome when available
  control/submission-failed.json               # explicit rejectionの場合だけ
  control/submission-binding.json              # 取得できたjob IDとのbinding
  control/job-entry.json                       # PJM側のrun IDとPJM_JOBID
  control/lifecycle/rendezvous.json
  control/lifecycle/summary.json
  control/phase/summary.json                    # profileだけ
  control/collection-status.json
  metrics/fragments/<producer>.tmp              # write中
  metrics/fragments/<producer>.jsonl            # 正常close後のrename先
  domain/sglang/node-<node>.jsonl
  domain/worker/global-rank-<rank>.jsonl
  source/sif-sidecar.json
  source/model-manifest.json
  source/dataset-manifest.json
  source/env/job-entry.snapshot
  source/env/host-node-<node>.snapshot
  source/env/sglang-node-<node>.snapshot         # score bundleだけ
  source/env/worker-rank-<rank>.snapshot         # score bundleだけ
  source/cache/node-<node>-input-manifest.json
  source/topology/node-<node>.json
  source/scheduler/pjstat.stdout
  source/scheduler/pjstat.stderr
  source/scheduler/pjstat-version.txt
  source/primary-result.json
  source/native-qualification-result.json          # qualification OFF/ONだけ
  source/workload-warmup.json
  source/workload-measured.json
  telemetry/node-<node>/gpu-samples.jsonl        # profileだけ
  profile/raw/...                                # profileだけ。profilerの固定出力先
  logs/pjm-job.log
  logs/benchmark.stdout
  logs/benchmark.stderr
  control/scheduler-terminal.json
  merge/collection-validation.json
  merge/metrics.jsonl
  archive-staging/run-artifacts.tar.zst
  archive-staging/artifact-manifest.json
  archive-staging/profile-raw.tar.zst            # profileだけ
```

regular runだけが上記`control/submission-*`、binding、job-entry、`control/scheduler-terminal.json`を所有する。
qualification childではそれらを作らず、代わりに
`control/qualification-parent-ref.json`（allocation ID、parent manifest hash、job ID、arm ordinal）を置く。
qualification parentは次の固定layoutを所有する。

```text
<runtime>/allocations/<allocation>/
  control/qualification-allocation-manifest.json
  control/protocol.toml
  control/child-source-validation-index.json   # child source validationsのcontent-addressed index
  control/submission-request.json
  control/submission.stdout
  control/submission.stderr
  control/submission-result.json | submission-failed.json
  control/submission-binding.json
  control/job-entry.json                           # allocation ID/job ID/ordered child IDs
  source/env/job-entry.snapshot                    # NUL区切りbyte-exact parent source
  source/scheduler/pjstat.stdout
  source/scheduler/pjstat.stderr
  source/scheduler/pjstat-version.txt
  logs/pjm-job.log
  control/collection-status.json
  control/scheduler-terminal.json
  archive-staging/allocation-evidence.tar.zst
```

parent evidenceは上記`control/`、`source/`、`logs/`を一つに収録し、`logs/pjm-job.log`をrequired memberとするが、
run metricsは持たない。各childはparent manifest
hashだけを参照し、parent、二child、canary result、該当するprofile attachmentを同じworkflow artifact setへ
まとめる。

これは全purposeのallowlist unionであり、bench archiveにprofile-only pathがないのは正常である。`finalize` commandは
contract applicabilityからexact expected member setを作り、任意file探索もempty placeholder作成もしない。

manifestにexpected pathを列挙し、node/global rankをfilenameから推測せずrecord内部との一致も検査する。
node-local ephemeral endpointはshared layoutに含めず、node wrapperが作ったdirectory内の
`events.sock`と`control.sock`に限定する。job終了時にsocketをarchiveせず、最終summaryだけをarchiveする。
runごとのarchive対象は次の三fileを基本にする。

```text
run-artifacts.tar.zst   # 下記expected source/log/fragment
metrics.jsonl           # 最終raw sidecar
artifact-manifest.json  # file path、size、SHA-256、contract
```

bundle memberはrun variant別のmanifest駆動exact allowlistとする。regularは`control/`のmanifest、contract、
protocol、submission raw stdout/stderr、実在するresult/failed outcome、binding、job-entry、scheduler-terminalを
含む。さらに全runでSIF sidecar、model/dataset manifest、該当するenvironment/cache/topology、PJM accounting raw/version、PJM batch
log、benchmark stdout/stderr、purpose別authoritative result/workloadをexpected fileとして列挙する。qualification
childはparent-owned memberを含めず、`qualification-parent-ref.json`だけを含む。
profileだけphase/telemetry、
qualification OFF/ONだけ`native-qualification-result.json`を含める。いずれもexpected producerの
`control/collection-status.json`、完成および残っていれば途中producer fragment、applicableな`domain/`、`source/`、`telemetry/`、`logs/`、
`merge/collection-validation.json`、自己を除くmember path/hash/sizeを持つ`bundle.manifest.json`を含める。
最終`metrics.jsonl`とartifact manifestはbundle内へ重複収録しない。

`run-artifacts.tar.zst`はmanifestで列挙したmemberを収録し、作成後の実byte列をhashする。再実行時にtar headerや
compressor versionまで同一にして同じbundle hashを再現することは要求しない。同一run attemptでは既に作った
bundleをそのまま転送し直す。
最終`metrics.jsonl`は完全一致rowを一件へし、異値conflictは全rowを残したまま`(run,key,value)`の
UTF-8 byte順にsortする。各lineはobject member順`run`,`key`,`value`、空白なし、UTF-8、LF終端へ固定する。

manifestは通常のtemporary writeとrenameで完成させる。基本三fileとapplicable profile artifactをGitHub Artifactへ
uploadできた時点でGenkai runtimeはcleanupしてよい。archive jobはhomeの
移管と同じくremote再読取checksum、transaction、reconciler、別delivery storeは実装しない。

BigQuery publishはhomeの`metrics.jsonl`または同じGitHub Artifactから再実行できればよい。invalidなrunも
通常のrun/attempt directoryへraw artifactを残し、comparison viewから除外する。物理的なquarantine systemは
作らない。通信・archive失敗はworkflow failureとして扱い、少数のarchive job状態だけをData Statusへ投影する。


成功後のGitHub Artifact削除である。Qwenはこの仕組みをまだ持たず、2026-08-09時点で134 artifact、圧縮後

これは実験用projectであり、極稀な転送failureやrun lossまで多重に防がない。通常pathで対象値を生成・回収する
ことを優先する。推奨option Rは次である。

```text
run -> build log、execution log、results、metrics.jsonlをGitHub Artifactへupload
    -> profileならraw GPU telemetryと、D-004でapplicableになった場合だけprofiler traceを別artifactへupload
    -> /srv/qwen-artifacts/<run>/<attempt>/ へscp
    -> SCP commandが成功した時だけ、そのworkflow runのGitHub Artifactを削除
```

SCP失敗ではarchive jobをfailureにし、GitHub Artifactを残す。job rerunで再転送できる。
remote transaction、content-addressed object store、put-if-absent、receipt DAG、容量reservation、複数quota、
remote再読取checksum、全境界crash recovery、独立backupは実装しない。runtimeはGitHub Artifact upload成功後に
現行どおりcleanupしてよい。

normal benchのPrimaryResultWriter、qualification OFF/ONのnative result、parent/child/canary evidenceを必ず
results artifactへ含める。Qwen profileのprofiler familyはD-004でapplicableと確定したprotocolだけrequiredにする。
その場合はprofiler wrapperへ明示引数`--profile-output-dir <runtime>/runs/<run>/profile/raw`を渡し、exact profiler
argv、tool metadata、trace rawを`profile-raw.tar.zst`へまとめてuploadする。D-004がprofiler非必須と決めたprotocolや
未回答中のrunでprofiler file/placeholderを要求せず、現在のtransparentな`exec-profiler.sh`を「取得済みprofiler」と扱わない。
bench/qualificationではprofile producer、artifact、placeholder、absence eventを作らず、対応consumerは
原因/status/Evidenceなしの`NULL`だけである。

model約400 GBとSIFをhomeへ複写することは必須にしない。model/dataset/tokenizer/cacheのmanifest、SIF source
run、build/source commit、SIF content hashをrun evidenceへ残し、実体は現行Genkai cacheをbest effortで使う。

推奨retentionはhomeのrun artifactをproject終了まで自動expireせず、容量が問題になった時だけ人が整理する
単純なpolicyである。専用quarantineは作らず、invalidなrun/attempt directoryもそのまま残す。Data Statusはarchive job最終成功時刻、
未移管Artifact数、home空き容量だけを表示する。完全replayはhome artifact tree、qualification parent artifact、
Git管理source/contract/protocol/decision/SQLから空datasetを再構築できることを意味する。

### 状態と失敗規則

runnerは`pjsub`のstdout/stderrと取得できたjob IDを保存する。PJM supervisorはentry直後にsubject ID、
`PJM_JOBID`、qualificationならordered child IDsを`job-entry.json`へ書く。terminal後に両者を照合し、
片方が欠けるattemptや不一致はcomparison不適格にする。稀なrunner crashからsubmit状態を自動復旧しない。

runの処理状態は次の単純なfile flowで追う。

```text
regular:
run manifest -> submit result/job-entry -> compute log/source/fragments
             -> terminal accounting -> post-run validation -> run artifact + metrics.jsonl
             -> GitHub Artifact upload -> runtime cleanup -> home SCP -> BigQuery publish

qualification:
allocation manifest -> submit result/job-entry -> child OFF source/log/metrics
                                           \-> child ON source/log/metrics
                    -> terminal accounting -> one qualification artifact set
```

- post-run validatorはvariant別expected file、producer、node/rankとrequired keyを一度検査する。正常終了runで
  欠けていればcollection failureとする。失敗・timeout runの途中fileは読めるcomplete JSON lineだけを
  best effortで回収し、欠損を空文字やzeroで補わない。
- 同一run/keyの異値は両方を保持し、conflictとしてpublishしてcomparisonから隔離する。
- foreign run、unknown key、owner違反、canonical encoding違反は入力をbundleへ保全するが、信頼できる
  metrics sidecarへ混ぜない。専用quarantine状態は作らない。
- process timeout、signal、spawn errorでもrunnerはscheduler terminal後にprimary result、domain event、
  GPU sampleをreduceしてfinalizeできる範囲でraw log/sourceをartifactへ含める。任意のcrash境界から完全再開
  できることは要求しない。
- cleanupはGitHub Artifact upload成功後に許す。publish失敗はhomeまたはGitHub Artifactの
  `metrics.jsonl`から手動再送し、computeを再実行しない。

### gateを証明するtest

| gate | 必須test/fixture | 証明すること |
|---|---|---|
| G0 依存境界 | `test_dependency_boundaries.py`、CLI subprocess import trace/egress guard | domain/probe/recording/collect/stageはpublisher/credential/networkをimportせず、node-agentはapproved control transport以外のnetwork client、credential、DNS、外部egressを使わない |
| G1 値契約 | `test_domain.py`、RFC 8785/decimal/env byte fixture | 空、zero、欠損、非有限拒否、単射escape、canonical byte |
| G2 submit/binding | `test_submission.py`、regular/qualificationのsuccess/reject/mismatch fixture | exact argv/stdout/stderr/job ID保存、allocation/job/ordered child join、不一致の比較除外 |
| G3 local file | `test_recording.py`、正常終了・program failure・truncated JSON fixture | buffered writerのclose/rename、expected file検査、読めるcomplete lineのbest-effort回収 |
| G4 ownership/merge | `test_merge.py`、variant別producer graph fixture | owner一意性、family routing、exact dedupe、異値conflict、required欠損、qualification parentのjob envを二childへjoinしてraw複写しない。profileでprofile-only collector失敗はreasonあり、benchの同列とqualification OFFのON専用列は理由なしNULL、qualification ONの同collector失敗はreasonあり |
| G5 process観測 | `test_stage_runner.py` | actual argv/cwd、process-group signal forwarding、exit/signal/spawn error、monotonic境界 |
| G6 archive/送達 | workflow integration fixture | happy pathで全expected artifactがhomeにあり、upload失敗はattempt failure、SCP失敗ではGitHub Artifactが残る。homeのJSONLをBigQueryへ再送できる |
| G7 Qwen意味 | `test_observation_contract.py`、`test_primary_result.py`、`test_distributed_events.py`、`test_lifecycle_control.py`、variant argv/source-diff fixture | purpose別authoritative result、OFFはnative result/ONはnative+primary、OFFはplanned workloadだけ/ONはactual manifest、primary missing、node/rank cardinality、lifecycle単一owner、qualification ONの差分がinstrumentation patchだけであること |
| G8 GPU phase | profile用`test_gpu_reducer.py`、2-node `test_phase_protocol.py` | benchではsampler/phase messageなし、profileではrank 0/1 phase receiver、共有UDS不使用、wrong-run/stale message、ACK順、marker欠損、gap cutoff、weighted mean/p95/energy |
| G9 workflow移行 | legacy/canary workflow integration fixture | legacy CLIとv2 CLI、timeout後のbest-effort source回収、finalize/publish手動再実行 |
| G10 consumer contract | contract生成SQL fixture、variant別BigQuery golden rows、Sheet parameter/refresh smoke test | 全列lineage、profile attemptのscore統計除外、非適用列がreason/EvidenceなしのNULL、pair-grain eligibility、同bundleでbase/workload/runtimeだけ変えたbench拒否、失敗attempt残存、protocol分離、estimand、definition非互換・片側env欠損TVF、freshness/size |
| G11 非擾乱 | D-001のreference OFF/final ON paired runと別allocation canary | canary result、planned workload/common-source適格性、bundle reliability、事前dimensioningしたoverhead採用基準を同時に満たすこと |

`test_observation_contract.py`はcontractのschemaとowner/source/capture/absenceの完全性を検査し、
生成した観測マップ表がcommit済み表とbyte一致することを検査する。これが「この値はどこから取るか」
を、一つの正本から人間向け表と実行時validationの双方へ機械的に保証する。

### release単位

移行はkeyごとに二重化せず、run contract単位で行う。

1. legacyのcollector event/result/GPU集約をgolden fixtureとして凍結する。
2. 専用canary protocolだけでpre-submit manifestとlocal metrics exportを導入し、legacy runの送信経路は触らない。
3. canaryでrunner finalize/archive/publishの正常系と代表的なmissing-file failureを通す。
4. node-local endpoint、worker-ready、workload manifest、primary resultをbench候補へ、phase control、
   raw GPU sampleをprofile purposeへ追加する。
5. benchへ必要な最小bundleだけD-001へ掛け、2 node TP=8で意味・result同値性・非擾乱gateを通したbundleだけbenchへ追加する。
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
   JSONL export、runner merge/publish、workflow artifact upload/home移管を導入する。
4. ingestion-time partition/`run,key` clusterのraw table、distinct/singleton/conflict/reason items、evidence
   item/family status、archive attempt/status、exact-contract run/dimension/point/stats/paired view、二run table
   functionをconsumer column contractから生成しfixtureを通す。
5. node-local event endpoint、rank 0 result ownership、node-scoped full configurationを実装し、
   397B runのconfiguration欠落と競合を修正する。manifestが固定したexact contractだけをv2 viewへ
   入れ、contractなしはlegacy、異なるcontractのproducer混在はinvalidにする。
6. warmup/measured workload、cache manifest、primary result validatorを追加し、2 node TP=8、
   global worker rank 0..7のcompletenessを確認する。
7. profile purposeだけにpersistent control-forwarder phase eventとnode-local raw GPU sampleを追加し、
   benchでは対応consumer列を理由なしのNULLにする。どちらのtimed pathからもBigQueryを外す。
8. Connected SheetsのRuns/Replicates/Evidence/Config Diff/Environment/Data Statusをbounded consumerへ
   切り替え、freshness/result-size smoke testを通す。旧key consumerがなくなってからthroughput、model名、
   GPU総数等の派生・部分集合keyの新規生成を止める。
9. homeのrun artifact tree、qualification parent artifact、Git管理source/contract/protocol/decision/SQLから
   空datasetを再構築し、raw row集合と主要view/Sheetの結果を照合してから旧viewを廃止する。

冗長keyの停止をview作成より先に行わない。

## 完了条件

- 性能に影響する変更がすべてremote到達可能なcommitにあり、commit bodyから仮説が分かる。
- SIF reuseでもsource-track head/base、original build commit、SGLang commit、content identityをrole付きで区別できる。
- modelはpinned revisionまたは事前作成manifestへ結び、config/index/tokenizerの1 byte変更とshard membership/path/size
  不一致を検出する。400 GB shard本文の同size byte変更はrunごとの検出対象にしない。dataset、cache、workloadの
  1 byte/order変更は各manifestが検出する。
- 正当な空文字、未観測、collector失敗、collector未到達、真のゼロをfixtureで区別できる。
- canonical JSONがprocessやobject挿入順に依存せず同じbyte列になる。
- 各key patternのownerが一意で、distinct後のrun/key conflictがゼロである。
- 実2 node runでnode-scoped full configが届き、viewが一致fieldとnode差をlosslessに表示できる。
- expected node rank、global worker rank 0..7、host、GPU UUID、TP、nnodes、MPI processの不一致を
  理由別に検出できる。
- prompt順序だけが違う同token総数workloadをmanifest identityで分離できる。
- result欠損、不正JSON、returned count/token不一致、非有限時間をcomparison対象外にできる。
- viewの四つの代数的throughputがSGLang artifactと許容誤差内で一致する。
- profileのphaseへ既知delayを入れたfixtureでinit、prepare、warmup、generation、shutdownを分離できる。
- profileでgeneration marker欠損時にGPU aggregateを全stage値で代用せず、benchの同列は理由なしのNULLになる。
- profileでexpected GPU gridの一台欠損をcoverage 0とし、比較用fleet projectionをNULL+reasonにできる。
- count/sizeが同じ別cacheをmanifest identityで区別できる。
- BigQuery停止時もhomeまたはGitHub ArtifactのJSONLから同じlogical row集合を再送できる。
- 成功・失敗・timeout runのexpected log/applicable artifact/JSONLをGitHub Artifactへuploadし、upload成功後は
  runtimeを削除できる。
- Qwenの全core/source/resultとprofile runのapplicable raw attachmentがhome pathから読め、bench
  archiveにはprofile pathもabsence recordもなく、対応Sheet列が原因なしNULLだけになる。
- bench/profileが分離され、profile-only列はbenchで理由なしのNULLになり、bench bundleの計測影響はD-001内である。
- 同一protocolの反復がpair ID、実行順、blockを含め復元でき、全estimand点、median、MADがSheetに表示される。
- treatment値を含む同一条件だけが同じreplicate groupへ入り、宣言したtreatment差だけが同じ
  comparison set内で許される。
- required dimensionが欠けたfailed/invalid runもpre-run planned groupのattempted nへ一度だけ入り、NULL
  fingerprint groupへ脱落・混入しない。
- 別protocolが同じhuman group labelを使ってもcontract/protocol/policy namespaceにより統計が混ざらない。
- 各run/evidence/Sheet列がcontract raw keyまたはGit/artifact dimensionへ一意に遡れ、cast失敗や異値競合を
  aggregateが隠さない。
- 二run diffがcell parameter付きtable functionで計算され、NULL/同一/不存在runを拒否し、全run pairを
  materializeせず、full envをparameterなしでSheetへextractしない。
- config definition/policy非互換をequality NULLで表示し、片側env snapshot欠損をadded/removedでなく
  unknownとして表示する。
- Data Statusが各objectのquery時刻、最新publish、未移管Artifact数、archive job最終成功時刻、home空き容量、
  SQL/contract identity、conflict/incomplete件数を表示し、
  scheduled refreshのowner変更・停止とresult-size超過をsmoke testで検出できる。
- Sheets利用者がJSON展開なしでcommit、completeness、一次結果、反復統計、artifactへ到達できる。
