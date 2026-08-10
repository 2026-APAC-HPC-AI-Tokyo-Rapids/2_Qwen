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

本projectは実験用であり、mission-criticalな送達systemではない。正常終了するrunで取得対象が最初から
実装されていないことは許さないが、極稀なcollector、node、転送failureを救うための多段retry、fallback source、
field別reason registry、耐久journalは作らない。値のsource、取得時点、schemaは厳密にする一方、取得できなかった
値はconsumerで`NULL`、run全体は単一の`collection_complete=false`で十分とする。失敗runは保存できたlogを残し、
必要なら人が再実行する。このsimple failure modelは以降の節にある個別の旧reason表現より優先する。
厳密さは実装時のcontract testに寄せ、実行時は各sourceを一度取得しterminal後に一度finalizeするだけにする。
同じ値を複数validatorで囲むruntime safeguardは追加しない。

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

### 調査根拠

現行fileに加え、`9109ede`（BigQuery収集導入）、`e1d5868`/`fdd81c1`（過剰metric削除とaggregate化）、
`a923409`（commandを一metric化）、`cc63e9d`（PJM log保持）、`8ac88e3`/`44143c7`（2-node実行とteardown）、
`b6daf9c`（bench/profile分離）までの関連commit列を確認した。過去のCodex threadで確定した「欠測はNULLだけ」
「OpenFOAM移管を基準」「稀な失敗は許容」は本書の決定状態へ要約し、会話本文やcommit messageをrun metricへ
複製しない。

## raw relationの契約

raw tableは現在の三列を維持する。

```text
run    STRING NOT NULL
key    STRING NOT NULL
value  STRING NOT NULL
```

`run`は一runの全producerが共有するglobally uniqueでimmutableなIDである。job投入前に
OS CSPRNGの16 byteを32文字のlowercase hexで表す128-bit run IDを一つ生成する。
manual経路でも同じrun ID規則を使う。
`pjsub`返却後にしか分からないPJM job IDはrun IDへ
含めず、submission resultとjob-entryのjob IDをquery時に結ぶ。完全導出できるbinding recordは作らず、初期run manifestを書き換えない。
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
- collectorは成功時にfragmentを閉じ、失敗時はnonzeroで終わる。別のcollection status、原因、error count、
  started/final耐久journalは作らない。finalizerはvariant別expected fragment/fileの存在から
  `collection_complete`を一度導出し、既に完成したsource/logだけを収録する。

contract/protocolのIDやhashはmetricへ保存しない。既存keyの意味、unit、JSON shapeを変える場合は
新keyにし、細かい仕様はソースと文書を正本にする。

各keyまたはkey patternは、owner、logical type、unit、cardinality、source、scope、
取得時点、required/optionalを一つのGit管理registryで定義する。動的keyの
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
effective、plannedとobserved、異なるnode rank、別phaseの時間は独立事実である。公式producer
との照合、schema routing、保存層をまたぐmanifest索引などの正当な重複は、registryの
`duplication_justification`へ理由と検証式を書く。

全環境変数の保存は、取り漏らしのないforensic比較という要件に基づく明示的な非最小例外
である。個別env値を別namespaceへ再掲せず、標準Sheetでは比較関連projectionだけを出す。

## raw metrics exportと送達

v2は任意のcrash境界から一件も失わず再開できることを目的にしない。release gateは、成功した各run
variantの正常pathでcontractが要求する設定、性能値、raw log、PJM accounting、profile runのprofile rawが
Artifactへ収録され、finalizerで確認できることである。失敗・中断runは完成済みfileだけを残し、partial recordの
復旧処理は作らない。supervisorがterminal後のfinalizerまで到達した失敗runは、そこで存在する完成済みlog/sourceと
`collection_complete=false`を通常の一artifact setへpackageする。finalizer前のcrashまたはpackage自体の失敗だけは
完成artifactを作らない。原因fieldは追加しない。

`metrics.jsonl`はBigQuery fact tableと同じ`run/key/value`を持つrun単位のraw exportである。artifact projectionは
subject artifact setからPublisherが別relationへ送る。別event
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
  -> merge済みfragmentを削除し、log/sample/source/resultのrun bundleとartifact-manifest.jsonを作成
  -> run bundle、merged JSONL、artifact manifestをGitHub Artifactへupload
  -> upload成功後にGenkai runtimeを削除可能
  -> archive jobがhome RAIDへscpし、runnerがBigQueryへ送信
```

- 一つのJSON objectを一lineで書き、producer終了時にflush/closeする。各lineやdirectoryごとの`fsync`は
  要求せず、必要なら一定間隔のbuffer flushを使う。
- expected producer/file集合はimmutable contractのvariant applicability、protocol、bundle selectionからjob投入前に一回導出する。manifestへ再掲しない。scheduler
  accountingとobserved placementは、その期待集合が実際に成立したかを後から検証するsourceである。
- mergeはrun ID、JSON、owner、expected node、一意性を検査し、欠損や異値競合を隠さない。
- BigQuery `insertId`はbest-effortの補助にすぎない。logical viewが完全一致rowをdeduplicate
  し、異値競合を隔離する。
- replayは同じrowを全件送ってよい。送信成功後もarchiveのJSONLを変更しない。publish失敗時はhomeまたは
  GitHub ArtifactのJSONLからworkflowを手動再実行する。
- upload失敗はattempt failureとする。upload成功後はruntimeを削除してよく、home転送のSCPが
  失敗した時だけGitHub Artifactを残す。

artifact identityの自己参照を避ける。全fragmentを一度mergeして`metrics.jsonl`を確定した後、log、source、
resultからrun bundleを作る。fragmentはmerged JSONLの完全な部分集合なのでartifactへ保存しない。
profileではraw GPU sampleとapplicableなprofiler rawだけを別のprofile raw tarへ
収録し、bundle、metrics sidecar、profile raw tarのpath/size/hashを
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
   append/flush、data fragmentのcompletionを所有する。別のcollector status fileは作らない。
3. `StageRunner`だけがspawnするactual argv、monotonic timer、signal、child wait statusを所有し、
   観測したfactを`Recorder`へ渡す。
4. runner側`merge`がexpected producer、run ID、owner、一意性、required key、fragment完了を検証し、
   archive対象のmerged JSONLを作る。node-scoped full factをrun-scopedへ変換しない。
5. `ArtifactStore/RunPackager`がrun bundle/manifestの作成と検証を所有する。GitHub Artifact uploadと
   home移管はworkflowが担当し、汎用reconciler、quarantine service、delivery ledgerは作らない。
6. runner側`publish`だけがBigQuery credentialと、一attempt一送信・同じartifact setを入力にした手動workflow
   再実行を所有する。collector、SGLang、PJM
   processはBigQuery、artifact upload、HTTP/cloud client等の外部送達networkを使わない。唯一、
   `node-agent`だけがprotocolの個別fieldで許可した明示rendezvous endpointへのrun内control transportを使える。
   credential、DNS discovery、外部egressを許可しない。

SGLang本体へ必要なhookは、version付きlocal structured eventを定義済み境界でだけ書く薄いadapterとする。
metrics packageやBigQuery clientをimportせず、送達、再試行、key展開を知らない。実装上は
`domain.py`、`recording.py`、`submission.py`、`stage_runner.py`、`merge.py`、`artifact_store.py`、`publish.py`、domain別
probe/parserを深いmodule境界にし、CLIはそのcomposition rootに限定する。

QwenのSGLang patch内で唯一のresult writerとして、rank 0の`PrimaryResultWriter`だけは
`backend.generate` return後にprimary artifactをtemporary fileからrenameして完成させる。これはmetrics writerでも
bundle storeでもなく、metrics keyを知らないsource-artifact writerである。schemaはcontract、run、
invocation ID、generation duration、returned/error count、observed input/output totalsに固定し、runnerの
primary-result parserだけが対応raw keyのownerになる。

Qwen用とOpenFOAM用を直ちに第三の共通packageへ統合しない。まず同じcontractとconformance
fixtureを両repositoryで満たし、interfaceが安定してから、canonical encoder、registry validator、
merge validatorだけを抽出する。domain collectorやpipeline固有の失敗処理は共通化しない。

実装開始前の設計正本として、各factの権威source、producer、取得時点、raw key/artifact、validation、
欠損時の扱いは[観測マップ](./01-v2-observation-map.md)、field-level shapeは
[contract schema handoff](./02-v2-contract-schemas.md)に置く。最初の実装sliceでこれらを一つの
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

## 実験データの独立性

一回のjobを一つのrun IDで記録し、各runは他のrunから独立した実験データとして扱う。
block、実行順、replicate index、pair ID、group/set ID、planned attempt、stopping ruleなど、
反復・ペア実行を管理するfield、metric key、manifest fieldは作らない。

Qwen/SGLangの実際のsource、workload、個別実行設定、配置、結果、時刻、artifact identityは各run自身が記録する。
複数runの比較や統計が必要な場合は、保存済みの個別keyを利用者がその都度選んでqueryする。
事前grouping、pairing、fingerprint、attempted count、median/MADなどの反復統計をv2の標準viewやSheetsへ持たせない。


## 分散producer契約

現行の共有filesystem上Unix domain socketをnode間event transportとして使わない。Unix
socket endpointは各node localに作り、各nodeのeventはそのnodeのJSONLへ書く。

- rank 0だけがworkload manifest、benchmark result、run-scoped phaseのauthoritative
  producerになる。
- 各nodeはresolved args、environment、actual argv、affinity、GPU mapping、telemetryを
  node scopeで出す。
- rawはnode/rank-scoped full factを正本として保持する。viewだけが全expected scopeの一致を
  検証したfieldをcommon表示し、rawにcommon/deltaの第二表現を作らない。
- expected node rank `0..protocol.nodes-1`、global worker rank `0..protocol.tensor_parallel_size-1`、distinct hostname、GPU UUID集合を
  manifestとschedulerへ照合する。
- 各nonzero node rankのcoordinated teardownと早期終了は、そのnode-local clock上のchild exitと
  `lifecycle.shutdown.start`受信時刻の前後で区別する。

controlを二familyへ分ける。`lifecycle control`はprotocolの個別設定で指定したmulti-node実行を正しく終了させる共通correctness harnessとして
bench/profileで共通にし、startup readyと`lifecycle.shutdown.start`の通常handshakeを
最終summary logへ記録する。`GPU phase control`はprofile-onlyで、warmup/measured start/endとnode受信時刻を
profile summaryへ記録する。両者のproducer/completenessを混ぜない。

nonzero node rankの`Engine()` constructorはdistributed completionまで内部でblockするため、constructor後を
configuration producerにしない。SGLang内部で最終runtime値を解決した後、各participantがwaitへ入る
直前のhookからnode fragmentをemitする。以下のGPU phase marker/control forwarderはprofile protocolだけで
起動し、benchではproducer graphから外してNULLにする。profileではmarkerをSGLang worker内部へ広げず、benchmark
開始前に接続済みの小さなcontrol forwarderを使う。rank 0 offline hookが`invocation_id`、phase、
start/end、sequenceを各node monitorへbroadcastし、各monitorがnode-local受信時刻を記録して
ACKする。最終summaryをartifactへ残し、異なるnodeのmonotonic値同士を減算しない。この実行設定
ではnative-worker markerとcontrol-forwarder markerを同じmetricへ混在させない。

profileのgeneration順序は`start broadcast -> expected protocol.nodes ACK -> rank0 timer start -> generate ->
timer end -> end broadcast -> expected ACK`へ固定する。start ACK完了からgenerate開始までのleadと、
generate終了からend ACK完了までのtrailをrank 0 clockでartifactへ残すため、GPU observation windowが
generationの保守的supersetであることを判定できる。warmup/measuredのsummaryは通常result writerと同じ
rank 0 monotonic-ns境界を使い、保存済みwall-time scalarをexact `(end-start)/1e9`で検証する。scalarは全variant共通
result、timestamp pairはprofile window containmentのためだけに併存し、別timerを二重取得しない。全purposeでgeneration timer終了後に
`lifecycle.shutdown.start(invocation_id)`を全expected nodeへ一度だけbroadcastし、通常ACK後に
`backend.shutdown()`を呼ぶ。

保存raw factは次の形にする。

- `run.stage.benchmark.launch`: stageが起動したouter MPI `argv/cwd/stderr_mode`のcanonical JSON object
- `parallel.node.<node_rank>.process_lifecycle`: child start/end、raw wait statusまたはspawn error、
  `<boot_id>:CLOCK_MONOTONIC` clock IDだけのJSON object。exit/signal/kindは保存せずviewで導出
- node topology artifact: node wrapper/Singularity/SGLang argv、host process affinity、static GPU mappingを所有する
- node config event artifact: resolved BenchArgs/ServerArgsだけを所有する。argv/affinity/GPUを複写しない
- worker-ready event artifact: global rank別のruntime backend、device/UUID、worker affinityを所有する

shell表示はviewで作る。同じnode固有fieldをrun-scoped keyへ送らない。
wrapperはshutdown受信や分類結果を自身のfragmentへ複写しない。node-agentの最終summaryをshutdown受信の正本、
wrapper fragmentをchild lifecycleの正本にする。summaryはrun、node rank、invocation ID、sequence、local
monotonic timestamp、`CLOCK_MONOTONIC` clock IDを持つ。reducerがshutdown invocation ID一致とnodeごとの一意性を検証し、viewはそのvalidな一件と
child lifecycleをrun/node rankでjoinする。clock IDも一致する時だけshutdown受信より前のchild exitをearly
failure、受信後をcoordinated teardownと導出する。受信record 0/複数、invocation/clock不一致では分類を
作らずNULLにする。別nodeのtimestampを直接比較せず、rank 0 completion file到着だけでnonzero node rankの早期死を
上書きしない。

## Qwen/SGLangで保存する原始事実

### run、stage

- `run.started_at`、`run.finished_at`、`run.elapsed_sec`
- `run.exit_code`。`run_outcome`は0=`succeeded`、非0=`failed`、欠損=`NULL`としてviewで一度導出する
- `run.stage.benchmark.launch`。node別actual argvはnode topology artifactだけが所有する
- `perf.stage.benchmark.wall_time_sec`とexit/signal/spawn errorを区別するstage termination object
- `scheduler.request`: runner submit boundaryの個別実行設定、PJM script identity、exact `pjsub` argv
- `scheduler.accounting`: 固定した`pjstat --history -S --choose ... --data --delimiter "\t"`の結果
- regular runだけ、submission resultの`scheduler.submit_job_id`とjob-entryの
  `scheduler.job_entry_job_id`を別sourceのscalar indexとして保存し、equalityはviewで導出する。binding booleanは作らない
- benchでもmodel load前に一度だけ、node別topology artifactへlogical index、PCI bus ID、UUID、model、
  memory bytes、driver versionと設定値を保存する。run-scoped `hardware.gpu.nodes` raw factは作らない
- profileだけ、minimal CPU/settingsに含まれないfull NUMA/memory raw artifactを追加する。captureのrun ID/hostnameを
  minimal hardwareの`(node_rank,hostname)`と照合し、meminfo kB→bytesとcpulist canonicalizationを固定する。
  `hardware.cpu_numa.nodes` raw factは作らずconsumerへ直接投影し、benchではNULL
- worker affinity/cpusetは`worker_ready`のglobal-rank-scoped runtime fact

run lifecycleはPJM entryからcompute source/log回収終了までを取る。started/finishedはUTC wall clock、elapsedは同じsupervisor境界の
monotonic clockで取得するため、
wall-clock jumpとtimestamp精度の影響を受けず厳密な派生関係ではない間だけ残す。同一scope、clock、
precisionへ統一した時点でelapsedまたはtimestamp pairの一方をview派生へ移す。

### SIF、model、dataset provenance

SIF内manifestにはbuild inputsとして次を固定する。

- buildを定義した`2_Qwen`のrepository ID、object hash algorithm、full commit
- `5_SGLang_Src`のrepository ID、object hash algorithm、resolved full commit
- base OCI digest、resolved package/toolchain manifest
- workflow run/attemptはSIF/sidecarに入れずoperational build rowだけが所有する

nccl-testsはmoving masterを使わずcommit
`717b68318278e93f371d8ffb46b076069d7c7851`へ固定し、MPI対応かつJSON出力対応でbuildする。
SIF build testは`all_reduce_perf`、`all_gather_perf`、`reduce_scatter_perf`の存在、固定commit、MPI support、
`-J` optionの利用可能性を検証する。実NCCL library versionはrun時に`software.nccl.version`へ記録する。

SIF完成後に実際の参照先binaryをSHA-256で検証し、そのcontent identityとembedded manifestの
identityをsidecarへ保存する。SIF全体hashをSIF自身へ埋め込む自己参照にはしない。
sidecarはscore bundle `source_paths`ごとのfinal tree stateをpath順のabsentまたはmode/blob OIDとして一度持ち、
SIF semanticへ含める。referenceはemptyとするため、
別instrumentation source bytesを同じsemanticとして扱わない。
v1のbundle `source_paths`とin-process hookはすべて`sif_build_definition` repositoryに属し、各hook `source_path`を
bundle集合とsidecar tree entryへ含める。upstream `sglang_source`を直接変更する場合は、曖昧に混ぜず別contractで
そのrepositoryのdiff/treeを追加する。
recipeの`From:`はdigest-pinned referenceにし、buildが消費したOCI manifest/config/layer digestを
sidecarへ記録する。build scriptがSIFとsidecarを書き終えて成功した後だけcacheへ公開し、reuse時に
両者のidentity一致を検査する。任意のcrash境界から二fileを復旧するtransactionは作らない。

model sourceはtagged unionにする。Hub由来なら`hub_revision`としてrepositoryとfull immutable revision、local
prebuiltなら`source/model-manifest.json`自身の`prebuilt_manifest` recordに各shardの
relative path/size/SHA-256を必須にし、完全導出できるmanifest IDはrecordへ複写しない。別のprebuilt manifest artifact/pathは参照しない。runの
`source/model-manifest.json`は選んだsource record、`config.json`と`model.safetensors.index.json`のSHA-256、indexが
参照するsorted unique shardのrelative path/size、tokenizer/chat-template fileのpath/size/SHA-256を持つ。runごとは
small file hashとshard path/sizeをsource recordへ照合し、400 GBのshard本文を再hashしない。

datasetはworkflowのtracked protocolでfull repository revisionをpinし、`hf download --revision <full-revision>`を
使う。`source/dataset-manifest.json`へrepository/full revision、relative file name、size、SHA-256を保存する。
moving `main`や共有pathだけをidentityにしない。runが一度所有し、provision時にmanifestと実体の一致を検証する。

raw索引は`artifact.model.manifest_sha256`、`artifact.dataset.manifest_sha256`だけを持つ。
repository/revision/memberはmanifest内の完全recordが正本であり、個別raw keyへ再掲しない。

raw tableはrun自身の再現に必要なcommitとmanifest identityだけを索引化する。source manifestはrun bundleの固定member
から到達し、bundle URIはworkflow operational joinが所有する。SIFや全model shardを各benchmark runで再hashしない。

### 実workload identity

rank 0は、実際に選ばれたrequest順序を持つcanonical manifestをartifactへ残す。各recordは
少なくとも次を含む。

- dataset rowならdataset manifest identity+元JSON array index、random warmupならgenerator ID+seed+ordinalのsource identity
- 実generateへ渡すexact prompt payload。現行text pathはUTF-8 textとtokenizerが観測したinput token count、
  将来token-ID pathならinteger arrayのtagged union
- requested output length
- sampling parameter
- request順序

actual input promptは比較に必要なため保存するが、生成本文は保存しない。manifest identityはordered requestの
exact prompt payloadを含む全recordから作る。publisherはartifactをparseし、request ordinal/source identity/
prompt payload/requested output length/sampling
parameterを`qwen_workload_requests_raw_v2`へqueryable copyとして投入する。manifest hashだけではworkloadの
長さ分布や内容をSheet/agentが判断できないため、このrelationはartifactとBigQueryの保存境界をまたぐ正当な重複である。
workload manifest identityは`qwen_workload_manifests_raw_v2`のsource SHA-256だけが所有し、通常Factへ複写しない。
protocolの単一expected workload identityは
`SHA-256(JCS({warmup:<warmup manifest file SHA-256>,measured:<measured manifest file SHA-256>}))`とし、regularはrun、

warmup manifest内のmodeもmeasured前のcache/stateを変える一致必須条件にする。selected count、planned input/
output token total、modeはmanifestの完全な部分集合なのでconsumer viewで導出し、raw keyやPrimaryResultへ
複写しない。総token数が等しくてもrequest順序や長さ分布が異なればmanifest identityが変わる。

### benchmark結果の原始観測値

- `result.request.returned_count`
- `result.request.error_count`
- `result.token.input.total`
- `result.token.output.total`
- `perf.benchmark.generation.wall_time_sec`

returned/error countは実際に返ったresponse配列と各objectを検証して作る。返らなかった件数は
viewで`workload.requests.length - returned_count`として導出する。generation開始前の予定request数を
`successful_requests`と呼ばない。result fileは存在だけでなく、非空、valid JSON、期待row数、
有限で正のlatency、token整合を検査する。

runtime backendの文字列出力はgenerate return直後にdecodeし、各responseの
`meta_info.prompt_tokens`と`completion_tokens`からobserved totalsを作る。malformed/error responseは
`error_count`へ分類し、取れないtoken値はNULLにする。generate全体が
例外ならreturned/error/token countを0と捏造せず、generation lifecycle failedとkey不在で表す。

bench/profileのprimary resultは、rank 0がこの検証直後に作るartifactを唯一のsourceとする。
temporary fileをwrite/closeしてrenameしてからprofile固有取得または
shutdownへ進む。`get_server_info()`は一次factを持たずprofileだけで呼ぶ。通常benchはprimary result完成後に
直ちにlifecycle shutdownへ進む。予定件数を成功件数として扱うnative `result_filename`は使わない。
この限定的な二sourceは`duplication_justification`へ検証式とともに登録する。

benchmark validityは少なくとも`error_count = 0`、`returned_count = workload.requests.length`、observed
input totalとplanned input totalの一致、有限かつ正のgeneration wall timeを要求する。現在の
`ignore_eos`固定protocolではobserved output totalとrequested output totalの一致も要求する。
nullableなordered output identityは通常bench/profileの`benchmark_valid`へ追加せず、開発時の一時比較の
evidence presence/result equivalenceにだけ要求する。

request/input/output/total throughputは上記の原始値からviewで計算する。資格確認/primary artifact内へ
代数的throughputを複写しない。一次比較に使わない`last_gen_throughput`も取得・保存しない。

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

### MPI・NCCL通信計測

bench/profileのprimary generationが成功した場合に限り、同じallocation、MPI process数、GPU/process、transport環境で
IMB-MPI1とnccl-testsを本処理後に実行する。cluster固有launcherは後続のジョブ組込みが完全argvを組み立て、
共通capture runnerはargvを変更せずstdout/stderr/argv/terminationを保存する。

```text
IMB-MPI1 -npmin <generationと同じMPI process数> -msglog 0:22 -iter 1000 -time 1 \
  -iter_policy dynamic PingPong Sendrecv Allreduce
```

0 byteおよび1 byteから4 MiBまでの2冪を対象にし、rawは`source/network/imb-mpi1/`へ置く。pure text parserは
complete sectionだけからIMB自身のprocess count、message size、iterations、time、bandwidthを復元し、独自平均や再計算値を作らない。

nccl-testsは`network_nccl_all_reduce`、`network_nccl_all_gather`、`network_nccl_reduce_scatter`の独立stageとし、
`all_reduce_perf`、`all_gather_perf`、`reduce_scatter_perf`を次の共通引数で実行する。

```text
-b 8 -e 1G -f 2 -w 5 -n 20 -m 1 -N 1 -d float -c 1 -a 1 -T 300
-g <generationと同じGPU数/MPI process> -J <collective別JSON path>
```

全MPI process・全GPUで一communicatorを作り、`NCCL_TESTS_SPLIT`は設定しない。JSONを正本としてparseし、
stdout/stderr/argvも`source/network/nccl-tests/<collective>/`へ保存する。out-of-place/in-placeを別rowにし、
nccl-tests出力の`algbw`と`busbw`を意味の異なる値としてそのまま保存する。

固定幅factsは`network.imb_mpi1.*`、`software.imb_mpi1.version`、`network.nccl_tests.*`、
`software.nccl_tests.commit`、`software.nccl.version`とstage termination/wall timeだけである。
IMBは`mpi_network_results_v2`へ`(run_id,benchmark,process_count,message_bytes)` grain、NCCLは
`nccl_network_results_v2`へ`(run_id,collective,message_bytes,buffer_mode)` grainで投影し、summary keyは作らない。

generation失敗時は通信stageを一つも起動しない。通信stageの失敗はgeneration/run成功を覆さないが
`collection_complete=false`とし、parse不能・truncated sourceから結果rowを作らない。NCCL三stageは一つが失敗しても残りを続行し、
retryしない。generation wall timeへ通信時間を含めず、run全体時間だけが含む。ASPIRE 2A/2A+、Gadi、玄界のlauncher、
module、host mapping、Singularity、walltime、PJM/PBS/workflow編集は後続作業とする。

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
logical indexとnullable measurement値を持つ。control markerを各nodeの同じnode-local clockで受け、expected
nodeのstart/endが一つでも欠ければgeneration aggregateを作らない。poll status/reason fieldは作らない。node GPU windowは同じ
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

viewは`collection_complete`、`benchmark_valid`、`score_comparison_eligible`、
`run_outcome`だけを独立に導出する。configuration/lifecycle/provenance/topology/telemetryごとのstatusやreasonは
作らず、個々の値はRunsまたはSourcesで直接確認する。profile telemetryはsample/valid duration/coverageと
nullable aggregate値だけを表示し、complete booleanを作らない。

telemetry欠損だけをbenchmark failureへ読み替えない。各runでnode rank `0..nodes-1`、distinct host数、
GPU/node、unique GPU UUID数、TP、nnodes、MPI process数をprotocolの個別fieldへ照合する。現在の具体設定値は
2 nodes、1 MPI process/node、4 GPU/node、TP=8である。configuration eventやworkload manifestが欠けた成功runはthroughput値を残すが、
score comparison eligibleにはしない。

`run_purpose`は各run自身の用途として保存するが、group/set IDは作らない。
`score_comparison_eligible`は`run_purpose=bench`と通常のresult/source/placement gateだけから導出する。
開発時の非擾乱確認はrunやeligibilityとして記録しない。profileはthroughputを算出・表示してもscore stats、best、rankingへ入れない。

## bench/profile trackの分離

重いprofilerやGPU pollingを挟むrunと公式性能を測るrunは分離する。

- `bench`: score、環境比較、通常の結果・provenance・placementを保存する。profiler、GPU polling、
  verbose traceを入れない。
- `profile`: benchのexact commitへprofile専用overlayを載せる診断track。score集計、
  環境rankingへ混ぜない。

key catalogのapplicabilityは`bench|profile`だけを持つ。非適用でも取得失敗でもconsumer値はNULLだけとし、
field別status、欠損placeholder、reasonを作らない。required sourceの有無はrun-level
`collection_complete`へ一度だけ反映する。

## BigQuery viewとスプレッドシート

BigQueryには通常bench/profileのrun、source、artifact、workload、result、placement、profile projectionだけを公開する。
開発時の非擾乱確認、計測なし／ありの組、allocation、判定recordを表すrelationは作らない。

Runsの実行設定はnodes、MPI processes、GPU/node、tensor parallel size、resource group、transport、
address source、portを個別列にする。設定を束ねるID/object/manifest列は作らない。

Connected Sheetsの既存tab設計は維持する。通信結果の表は全実装完了後に設計し、現段階では
`mpi_network_results_v2`と`nccl_network_results_v2`を将来の`MPI Network`、`NCCL Network` tabの入力候補と
することだけを固定する。JSON展開やsummary列は要求しない。固定幅source欠損はNULL、long relationは実在rowだけを返す。

## 開発時の非擾乱確認

experiment-metrics-v2のcollector実装中に、代表2-node TP=8設定で計測なし／ありを同条件で実行し、
throughput、stage wall、ordered output identityを見て実用上の低下や数値変化がないことを確認する。
これは実装作業の一時チェックであり、確認後は出力を破棄する。

v2には専用run purpose、arm、pair、allocation、manifest、artifact、Fact、metric key、BigQuery relation、
Sheets tab/column、eligibility、Git管理decision record、fixtureを作らない。通常benchは確認済みbundleを直接使い、
一時チェックの結果をrunへjoinしない。


## 決定事項と再open条件

コード監査で、profile tool contract D-004が残っていることが分かった。現行
`scripts/exec-profiler.sh`はtransparent wrapperであり、SGLang側にはTorch Profilerと
`CUDA_PROFILER`/Nsight系の複数経路があるため、tool、activity、step範囲、required raw file集合はコードだけでは
一意に決まらないためD-004で初版profileのprofilerを次の最小contractへ固定した。transparent wrapperだけを
「profiler取得済み」とは扱わない。

2026-08-09にSGLang `5ffba99b631f1f442371cc7a7d79391409197724`を再監査した。`offline_throughput`が直接公開するのは
`--profile`、`--profile-activities`、`--profile-steps`であり、通常warmupは常にprofile外、measured generation直前に
profileを開始する。stepはrequest数でなくscheduler forward数である。現行経路はoutput directoryを
`SGLANG_TORCH_PROFILER_DIR`から読み、stack/shapeは何も指定しないとともにtrue、profile IDはtimestamp、mergeと
stage別profileはfalseになる。legacy Torch Profilerは一つのprofile requestを全schedulerへfan-outし、現在の
DP/PP/EP=1の現在設定ではglobal TP rankごとに一つのgzip Chrome traceを書く。従ってexpected trace数は現在のTP=8という
固定値でなくprotocolの`tensor_parallel_size`から展開する。

コード・artifact・情報設計の監査では次が不足していた。(1) timestamp filenameとoutput directoryを
offline CLIで固定できず、事前展開するsource path contractを閉じられない、(2) 現行monitorは最初の一traceだけで終了し、
0件では待ち続けるため全rank completenessの正本にならない、(3) `SGLANG_PROFILE_V2=true`のmanual start/stopは未実装、
(4) traceをraw tarへ置くだけではBigQueryからSheetへ診断値を渡せない、である。採用する初版contractは次である。

```text
tool                     = SGLang built-in PyTorch Profiler (legacy path)
activities               = [CPU, GPU]
window                   = warmup外、measured generation先頭5 scheduler forward
with_stack               = false
record_shapes            = false
merge/profile-by-stage   = false
required raw grain       = global TP rank
required raw set         = protocol.tensor_parallel_size件のgzip Chrome trace
```

profile overlayは`output_dir`と`profile_id`だけを`offline_throughput`から明示できる小変更にし、profile IDを
`qwen-profile`、pathを`profile/raw/torch/qwen-profile-TP-<global_rank>.trace.json.gz`へ固定する。
`SGLANG_PROFILE_V2=0`、`SGLANG_PROFILE_WITH_STACK=0`、`SGLANG_PROFILE_RECORD_SHAPES=0`もprotocolで明示する。
backend終了後のfinalizerがprotocolの`tensor_parallel_size`から展開したpathを一度検査し、現行の任意一file monitorは正しさに使わない。
全traceをmergeした派生trace、Nsight Systems/Compute、CUDA profiler、memory snapshot、CUDA graph capture trace、
別tool fallback、retryは初版へ入れない。

opaque traceとrun/global rankを結ぶため、実在する各traceについてだけclosed `profiler-capture/v1`
`{schema_version,run_id,global_rank,invocation_id,source_member,size_bytes,sha256}`を同じprofile rawに一件置く。
tool/version/argv/outcome/reason/statusやexpected rank集合は複写しない。publisherはtraceから一度だけ
`(run,global_rank,activity,event_category,event_name,call_count,inclusive_duration_sum_ns,source_member,source_sha256)` grainの
`qwen_profiler_events_raw_v2`を作る。per-event trace本文、平均、構成比、rank合計、top-N、CPU/GPU/NCCL分類、run-wide
summaryは保存せずquery時に導出し、bounded `Profiler` Sheet tabがprofile run同士を比較する。trace欠損時は既存
`collection_complete=false`、固定幅consumerはNULL、long relationは実在rowだけとし、profiler専用complete/reason/placeholderを作らない。

D-004のreopen triggerは、固定したSGLang commitでlegacy profiler経路が削除されるか、global TP rankとtraceの
一対一対応が成立しなくなった時だけとする。profile traceの稀な欠損は再設計理由にせず、通常どおりNULL/0 rowで残す。

D-003は2026-08-09にユーザー回答「基本的にはマルチノード対応型の収集。Qwen推論も同じ」で確定した。
2026-08-10の追加回答により、実行設定を一つの抽象やIDへ束ねず、nodes/worker/GPU cardinalityその他を
protocolの個別fieldから展開し、Sheetsでも各値を別列にする。affected fields/viewsはprotocolの個別実行設定、
scheduler request、producer node/global-rank axes、topology/host/GPU consensus、lifecycle/phase ACK集合、
participant termination、collection/score eligibilityである。OpenFOAMは1/2/4-node実機canary、Qwenは
個別field駆動のcardinalityと現在の2-node TP=8実機設定を対象にする。
reopen triggerはSGLangまたはschedulerの配置意味が変わり、v1の一node一MPI wrapper/full-GPU TP制約で表せなくなった時だけとする。
collector実装中の非擾乱確認は代表2-node TP=8設定で一時的に行い、確認後に出力を破棄する。

## 決定状態

| 状態 | 決定 |
|---|---|
| 確定 | raw relationは`run/key/value`の三列、compound valueはcanonical JSON |
| 確定 | 定義した各process scopeの環境変数は機密値を含めfilterせずbyte-exactに全件保存 |
| 確定 | native object/structured sourceを優先し、log regexはfixture付きの最終手段 |
| 確定 | compute側はlocal raw fileだけ、finalize/publishはrunner、Artifact upload/home移管はworkflowが担当 |
| 確定 | observed totalsから派生するthroughputやnode full factsのcommon copyをrawへ重複保存しない |
| 確定 | bench/profileを分離し、取得できなかった値とprofile専用familyは値だけNULL、原因/status/欠損用recordなしとする。run-level statusは`collection_complete`だけ |
| 確定 | 実験用途として多段retry、fallback source、field別reason registryを作らない |
| 確定 | artifact移管はOpenFOAMのGitHub Artifact -> Tailscale/SSH -> home RAIDを基準とし、Qwenへ同じ経路を移植する |
| 確定 | home artifactは自動expireせず、容量が実際に問題になった時だけ手動整理する。専用quarantineは作らない |
| 確定 | D-003: Qwen collectorはmulti-node対応とし、expected node/worker/GPU集合をprotocolの個別fieldから展開する。実行設定を束ねるID/object/manifestは作らず、Sheetsでは各値を別列にする。現在の実機設定は2 node、4 GPU/node、TP=8 |
| 確定 | D-004: profileはSGLang legacy PyTorch ProfilerのCPU+GPU、warmup外の先頭5 scheduler forward、stack/shape/mergeなし、global TP rank別traceを使う |
| 確定 | 非擾乱確認はcollector実装中だけ行い、結果・専用run・metric・artifact・relation・decision recordを保存しない |

## 実装ハンドオフ

公開面はbench/profileの通常runだけを扱う。

```python
Fact(key: str, logical_value: LogicalValue)
init_run(settings_path, runtime_root, identities) -> RunManifestPath
Recorder.open(manifest_path, registry_path, producer_id, runtime_dir)
collect_once(manifest_path, runtime_dir, producer_id, probe) -> Path
submit_run(manifest_path, runtime_dir, execution_fields, pjsub_argv) -> SubmissionOutcome
run_stage(stage_name, argv, *, cwd, stderr_mode, recorder) -> Termination
capture_network_stage(stage_name, argv, *, cwd, artifact_dir, recorder) -> Termination
finalize_run(manifest_path, runtime_dir) -> FinalizeOutcome
publish_run(manifest_path, artifact_set_path, destination) -> PublishOutcome
```

CLIのpurposeは`bench|profile`だけとする。非擾乱確認、arm、pair、allocation、childを表すoption/subcommandは
追加しない。owner scopeはrun/build、source storageはfact fragment、run bundle、profile rawだけである。

runtime layoutは`runs/<run-id>/`の下に`control/`、`domain/`、`source/`、`logs/`、
`profile/raw/`、`archive-staging/`を持つ。artifact setは`run-artifacts.tar.zst`、
`metrics.jsonl`、profileだけの`profile-raw.tar.zst`、`artifact-manifest.json`で閉じる。
開発時の非擾乱確認用file/directoryはこのlayoutへ入れない。

### test gate

manifest、submission/job ID、environment、merge、artifact、Qwen/SGLang意味、placement/lifecycle、profile、再構築に加え、
IMBの0 byte/科学表記/欠損列/複数process-count section/警告/truncated fixtureと、3 collectiveのNCCL JSON fixtureで
buffer mode、datatype、reduction op、time、algbw、busbw、errorを検証する。command失敗、timeout、JSON不在、部分出力は0 rowかつ
terminationと`collection_complete=false`だけになること、raw artifactから同一row集合を再構築できることもtest gateに含める。
catalog/schema testは`lane`、`qualification*`、`block_id`、`block_order`、`replicate_index`、`pair_id`、
`ib_*_bw`、`ib_*_lat`、network用contract/protocol ID・hashがないことを保証する。Connected Sheetsの詳細testは表設計時へ延期する。
非擾乱確認は実装者がその場で実行・確認し、
fixture、golden decision、publish対象にはしない。

### release単位

1. legacy output、evaluator、baseline logをgolden fixtureとして凍結する。
2. manifest、local metrics export、structured hooks、complete result、topology sidecarをv2 fixtureと最初のrunで通す。
3. benchへ必要な最小bundleをcollector実装中に代表2-node TP=8設定で計測なし／あり比較へ掛ける。
   意味・数値同等性・非擾乱を確認したら結果を破棄し、確認済みcollectorをbenchへ追加する。
4. v2 view/Sheetを切り替え、legacy-only consumerがゼロになってから旧sender/subset keyを削除する。

rollbackは実行設定と実装を直前版へ戻して新しいrunを投入する。既存runのmanifest/rowを書き換えず、
一run内でlegacy producerとv2 producerを混在させない。

## 実装順序

1. 現行key、producer、consumer、artifact保存先をinventory化し、key registryとexpected producer graphを作る。
2. clean checkout、remote到達性、SIF/build sidecar、manual provenance gateを作る。
3. BigQuery credentialをrunnerだけへ移し、Fact/Recorder、JSONL export、runner merge/publish、
   GitHub Artifact upload/home移管を導入する。
4. raw table、conflict検査、Sources lineage、artifact status、run/point viewを
   key catalogから生成しfixtureを通す。
5. actual workload、primary result、environment、topology、lifecycleをstructured sourceとして追加する。
6. canonical evaluatorを作り、legacy validity/throughputとbaseline golden logを照合する。
7. D-003の個別実行設定、node/rank/GPU projectionを追加し、各設定fieldとの一致を検証する。
8. D-004のprofile raw/capture/event projectionを追加する。
9. 共通network capture、IMB/NCCL pure parser、固定幅facts、二つのnetwork long relationを実装し、round-tripと
   失敗時0-row規則をfixtureで通す。SIFは固定nccl-tests commitをbuild/testするがcluster固有ジョブへは組み込まない。
10. Connected Sheetsをbounded consumerへ切り替え、旧key consumerがなくなってから冗長key生成を止める。
11. home run artifact treeとGit管理source/SQLから空datasetを再構築して旧viewを廃止する。

## 完了条件

- source/build commit、toolchain、SIF/binary identityを追跡できる。
- key ownerが一意で、distinct後のrun/key conflictがゼロである。
- BigQuery停止時もartifact JSONLから同じlogical row集合を再送できる。
- success/failure/timeout runのexpected source/logを保存し、artifact移管を再実行できる。
- benchmark validityとthroughputがlegacy golden logに一致する。
- workload、result、stage wall、node/rank/GPU placementを独立sourceから再構築できる。
- completeなIMB/NCCL rawからmessage size別rowをlosslessに再構築でき、通信stage失敗がgeneration成功を変更しない。
- requested/observedのnodes、MPI、GPU/node、TP、resource、transportが個別列で照合できる。
- profile-only列はbenchで理由なしのNULLとなり、profile raw/projectionをhomeから再構築できる。
- collector実装中の一時比較で性能・数値非回帰を確認し、その比較結果がv2のrun、key、artifact、
  relation、Sheet、decision recordに保存されない。
- Runs/Sources/Config Diff/Environment/Topology/Profile Counters/Data Statusが
  JSON展開なしで主要結果とlineageを表示する。
