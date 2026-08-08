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

`run`は全producerが共有するglobally uniqueでimmutableなIDである。Actions経路では
repository、run ID、attempt、PJM job IDを含め、manual経路ではclusterを含むUUIDを使う。
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
- collector lifecycle keyはsupervisorだけが所有する。collector呼出し前に
  `collector.<name>.started`を耐久appendし、呼出しから戻った後にstatusとerror countを持つ
  `collector.<name>.final`を一度だけappendする。どちらのkeyも更新しない。
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

```text
compute producer
  -> node別のmetrics JSONLへ耐久append
  -> node別fragmentをatomic completion
  -> runnerがexpected nodeを検証・merge
  -> runnerがBigQueryへ全行をat-least-once送信
  -> fragment、merged JSONL、全logをarchive
  -> remote保存を検証してからruntimeを削除
```

- 一つの完全なJSON lineを`O_APPEND`またはlock下で書き、flush/fsync後に完了とする。
- expected producer集合はscheduler allocationから固定する。
- mergeはrun ID、JSON、owner、expected node、一意性を検査し、欠損や異値競合を隠さない。
- BigQuery `insertId`はbest-effortの補助にすぎない。logical viewが完全一致rowをdeduplicate
  し、異値競合を隔離する。
- replayは常に同じrowを全件送ってよい。送信成功後もJSONLをtruncateしない。
- archive/upload失敗時はruntimeをquarantineし、後続reconcilerが回収する。
- remote checksumとmanifest確認前に唯一のlocal copyやGitHub artifactを削除しない。

timed benchmark中はBigQueryへ接続しない。送信用credentialはrunnerだけが保持するため、
PJM processの性能条件へ混入しない。各snapshotはそのscopeに実在する環境をfilterなしで
保存し、機密値も例外扱いしない。

## 環境snapshot

同じ名前でも時点、node、processが違えば意味が異なるため、次を分ける。

- `env.job_entry.*`: PJM entry直後、module操作やunsetより前の全環境
- `env.host.node.<node_rank>.*`: module設定後、各nodeのbenchmark exec直前の全環境
- `sglang.env.node.<node_rank>.*`: 各container内、SGLang初期化直前の全環境

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
- mergeはfull node snapshotからrank-invariantなcommon fieldsとnode-specific deltaを
  losslessly構成する。common値と同じ値をnode deltaへ再掲しない。
- expected rank集合、distinct hostname、GPU UUID集合をschedulerと照合する。
- rank 1のcoordinated teardownと、generation前の異常終了をphase/result有無で区別する。

rank 1の`Engine()` constructorはdistributed completionまで内部でblockするため、constructor後を
configuration producerにしない。SGLang内部で最終runtime値を解決した後、rank 1がwaitへ入る
直前のhookからnode fragmentをemitする。generationのnode-local markerもoffline driverからでは
なく、各worker/scheduler内部hookでemitする。内部hookを置けない版だけはrank 0から各node-local
samplerへcontrolを送り、そのlocal receipt時刻をmarkerにする。いずれもexpected node全ての
start/endを要求する。

metricは次の形にする。

- `run.launch_argv`: stageが起動したouter MPI argvのcanonical JSON配列
- `parallel.node.<node_rank>.benchmark_argv`: 各nodeのSingularity/SGLang argv
- `sglang.benchmark_arg.*`: rank間一致を確認したcommon resolved BenchArgs
- `sglang.server_arg.*`: rank間一致を確認したcommon resolved ServerArgs
- `parallel.node.<node_rank>.*`: node rank、dist init、MPI rank、affinity、GPU mapping、
  node固有resolved delta

shell表示はviewで作る。同じnode固有fieldをrun-scoped keyへ送らない。

## Qwen/SGLangで保存する原始事実

### run、stage、allocation

- `run.started_at`、`run.finished_at`、`run.elapsed_sec`
- `run.outcome`、`run.exit_code`
- `run.launch_argv`とnode別actual benchmark argv
- `perf.stage.benchmark.wall_time_sec`とstage exit code
- `scheduler.allocation`: requested/allocated node、host、process、GPUのJSON object
- `hardware.nodes`: hostnameごとのCPU、NUMA、cpuset、GPU UUIDと静的構成のJSON配列

started/finishedとelapsedは、現実装ではclockと範囲が一致せず厳密な派生関係ではないため
残す。timer範囲を統一した後も、運用・Sheets用の意図的な索引として扱う。

### SIF、model、dataset provenance

SIF内manifestにはbuild inputsとして次を固定する。

- buildを定義した`2_Qwen` commit
- `5_SGLang_Src`のresolved commit
- base OCI digest、resolved package/toolchain manifest
- original build workflow/run

SIF完成後に実際の参照先binaryをSHA-256で検証し、そのcontent identityとembedded manifestの
identityをatomic sidecarへ保存する。SIF全体hashをSIF自身へ埋め込む自己参照にはしない。

modelは全shardのimmutable manifest、datasetはpinned repository revisionまたはfile manifest、
tokenizerとchat templateはmodel artifact identityへ含める。共有pathだけをidentityにしない。
provision時にmanifestと実体の一致を検証する。

raw tableは比較groupとartifact到達に必要なcommit、manifest identity、artifact URIだけを
索引化する。SIFや全model shardを各benchmark runで再hashしない。

### 実workload identity

rank 0は、実際に選ばれたrequest順序を持つcanonical manifestをartifactへ残す。各recordは
少なくとも次を含む。

- dataset source IDまたはrow ID
- prompt token content digest
- input token length
- requested output length
- sampling parameter
- request順序

prompt本文と生成本文はBigQueryへ保存しない。BigQueryには次だけを索引化する。

- `artifact.workload.manifest_sha256`
- `workload.request.selected_count`
- `workload.token.input.planned_total`
- `workload.token.output.requested_total`

このcountとtoken totalsはresult completenessをartifact読取なしで判定するためのjustified index
である。総token数が等しくてもrequest順序や長さ分布が異なればmanifest identityが変わる。

### benchmark結果の原始観測値

- `result.request.returned_count`
- `result.request.failed_count`
- `result.token.input.total`
- `result.token.output.total`
- `perf.benchmark.generation.wall_time_sec`
- `run.stage.benchmark.result_present`

returned/failed countは実際に返ったresponseを検証して作る。generation開始前の予定request数を
`successful_requests`と呼ばない。result fileは存在だけでなく、非空、valid JSON、期待row数、
有限で正のlatency、token整合を検査する。

benchmark validityは少なくとも`failed_count = 0`、`returned_count = selected_count`、observed
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

各nodeのsamplerはsampleを実行中に圧縮可能なJSONLへ追記し、SIGKILLでも途中まで残す。
sampleはnode-local monotonic timestamp、GPU UUID、logical index、値、poll errorを持つ。
phase markerも各nodeの同じclock domainへ渡す。expected nodeのgeneration start/endが一つでも
欠ければgeneration aggregateを作らない。

BigQueryにはphase/node/GPU scopeを明示して、次だけを集約する。

- valid sample countとvalid observed duration
- generation GPU utilization mean/p95
- memory max
- energyとpower-valid duration
- temperature max、SM clock min

power meanはenergy/durationからviewで導出する。phase markerが欠けた場合はgeneration aggregateを
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

## 実装順序

1. 現行key、producer、consumer、artifact保存先をinventory化し、key registryと
   `run.metrics_contract`を作る。
2. clean checkout、remote到達性、SIF/model/dataset sidecar manifest、manual provenance
   gateを作る。
3. BigQuery credentialをrunnerだけへ移したうえで、現行metric集合のままJSONL WAL、node
   merge、runner replay、archive/reconcilerを導入する。
4. `raw_distinct`、conflict、completeness、v2 comparison viewを旧collectorと並行して作る。
5. node-local event endpoint、rank 0 ownership、common/delta mergeを実装し、397B runの
   configuration欠落と競合を修正する。
6. workload/cache manifestとresult validationを追加し、2 node TP=8 completenessを確認する。
7. observable phase eventとnode-local raw GPU sampleを追加し、timed pathからBigQueryを外す。
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
- 実2 node runでnode-local eventが届き、common値とnode deltaをlosslessに復元できる。
- expected rank、host、GPU UUID、TP、nnodes、MPI processの不一致を理由別に検出できる。
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
