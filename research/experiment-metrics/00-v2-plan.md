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
複製しない。

## raw relationの契約

raw tableは現在の三列を維持する。

```text
run    STRING NOT NULL
key    STRING NOT NULL
value  STRING NOT NULL
```

`run`は一childの全producerが共有するglobally uniqueでimmutableなIDである。各submission subjectにはjob投入前に
OS CSPRNGの16 byteを32文字のlowercase hexで表す128-bit `subject_nonce`を一つ生成する。regularでは
`submission_subject_id=run_id`、qualificationではglobally uniqueなallocation IDをsubject IDとし、二child run IDも
それぞれ独立なCSPRNG 16 byteからopaqueに作る。allocation IDとarmはmanifest fieldだけが所有し、child IDへencodeしない。
manual経路でも同じsubject/child規則を使う。
`pjsub`返却後にしか分からないPJM job IDはrun/allocation IDへ
含めず、submission resultとjob-entryのjob IDをquery時に結ぶ。完全導出できるbinding recordは作らず、初期run manifestを書き換えない。
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
- collectorは成功時にfragmentを閉じ、失敗時はnonzeroで終わる。別のcollection status、原因、error count、
  started/final耐久journalは作らない。finalizerはvariant別expected fragment/fileの存在から
  `collection_complete`を一度導出し、既に完成したsource/logだけを収録する。

`run.metrics_contract`は、emitterとviewの非互換schemaを識別するための小さな運用索引として
意図的に保存する。既存keyの意味、unit、JSON shapeは変更せず、非互換変更は新contract
または新keyにする。

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
完成artifactを作らず、qualification setではchildの`artifact_manifest=null`だけにする。原因fieldは追加しない。

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
   `node-agent`だけがlane contractで許可した明示rendezvous endpointへのrun内control transportを使える。
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

## 実験protocolと比較条件

一次estimandは、固定workloadをwarmup後に生成したtotal token throughputとする。二次
estimandはdistributed engine init、input prepare、warmup generation、generation、engine
shutdown、process lifecycle全体のwall timeとする。結果を見て指標を選び直さない。

実験protocol、予定試行数、停止規則、A/B順序はGitへcommitする。実行時に割り当てられた
次の値はregular bench/profileではGitだけから実行事実を復元できないためraw metricとして残す。

- `experiment.block_id`
- `experiment.block_order`
- `experiment.replicate_index`
- `experiment.pair_id`: 欠損・置換後もA/B対応を推測しないprotocol割当ID
- `experiment.planned_replicate_group_id`: submit前に割り当てる非NULL arm/group ID
- `experiment.planned_comparison_set_id`: submit前に割り当てる非NULL比較集合ID

qualification childはblock ID/order/replicate indexを複写せず、順にallocation ID、parent/setのordered-child offset、
protocol pair array offsetからviewで導出する。pair ID、planned comparison set、must-match/treatment tupleだけをchildへbindする。

探索比較の既定は各条件5 attemptとし、記述統計だけに使う。同じ時間block内でA/Bを
事前乱数化し、順序を均衡させる。5 attemptや有効run 3件を、equivalence、non-inferiority、
release gateの根拠へ流用しない。意思決定を伴うprotocolは、estimand、margin、想定効果、
分散、alpha、powerからplanned attempt数を事前dimensioningする。invalid/failed runを遅かった
という理由で除外・置換せず、最大attempt数と置換規則も事前にcommitする。

比較次元はGit管理SQLで次へ分類する。

- 一致必須: workload、purpose別のSIF semantic/model/dataset、run-time MPI/container launcher identity、effective args、cache input、resource shape、
  instrumentation mode
- 意図的に変えるtreatment: 対象commit、cluster、node/GPU type等
- nuisance: hostname、GPU UUID、開始日時、queue待ち、個別node identity
- outcome: throughput、phase時間、resource値

protocol commitはtreatmentにするkey集合と一致必須key集合を事前宣言し、runner manifestがexact policyと
`dimension_policy_hash`をsealする。contractはdimensionごとのallowed classだけを制約し、protocolごとに
変わり得るclassを固定しない。viewは次の二つを導出し、fingerprint自体はrawへ保存しない。hash入力は
contract ID/hash、protocol commit/hash、dimension policy hash、dimension name、logical type、canonical
valueのsorted tupleである。required dimensionがmissing/conflictingならfingerprintを作らずNULLにする。

- `observed_replicate_fingerprint`: 実観測した一致必須条件とtreatment値を含むnullable検証結果。
- `observed_comparison_set_fingerprint`: 実観測値から宣言済みtreatment次元だけを除いたnullable検証結果。

attempted countのgroupにはnullableなobserved fingerprintを使わない。pre-run protocol assignmentが必ず
非NULLのplanned group/set IDをsealし、failed/invalid runもそこへ残す。eligible countだけが、実観測
fingerprintとplanned policyの一致を要求する。
`namespace={contract_id,contract_hash,protocol_id,protocol_hash,dimension_policy_hash}`とする。
`planned_replicate_group_id`は`SHA-256(JCS({namespace,purpose,must_match,treatment}))`、
`planned_comparison_set_id`は`SHA-256(JCS({namespace,purpose,must_match}))`で生成する。
dimension arrayはplanned-treatmentと同じcanonical entry/name byte順とし、arm/pair/block/order/replicateをcomparison-set
hashへ入れない。qualification armはprotocol指定treatmentとしてtreatment ID/paired contrastには使う。
`group-1`等の任意labelは表示属性にすぎない。

commit、artifact、cluster等がtreatmentならcomparison set内では差を許すが、replicate groupは
必ず分離する。nuisanceとoutcomeはどちらにも含めない。競技提出用bestと、環境比較用の全点、
median、MAD、ratio、不確実性を分ける。

上記は通常の性能比較policyである。計測擾乱専用protocolだけは、`instrumentation_mode`、
score bundle、qualified source diff、reference/final SIF semanticだけをtreatmentへ
再分類する。これを行わずにon/offを同じcomparison setへ入れたり、通常比較でmode差を許したりしない。
adopt済みscore benchのenvironment比較ではdecision contextのreference、final SIF semantic、runtime launcher semantic、score bundle、
control-plane blobsをmust-matchとし、current `source_track_head_commit`、`sif_manifest_identity`、`sif_artifact_sha256`、
outer sidecar hash、build-definition commitはlineage/nuisanceにする。これにより同じruntime semanticの文書commit/repackageだけでgroupを分割しない。

## 分散producer契約

現行の共有filesystem上Unix domain socketをnode間event transportとして使わない。Unix
socket endpointは各node localに作り、各nodeのeventはそのnodeのJSONLへ書く。

- rank 0だけがworkload manifest、benchmark result、run-scoped phaseのauthoritative
  producerになる。
- 各nodeはresolved args、environment、actual argv、affinity、GPU mapping、telemetryを
  node scopeで出す。
- rawはnode/rank-scoped full factを正本として保持する。viewだけが全expected scopeの一致を
  検証したfieldをcommon表示し、rawにcommon/deltaの第二表現を作らない。
- expected node rank `0..lane.nodes-1`、global worker rank `0..lane.tensor_parallel_size-1`、distinct hostname、GPU UUID集合を
  manifestとschedulerへ照合する。
- 各nonzero node rankのcoordinated teardownと早期終了は、そのnode-local clock上のchild exitと
  `lifecycle.shutdown.start`受信時刻の前後で区別する。

controlを二familyへ分ける。`lifecycle control`はlane指定のmulti-node実行を正しく終了させる共通correctness harnessとして
bench/profileとqualification両armで共通にし、startup readyと`lifecycle.shutdown.start`の通常handshakeを
最終summary logへ記録する。`GPU phase control`はprofile-onlyで、warmup/measured start/endとnode受信時刻を
profile summaryへ記録する。両者のproducer/completenessを混ぜない。

nonzero node rankの`Engine()` constructorはdistributed completionまで内部でblockするため、constructor後を
configuration producerにしない。SGLang内部で最終runtime値を解決した後、各participantがwaitへ入る
直前のhookからnode fragmentをemitする。以下のGPU phase marker/control forwarderはprofile protocolだけで
起動し、benchではproducer graphから外してNULLにする。profileではmarkerをSGLang worker内部へ広げず、benchmark
開始前に接続済みの小さなcontrol forwarderを使う。rank 0 offline hookが`invocation_id`、phase、
start/end、sequenceを各node monitorへbroadcastし、各monitorがnode-local受信時刻を記録して
ACKする。最終summaryをartifactへ残し、異なるnodeのmonotonic値同士を減算しない。このlane
ではnative-worker markerとcontrol-forwarder markerを同じmetricへ混在させない。

profileのgeneration順序は`start broadcast -> expected lane.nodes ACK -> rank0 timer start -> generate ->
timer end -> end broadcast -> expected ACK`へ固定する。start ACK完了からgenerate開始までのleadと、
generate終了からend ACK完了までのtrailをrank 0 clockでartifactへ残すため、GPU observation windowが
generationの保守的supersetであることを判定できる。warmup/measuredのsummaryは通常result writerと同じ
rank 0 monotonic-ns境界を使い、保存済みwall-time scalarをexact `(end-start)/1e9`で検証する。scalarは全variant共通
result、timestamp pairはprofile window containmentのためだけに併存し、別timerを二重取得しない。全purposeでgeneration timer終了後に
`lifecycle.shutdown.start(invocation_id)`を全expected nodeへ一度だけbroadcastし、通常ACK後に
`backend.shutdown()`を呼ぶ。この一往復のstage-wall影響はD-001 bundleに含める。

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

### run、stage、allocation

- `run.started_at`、`run.finished_at`、`run.elapsed_sec`
- `run.exit_code`。`run_outcome`は0=`succeeded`、非0=`failed`、欠損=`NULL`としてviewで一度導出する
- `run.stage.benchmark.launch`。node別actual argvはnode topology artifactだけが所有する
- `perf.stage.benchmark.wall_time_sec`とexit/signal/spawn errorを区別するstage termination object
- `scheduler.request`: runner submit boundaryのlane、PJM script identity、exact `pjsub` argv
- `scheduler.accounting`: 固定した`pjstat --history -S --choose ... --data --delimiter "\t"`の結果
- regular runだけ、submission resultの`scheduler.submit_job_id`とjob-entryの
  `scheduler.job_entry_job_id`を別sourceのscalar indexとして保存し、equalityはviewで導出する。binding booleanは作らない
- benchでもmodel load前に一度だけ、node別topology artifactへlogical index、PCI bus ID、UUID、model、
  memory bytes、driver versionと設定値を保存する。run-scoped `hardware.gpu.nodes` raw factは作らない
- profileだけ、minimal CPU/settingsに含まれないfull NUMA/memory raw artifactを追加する。captureのrun ID/hostnameを
  minimal hardwareの`(node_rank,hostname)`と照合し、meminfo kB→bytesとcpulist canonicalizationを固定する。
  `hardware.cpu_numa.nodes` raw factは作らずconsumerへ直接投影し、benchではNULL
- worker affinity/cpusetは`worker_ready`のglobal-rank-scoped runtime fact

regularのrun lifecycleはPJM entryからcompute回収終了まで、qualificationは各child orchestration開始直前から
そのchildのsource/log回収終了までを個別に取る。allocation-wideのPJM entry/accounting/terminalは親projectionだけが
所有し、child lifecycleへ複写しない。started/finishedはUTC wall clock、elapsedは同じsupervisor境界の
monotonic clockで取得するため、
wall-clock jumpとtimestamp精度の影響を受けず厳密な派生関係ではない間だけ残す。同一scope、clock、
precisionへ統一した時点でelapsedまたはtimestamp pairの一方をview派生へ移す。

### SIF、model、dataset provenance

SIF内manifestにはbuild inputsとして次を固定する。

- buildを定義した`2_Qwen`のrepository ID、object hash algorithm、full commit
- `5_SGLang_Src`のrepository ID、object hash algorithm、resolved full commit
- base OCI digest、resolved package/toolchain manifest
- workflow run/attemptはSIF/sidecarに入れずoperational build rowだけが所有する

SIF完成後に実際の参照先binaryをSHA-256で検証し、そのcontent identityとembedded manifestの
identityをsidecarへ保存する。SIF全体hashをSIF自身へ埋め込む自己参照にはしない。
sidecarはscore bundle `source_paths`ごとのfinal tree stateをpath順のabsentまたはmode/blob OIDとして一度持ち、
SIF semanticへ含める。referenceはempty、qualification ONはsource diffのnew stateとexact一致させるため、
別instrumentation source bytesを同じsemanticとして扱わない。
v1のbundle `source_paths`とin-process hookはすべて`sif_build_definition` repositoryに属し、各hook `source_path`を
bundle集合とsidecar tree entryへ含める。upstream `sglang_source`を直接変更する場合は、曖昧に混ぜず別contractで
そのrepositoryのdiff/treeを追加する。
recipeの`From:`はdigest-pinned referenceにし、buildが消費したOCI manifest/config/layer digestを
sidecarへ記録する。build scriptがSIFとsidecarを書き終えて成功した後だけcacheへ公開し、reuse時に
両者のidentity一致を検査する。任意のcrash境界から二fileを復旧するtransactionは作らない。

model sourceはtagged unionにする。Hub由来なら`hub_revision`としてrepositoryとfull immutable revision、local
prebuiltなら`source/model-manifest.json`自身の`prebuilt_manifest` recordに各shardの
relative path/size/SHA-256を必須にし、完全導出できるmanifest IDはrecordへ複写しない。別のprebuilt manifest artifact/pathは参照しない。regular runまたはqualification parentの
`source/model-manifest.json`は選んだsource record、`config.json`と`model.safetensors.index.json`のSHA-256、indexが
参照するsorted unique shardのrelative path/size、tokenizer/chat-template fileのpath/size/SHA-256を持つ。runごとは
small file hashとshard path/sizeをsource recordへ照合し、400 GBのshard本文を再hashしない。

datasetはworkflowのtracked protocolでfull repository revisionをpinし、`hf download --revision <full-revision>`を
使う。`source/dataset-manifest.json`へrepository/full revision、relative file name、size、SHA-256を保存する。
moving `main`や共有pathだけをidentityにしない。regularはrun、qualificationはparentが一度所有し、provision時にmanifestと実体の一致を検証する。

raw索引は`artifact.model.manifest_sha256`、`artifact.dataset.manifest_sha256`だけを持つ。
repository/revision/memberはmanifest内の完全recordが正本であり、個別raw keyへ再掲しない。

raw tableは比較groupに必要なcommitとmanifest identityだけを索引化する。source manifestはrun bundleの固定member
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
qualificationはallocation parentが所有する二manifestから同じ式で導出する。

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

normal bench/profileとqualification ONのprimary resultは、rank 0がこの検証直後に作るartifactを唯一のsourceとする。
temporary fileをwrite/closeしてrenameしてからprofile固有取得または
shutdownへ進む。`get_server_info()`は一次factを持たずprofileだけで呼ぶ。通常benchはprimary result完成後に
直ちにlifecycle shutdownへ進む。qualificationではscore referenceに含める共通benchmark harnessの
`QualificationResultWriter`が、OFF/ONとも同じresponse validationから`source/qualification-result.json`を作る。
これはD-001 estimandの唯一のsourceで、予定件数を成功件数として扱うnative `result_filename`は使わない。
ONのPrimaryResultはbench採用予定writer自体を検証するためだけに独立に残し、qualification resultとの一致を要求する。
この限定的な二sourceは`duplication_justification`へ検証式とともに登録する。

benchmark validityは少なくとも`error_count = 0`、`returned_count = workload.requests.length`、observed
input totalとplanned input totalの一致、有限かつ正のgeneration wall timeを要求する。現在の
`ignore_eos`固定protocolではobserved output totalとrequested output totalの一致も要求する。

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

viewは`collection_complete`、`benchmark_valid`、`score_comparison_eligible`、`qualification_arm_eligible`、
`run_outcome`だけを独立に導出する。configuration/lifecycle/provenance/topology/telemetryごとのstatusやreasonは
作らず、個々の値はRunsまたはSourcesで直接確認する。profile telemetryはsample/valid duration/coverageと
nullable aggregate値だけを表示し、complete booleanを作らない。

telemetry欠損だけをbenchmark failureへ読み替えない。各laneでnode rank `0..nodes-1`、distinct host数、
GPU/node、unique GPU UUID数、TP、nnodes、MPI process数をlane resolved objectへ照合する。現在の具体laneは
2 nodes、1 MPI process/node、4 GPU/node、TP=8である。configuration eventやworkload manifestが欠けた成功runはthroughput値を残すが、
score comparison eligibleにはしない。

`run_purpose`はplanned/observed双方の必須比較次元で、planned group/set IDのhash入力にも必ず含める。
`score_comparison_eligible`は`run_purpose=bench`かつlane/bundleに対応するD-001 decisionがadoptの場合だけ
trueになり、`qualification_arm_eligible`は`run_purpose=qualification`の一runだけから導出する。
`qualification_pair_eligible`はrun列やraw factへ複写せず、両arm、parent allocation binding、model/dataset/workloadと
instrumentation対象外runtimeの一致をjoinしたpair grainで一度だけ導出する。reference/final sourceとSIFは宣言済みtreatmentであり
common-source一致へ含めない。
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
`applicable_run_variants`（`bench`、`profile`、`qualification:off`、`qualification:on`の集合）だけを持つ。
expected producer graphはrun variantごとに作る。非適用でもcollector failureでもconsumer値は`NULL`だけとし、
field別status、欠損用placeholder、reasonを作らない。required sourceの有無はrun-level
`collection_complete`を一度導出するだけである。

| family | bench | profile | benchで取らない時のRuns値 |
|---|---|---|---|
| source/SIF/model/dataset/workload/cache、exact outer/node argv、job/node env、scheduler、termination、purpose-authoritative result/throughput、rank/host/CPU/GPU/TP topology、common lifecycle control | required | required | 欠損ならbench不適格 |
| container launch/worker env、effective-config event、PrimaryResultWriter | 全v2 benchでrequired。decision pending中はaudit-only、adoptedだけscore | required | 欠損ならapplicable bench/profile不適格 |
| GPU polling/aggregate、phase control marker、profiler、verbose backend trace、full placement dump | not expected | protocol指定時required | `NULL` |

benchで取得しないprofile専用familyはexpected producer graphから外し、raw key、status、placeholderを作らず、
typed/Sheet値だけを`NULL`にする。benchの`collection_complete`分母へ含めない。bench/profileそれぞれでrequiredと
宣言したfamilyのcollector failureはrun-level `collection_complete=false`にするが、原因分類はしない。
profile値を同じprofile baseのbench runへ移植しない。

qualificationは同じpurpose内でもarmごとにgraphが異なるため、expected graphを
subject manifest top-levelの`(run_purpose, qualification_arm)`で解決する。armはplanned-treatment tupleへ複写せず、
protocol指定のtreatment ID/paired contrastにだけ反映する。

- `qualification/OFF`: reference側source/SIFと両armでrequiredなmodel/dataset、parentがarm 1前に一度生成したactual ordered workload
  manifest、job/node env、scheduler、termination、neutral qualification result、外側timer、common lifecycle controlをrequiredにする。
- `qualification/ON`: final側source/SIF、OFFと同じmodel/dataset/workload/common runtime familyに加え、D-001対象の
  exact minimal score bundleだけをrequiredにする。source/SIF pairは意図的treatmentであり同値条件にしない。

両armはinstrumentation対象外のneutral request-manifest harnessから同じmanifest byteを順番通り消費する。
manifestはparent allocation artifactが一度所有し、child rawへ複写せずallocation IDでjoinする。

両armともprofile-only列はNULLだけを返す。OFFでON専用列も同じくNULLだけとする。最終benchはlane ID、
reference source object、decisionのqualification evidence、final SIF semantic、workload、実score bundle SHA-256と
実control-plane file blob集合がadopt済みD-001 decision contextと
一致し、profile-only producer IDを含まない時だけそのbundleを選べる。

Git運用はlong-livedな相互mergeではなく、`bench -> profile`を基本方向にする。profile側はbenchの更新を
merge/rebaseして追従する。profileで見つけた最適化はprofile hookを含まない独立commitとしてbenchへ実装し、
その後benchからprofileへ取り込む。profile branch全体をbenchへmergeしない。必要なら二つのworktreeを使う。
branch名はmoving labelなのでidentityにせず、run manifestは次のcanonical source-track recordをsealする。

```text
{schema_version="source-track/v1", object_hash_algorithm,
 base_role="score_reference"|"profile_base", base_commit, head_commit}
```

bench/qualificationはscore reference、profileはprofile baseを選ぶ。run purpose、qualification arm、repository IDはsubject
manifest top-levelが所有し、nested recordへ複写しない。

`source_track_manifest_hash = SHA-256(canonical source-track record)`をsource-track projectionのidentityにする。hashをrecord自身へ入れて
再帰させない。全v2 benchはscore bundleを持ち、score referenceがbench headのancestorであることを要求する。
decision pending中のbenchもaudit-onlyで同じbundle/source validationを行う。profileではprofile baseがprofile headの
ancestor、qualification OFFではhead=score referenceを要求する。
qualification ONではscore referenceがsource headのancestorであり、そのbaseからheadへ
qualification validation時にcanonical tree diff v2を計算する。このvariantだけがdecoded path集合をscore bundleの
`source_paths`へexact一致させ、protocolの`control_plane_files`をbase/headとparentの実行checkoutで同じblob OIDへ一致させる。
SIF sidecarの
`source_commits[commit_role="sif_build_definition"]` OIDをqualification ON source headと一致させる。profile-only overlay、
profiler flag、workload変更やcontrol-plane実装変更はdiffに含めない。algorithm/version、base/head、canonical entriesは
`control/source-track-diff.json`へ一度sealする。entriesのJCS SHA-256は採否decisionのcross-run join anchorへ一度だけ置き、
source artifactへhash/一致booleanを複写しない。

SIF生産時はclean checkoutのHEAD/treeをsidecar `sif_build_definition` OIDと検査する。adopt decisionを使わない
bench/profile/qualificationでは、そのOIDをrunのsource-track headとも一致させる。採用済みSIFを再利用する
score benchは実行runnerの現在HEADとsidecarの一致を要求せず、decisionのfinal SIF semantic SHA-256、
score bundle SHA-256、qualification時と同じcontrol-plane file blobsをexact照合する。qualified source diff SHA-256は
qualification evidenceとしてjoinするが、benchのcurrent HEADから再計算しない。submodule pointer、file mode、symlink、LFS object
identityもqualification diff検証対象にする。全variantのhead/base commitはremote reachableでなければ投入しない。
base/headから決定的に計算できるdiff hashはrawへ重複保存しない。

`instrumentation_bundle_hash`は、score/profile pathで有効なhook、event、handshake protocol、sample interval、sink、
buffer、timeout、wrapper/node-agent executable hash、observation contract version、env capture scope、source path allowlistを
列挙したcanonical manifestのcontent hashである。`lightweight`等の
labelだけでは同一視しない。`source_track_head_commit`、`score_reference_commit`、`profile_base_commit`等のconsumer列はsource-track recordから
導出し、同じcommit tupleを別raw keyへ複写しない。profile runはprofile baseが同じでもbundle、SIF、argv、
environmentを独立に保存し、「profileで速かった」をbenchの性能値へ読み替えない。
score bundle hashはpurpose/arm/run ID/absolute runtime rootを含まないsemantic identityで、sinkとargvはrelative
templateにする。qualification ONと採用後benchは同じbundle bytesを参照し、actual argv/sinkはrun sourceで観測する。

## BigQuery viewとスプレッドシート

logical schemaを増やさず、raw tableはdaily ingestion-time partition、cluster順`run,key`で作り、
`require_partition_filter=false`とする。`_PARTITIONTIME`はUTC delivery partition bucketであってrun時刻にも
exact publish時刻にもならず、replayで変わり、streaming write-optimized storageでは一時的にNULLにも
なり得るためcomparison/freshness factへ使わない。project中はpartition expirationを設定しない。
recent publish確認だけ`(_PARTITIONDATE >= ... OR _PARTITIONTIME IS NULL)`を使い、run lookupは
cluster filterを使う。semanticなdedupe/conflict/comparisonは正しさのため全partitionを対象にする。
artifact projectionの`*_raw_v2`もdaily ingestion-time partitionとし、subject/runおよび各row grainの先頭fieldで
clusterする。値のidentityやcomparisonにpartition時刻を使わない。

Git管理SQLは次の非二乗relationを作る。

| object | row grain | 役割 |
|---|---|---|
| `raw_distinct` | unique `(run,key,value)` | replayで生じた完全一致rowだけをdeduplicate |
| `run_key_singletons` | `(run,key)` | distinct valueが一つだけのtyped pivot入力。異値keyを通さない |
| `run_key_conflicts` | `(run,key,value)` | 異値が二つ以上あるkeyの各distinct valueを一行ずつ保持し、値を選ばない |
| `qwen_run_fact_pivot_v2` | one row per run | conflict-free Fact refだけをtyped固定列へpivot |
| `qwen_run_facts_v2` | one row per run | fact pivotへconflict-free projection refとGit-managed dimension refをconsumer bindingどおりjoinした最終Runs relation |
| `qwen_run_commits_v2` | `(run,repository,commit_role,object_hash_algorithm,commit)` | source-trackとSIF provenance projectionからalgorithm付きfull object IDを導出し、Git subject/body/URLをjoin |
| `qwen_run_sources_v2` | `(run,consumer_column)` | 実在する値だけについてauthoritative raw key/artifact member/projection IDとtransform IDを示すlineage。欠損placeholderは作らない |
| `qwen_source_tracks_raw_v2` | `(run)` | compact source track、base/reference/head commit、record/source hashを一度だけ保持 |
| `qwen_sif_provenance_raw_v2` | `(source_sha256)` | SIF artifact identity、role付きsource commits、OCI/recipe/patch/lock/build command/toolchain、embedded manifest、runtime file identityを一度だけ保持 |
| `qwen_runtime_launchers_raw_v2` | `(subject_kind,subject_id)` | 実行時mpirun/Singularityのresolved path/hashとexact version outputをregularまたはqualification parentが一度だけ保持 |
| `qwen_workload_manifests_raw_v2` | `(subject_kind,subject_id,workload_kind)` | manifest identity、dataset identity、nullable warmup mode、source member/hashを一度だけ保持 |
| `qwen_workload_requests_raw_v2` | `(subject_kind,subject_id,workload_kind,ordinal)` | ordered requestのtagged source、exact prompt payload、requested output count、sampling paramsを保持。identityはparent manifest join |
| `qwen_minimal_hardware_raw_v2` | `(subject_kind,subject_id,node_rank)` | regular runまたはqualification allocationのfull minimal hardwareとsource member/hashを一度だけ保持 |
| `qwen_node_launch_raw_v2` | `(run,node_rank)` | node wrapper/Singularity/SGLang argv、rank placement、host affinityを一度だけ保持 |
| `qwen_node_configs_v2` | `(run,node_rank,config_kind,field_name)` | node-scoped BenchArgs/ServerArgs Factのtyped field arrayをlosslessに展開する唯一のlong consumer。別raw copyを作らない |
| `qwen_topology_raw_v2` | `(run,node_rank,scope,gpu_index)` | node topology artifactからhost/container別GPU identity/settingsをqueryableに転記。source member/hashを保持 |
| `qwen_participant_terminations_v2` | `(run,node_rank)` | nonzero node rankのunique lifecycleと、waitedの場合だけ同node shutdownを結んだ観測済み終了分類。Runsへのarray/aggregateやmissing placeholderを作らない |
| `qwen_gpu_samples_raw_v2` | `(run,node_rank,sequence,gpu_uuid)` | profile raw tarのsampleをexact schemaでqueryableに転記。source member/hashを保持 |
| `qwen_profile_hardware_raw_v2` | `(run,node_rank,record_kind,numa_node_id)` | profile hardware artifactのnode summary/NUMA nodeをqueryableに転記。source member/hashを保持 |
| `qwen_lifecycle_summaries_raw_v2` | `(run)` | lifecycle summaryのnode ready/shutdown recordを一つのclosed rowとして保持 |
| `qwen_phase_summaries_raw_v2` | `(run,invocation_id)` | profile phase summaryのrank 0 boundaryとordered node-local windowsを一つのclosed rowとして保持 |
| `qwen_qualification_allocations_raw_v2` | `(allocation_id)` | parent artifactのscheduler、decoded env、二job IDとfull bundle member arrayを保持。hardware/workloadはsubject-keyed relationへ分離 |
| `qwen_qualification_allocations_v2` | `(allocation_id)` | parent manifest/artifactからnamespace、ordered child IDs、binding、scheduler request/accounting/terminal、observed control-plane hash、job-entry envを一度だけ投影。同じ`(namespace,pair_id)`が複数なら全候補をconflictへ出しskeleton joinへ0行を返す |
| `qwen_instrumentation_decisions_v2` | `(decision_id,decision_sha256)` | Git管理のclosed decision record。adopt済みscore benchだけmanifest ID/hashでexact join |
| `qwen_instrumentation_decision_context_v2` | `(decision_id,decision_sha256)` | decision namespaceから元qualification protocolへexact joinしたplanned target+observed evidence view。保存copyなし |
| `qwen_comparison_dimensions_v2` | `(run,dimension)` | `must_match/treatment/nuisance/outcome`とcanonical value |
| `qwen_qualification_estimand_points_v2` | `(contract_hash,protocol_hash,dimension_policy_hash,planned_comparison_set_id,pair_id,arm,estimand_id)` | protocol schedule基底のplanned skeleton。allocation/run/valueはnullable payloadとしてLEFT JOIN |
| `qwen_replicate_points_v2` | `(run,estimand_id)` | 実bench runを直接基底にし、protocol/group/treatment、attempt/order/value、eligibleを全点表示 |
| `qwen_replicate_stats_v2` | `(contract_hash,protocol_hash,dimension_policy_hash,planned_replicate_group_id,estimand_id)` | bench attempted全件とobserved一致済みeligible n、median、MAD、CV。bestとは別 |
| `qwen_paired_points_v2` | `(analysis_purpose,contract_hash,protocol_hash,dimension_policy_hash,planned_comparison_set_id,pair_id,estimand_id,baseline_treatment_id,candidate_treatment_id)` | bench通常比較またはqualificationのprotocol指定pair。profileは行を出さない |
| `qwen_paired_comparisons_v2` | `(analysis_purpose,contract_hash,protocol_hash,dimension_policy_hash,planned_comparison_set_id,estimand_id,baseline_treatment_id,candidate_treatment_id,estimator_id)` | ratio方向、pair数、interval、decision。未決定基準はNULL |
| `qwen_env_facts_v2` | `(run,scope_kind,scope_instance,escaped_name)` | 各scopeのraw full snapshotだけをdecodeしたbyte-exact name/value |
| `qwen_run_diff_v2(left_run,right_run)` | table function、一declared dimension一行 | 二runのvalue/equality/class。片側欠損はNULL |
| `qwen_env_diff_v2(left_run,right_run)` | table function、一scope/name一行 | full outer joinしたnullable左右値と、両値がある時だけのequality |
| `qwen_archive_attempts_v2` | `(github_run_id,github_run_attempt)` | workflowが書くupload/SCP/publish outcomeと時刻、GitHub Artifact保持有無、home空き容量 |
| `qwen_sheet_status_v2` | one row | SQL/contract identity、query時刻、latest publish/archive、未移管Artifact数、home空き容量 |

environmentの`scope_kind`はcontract enum `job_entry|host_exec_node|sglang_launch_node|worker_global_rank`、
`scope_instance`はnullable node/global rank等をfield名付きcanonical JSONにしたinjective IDで、Sheetにはtyped
scope列も出す。display hostnameだけをjoin keyにしない。`escaped_name`はbyte nameの単射表現であり、同名でも
scopeが違えば別rowである。

archive workflowは一attempt一rowのoperational recordだけを残す。hash chain、delivery FSM、receipt storeは
作らない。Data Statusはこのtableの`MAX`時刻と最新attempt、GitHub artifact inventory、home `df` snapshotを
表示する。scientific factへdelivery状態を複写しない。

artifact sourceが科学的正本で、BigQueryの`*_raw_v2`はSheet/agentから値をqueryできるようpublisherが
contract-declared row schemaへ一度projectionする保存境界である。direct projectionはsource member/hash、child projectionは
明示したparent projection/joinを持ち、別のmetric aliasやaggregateを作らない。
それ以外の上表は独立した保存メトリクス集合ではなくlogical query interfaceである。
各projectionのphysical append tableはreplay rowを許し、一source instanceを一load jobでappendする。logical
`*_raw_v2` relationは全列完全一致を一件へdedupeする。同じsource-instance grainに異なるsource hashがあれば
そのinstance全rowを、同じdeclared row grainに異なるpayloadがあればその全variantを`*_conflicts_v2`へ残し、consumer入力から外す。GPU sample数や
energyは必ずlogical relationから計算し、再publishで倍増させない。
相互に列が重なるprojectionも別のraw producerやscheduled copyを作らない。lineageはcontractに定義した
consumer columnのordered source refsへ直接展開し、欠損用rowやreason rowを生成しない。

`qwen-v2.toml`にはraw contractと同じcommitで`[[consumer.run_column]]`を置く。各固定列は
`name,type,unit,source_refs,scope_cardinality,reducer_id,conflict_rule,transform_id,applicable_run_variants,
dimension_name,allowed_classes`を宣言する。`source_refs`はordered tagged listで、各entryに
`kind=fact|projection|dimension`、ID、必要なら`json_path`とvariant別`bindings`を持つ。projection bindingはrun-keyedの
`run_id -> run_id`、regular subject-keyedの`run_id -> subject_id`、qualification childの
`block_id -> allocation subject_id|allocation_id`、build projectionの
`artifact.sif_manifest_sha256 -> source_sha256`を明示する。sourceが欠損または当該variantで
非適用ならRuns列を`NULL`にする。複数sourceの
派生列はordered source listを持つ。
`dimension`はGit管理のclosed `contract|protocol|instrumentation_decision_context` relation、projectionは
conflict-free logical relationを入力にし、archiveをSQLから直接読まない。
dimension bindingはcontractならfact pivotの`(contract_id,contract_hash)`からdimensionの
`(contract_id,contract_sha256)`へ、protocolなら
`(contract_id,contract_hash,protocol_id,protocol_hash,dimension_policy_hash)`から
`(contract_id,contract_sha256,protocol_id,protocol_sha256,dimension_policy_hash)`へexact joinする。
branch、latest、IDだけへのfallbackは作らない。
adopt済みinstrumentation decisionはrun manifestの`(instrumentation_decision_id,instrumentation_decision_sha256)`から
Git-managed decision context viewの`(decision_id,decision_sha256)`へ一段exact joinする。contextはdecision namespaceから元qualification
protocolへexact join済みで、artifact内copyをSQLから読まない。
生成SQL、Sources lineage、Sheet headerだけがこれを読み、「主要field」の手動選択や手書きpivotを許さない。
contract testは各consumer列についてexactly one definition、variant applicability、input sourceへの到達を検査する。

同じcontract/protocolに`[[consumer.estimand]]`を置き、`estimand_id`、source/transform、logical type、unit、
改善方向、eligibility rule ID、許すstatistic/estimator/interval、`applicable_run_variants`を固定する。
comparison dimensionとestimandはsource列のapplicabilityを継承する。当該variantで
NULLとなる列をdimension/estimand、observed fingerprint、config diff expected gridへ入れない。
point viewはprofile runを展開せず、bench/qualification runとそのpurposeにapplicableなestimandの直積だけを
展開する。source欠損は行を落とさずNULL valueにする。protocolは一次・二次estimandと
decision対象を事前指定し、view実装者が結果を見て選ばない。

全run pairのdiffを通常viewとしてmaterializeしない。agent/SQL利用者とConnected Sheetsは二つのcell
parameterを持つcustom query `SELECT * FROM dataset.qwen_run_diff_v2(@LEFT_RUN,@RIGHT_RUN)`、同env diffを
直接呼ぶ。NULL、同一、存在しないrunの指定はquery errorまたはzero rowとし、selection/status sentinelを
生成しない。valid config diffは左右contract/protocolのdeclared dimension gridのunionから始め、左右のdefinition
hash、logical type/unit/transform、policy classとnullableな左右値を表示する。互換な両値が実在する時だけ
equalityを計算し、それ以外はNULLにする。

env diffは左右の実在snapshotをfull outer joinし、両側の値をnullableで表示する。scope全体のsnapshotが
取れなかった場合も原因statusを増やさず、当該側をNULLにする。UI filter/
pivotだけで二run制約を実現しない。Sheetのfilterなしでfull environment extractを作らない。値はユーザー
指示どおり内部利用者へmask/redactせず表示するが、Sheetとdatasetの共有先を
同じ内部ACLへ限定する。

`qwen_run_facts_v2`は少なくとも次の固定列を持つ。ここで挙げる名前をcontractのconsumer projectionへ
個別登録し、概念名だけの列を作らない。aggregate統計をrun rowへ複写せず、replicate viewを別tabにする。

| group | columns |
|---|---|
| identity/protocol | `run_id`、`contract_id`、`contract_hash`、`protocol_id`、`protocol_hash`、`dimension_policy_hash`、`planned_replicate_group_id`、`planned_comparison_set_id`、`planned_attempts`、`max_attempts`、`stopping_rule_id`、`block_id`、`block_order`、`pair_id`、`replicate_index`、`lane_id`、`run_purpose`、`qualification_arm`、`instrumentation_mode`、`instrumentation_bundle_hash`、`instrumentation_decision_id`、`instrumentation_decision_sha256`、`started_at`、`finished_at`。qualificationでは`block_id=allocation ID`を唯一のparent join keyにする |
| source/artifact | `source_track_head_commit`、`score_reference_commit`、`profile_base_commit`、`source_track_manifest_hash`、SIF projection由来の`sglang_commit`、`sif_manifest_identity`、`sif_artifact_sha256`、run時`runtime_launcher_identity`、`model_manifest_identity`、`dataset_manifest_identity`、`measured_workload_manifest_identity`、`warmup_workload_manifest_identity`、`cache_manifest_identity`、qualification childだけallocation projectionからjoinする`qualification_parent_archive_hash`。bundle hashはartifact dimension、URIはworkflow operational joinが所有しraw metricへ戻さない |
| quality | `collection_complete`、`benchmark_valid`、`score_comparison_eligible`、`qualification_arm_eligible`。欠損原因やsubsystem別statusは列にしない |
| termination | `run_outcome`、`run_exit_code`、`benchmark_termination_kind`、`benchmark_exit_code`、`benchmark_signal`、`engine_termination_kind`、`engine_exit_code`、`engine_signal`、`scheduler_terminal_state`、`scheduler_exit_code`。participantはlong relationだけで表示 |
| workload validity | `measured_selected_count`、`planned_input_tokens`、`requested_output_tokens`、`returned_count`、`error_count`、`missing_response_count`、`observed_input_tokens`、`observed_output_tokens`、`warmup_mode` |
| topology/hardware | bench coreの独立lane値`lane_node_count`、`lane_gpu_per_node`、実観測の`observed_node_count`、`distinct_host_count`、`observed_mpi_process_count`、`observed_global_rank_count`、`observed_gpu_record_count`、`observed_gpu_per_node_min`、`observed_gpu_per_node_max`、`unique_gpu_uuid_count`、`effective_tp`、`effective_nnodes`、`placement_valid`、`affinity_valid`、`gpu_model_consensus`、`gpu_memory_bytes_consensus`、`gpu_driver_consensus`、`gpu_power_limit_w_consensus`、`gpu_max_sm_clock_mhz_consensus`、`gpu_compute_mode_consensus`、`gpu_mig_mode_consensus`、`cpu_model_consensus`、`kernel_release_consensus`、`cpu_numa_node_count_consensus`、`cpu_memory_bytes_consensus`、`cpu_governor_consensus`、`cpu_boost_enabled_consensus`と、profile-onlyの`numa_cpu_layout_identity`、`postrun_memory_available_bytes_min`。`expected_mpi_process_count`、`expected_global_rank_count`、`expected_gpu_total`はv1 lane値から完全導出できるため列にしない。profile-only列はbenchでNULL |
| effective args | `bench_args_identity`、`server_args_identity`と、contractがcurrent dataclassの比較関連scalar leafをexact nameで列挙した列、`runtime_backend_consensus`、`runtime_transport_consensus`。`主要field`という自由選択は置かない |
| outcome | `request_throughput`、`input_token_throughput`、`output_token_throughput`、`total_token_throughput`、`run_elapsed_sec`、`stage_benchmark_wall_sec`、`init_wall_sec`、`prepare_wall_sec`、`warmup_wall_sec`、`measured_generation_wall_sec`、`shutdown_wall_sec` |
| GPU projection | `generation_aggregate_present_gpu_count`、`generation_utilization_valid_gpu_count`、`generation_power_valid_gpu_count`、`generation_memory_valid_gpu_count`、`generation_temperature_valid_gpu_count`、`generation_clock_valid_gpu_count`、`generation_energy_included_gpu_count`、`generation_utilization_coverage_min`、`generation_power_coverage_min`、`generation_valid_sample_count_min`、`generation_utilization_valid_duration_sec_min`、`generation_power_valid_duration_sec_min`、`generation_fleet_utilization_mean`、`generation_gpu_utilization_mean_min`、`generation_gpu_utilization_mean_median`、`generation_gpu_utilization_mean_max`、`generation_gpu_utilization_p95_min`、`generation_gpu_utilization_p95_median`、`generation_gpu_utilization_p95_max`、`observed_generation_energy_j_sum`、`generation_memory_bytes_max`、`generation_temperature_c_max`、`generation_sm_clock_mhz_min`。入力不足の値はNULL |

`cache_manifest_identity`はscalar hashへ再hashせず、expected node rank順のclosed
`[{node_rank:uint,manifest_sha256:sha256},...]`とする。`cache_nodes/v1` consumer reducerがnode-scoped Factを
exact expected node集合で並べ、missing/conflict時はNULLにする。第二raw aggregateは作らない。

current SGLang commitからBenchArgs/ServerArgs dataclass field inventoryを生成し、contractへcommitしたinventoryと
byte比較する。追加・削除・rename・type/default変更が未分類ならreleaseを止める。全scalar leafを
`comparison_dimensions`へ出し、各fieldをmust-match/treatment/nuisanceへ分類する。Runsのwide列はその中から
contractがexact nameで列挙したものだけであり、実装者判断で増減しない。

`collection_complete`はrequired sourceの存在、`benchmark_valid`はerror count、selected/returned count、
planned/observed token、正有限generation timeの整合である。原因列は持たない。
env factsはmapの各scope full snapshotだけから作り、common/delta/removedやscope statusを追加しない。

GPU projectionはapplicabilityをexpected locator/grid生成より先に評価する。bench/qualificationでは
全GPU projectionをNULLにし、gridやplaceholderを作らない。
profileだけが`phase=measured_generation`とlane/manifestのexpected `(node_rank,local_gpu_slot)` Cartesian
gridを母集団にしてsample由来aggregateへleft joinする。observed topologyは各slotへexactly one unique GPU UUIDを
対応付ける。missing/duplicate mappingのslot値はNULLにする。
expected/observed GPU countはtopology列だけが所有し、GPU groupへ複写しない。aggregate present countと
各metricのvalid GPU countを分ける。coverage不足を別gateにせず、得られたintervalからの値とincluded GPU countを
表示して人間/agentが判断する。energyはmissing intervalへ外挿しない。
fleet utilization meanは`sum(utilization_integral) / sum(valid_duration)`で、per-GPU meanの単純平均にしない。
lane contractはnominal intervalだけを固定し、coverage/valid durationは判断材料として表示する。

GPU view projectionはconflict-free/deduped `qwen_gpu_samples_raw_v2`と`qwen_phase_summaries_raw_v2`を入力に
query時に導出し、raw aggregate keyを追加しない。Fact ref列は`run_key_singletons`、projection ref列は各logical relation、
dimension ref列はGit-managed relationをconsumer bindingでjoinする。contract typeから`SAFE_CAST`/JSON validationし、missing、conflict、cast-invalid、
非適用はいずれもwide列をNULLにし、原因列は増やさない。複数値を`MAX`、`ANY_VALUE`等で隠さない。
publish後のprojection conflictは該当consumerをNULLにするだけで、finalizerが確定済みのraw `collection_complete`を上書きしない。
commit metadataはGitから`qwen_run_commits_v2`へrole付きでjoinし、commit messageをrun metricやRunsの
曖昧な単数列へ複製しない。

qualification parentはprogram runではないので`run/key/value`へ擬似runとして入れない。publisherが
`qwen_qualification_allocations_raw_v2`へ投入し、その上の`qwen_qualification_allocations_v2`がallocation ID
一意性とchild referenceを検査して投影する。standalone parent summary JSONは作らない。qualification childのRuns/
paired viewはqualification childの`block_id`でjoinし、同じscheduler request/accounting/terminalを二childのraw keyへ
複写しない。allocation raw rowはdecoded job-entry env entries、submission/job-entryの二job IDと各source member/hashを
実値で持つ。minimal hardwareは同じsubject kind/allocation IDの`qwen_minimal_hardware_raw_v2`、ordered workload payloadは
workload manifest/request relationだけが所有し、allocation rowへnested copyを作らない。Environment/Hardware/Workload viewは
これらをjoinし、child rawへ複写しない。common GPU bindingは二childのtopology equalityから導出する。regular runの
scheduler/env factは自身のraw、minimal hardware/workloadはrun-subject projectionへ置く。

`qwen_replicate_points_v2`はbench pointを実run基底で表し、failed/invalidも削除せずNULL valueでapplicable estimandを残す。
`qwen_qualification_estimand_points_v2`はqualification pointを
基底にせず、qualification protocolのexact pair schedule × two arms × applicable estimandsをplanned skeletonとし、
allocation projectionをnamespace/pair IDで、そのordered childをarmで、child Fact/estimand valueをrun IDで順にLEFT JOINする。
未投入pairはallocation/childがNULL、未起動childはrun/valueがNULLのまま行を保持する。replicate pointsはprotocol、planned/max
attempt、stopping rule、planned group/set ID、pair/block ID、nullable observed fingerprintを持つ。statsは
planned groupでattempted nを数え、事前protocolのeligible ruleとobserved fingerprint一致を満たす点だけを
eligible nへ数える。treatment IDは宣言した全treatment dimensionのname/type/canonical value sorted tupleの
hashで、canonical JSONを表示属性にする。logical allocation viewは同じ`(namespace,pair_id)`を0-or-1 rowに制限し、複数allocationは
全候補をconflictへ出してskeletonへjoinしない。paired pointsはbenchではreplicate points、qualificationでは上記planned skeletonから
`analysis_purpose=score|qualification`を持ち、protocolが割り当てたpair IDだけを対応付け、欠損・置換後に
block/orderから推測しない。scoreは`score_comparison_eligible`、qualificationは
両armの`qualification_arm_eligible`、parent binding、model/dataset/workloadとinstrumentation対象外runtimeのequalityからpair grainで導出した
`qualification_pair_eligible`だけを使い、purposeを跨いでpairにしない。aggregateはbaseline/candidateを順序付きで保持しratioを
`candidate / baseline`と定義し、planned/attempted/eligible pair count、median、MAD、protocolが明示した場合だけの
interval、decisionを併記する。これらはpoint rowとversioned protocolからquery時に導出し、rawへaggregateの
第二正本を作らない。estimator/intervalはprotocolが事前指定した方法だけを使い、
`instrumentation_overhead_decision`は現在未決定のcriterionを確定するまでNULLにする。暫定式を埋めない。
breakingなcolumn/type/meaning変更は
既存v2 viewを書き換えずv3 objectを作る。

Connected Sheetsは次のtabだけをversioned consumer objectへ接続する。

1. `Runs`: one run/rowの固定列。JSON展開なしでquality、config、一次結果、artifactへ到達する。
2. `Replicates`: point viewとstats/comparison viewを別tableとして全点とrobust statisticsを示す。競技bestを同じ列で置換しない。
3. `Sources`: 選択runで実在する各表示列について、raw key、source、artifact member、transformを示す。
4. `Config Diff`: 左右run cell parameterからtable functionを直接呼ぶ。
5. `Environment`: 左右run cell parameterからscope-aware diffを直接呼び、full byte-exact envをmask/redactせず比較する。
6. `Node Config`: 選択runの`qwen_node_configs_v2`をnode/config kind/field grainでlosslessに表示する。
7. `Participant Termination`: 選択runの`qwen_participant_terminations_v2`をnonzero node rank grainで表示する。1-node runと未観測nodeのplaceholder行は作らない。
8. `Data Status`: contract/SQL identity、各query時刻、latest publish/archive、未移管Artifact数、home空き容量を表示する。

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
初版ではsample sizeやCIを導入せず、固定5 pairの簡素screenとしてのみ使う。

### 推奨する簡素な判定（ユーザー回答待ち）

初版はlaneごとに固定5 pairだけを事前commitし、同一allocation内でOFF/ONを隣接実行する。順序は
OFF→ONとON→OFFを交互にする。pilot、sample-size再計算、TOST、sequential stopping、replacement slot、別response
canaryは作らない。稀な失敗pairは欠測として残し、自動再投入しない。3 pair未満しか有効でなければ単に未決定とする。

OFF/ONは次だけを変える。

- `OFF`: reference SIF/code、両arm共通のneutral qualification resultと外側timer。
- `ON`: release予定SIF/code、benchで有効にするeffective-config event、PrimaryResultWriter、source file。

parentはarm 1前にactual ordered request manifestを一度生成する。OFF/ON双方のneutral harnessはその同じfile byteを
順番通り消費し、manifest自体はallocation artifactが所有する。これによりOFFでもplanned seedではなく実workload
identityを検証でき、childへ同じmanifestを複写しない。

各pairは同じallocation、node/GPU identity、TP/nnodes、model/dataset、ordered workload、cache input manifestと、
tracked MPI/container/resource launch policy、warmup policyを使い、engine/processはarmごとに再起動する。
ONにだけ必要なbundle宣言済みwrapper、event sink、patchはtreatmentとして許す。OFFで観測できないactual worker
affinity/runtime backendの一致は判定条件にせず、ON専用列はOFFでNULLにする。SIF copy、cache refresh、profile polling、
archiveはarm間に挟まない。片armが失敗しても保存できたlogを残し、そのpairの値をNULLにするだけで別runと再pairしない。

判定は有効pairについて次を人間とagentが同じSheetで直接確認する。

- neutral qualification resultから導出したtotal-token throughputの `throughput_ON / throughput_OFF` median
- stage benchmark wallの `wall_ON / wall_OFF` median
- neutral qualification resultとON primary resultのreturned/error/token totals一致
- model/dataset/workload/cache/topology identity一致
- ONで必要なsource fileが正常pathでは揃うこと

判定対象はqualificationのexact contract/protocol/policy namespaceが所有するlane、reference source、workload、
reference/score bundle path/SHA-256、control-plane file blob集合、exact 5-pair scheduleと、build後に初めて決まる
qualified source diff SHA-256、reference/final SIF semantic pair、eligible parentで一値のruntime launcher semanticである。
採否recordは`decision_id`、qualification namespace、qualified source diff SHA-256、observed SIF semantic pair、
observed runtime launcher semantic、`adopt|reject|inconclusive`をGit管理する。final head commitは
無関係な変更を含み得るためdecision identityにしない。instrumentation、bundle、実行control-plane file、runtime launcherのどれかが
変わった時だけ再判定し、driverや無関係なcommitが変わるたびに巨大なtupleを再qualificationしない。

推奨初期値は「throughput低下1%以内かつstage wall増加1%以内、result totals一致」であるが、閾値はD-001の
回答で確定する。これはformalな非擾乱証明ではなく、実験projectで明らかな計測負荷をbenchへ持ち込まないための
screenである。5 pairの個々の値とmedianを表示し、CI、power、closed reason enumは作らない。

D-001未回答中はinstrumented benchのoverhead decisionをNULLにする。profile runはscore比較へ混ぜない。

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

D-003は2026-08-09にユーザー回答「基本的にはマルチノード対応型の収集。Qwen推論も同じ」で確定した。
質問はQwen collectorを現在の2-node条件へ固定するか、node/worker/GPU cardinalityをlaneから展開するか、
推奨・採用回答とも後者である。affected fields/viewsはlane resolved object、scheduler request、producer node/global-rank
axes、topology/host/GPU consensus、lifecycle/phase ACK集合、participant termination、collection/score eligibilityである。
reopen triggerはSGLangまたはschedulerの配置意味が変わり、v1の一node一MPI wrapper/full-GPU TP制約で表せなくなった時だけとする。
残る質問順はQwen profile contract D-004、確定したscore laneのinstrumentation overhead D-001とする。

## 決定状態

| 状態 | 決定 |
|---|---|
| 確定 | raw relationは`run/key/value`の三列、compound valueはcanonical JSON |
| 確定 | 定義した各process scopeの環境変数は機密値を含めfilterせずbyte-exactに全件保存 |
| 確定 | native object/structured sourceを優先し、log regexはfixture付きの最終手段 |
| 確定 | compute側はlocal raw fileだけ、finalize/publishはrunner、Artifact upload/home移管はworkflowが担当 |
| 確定 | observed totalsから派生するthroughputやnode full factsのcommon copyをrawへ重複保存しない |
| 確定 | bench/profileを分離し、取得できなかった値とprofile専用familyは値だけNULL、原因/status/欠損用recordなしとする。run-level statusは`collection_complete`だけ |
| 確定 | 実験用途として多段retry、fallback source、field別reason registry、formalな大規模qualification campaignは作らない |
| 確定 | home artifactは自動expireせず、容量が実際に問題になった時だけ手動整理する。専用quarantineは作らない |
| 確定 | D-003: Qwen collectorはlane-driven multi-node対応とし、expected node/worker/GPU集合をlaneから展開する。現在の実機laneは2 node、4 GPU/node、TP=8 |
| 未決定 | D-004: 初版profileで必須にするprofiler、activity、step範囲、required raw file集合 |
| 未決定 | D-001: 固定5 pairの簡素screenで許すinstrumentation overhead閾値。推奨はthroughput/stage wallとも1%以内 |

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
collect_once(manifest_path, runtime_dir, producer_id, probe) -> Path
submit_subject(subject_manifest_path, runtime_dir, lane_adapter, pjsub_argv) -> SubmissionOutcome
run_stage(stage_name, argv, *, cwd, recorder) -> Termination
finalize_subject(subject_manifest_path, runtime_dir) -> tuple[FinalizeOutcome, ...]
publish_subject(subject_manifest_path, artifact_set_path, destination) -> PublishOutcome
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
`stage_runner.py`、`finalize_subject`は`artifact_store.py`、`publish_subject`は`publish.py`が
所有する。`cli.py`はargument parseとhandler dispatchだけを行い、manifest生成、scheduler subprocess、
probe lifecycleを実装しない。
`publish_subject`はregularなら一runのmetrics/projection、qualificationならset manifestを入口にparent projectionと
ordered child slotを保持したまま、`artifact_manifest`がnon-nullなchildの`metrics.jsonl`/applicable projectionだけを一callで送る。
null childにfileを要求せず、childだけを別callしてparent joinを欠かさない。

`collect_once`は完成fragment `Path`を返し、CLIはそのpathをstdoutへ出す。成否はprocess exitだけで表す。`SubmissionOutcome`は
`submitted|failed`とnullable job IDを持つ。曖昧submitを自動探索・復旧するstate machineは
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
| `scripts/run-qwen35-node.sh` | node rank、manifest、node-local endpointを`node_agent.py`へ渡すだけの薄いadapterにする。childをBash `wait`で監督しない |
| `scripts/run-qwen35.pjm` | manifest/runtime pathを明示してcompute collectorとstageだけを呼ぶ |
| `.github/workflows/genkai.yml` | pre-submit manifest、terminal後finalize/archive/publishを所有 |

実装fileは計画済みの`domain.py`、`recording.py`、`submission.py`、`stage_runner.py`、`merge.py`、
`artifact_store.py`、`publish.py`とdomain別probe/parserに限定する。`manager`、`service`、汎用event
bus、DI frameworkは追加しない。`submission.py`はintent、scheduler adapter、曖昧結果照合を隠す一つの
深いmoduleであり、collectorやpublisherをimportしない。Qwen固有fileは`node_agent.py`、pureな`sglang_events.py`、
`source_artifacts.py`、`gpu_telemetry.py`だけを許可する。node control protocolを`node_agent.py`へ閉じ、
共通基盤へ漏らさない。`node_agent.py`のchild supervisorだけが`os.waitpid(child_pid, 0)`を呼び、node lifecycleの
raw wait statusと同じnode-local monotonic境界を一度記録する。outer MPIだけを扱う`StageRunner`へこの責務を重ねない。

同じfamilyをsource fileとmetric fragmentの両方へ書かない。source ownerとraw ownerを次へ固定する。

| family | durability source / single writer | Fact owner |
|---|---|---|
| job/host probe、stage termination、node child lifecycle | 対応producerの`Recorder`へ即時append | 各compute/wrapper producer |
| SGLang node config、worker-ready | node agentがnode/global-rank JSONLへappend | terminal `sglang_events` reducer |
| lifecycle handshake | coordinator/node agentの最終`control/lifecycle/summary.json` | Publisherがexact projection、viewがreduce |
| phase handshake（profile） | coordinator/node agentの最終`control/phase/summary.json` | Publisherがinvocation projection、viewがreduce |
| GPU sample（profile） | node agent内samplerがnode telemetry JSONLへbuffered append | Publisherがsample projection、viewがreduce |
| primary result（bench/profile/qualification ON） | rank 0 `PrimaryResultWriter`のsource artifact | `source_artifacts` parser |
| neutral qualification result（OFF/ON） | 共通benchmark harnessのrank 0 writer | qualification parser。D-001 estimandだけを所有 |
| warmup/measured workload | rank 0 manifest writerのsource artifact | Publisherがmanifest/request projection、viewがreduce |

`PrimaryResultWriter`はtargetと同じdirectoryへtemporary fileをwrite/closeし、`os.replace`で完成させる。
write/renameが失敗したらstageをnonzeroとし、primary result由来のraw keyは作らない。
workload manifest writerも同じ小さなhelperを使う。retry競合やdirectory crashを扱う専用protocolは作らない。

機械可読正本は`qwen-experiment-metrics/contracts/qwen-v2.toml`一fileにする。producer entryがowner scopeとapplicable variant、
source entryがpath/schema/requiredness、fact entryがkey/type/unit/source/reducerを各一度所有する。lane/protocolはこのcontractに
定義済みのproducer IDからexpected graphを選ぶ。runtime manifestにはcontract IDとcontent hashを
sealし、raw tableにはroutingに必要なcontract IDだけを索引化する。観測マップの表はこのTOMLから
生成し、手書きproseだけを別に保つ。したがってregistryとMarkdown表の二つを正本にしない。

各`[[fact]]`のcanonical field listは
`id,key_pattern,source_id,schema_id,logical_type,unit,cardinality,capture_id,validator_id,reducer_id,
duplication_justification?`だけとし、producer/owner/applicability/requirednessをsource→producer joinから複写しない。
各`[[artifact_projection]]`は
`id,base_table_id,relation_id,conflict_relation_id,source_artifact_id,source_schema_id,row_schema_id,row_grain,
subject_scopes,publisher_id,transform_id,conflict_rule,source_member_field?,source_sha256_field?,
source_instance_grain?,source_instance_sha256_field?,parent_projection_id?,parent_join_columns?,duplication_justification`だけを持つ。
`subject_scopes`はrun variant付きrun ownerと、variantを持たないqualification-allocation owner/build ownerのtagged arrayである。

### CLIと設定境界

CLIは次の六orchestration use caseと一つのQwen固有internal entrypointだけを持つ。behaviorを環境変数で
切り替えず、manifestとruntimeは
明示引数、比較条件はGit管理protocol、BigQuery宛先はrunner側のGit管理設定から読む。

```text
init-subject   regular runまたはqualification parent+二childのmanifestを作成
submit         exact pjsub argvを一度実行し、stdout/stderrと取得できたjob IDを保存
collect        compute producerを一度実行し、成功時だけ完成data fragmentを残す
stage          target argvをspawnしterminationと時間を記録
finalize       terminal後にsubjectを検証し、expected artifactとmetrics.jsonlを作成
publish        subject artifact setをparseし、metricsとqueryable artifact projectionをBigQueryへ送達

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
  --qwen-qualification-result-path /run/qwen-source/qualification-result.json
```

qualification OFFへscore observation/primary flagを渡さず、qualification ONはneutral estimandとprimary reliability
artifactの両方を書く。normal bench/profileへqualification result pathを渡さない。argv allowlist testでvariant間の境界を固定する。

収集器はlaneのpositive `nodes`、`mpi_processes`、`gpus_per_node`、`tensor_parallel_size`からexpected producer
cardinalityを展開する。v1は`mpi_processes = nodes`かつ
`tensor_parallel_size = nodes * gpus_per_node`の一node一MPI wrapper/full-GPU TPに限定する。現在の実機laneは
2 nodes、2 MPI processes、4 GPU/node、TP=8である。rank 0 MPI wrapperが取得した`hostname`をadvertise address、
SGLang distributed portを`25000`、node-agent control portを`25001`、rendezvous待ちを60秒、ready待ちを120秒に固定する。
rank 0はaddress、固定port、run ID、protocol versionを`control/lifecycle/rendezvous.json`へ一度writeし、
全nonzero node rankはその一fileだけを待って明示endpointへ接続する。port衝突やtimeoutはそのrunを失敗とし、
自動retry、動的port allocator、複数discovery source、自動fallbackは作らない。

coordinatorはexpected node `0..lane.nodes-1`のidentity付きready ACKをmemory上で集めてからcontainer開始を許可し、
最終summaryだけをfileへ書く。bind/connect/ready timeout、stale/wrong-run messageではsolverを開始せず、
runはnonzeroで終了する。別のfailure classificationは保存しない。

primary result pathはrank 0だけwrite可能にbindする。allocation内producerはnode agent停止時にpurpose上
applicableなdomain/control/telemetryをshared runtimeへflush/closeしてからPJM scriptを終了する。
runnerの`finalize`は
node/GPU/workerを再観測せず、terminal accountingと既存sourceのpure parser/reducerだけを実行する。

submitは`pjsub ... -o <runtime>/runs/<run>/logs/pjm-job.log -j`へunique run pathを直接渡し、job ID parserは
current PJM fixtureの`^Job ([0-9]+) submitted\.$`に固定する。terminal captureはfinal
`pjwait <job-id>`のargv/stdout/stderr/exitと時刻を`scheduler-terminal.json`へ一度保存する。
`pjstat-version.txt`にはlane canaryで確認したversion表示commandとraw outputを残す。曖昧時のjob検索やblind
resubmitはしない。

### runtimeとarchive layout

runとsubmission subjectを分け、run側は次の固定layoutにする。任意file探索や「最後に見つかったJSON」を禁止する。

```text
<runtime>/runs/<run>/
  control/run-manifest.json                    # pre-submit immutable
  control/contract.toml                        # regularだけ。qualificationはparentが一度所有
  control/protocol.toml                        # regularだけ。qualificationはparentが一度所有
  control/source-track-diff.json               # qualification ONだけ: diff algorithm/base/head/entries
  control/instrumentation-bundle.json           # 実際に選択したcomponent/launch設定
  control/submission-request.json
  control/submission.stdout
  control/submission.stderr
  control/submission-result.json               # raw subprocess outcome when available
  control/job-entry.json                       # PJM側のrun IDとPJM_JOBID
  control/lifecycle/rendezvous.json
  control/lifecycle/summary.json
  control/phase/summary.json                    # profileだけ
  metrics/fragments/<producer>.tmp              # write中
  metrics/fragments/<producer>.jsonl            # 正常close後のrename先
  domain/sglang/node-<node-rank>.jsonl
  domain/worker/global-rank-<rank>.jsonl
  source/sif-sidecar.json
  source/runtime-launchers.json                  # regularだけ。qualificationはparentが一度所有
  source/model-manifest.json                   # regularだけ。qualificationはparentが一度所有
  source/dataset-manifest.json                 # regularだけ。qualificationはparentが一度所有
  source/env/job-entry.snapshot
  source/env/host-node-<node-rank>.snapshot
  source/env/sglang-node-<node-rank>.snapshot    # score bundleだけ
  source/env/worker-rank-<rank>.snapshot         # score bundleだけ
  source/cache/node-<node-rank>-input-manifest.json
  source/hardware/node-<node-rank>.json
  source/topology/node-<node-rank>.json
  source/topology/node-<node-rank>-host-nvidia-smi.stdout
  source/topology/node-<node-rank>-host-nvidia-smi.stderr
  source/topology/node-<node-rank>-container-nvidia-smi.stdout
  source/topology/node-<node-rank>-container-nvidia-smi.stderr
  source/topology/node-<node-rank>-nvidia-smi-topo.stdout
  source/topology/node-<node-rank>-nvidia-smi-topo.stderr
  source/scheduler/pjstat.stdout
  source/scheduler/pjstat.stderr
  source/scheduler/pjstat-version.txt
  source/primary-result.json
  source/qualification-result.json             # qualification OFF/ONだけ。closed neutral result
  source/workload-warmup.json                  # regularだけ。qualificationはparentが一度所有
  source/workload-measured.json                # regularだけ。qualificationはparentが一度所有
  telemetry/node-<node-rank>/gpu-samples.jsonl   # profileだけ
  profile/raw/...                                # profileだけ。profilerの固定出力先
  profile/raw/hardware/node-<node-rank>/...      # profileだけの追加sysfs/proc raw
  logs/pjm-job.log
  logs/benchmark.stdout
  logs/benchmark.stderr
  control/scheduler-terminal.json
  merge/metrics.jsonl
  archive-staging/run-artifacts.tar.zst
  archive-staging/bundle.manifest.json
  archive-staging/artifact-manifest.json
  archive-staging/profile-raw.tar.zst            # profileだけ
```

regular runだけが上記`control/submission-*`、job-entry、`control/scheduler-terminal.json`を所有する。
qualification childではそれらを作らず、child manifest自身はallocation ID、parent manifest hash、
top-level qualification arm、pair IDを持つ。block/order/replicateは親とprotocolのarray offsetから導出する。job IDやarm ordinalは持たない。
完全導出できるbinding fileやparent-ref fileは作らない。
qualification parentは次の固定layoutを所有する。

```text
<runtime>/allocations/<allocation>/
  control/qualification-allocation-manifest.json
  control/contract.toml
  control/protocol.toml
  control/submission-request.json
  control/submission.stdout
  control/submission.stderr
  control/submission-result.json
  control/job-entry.json                           # allocation ID/job ID/ordered child IDs
  source/env/job-entry.snapshot                    # NUL区切りbyte-exact parent source
  source/runtime-launchers.json                    # mpirun/Singularity実体を一度
  source/hardware/node-<node-rank>.json             # arm 1前に一度
  source/model-manifest.json                         # OFF/ON共通identityを一度
  source/dataset-manifest.json                       # OFF/ON共通identityを一度
  source/workload-measured.json                    # OFF/ONが共通消費するactual ordered request
  source/workload-warmup.json                      # OFF/ONが共通消費するactual warmup request
  source/scheduler/pjstat.stdout
  source/scheduler/pjstat.stderr
  source/scheduler/pjstat-version.txt
  logs/pjm-job.log
  control/scheduler-terminal.json
  archive-staging/allocation-artifacts.tar.zst
  archive-staging/qualification-set-manifest.json
```

qualification parent sourceは上記`control/`、`source/`、`logs/`を`allocation-artifacts.tar.zst`へ一度収録し、
`logs/pjm-job.log`をrequired memberとするがrun metricsは持たない。各childはparent manifest hashだけを参照する。
qualification child tarへcontract/protocol/model/dataset/workload bytesを複写せず、child manifestのhash/allocation IDから
parent copyへjoinする。
`qualification/<allocation>/allocation/`と`qualification/<allocation>/runs/<child>/`へ分ける。

これは全purposeのallowlist unionであり、bench archiveにprofile-only pathがないのは正常である。`finalize` commandは
contract applicability、protocol、bundle selectionからexact expected member setを一回導出し、任意file探索もempty placeholder作成もしない。
expected pathをmanifestへ再掲せず、node/global rankをfilenameから推測せずrecord内部との一致も検査する。
node-local ephemeral endpointはshared layoutに含めず、node wrapperが作ったdirectory内の
`events.sock`と`control.sock`に限定する。job終了時にsocketをarchiveせず、最終summaryだけをarchiveする。
runごとのarchive対象は次の三fileを基本にする。

```text
run-artifacts.tar.zst   # 下記expected control/source/domain/log/result
metrics.jsonl           # 最終raw sidecar
artifact-manifest.json  # outer file role、size、SHA-256。filenameはroleから導出
```

bundle memberはrun variant別のmanifest駆動exact allowlistとする。regularは`control/`のmanifest、contract、
protocol、submission raw stdout/stderr、実在するsubmission result、job-entry、scheduler-terminal、PJM accounting raw/version、
PJM batch logを含む。全program runはSIF sidecar、regularまたはqualification parent-owned runtime launcher、該当するchild-owned environment/cache/topology、benchmark stdout/stderr、
purpose別authoritative resultをexpected fileとして列挙し、regularだけmodel/dataset/workload manifestも含める。qualification
childはparent-owned contract/protocol、model/dataset/workload、job-entry env、runtime launcher、minimal hardware、scheduler/accounting/PJM batch logを
含めず、child manifestのparent fields/allocation IDから一度joinする。
profileだけphase summary、
qualification OFF/ONだけ`qualification-result.json`を含め、`qualification-result/v1`のrequired fieldを要求する。いずれも
applicableな`domain/`、`source/`、`logs/`、自己を除くmember path/hash/sizeを持つ
`archive-staging/bundle.manifest.json`を作り、tar rootの`bundle.manifest.json`として含める。
最終`metrics.jsonl`、artifact manifest、merge元fragmentはbundle内へ重複収録しない。

`telemetry/**`は`profile-raw.tar.zst`だけが所有する。D-004でprofilerがapplicableなら同じtarへprofiler trace、
tool metadata、exact argvも加え、非適用ならGPU telemetryだけを収録する。run bundleへ同じraw byteを重複収録しない。

`run-artifacts.tar.zst`はmanifestで列挙したmemberを収録し、作成後の実byte列をhashする。再実行時にtar headerや
compressor versionまで同一にして同じbundle hashを再現することは要求しない。同一run attemptでは既に作った
bundleをそのまま転送し直す。
最終`metrics.jsonl`は完全一致rowを一件へし、異値conflictは全rowを残したまま`(run,key,value)`の
UTF-8 byte順にsortする。各lineはobject member順`run`,`key`,`value`、空白なし、UTF-8、LF終端へ固定する。

manifestは通常のtemporary writeとrenameで完成させる。基本三fileとapplicable profile artifactをGitHub Artifactへ
uploadできた時点でGenkai runtimeはcleanupしてよい。archive jobはhomeの
移管と同じくremote再読取checksum、transaction、reconciler、別delivery storeは実装しない。

BigQuery publishはhomeのsubject artifact setまたは同じGitHub Artifactから再実行できればよい。invalidなrunも
通常のrun/attempt directoryへraw artifactを残し、comparison viewから除外する。物理的なquarantine systemは
作らない。通信・archive失敗はworkflow failureとして扱い、少数のarchive job状態だけをData Statusへ投影する。


成功後のGitHub Artifact削除である。Qwenはこの仕組みをまだ持たず、2026-08-09時点で134 artifact、圧縮後

これは実験用projectであり、極稀な転送failureやrun lossまで多重に防がない。通常pathで対象値を生成・回収する
ことを優先する。推奨option Rは次である。

```text
run -> build log、execution log、results、metrics.jsonlをGitHub Artifactへupload
    -> profileなら同じlogical artifact setの別outer fileへraw GPU telemetryと、D-004でapplicableになった場合だけprofiler traceを収録
    -> /srv/qwen-artifacts/<github_run_id>/attempt-<github_run_attempt>/ へscp
    -> SCP commandが成功した時だけ、そのworkflow runのGitHub Artifactを削除
```

SCP失敗ではarchive jobをfailureにし、GitHub Artifactを残す。job rerunで再転送できる。
remote transaction、content-addressed object store、put-if-absent、receipt DAG、容量reservation、複数quota、
remote再読取checksum、全境界crash recovery、独立backupは実装しない。runtimeはGitHub Artifact upload成功後に
現行どおりcleanupしてよい。

normal benchのPrimaryResultWriter、qualification OFF/ONのneutral qualification result、parent/child artifactを必ず
results artifactへ含める。Qwen profileのprofiler familyはD-004でapplicableと確定したprotocolだけrequiredにする。
その場合はprofiler wrapperへ明示引数`--profile-output-dir <runtime>/runs/<run>/profile/raw`を渡し、exact profiler
argv、tool metadata、trace rawを`profile-raw.tar.zst`へまとめてuploadする。D-004がprofiler非必須と決めたprotocolや
未回答中のrunでprofiler file/placeholderを要求せず、現在のtransparentな`exec-profiler.sh`を「取得済みprofiler」と扱わない。
bench/qualificationではprofile producer、artifact、欠損用eventを作らず、対応consumerは`NULL`だけである。

model約400 GBとSIFをhomeへ複写することは必須にしない。model/dataset/tokenizer/cacheのmanifest、SIF source
run、build/source commit、SIF content hashをrun artifactへ残し、実体は現行Genkai cacheに存在するものを使う。

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
run manifest -> submit result/job-entry -> compute log/source/ephemeral fragments
             -> terminal accounting -> post-run validation -> run artifact + metrics.jsonl
             -> GitHub Artifact upload -> runtime cleanup -> home SCP -> BigQuery publish

qualification:
allocation manifest -> submit result/job-entry -> child OFF source/log/metrics
                                           \-> child ON source/log/metrics
                    -> terminal accounting -> one qualification artifact set
```

- finalizerはvariant別expected file、producer、node/rankとrequired keyを一度検査する。正常終了runで
  欠けていれば`collection_complete=false`とする。失敗・timeout runのtmpは手元に診断用として残り得るが
  artifact/merge入力にせず、partial line recoveryや空文字/zero補完はしない。
- 同一run/keyの異値は両方を保持し、conflictとしてpublishしてcomparisonから隔離する。
- foreign run、unknown key、owner違反、canonical encoding違反はfragmentをmetrics sidecarへ混ぜない。
  専用quarantine状態や二次保存経路は作らない。
- process timeout、signal、spawn errorでもrunnerはscheduler terminal後にprimary result/domain eventをparseし、
  保存済みGPU sample/summaryをpackageできる範囲でraw log/sourceをartifactへ含める。GPU集約はBigQuery viewだけで行う。任意のcrash境界から完全再開
  できることは要求しない。
- cleanupはGitHub Artifact upload成功後に許す。publish失敗はhomeまたはGitHub Artifactの
  同じsubject artifact setから手動再送し、computeを再実行しない。

### gateを証明するtest

| gate | 必須test/fixture | 証明すること |
|---|---|---|
| G0 依存境界 | `test_dependency_boundaries.py`、CLI subprocess import trace/egress guard | domain/probe/recording/collect/stageはpublisher/credential/networkをimportせず、node-agentはapproved control transport以外のnetwork client、credential、DNS、外部egressを使わない |
| G1 値契約 | `test_domain.py`、RFC 8785/decimal/env byte fixture | 空、zero、欠損、非有限拒否、単射escape、canonical byte |
| G2 submit/binding | `test_submission.py`、regular/qualificationのsuccess/reject/mismatch fixture | exact argv/stdout/stderr/job ID保存、allocation/job/ordered child join、不一致の比較除外 |
| G3 local file | `test_recording.py`、正常終了・program failure fixture | buffered writerのclose/rename、完成fragmentだけがmerge入力になり、merge後fragmentと失敗tmpをartifactへ入れないこと |
| G4 ownership/merge | `test_merge.py`、variant別producer graph fixture | owner一意性、family routing、exact dedupe、異値conflict、required欠損、qualification parentのjob envを二childへjoinしてraw複写しない。欠損・非適用のconsumer値はNULL、required欠損時だけ`collection_complete=false` |
| G5 process観測 | `test_stage_runner.py` | actual argv/cwd、process-group signal forwarding、exit/signal/spawn error、monotonic境界 |
| G6 archive/送達 | workflow integration fixture | happy pathで全expected artifactがhomeにあり、upload失敗はattempt failure、SCP失敗ではGitHub Artifactが残る。homeのartifact setをBigQueryへ再送できる |
| G7 Qwen意味 | `test_observation_contract.py`、`test_primary_result.py`、`test_distributed_events.py`、`test_lifecycle_control.py`、variant argv/source-diff fixture | purpose別authoritative result、OFF/ONがparentの同一actual workloadを順番通り消費、ONはneutral qualification result+primary、primary欠損、node/MPI/worker/GPU bijection、1-node participant 0行、4-node participant別終了分類、lifecycle単一owner、qualification ONの差分がinstrumentation patchだけであること |
| G8 GPU phase | profile用`test_gpu_reducer.py`、1/2/4-node cardinalityの`test_phase_protocol.py` | benchではsampler/phase messageなし、profileでは全expected nodeのphase receiver、共有UDS不使用、wrong-run/stale message、node-local ACK/clock順、marker欠損、gap cutoff、weighted mean/p95/energy |
| G10 consumer contract | contract生成SQL fixture、variant別BigQuery golden rows、Sheet parameter/refresh smoke test | 全列lineage、profile attemptのscore統計除外、非適用・欠損列がNULL、pair-grain eligibility、失敗attempt残存、protocol分離、estimand、片側env欠損、freshness/size |
| G11 非擾乱 | D-001のreference OFF/final ON固定5 pair | 個々のthroughput/stage wall、median、result totals、identity一致を表示し簡素screenを再現できること |

`test_observation_contract.py`はcontractのschemaとowner/source/capture/applicability/validatorの完全性を検査し、
生成した観測マップ表がcommit済み表とbyte一致することを検査する。これが「この値はどこから取るか」
を、一つの正本から人間向け表と実行時validationの双方へ機械的に保証する。

### release単位

移行はkeyごとに二重化せず、run contract単位で行う。

1. legacyのcollector event/result/GPU集約をgolden fixtureとして凍結する。
2. pre-submit manifest、local metrics export、node-local endpoint、worker-ready、workload manifest、primary resultを
   一つのv2 fixtureと最初のv2 runで通す。phase control、
   raw GPU sampleをprofile purposeへ追加する。
3. benchへ必要な最小bundleだけD-001へ掛け、現在の2-node TP=8 score laneで意味・result同値性・非擾乱screenを通したbundleだけbenchへ追加する。
4. v2 viewとSheetを切り替え、legacy-only consumerがゼロになってから旧`send()`、fallback result、
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
4. ingestion-time partition/`run,key` clusterのraw table、distinct/singleton/conflict、Sources lineage、
   archive attempt/status、exact-contract run/dimension/point/stats/paired view、二run table
   functionをconsumer column contractから生成しfixtureを通す。
5. node-local event endpoint、rank 0 result ownership、node-scoped full configurationを実装し、
   397B runのconfiguration欠落と競合を修正する。manifestが固定したexact contractだけをv2 viewへ
   入れ、contractなしはlegacy、異なるcontractのproducer混在はinvalidにする。
6. warmup/measured workload、cache manifest、primary result validatorを追加し、laneから展開したnode/worker/GPU
   cardinalityを1/2/4-node fixtureで確認し、現在の2-node TP=8 laneでglobal worker rank 0..7を実機確認する。
7. profile purposeだけにpersistent control-forwarder phase eventとnode-local raw GPU sampleを追加し、
   benchでは対応consumer列を理由なしのNULLにする。どちらのtimed pathからもBigQueryを外す。
8. Connected SheetsのRuns/Replicates/Sources/Config Diff/Environment/Node Config/Participant Termination/Data Statusをbounded consumerへ
   切り替え、freshness/result-size smoke testを通す。旧key consumerがなくなってからthroughput、model名、
   GPU総数等の派生・部分集合keyの新規生成を止める。
9. homeのrun artifact tree、qualification parent artifact、Git管理source/contract/protocol/decision/SQLから
   空datasetを再構築し、raw row集合と主要view/Sheetの結果を照合してから旧viewを廃止する。

冗長keyの停止をview作成より先に行わない。

## 完了条件

- 性能に影響する変更がすべてremote到達可能なcommitにあり、commit bodyから仮説が分かる。
- SIF reuseでもsource-track head/base、original build commit、SGLang commit、content identityをrole付きで区別できる。
- modelはfull Hub revisionまたは一度だけ作るper-shard hash付きprebuilt manifestへ結び、runごとは
  config/index/tokenizerの1 byte変更とshard membership/path/size不一致を検出する。400 GB shard本文はrunごとに
  再hashしない。dataset、cache、workloadの1 byte/order変更は各manifestが検出する。
- 正当な空文字、未観測、collector失敗、collector未到達、真のゼロをfixtureで区別できる。
- canonical JSONがprocessやobject挿入順に依存せず同じbyte列になる。
- 各key patternのownerが一意で、distinct後のrun/key conflictがゼロである。
- synthetic 1/2/4-node fixtureで同じcollector graphがlane指定のexpected node/worker/GPU集合を展開でき、
  現在の実2-node TP=8 runでnode-scoped full configが届き、viewが一致fieldとnode差をlosslessに表示できる。
- expected node rank、global worker rank、host、GPU UUID、TP、nnodes、MPI processの不一致を検出できる。
- prompt順序だけが違う同token総数workloadをmanifest identityで分離できる。
- result欠損、不正JSON、returned count/token不一致、非有限時間をcomparison対象外にできる。
- viewの四つの代数的throughputがSGLang artifactと許容誤差内で一致する。
- profileのphaseへ既知delayを入れたfixtureでinit、prepare、warmup、generation、shutdownを分離できる。
- profileでgeneration marker欠損時にGPU aggregateを全stage値で代用せず、benchの同列は理由なしのNULLになる。
- profileでexpected GPU gridの一台欠損時に比較用fleet projectionをNULLにできる。
- count/sizeが同じ別cacheをmanifest identityで区別できる。
- BigQuery停止時もhomeまたはGitHub ArtifactのJSONLから同じlogical row集合を再送できる。
- 成功・失敗・timeout runのexpected log/applicable artifact/JSONLをGitHub Artifactへuploadし、upload成功後は
  runtimeを削除できる。
- Qwenの全core/source/resultとprofile runのapplicable raw attachmentがhome pathから読め、bench
  archiveにはprofile pathも欠損用recordもなく、対応Sheet列がNULLだけになる。
- bench/profileが分離され、profile-only列はbenchで理由なしのNULLになり、bench bundleの計測影響はD-001内である。
- 同一protocolの反復がpair ID、実行順、blockを含め復元でき、全estimand点、median、MADがSheetに表示される。
- treatment値を含む同一条件だけが同じreplicate groupへ入り、宣言したtreatment差だけが同じ
  comparison set内で許される。
- required dimensionが欠けたfailed/invalid runもpre-run planned groupのattempted nへ一度だけ入り、NULL
  fingerprint groupへ脱落・混入しない。
- 別protocolが同じhuman group labelを使ってもcontract/protocol/policy namespaceにより統計が混ざらない。
- 各run/Sources/Sheet列がcontract raw keyまたはGit/artifact dimensionへ一意に遡れ、cast失敗や異値競合を
  aggregateが隠さない。
- 二run diffがcell parameter付きtable functionで計算され、NULL/同一/不存在runを拒否し、全run pairを
  materializeせず、full envをparameterなしでSheetへextractしない。
- config definition/policy非互換または片側env snapshot欠損では、左右値/equalityをNULLにする。
- Data Statusが各objectのquery時刻、最新publish、未移管Artifact数、archive job最終成功時刻、home空き容量、
  SQL/contract identity、conflict/run-level collection incomplete件数を表示し、
  scheduled refreshのowner変更・停止とresult-size超過をsmoke testで検出できる。
- Sheets利用者がJSON展開なしでcommit、completeness、一次結果、反復統計、artifactへ到達できる。
