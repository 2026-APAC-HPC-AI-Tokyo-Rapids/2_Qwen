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
  create one bench/profile run manifest
  verify source/artifact sidecars and independent scheduler/execution fields
  record exact pjsub argv and submit without BigQuery credentials
  save stdout/stderr, outcome, and returned job ID

PJM supervisor
  snapshot job-entry environment
  write run ID and PJM_JOBID
  launch one StageRunner/mpiexec/node-wrapper set

node wrapper / node-agent
  record topology, cache input, exact argv, hardware, and host environment
  wait for expected-node readiness before starting containers
  bench/profile: snapshot score config and container state
  profile only: append raw GPU samples and local phase markers
  flush/close local sources before script exit

SGLang score hook
  emit launch-effective ServerArgs and worker-ready facts

rank-0 result hook
  persist warmup/measured workload manifests
  write the authoritative primary-result artifact
  profile only: emit distributed GPU phase boundaries

Actions runner
  parse/reduce completed sources after terminal state
  merge, validate, upload run artifact, and publish run projections
```

全producerは`run-manifest.json`のrun IDを読む。各processで再計算しない。target hookはlocal structured
eventを書くだけで、metric key、BigQuery、retryを知らない。

## fact別の取得契約

取得できなかった値はkeyを出さずconsumerでNULLにする。required sourceの欠損は
`collection_complete=false`に反映するだけで、原因、family status、placeholder keyを作らない。

| fact family | authoritative source / producer | raw destination | validation / applicability |
|---|---|---|---|
| run identity | runner / submit前 | `control/run-manifest.json` | run IDとpurposeを照合。contract/protocol identityは記録しない |
| source track | Git object / runner | run manifestとsource projection | clean checkout、full OID、base ancestor、実control-plane hash |
| instrumentation bundle | Git管理bundleとSIF sidecar / runner | bundle member/hash | bench/profileの通常bundleだけ。開発時比較の情報は入れない |
| scheduler | exact pjsub argv/stdout/stderr、job-entry、pjstat / runner・supervisor | scheduler fact/source | submit、entry、accountingのjob ID一致 |
| environment | job entry、host exec、SGLang launch、workerのbyte環境 / 各scope owner | snapshotとscope別fact | byte-exact、expected node/rankを保持 |
| SIF/toolchain | build sidecar、完成SIF、runtime file / build packager | SIF provenance projection | full commit、digest、file hash、role一意性 |
| model/dataset/cache | pinned manifestとstaging input / provision verifier | source manifestとartifact identity | revision、member path/size/hashを照合 |
| scheduler request | 実行設定の独立fieldとexact pjsub argv / runner | scheduler request | nodes、MPI、GPU/node、TP、resource、transport等を独立に照合 |
| hardware/topology | lscpu/sysfs、MPI/container/worker observation / node wrapper | node/rank projection | expected node/rank/GPU bijection |
| lifecycle | node-agent rendezvousとsummary | lifecycle projection | run/version/freshness、expected node ACK、shutdown一意性 |
| workload | generateへ渡したordered request / rank-0 result hook | ordered workload artifact | warmup/measuredを分離し、summary keyは作らない |
| primary result | 実response検査結果 / rank-0 result hook | `source/primary-result.json` | returned/error/token/output identity/wall。throughputはquery時導出 |
| stage/process | StageRunnerのmonotonic timerとwait status | stage termination fact | argv/cwd/exit/signal/timeout、clock scopeを保持 |
| MPI network capture | generation成功後のIMB argv/stdout/stderr / network capture | `source/network/imb-mpi1/`、固定幅facts、`mpi_network_results_v2` | generationと同じMPI process/allocation/transport。complete sectionだけをpure parse。失敗/timeout/truncatedは0 row、termination保持、`collection_complete=false` |
| NCCL network capture | generation成功後のcollective別argv/JSON/stdout/stderr / network capture | `source/network/nccl-tests/<collective>/`、固定幅facts、`nccl_network_results_v2` | generationと同じMPI/GPU数、全GPU一communicator、`NCCL_TESTS_SPLIT`なし。3 stageは独立継続、JSON不在/parse不能は当該stage 0 row、retryなし |
| GPU telemetry | nvidia-smi sampleとphase marker / profile agent | profile raw/projection | profileだけ。benchはproducerなし、consumer NULL |
| artifact | hash済みmemberとouter manifest / publisher | artifact index | expected member、size/hash、role、一意なownerを照合 |

## 個別実行設定とconsumer

nodes、MPI processes、GPU/node、tensor parallel size、resource group、transport、address source、portは
実行設定の独立field、BigQuery/Sheetsの独立列として持つ。これらを束ねるID、object、manifestは作らない。

benchの比較適格性は通常runのsource/workload/result/placement/completenessから導出する。
profileはscore統計へ入れない。complete recordから導出できるfirst/last/count/average、throughput、
構成比、rank/run合計、top-Nはraw keyとして保存しない。

## 開発時の非擾乱確認

collector実装中に代表設定で計測なし／ありを同条件で実行し、throughputとstage wallが実用上遅くなって
いないことを確認する。確認後は出力を破棄する。

この確認のためのrun purpose、arm、pair、allocation、manifest、artifact、metric key、relation、Sheet列、
eligibility、decision record、fixtureをv2へ追加しない。通常benchへ結果をjoinしない。
