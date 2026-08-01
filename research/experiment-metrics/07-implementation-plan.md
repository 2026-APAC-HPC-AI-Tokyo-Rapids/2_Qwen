# 実装順序と検証計画

## 現在確定している設計

- BigQuery raw schemaとsenderはOpenFOAM互換にする。
- Qwen専用dataset/tableは`qwen_experiment_metrics.qwen_metrics`とする。
- collectorは独立uv projectとしてruntime bundleへcopyする。
- SGLangの実効引数はcommand lineから再構築せず、Engine内のobjectから取得する。
- nested objectはscalar keyへ再帰展開し、JSON valueを保存しない。
- GPU raw sampleは保存せず、1秒samplingのrun-level集約だけを送る。
- 第1段階は`nvidia-smi`、DCGMは実機利用可能性確認後の第2段階にする。
- 通常benchmarkの成否をmetrics失敗で変えない。

## Phase 1: 共通基盤

1. `qwen-experiment-metrics/`をuv projectとして作る。
2. OpenFOAMから`Metric`, scalar codec, key validator, credential capture,
   config loader, REST sender, `CommandRunner`, `pjstat` readerを移植する。
3. package名、CLI名、BigQuery設定をQwen用に変更する。
4. stdout sinkとunit testを移植する。
5. generic nested scalar flattenerを追加し、dict/list/None/Enum/invalid keyのtestを書く。

このphaseはSIFを変更しない。

## Phase 2: host collector

1. run ID、run start/end、argv collectorを実装する。
2. host platform、全環境変数、CPU、memory collectorを移植・拡張する。
3. static NVIDIA GPU query parserを実装する。
4. PJM scheduler collectorを移植する。
5. SIF、script、model config、dataset、result artifactのreader/hashを実装する。
6. 現行`results.jsonl` parserとbenchmark result mappingを実装する。

fixtureだけで全parserをtestし、login nodeにGPUがないことをtest失敗理由にしない。

## Phase 3: GPU stage wrapper

1. Python supervisorでsubprocess、monotonic timing、signal伝播を実装する。
2. `nvidia-smi` samplerをstdout pipeで起動し、raw sampleをmemory集約する。
3. mean/max/p50/p95、power積分、P-state countを実装する。
4. missing field、`N/A`、途中終了、0 sample、複数GPU、MIG fixtureをtestする。
5. Unix socket event receiverを追加する。

sampler overheadはActionsのGPU jobで、samplingなし/ありの短い同条件runを複数回行って
確認する。比較runは同時投入できる場合は並行させる。

## Phase 4: SGLang instrumentation

upstream sourceへのpatchを最小限で追加する。

1. Engine初期化後に`BenchArgs`, 実効`ServerArgs`, container env,
   workload shapeをconfiguration eventとして送る。
2. benchmark終了後にresult dictとoutput hashをresult eventとして送る。
3. socket未設定・接続失敗・collector切断時はwarningだけで通常動作を続ける。
4. `backend=engine`, `runtime`, Ray proxyでevent取得経路が壊れないようtestする。
5. patch適用後のSGLang commitとevent schemaをSIF testで確認する。

このpatchを含めるcommitは先頭に`[require-build]`を付ける。

## Phase 5: workflow統合

1. provision jobでcollector projectをcopyし`uv sync --frozen --no-dev`する。
2. PJM submit前に共有`RUNTIME_DIR`へ`cd`し、既知の`NO CURRENT DIR`問題も同時に解消する。
3. GitHub metadataとcredential Base64を`pjsub -x`へ追加する。
4. PJM scriptをrun/platform/static GPU/stage/run-endへ配線する。
5. Singularity commandへmetrics socket directoryとsocket path envを渡す。
6. PJM終了後のrunner stepでscheduler metricsを送る。
7. cleanupとartifact uploadは現在の`if: always()`を維持する。

## Phase 6: BigQuery

1. Tokyoに`qwen_experiment_metrics` datasetを作る。
2. `qwen_metrics` tableを3列、ingestion-time日次partition、
   `run,key` clusterで作る。
3. OpenFOAMと同じservice accountへ必要最小限のinsert権限を付与する。
4. repository secretは既存名を再利用できるか確認し、必要なら登録する。
5. stdout sinkでend-to-end dry run後、実tableへ1 runだけ送る。
6. row数、空文字、scalar key、重複、scheduler後送信をqueryで検証する。
7. raw table以外のviewや追加resourceを作らないことを確認する。

使っていなくても継続課金するserviceは、この計画には追加しない。

## Phase 7: optional DCGM

通常のSGLang変更ではない調査であっても、実機確認はActions/PJM経由で履歴を残す。

1. 計算ノード上の`dcgmi`/library/host engineの有無とversionを確認する。
2. H100/full GPU/MIGそれぞれでsupported metric groupを確認する。
3. `nvidia-smi` samplerとの同時利用可否とoverheadを測る。
4. 利用可能なfieldだけschemaへ追加し、unsupportedを空文字にする。

## 受け入れ条件

- unit testでBigQuery rowが常に3列・scalarになる。
- credentialが`env.*`、log、artifact、socket eventへ出ない。
- benchmark commandと終了codeがcollectorなしの場合と同じである。
- SGLang default/auto解決後のbackend値がBigQueryで確認できる。
- `BenchArgs`, `ServerArgs`, request dictのnested値が個別keyになる。
- GPU static構成とruntime aggregateがUUIDで対応する。
- raw GPU sample、prompt、生成本文が保存されない。
- benchmark失敗runでも開始済みbatchとGPU集約が可能な範囲で残る。
- sender失敗だけではActions/PJM benchmarkが失敗しない。
- PJM終了後にscheduler accountingが同じrun名へ追加される。

## 実装時に先に確認する小さな不確実性

- 現行SIFで使われる正確なSGLang commitと、調査snapshotとの差
- Genkaiのdriverが受け付ける`nvidia-smi --query-gpu` field名
- full H100 jobでDCGM profilingがuser権限から利用可能か
- Singularity bind越しのUnix socket通信
- model local metadataからHugging Face commitを確実に得られるか

これらは設計を未確定にする理由ではない。unsupported時は空文字にする契約を維持し、
実装fixtureとActions上の短いprobeで取得方法だけを確定する。
