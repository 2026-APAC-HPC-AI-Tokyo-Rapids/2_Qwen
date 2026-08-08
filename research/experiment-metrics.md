# Qwen / SGLang 実験メトリクス収集計画

## 目的

Genkai 上の Qwen / SGLang benchmark について、再現条件、実効設定、CPU・GPU
資源使用量、性能、成否を run 単位で BigQuery へ記録する。

保存形式と送信規則は
[`../1_OpenFOAM`](../../1_OpenFOAM/research/experiment-metrics.md) の実装を継承する。
Qwen 固有部分は、SGLang 内部で解決された引数、NVIDIA GPU の実行中 telemetry、
LLM benchmark 結果である。

## 継承する契約

- BigQueryへ送るのは検索・比較できる実測値、実効設定、識別子、成否などのscalarを
  基本とし、artifact本文やその代理となるcontent hashをmetricにしない。
- raw table は `run STRING NOT NULL`, `key STRING NOT NULL`,
  `value STRING NOT NULL` の3列だけにする。
- dictはscalar keyへ展開する。list/tupleは順序を持つ設定値を欠落させないための
  限定的な例外として、1つのcompact JSON文字列にする。
- 通常の成功 run では `(run, key)` を一意にする。
- 同じ取得元から同時に得た値を1 batchにし、取得後すぐ BigQuery へ送る。
- BigQuery transport用のdisk queueやspoolは作らない。複数nodeからrank 0へ渡す
  GPU集約batchだけは共有runtime directoryを一時handoffに使い、読取後すぐ削除する。
- 取得・送信失敗は benchmark の終了 codeを変更せず、取得不能値は空文字にする。
- collector は値同士の照合、baseline判定、pass/fail判定をしない。
- 計算環境の環境変数は問答無用で全件保存する。
- BigQuery 認証情報は collector 起動時に環境からメモリへ移し、環境メトリクスには
  混入させない。
- BigQuery REST API `tabledata.insertAll`、memory batch、同じ再試行規則、
  deterministic `insertId` を OpenFOAM 実装から再利用する。

送信先は次を予定する。

```text
apac-hpc-ai-tokyo-rapids-26.qwen_experiment_metrics.qwen_metrics
```

dataset は Tokyo (`asia-northeast1`)、table は ingestion-time で日次 partitionし、
`run`, `key` で clusterする。GCP resource の作成・変更は実装段階で行い、
この research 作業では行わない。

## Qwen 固有の収集モジュール

| 文書 | 実装単位 | 主なキー | 取得時点 |
|---|---|---|---|
| [01-run-and-source.md](experiment-metrics/01-run-and-source.md) | `run_source` | `run.*`, `software.*`, `artifact.*` | 開始・終了 |
| [02-platform-and-scheduler.md](experiment-metrics/02-platform-and-scheduler.md) | `platform`, `scheduler_pjm` | `env.*`, `system.*`, `hardware.cpu.*`, `scheduler.*` | 開始前・PJM終了後 |
| [03-sglang-configuration.md](experiment-metrics/03-sglang-configuration.md) | `sglang_event` | `sglang.*`, `model.name` | 開始前・Engine初期化後 |
| [04-gpu-telemetry.md](experiment-metrics/04-gpu-telemetry.md) | `nvidia_gpu` | `hardware.node.*.gpu.*`, `resource.stage.benchmark.node.*.gpu.*` | benchmark終了後 |
| [05-benchmark-result.md](experiment-metrics/05-benchmark-result.md) | `benchmark_result` | `perf.*`, `result.*` | benchmark終了直後 |
| [06-lifecycle-and-artifacts.md](experiment-metrics/06-lifecycle-and-artifacts.md) | `stage_runner`, `artifact` | `run.stage.*`, `artifact.*` | 各値の確定直後 |
| [07-implementation-plan.md](experiment-metrics/07-implementation-plan.md) | 全体 | 実装順序、テスト、導入条件 | 実装前 |

## SGLang 引数を取得する方針

2026-07-31 に upstream SGLang commit
`585a7d05e384c79ee17bf3185c7469fe8ce75946` を調査した。
`offline_throughput` は次の経路を通る。

```text
argparse.Namespace
  ├─ BenchArgs.from_cli_args()
  └─ ServerArgs.from_cli_args()
       └─ ServerArgs.__post_init__() でモデル・GPU依存の既定値を解決
            └─ dataclasses.asdict(server_args)
                 └─ Engine(**kwargs)
```

したがって、PJM scriptに書かれた command lineだけでなく、次の3層を別々に保存する。

1. `run.command`: 実際に起動したargvのshell表現
2. `sglang.benchmark_arg.*`: defaultを含む `BenchArgs`
3. `sglang.server_arg.*`: Engine初期化後の実効 `ServerArgs`

SGLang 側の小さな instrumentation patch が、Engine初期化直後とbenchmark終了直後に
構造化eventを Unix domain socketへ送る。host側 collectorはeventを受信し、
dictを再帰的にscalar recordへ展開し、list/tupleは順序を持つ実設定値として1つの
compact JSON valueにして、直ちにBigQueryへ送る。
`backend.get_server_info()`由来の`sglang.runtime.*`と`model.config.*`は保存せず、
モデルはhostの`MODEL_PATH`末尾から得る`model.name`だけを保存する。

socketはprocess間通信endpointであり、metric dataをdiskへ保存する一時ファイルではない。
JSON object全体はBigQueryの`value`へ保存しない。list/tupleだけは設定値の順序と
要素境界を保持しつつ行数を抑えるため、compact JSON配列として保存する。
`results.jsonl`は従来どおり人間とActions向けのbenchmark成果物として残す。

## GPU計測の境界

GPU構成は実行前に1回取得する。GPU utilization、memory、power、temperature、clockは
benchmark processと並行して1秒間隔で取得し、raw sampleは保存も送信もしない。
終了時にGPUごとの件数、平均、最大、percentile、energy積算値へ縮約し、1 batchで送る。

この集約範囲にはmodel load、warmup、timed generation、shutdownを含める。
SGLangが報告する`total_latency`はtimed generationだけなので、両者を同じ時間として
扱わない。

第1段階はGenkaiに標準装備される`nvidia-smi`を使う。DCGMが計算ノードで利用可能と
確認できた場合だけ、第2段階としてTensor Core、DRAM、PCIe、NVLinkのprofiling値を
追加する。利用可否を公称値から推測しない。

## 収集しないもの

- prompt本文、生成本文、tokenごとのlog
- GPUのraw 1秒sample
- SGLang requestごとのlatency時系列
- profiler traceやkernel counterのBigQueryへの展開
- stdout/stderr全文
- baselineとの差、合否、性能回帰判定
- 公称hardware値を実測値として代入した値

生成結果の本文はBigQueryへ保存しない。必要な比較はActions artifactの
`results.jsonl`を対象に行い、collector側で内容識別子を生成しない。

## 現行構成に対して必要になる変更

実装段階では、少なくとも次が必要である。

- repository直下に独立したuv project `qwen-experiment-metrics/` を追加
- OpenFOAM collectorのsender、credential、model、PJM readerを移植
- `.github/workflows/genkai.yml` でcollectorをruntime bundleへprovision
- `GITHUB_RUN_ATTEMPT`, `GITHUB_REPOSITORY` と認証情報をPJMへ搬送
- `scripts/run-qwen35.pjm` をcollectorのlifecycle supervisorへする
- `offline_throughput` の構造化event送信用SGLang patchを追加
- host側stage wrapperにUnix socket受信とGPU samplerを実装
- PJM終了後にrunnerからscheduler accountingを送信

SGLang patchをSIFへ含めるcommitは`[require-build]`が必要である。collector、workflow、
runtime scriptだけを先に実装・testするcommitでは既存SIFを再利用できる。

## 調査根拠

- OpenFOAMの設計・実装・Git履歴
- このrepositoryと旧`~/2_Qwen`をcwdとするCodex会話履歴
- [SGLang offline throughput source](https://github.com/sgl-project/sglang/blob/585a7d05e384c79ee17bf3185c7469fe8ce75946/python/sglang/benchmark/offline_throughput.py)
- [SGLang ServerArgs source](https://github.com/sgl-project/sglang/blob/585a7d05e384c79ee17bf3185c7469fe8ce75946/python/sglang/srt/server_args.py)
- [SGLang Engine source](https://github.com/sgl-project/sglang/blob/585a7d05e384c79ee17bf3185c7469fe8ce75946/python/sglang/srt/entrypoints/engine.py)
- [SGLang benchmark guide](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md)
- [NVIDIA nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/)
- [NVIDIA DCGM profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)
- [Genkai NVIDIA software guide](https://www.cc.kyushu-u.ac.jp/scp/en/system/genkai/software/nvidia/)
- [BigQuery tabledata.insertAll](https://cloud.google.com/bigquery/docs/reference/rest/v2/tabledata/insertAll)
