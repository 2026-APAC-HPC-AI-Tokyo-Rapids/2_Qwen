# Qwen / SGLang 実験メトリクス収集計画

## 目的

Genkai 上で実行される Qwen / SGLang ベンチマークを対象に、再現条件、実効設定、CPU・GPU の資源使用量、推論性能、およびジョブ成否の情報を実行単位（run）で BigQuery へ記録する。

Qwen では、SGLang 内部で動的に解決された実効引数、NVIDIA GPU の実行時テレメトリ、および LLM ベンチマーク特有の性能指標を収集する。

## データ保存と送信の契約

- BigQuery へ送信するデータは、後からの検索・比較が容易な実測値、実効設定、識別子、成否フラグなどのスカラー値を基本とし、成果物（artifact）の本文やその代替となるコンテンツハッシュ自体はメトリクスとして扱わない。
- 生データテーブル（raw table）は、`run STRING NOT NULL`、`key STRING NOT NULL`、`value STRING NOT NULL` の3列構成に限定する。
- 辞書（dict）はスカラーキーへ再帰的に展開する。リストやタプル（list/tuple）は、順序性を持つ設定値の構造を欠落させないための限定的な例外として、単一のコンパクトな JSON 文字列としてシリアライズする。
- 正常終了した通常の実行（run）においては、`(run, key)` の組み合わせが一意となることを保証する。
- 同一の収集元から同時に取得した値は1つのバッチにまとめ、取得後速やかに BigQuery へ送信する。
- BigQuery 送信用のディスクキューやスプールファイルは設けない。複数ノードからランク0（Rank 0）へデータを集約する GPU テレメトリバッチに限り、共有ランタイムディレクトリを一時的な受け渡し領域として使用し、読み取り完了後直ちに削除する。
- メトリクスの取得や送信に失敗した場合でもベンチマーク本体の終了コードは変更せず、取得不能な値は空文字列として扱う。
- コレクター自身は、値同士の妥当性照合、ベースライン比較、合否判定（pass/fail 判定）を行わない。
- 計算ノードの環境変数は、フィルタリングを行わず全件保存する。
- BigQuery の認証情報はコレクター起動時に環境変数からメモリ上へ直ちに退避させ、環境変数メトリクスへ混入させない。
- BigQuery REST API の `tabledata.insertAll` エンドポイント、インメモリバッチ処理、再試行規則、および決定論的な `insertId` 生成ロジックを実装する。

送信先のテーブルは以下を予定する。

```text
apac-hpc-ai-tokyo-rapids-26.qwen_experiment_metrics.qwen_metrics
```

データセットのロケーションは東京リージョン（`asia-northeast1`）とし、テーブルは取り込み日時（ingestion-time）による日次パーティション、ならびに `run` および `key` によるクラスタリングを設定する。なお、GCP リソースの作成・変更は実装フェーズで実施するものとし、本調査・設計フェーズでは行わない。

## Qwen 固有の収集モジュール

| 文書 | 実装単位 | 主なキープレフィックス | 収集タイミング |
|---|---|---|---|
| [01-run-and-source.md](experiment-metrics/01-run-and-source.md) | `run_source` | `run.*`, `software.*`, `artifact.*` | ジョブ開始時・終了時 |
| [02-platform-and-scheduler.md](experiment-metrics/02-platform-and-scheduler.md) | `platform`, `scheduler_pjm` | `env.*`, `system.*`, `hardware.cpu.*`, `scheduler.*` | ジョブ開始前・PJM 終了後 |
| [03-sglang-configuration.md](experiment-metrics/03-sglang-configuration.md) | `sglang_event` | `sglang.*`, `model.name` | ジョブ開始前・Engine 初期化後 |
| [04-gpu-telemetry.md](experiment-metrics/04-gpu-telemetry.md) | `nvidia_gpu` | `hardware.node.*.gpu.*`, `resource.stage.benchmark.node.*.gpu.*` | ベンチマーク終了後 |
| [05-benchmark-result.md](experiment-metrics/05-benchmark-result.md) | `benchmark_result` | `perf.*`, `result.*` | ベンチマーク終了直後 |
| [06-lifecycle-and-artifacts.md](experiment-metrics/06-lifecycle-and-artifacts.md) | `stage_runner`, `artifact` | `run.stage.*`, `artifact.*` | 各値の確定直後 |
| [07-implementation-plan.md](experiment-metrics/07-implementation-plan.md) | 全体 | 実装手順、テスト戦略、移行要件 | 実装着手前 |

## SGLang 引数を取得する方針

2026-07-31 時点における upstream SGLang のコミット `585a7d05e384c79ee17bf3185c7469fe8ce75946` を精査した結果、`offline_throughput` の引数解決は以下の経路を辿ることが確認された。

```text
argparse.Namespace
  ├─ BenchArgs.from_cli_args()
  └─ ServerArgs.from_cli_args()
       └─ ServerArgs.__post_init__() でモデル・GPU依存の既定値を解決
            └─ dataclasses.asdict(server_args)
                 └─ Engine(**kwargs)
```

したがって、PJM スクリプトに記述された静的なコマンドライン文字列だけでなく、以下の3層を明確に分離して記録する。

1. `run.command`: 実際に実行された argv のシェル文字列表現
2. `sglang.benchmark_arg.*`: デフォルト値を含めて解決された `BenchArgs`
3. `sglang.server_arg.*`: Engine 初期化処理を経て確定した実効 `ServerArgs`

SGLang 側へ最小限の計測用パッチを適用し、Engine 初期化直後およびベンチマーク終了直後に構造化イベントを Unix ドメインソケット経由でホスト側へ送信する。ホスト側のコレクターがこのイベントを受信し、辞書構造を再帰的にスカラーレコードへと展開する。リストやタプルは順序性を持つ設定値として単一のコンパクトな JSON 文字列に変換した上で、速やかに BigQuery へ送信する。
なお、`backend.get_server_info()` に由来する `sglang.runtime.*` や `model.config.*` の動的情報は保存せず、モデル名についてはホスト側の `MODEL_PATH` 末尾から抽出した `model.name` のみを保存する。

Unix ドメインソケットはプロセス間通信のエンドポイントであり、メトリクスデータをディスクへ書き出す一時ファイルではない。また、JSON オブジェクト全体を BigQuery の `value` 列へ格納することは避ける。リストやタプルに限り、設定値の順序関係と要素境界を保持しつつテーブルの行数肥大化を抑えるため、コンパクトな JSON 配列として記録する。
なお、`results.jsonl` は従来どおり人間による確認および GitHub Actions の成果物（artifact）として保持する。

## GPU 計測の境界

GPU のハードウェア静的構成は、ベンチマーク実行前に1回のみ取得する。GPU 使用率、メモリ使用量、消費電力、温度、クロック周波数などの動的指標は、ベンチマークプロセスと並行して1秒間隔でサンプリングするが、生サンプル（raw sample）をそのまま保存・送信することはしない。ベンチマーク終了時に、GPU ごとのサンプル数、平均値、最大値、パーセンタイル値、および積算電力量（エネルギー消費量）へと集約した上で、1つのバッチとして送信する。

この集約区間には、モデルのロード、ウォームアップ、時間計測対象の生成処理（timed generation）、およびシャットダウンの全区間が含まれる。SGLang が出力する `total_latency` は timed generation の区間のみを対象としているため、両者の計測時間を同一視してはならない。

第1段階としては Genkai に標準導入されている `nvidia-smi` を使用する。DCGM が計算ノード上で利用可能であることが実地で確認された場合に限り、第2段階として Tensor Core、DRAM、PCIe、NVLink の詳細プロファイリング値の取得を追加する。事前の仕様や公称スペックから利用可否を推測してはならない。

## 収集対象外とする項目

- プロンプト本文、生成テキスト本文、トークン単位の推論ログ
- GPU テレメトリの1秒単位の生サンプルデータ
- SGLang のリクエスト単位のレイテンシ時系列データ
- プロファイラトレースデータや詳細なカーネルカウンタ値の BigQuery への展開
- 標準出力（stdout）および標準エラー出力（stderr）の全文
- ベースラインとの差分値、合否判定フラグ、性能リグレッション判定結果
- 公称ハードウェアスペックを実測値として代用した静的数値

生成テキストの本文は BigQuery には保存しない。出力結果の比較・検証が必要な場合は、GitHub Actions の成果物として保全される `results.jsonl` を直接対象とし、コレクター側で独自の内容ハッシュや識別子を生成することは行わない。

## 既存構成に対する主な変更要件

実装フェーズにおいては、少なくとも以下の変更作業が必要となる。

- リポジトリ直下に独立した uv プロジェクト `qwen-experiment-metrics/` を新設
- 送信モジュール、認証情報ハンドラ、データモデル、PJM アカウンティングリーダーを実装
- `.github/workflows/genkai.yml` においてコレクター資材をランタイムバンドルへプロビジョニング
- `GITHUB_RUN_ATTEMPT`、`GITHUB_REPOSITORY`、および GCP 認証情報を PJM スクリプトへ伝播
- `scripts/run-qwen35.pjm` をコレクターのライフサイクルスーパーバイザーとして機能拡張
- `offline_throughput` から構造化イベントをソケット送信するための SGLang 向けパッチを追加
- ホスト側のステージラッパーに Unix ドメインソケット受信機能および GPU サンプラーを実装
- PJM ジョブ終了後に GitHub Actions ランナーからスケジューラのアカウンティング情報を送信

SGLang への計測パッチを SIF イメージへ組み込むコミットでは `[require-build]` による再ビルドが必須となる。一方、コレクター本体、ワークフロー定義、およびランタイムスクリプトのみを先行して実装・検証するコミットにおいては、既存の SIF イメージをそのまま再利用できる。

## 調査根拠・参照資料

- [SGLang offline throughput source](https://github.com/sgl-project/sglang/blob/585a7d05e384c79ee17bf3185c7469fe8ce75946/python/sglang/benchmark/offline_throughput.py)
- [SGLang ServerArgs source](https://github.com/sgl-project/sglang/blob/585a7d05e384c79ee17bf3185c7469fe8ce75946/python/sglang/srt/server_args.py)
- [SGLang Engine source](https://github.com/sgl-project/sglang/blob/585a7d05e384c79ee17bf3185c7469fe8ce75946/python/sglang/srt/entrypoints/engine.py)
- [SGLang benchmark guide](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md)
- [NVIDIA nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/)
- [NVIDIA DCGM profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)
- [Genkai NVIDIA software guide](https://www.cc.kyushu-u.ac.jp/scp/en/system/genkai/software/nvidia/)
- [BigQuery tabledata.insertAll](https://cloud.google.com/bigquery/docs/reference/rest/v2/tabledata/insertAll)
