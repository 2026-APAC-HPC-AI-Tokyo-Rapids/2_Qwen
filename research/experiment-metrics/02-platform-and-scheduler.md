# ホストプラットフォーム・環境変数・PJM

## ホストプラットフォーム情報

`platform` コレクターは、PJM スクリプト上で動作するコレクタープロセスから以下のホスト情報を取得する。

| キー | 取得元 |
|---|---|
| `system.cluster` | フォールバック定数 `genkai` |
| `system.os.id` | `/etc/os-release` |
| `system.os.version_id` | `/etc/os-release` |
| `system.kernel.release` | `os.uname()` |
| `system.architecture` | `os.uname()`（対応する環境変数がない場合のみ） |
| `hardware.cpu.model` | `lscpu --json` |
| `hardware.cpu.socket_count` | `lscpu --json` |
| `hardware.cpu.cores_per_socket` | `lscpu --json` |
| `hardware.cpu.threads_per_core` | `lscpu --json` |
| `hardware.cpu.available_logical_count` | プロセスのアフィニティマスク |
| `hardware.cpu.numa_node_count` | `lscpu --json` |
| `hardware.memory.total_bytes` | `/proc/meminfo` |

CPU の公称スペック値をハードコードで埋め込むことはしない。GPU ベンチマークであってもトークナイズ処理、データセットの前処理、プロセス管理等で CPU を利用するため、CPU モデル、アフィニティ、NUMA 構成、および PJM の CPU アカウンティング情報は確実に記録する。なお、瞬間的な CPU 使用率のサンプリングは第1段階では実施しない。

## 環境変数の取り扱い

PJM スクリプトの実行環境変数は、`env.<name>` 規則に従い、`os.environb` から全件を無加工で保存する。許可リスト（allowlist）や拒否リスト（denylist）による選別、秘密情報の自動フィルタリングは設けない。

ただし、BigQuery 送信に用いる以下の認証情報環境変数に限り、実験条件そのものではないため、CLI 起動直後にメモリ上へ退避させた上で環境変数テーブルから削除する。

```text
EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_JSON
EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_BASE64
```

SGLang の推論プロセスは Singularity の `--cleanenv` フラグ下で動作するため、ホストとは隔離された独立の環境変数群を持つ。SGLang 側の計測イベントからその全件を抽出し、以下のキープレフィックス配下へ保存する。

```text
sglang.env.<name>
sglang.env.encoded.<name-bytes-hex>
sglang.env.bytes.<name-token>
```

これにより、ホスト／PJM のジョブ実行環境と、コンテナ内の実際の推論プロセス環境との混同を防止する。

## PJM アカウンティング終了統計

`pjstat --history --data --delimiter` リーダーを用い、GitHub Actions ランナーが PJM ジョブの終了直後に以下の統計値を送信する。

| キー | PJM 項目 |
|---|---|
| `scheduler.state` | `st`（ジョブ状態） |
| `scheduler.started_at` | `sdt`（開始日時） |
| `scheduler.elapsed_sec` | `elp`（経過秒数） |
| `scheduler.exit_code` | `ec`（終了コード） |
| `scheduler.signal` | `sn`（シグナル番号） |
| `scheduler.pjm_code` | `pc`（PJM 詳細コード） |
| `scheduler.node_count.allocated` | `nnumu`（割り当てノード数） |
| `resource.job.memory.max_rss_bytes` | `mmszu`（最大 RSS バイト数） |
| `resource.job.cpu.user_time_sec` | `uctmut`（ユーザー CPU 時間秒） |
| `resource.job.cpu.system_time_sec` | `sctmut`（システム CPU 時間秒） |

要求 GPU 数、リソースグループ、ジョブ ID などの静的パラメータは PJM 環境変数側に網羅されているため、別名キーへの二重登録は行わない。この終了統計は GPU サンプラーとは測定時間枠が異なり、ジョブの全ライフサイクルを網羅したリソース消費量を表す。
