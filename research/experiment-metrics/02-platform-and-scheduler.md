# host platform・環境変数・PJM

## host platform

次を取得する。

| キー | 取得元 |
|---|---|
| `system.cluster` | fallback定数`genkai` |
| `system.os.id` | `/etc/os-release` |
| `system.os.version_id` | `/etc/os-release` |
| `system.kernel.release` | `os.uname()` |
| `system.architecture` | `os.uname()`、対応envがない場合のみ |
| `hardware.cpu.model` | `lscpu --json` |
| `hardware.cpu.socket_count` | `lscpu --json` |
| `hardware.cpu.cores_per_socket` | `lscpu --json` |
| `hardware.cpu.threads_per_core` | `lscpu --json` |
| `hardware.cpu.available_logical_count` | process affinity |
| `hardware.cpu.numa_node_count` | `lscpu --json` |
| `hardware.memory.total_bytes` | `/proc/meminfo` |

CPUの公称値は埋め込まない。GPU benchmarkでもtokenization、dataset処理、process管理に
CPUを使うため、CPU model、affinity、NUMA構成とPJMのCPU accountingは残す。
瞬間的なCPU utilizationのsamplingは第1段階では行わない。

## 環境変数

allowlist、denylist、秘密情報filterは設けない。

BigQuery送信用の次の変数だけは実験条件ではないため、CLI起動直後にメモリへ移して
環境から削除する。

```text
EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_JSON
EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_BASE64
```

SGLang processはSingularityの`--cleanenv`内にあり、hostとは異なる環境を持つ。
SGLang instrumentation eventからその全件を次へ保存する。

```text
sglang.env.<name>
sglang.env.encoded.<name-bytes-hex>
sglang.env.bytes.<name-token>
```

これにより、host/PJM環境と実際のbenchmark process環境を混同しない。

## PJM終了統計

runnerがPJM終了直後に送る。

| キー | PJM項目 |
|---|---|
| `scheduler.state` | `st` |
| `scheduler.started_at` | `sdt` |
| `scheduler.elapsed_sec` | `elp` |
| `scheduler.exit_code` | `ec` |
| `scheduler.signal` | `sn` |
| `scheduler.pjm_code` | `pc` |
| `scheduler.node_count.allocated` | `nnumu` |
| `resource.job.memory.max_rss_bytes` | `mmszu` |
| `resource.job.cpu.user_time_sec` | `uctmut` |
| `resource.job.cpu.system_time_sec` | `sctmut` |

要求GPU数、resource group、job ID等はPJM環境変数に含まれるため別名へ複製しない。
終了統計はGPU samplerと範囲が異なり、job全体を表す。
