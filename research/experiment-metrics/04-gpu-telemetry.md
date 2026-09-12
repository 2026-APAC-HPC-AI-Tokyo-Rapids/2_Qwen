# NVIDIA GPU 構成・実行時テレメトリ

## 対象

Genkai の Node Group B は NVIDIA H100 GPU を搭載している。ジョブへ実際に割り当てられたフル GPU または MIG（Multi-Instance GPU）インスタンスのみを計算ノード上で直接クエリし、公称スペックの静的な代入は行わない。

GPU キーのインデックスには、ジョブ内部から観測される論理インデックスを使用し、物理デバイスとの対応関係を特定できるように UUID および PCI バス ID を必ず併記して記録する。

## 静的構成

ベンチマークの開始前に、ホスト環境で `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` をシェルを介さず直接実行し、以下の静的構成情報を取得する。

```text
hardware.node_count.observed
hardware.gpu.count
hardware.gpu.count.observed
hardware.node.<node-rank>.hostname
hardware.node.<node-rank>.gpu.count
hardware.node.<node-rank>.gpu.<index>.logical_index
hardware.node.<node-rank>.gpu.<index>.uuid
hardware.node.<node-rank>.gpu.<index>.name
hardware.node.<node-rank>.gpu.<index>.pci_bus_id
hardware.node.<node-rank>.gpu.<index>.compute_capability
hardware.node.<node-rank>.gpu.<index>.memory.total_bytes
hardware.node.<node-rank>.gpu.<index>.power.limit_watt
hardware.node.<node-rank>.gpu.<index>.mig.mode.current
hardware.node.<node-rank>.gpu.<index>.driver_version
```

ドライバのバージョン差異等によりクエリ対象のフィールドが未対応である場合は、該当フィールドのみを空文字列として扱う。単位付きの CSV 文字列をそのまま保存することは避け、`nounits` で取得した数値をバイトやワットなどの標準単位へ変換して格納する。
`hardware.gpu.count.observed` には情報を取得できた全ノードの実測 GPU 数合計を記録する。`hardware.gpu.count` は、PJM で割り当てられたノード数と観測ノード数が完全に一致した場合に限り同一の値を保存し、不一致が検出された場合は空文字列とする。

MIG を使用する場合は、GPU インスタンス／コンピュートインスタンスの UUID およびプロファイル設定を独立したキー階層に記録する。フル GPU と MIG デバイスを同一の UUID 名前空間へ無差別に混在させてはならない。

## サンプリング範囲

ホスト側のステージラッパーが Singularity コンテナを起動する直前にサンプラープロセスを開始し、コンテナプロセスの終了直後にサンプラーを停止させる。計測区間には以下の全フェーズが含まれる。

```text
モデルロード → JITコンパイル／ウォームアップ → 時間計測対象の推論生成（timed generation） → Engine シャットダウン
```

サンプリング間隔の初期値は1秒とする。NVIDIA ドライバが報告する使用率（utilization）の値自体が直前のサンプリング期間における平均稼働割合であるため、過剰に高頻度なポーリングは行わない。

## 生サンプル（Raw Sample）

第1段階として、各 GPU から以下の動的指標を定期サンプリングする。

```text
timestamp
index
uuid
utilization.gpu
utilization.memory
memory.used
power.draw
temperature.gpu
clocks.sm
clocks.mem
pstate
```

これらの生サンプルはコレクタープロセスのメモリ上にのみ保持し、BigQuery や成果物ファイルへ生データのまま保存・送信することはしない。
各計算ノードは、ジョブ終了時に集約済みのスカラー統計値のみを共有ランタイムディレクトリへ一時ファイルとして書き出す。ランク0（Rank 0）が全ノード分の集約ファイルを読み取って BigQuery へ一括送信し、送信完了後速やかにこれらの一時受け渡しファイルを削除する。

## 実行レベル集約

各 GPU について、以下の集約統計量を算出して送信する。

```text
resource.stage.benchmark.node.<node-rank>.gpu.<index>.sample_count
resource.stage.benchmark.node.<node-rank>.gpu.<index>.sample_interval_sec
resource.stage.benchmark.node.<node-rank>.gpu.<index>.utilization.gpu_percent.mean
resource.stage.benchmark.node.<node-rank>.gpu.<index>.utilization.gpu_percent.max
resource.stage.benchmark.node.<node-rank>.gpu.<index>.utilization.gpu_percent.p50
resource.stage.benchmark.node.<node-rank>.gpu.<index>.utilization.gpu_percent.p95
resource.stage.benchmark.node.<node-rank>.gpu.<index>.utilization.memory_percent.mean
resource.stage.benchmark.node.<node-rank>.gpu.<index>.utilization.memory_percent.max
resource.stage.benchmark.node.<node-rank>.gpu.<index>.memory.used_bytes.mean
resource.stage.benchmark.node.<node-rank>.gpu.<index>.memory.used_bytes.max
resource.stage.benchmark.node.<node-rank>.gpu.<index>.power.draw_watt.mean
resource.stage.benchmark.node.<node-rank>.gpu.<index>.power.draw_watt.max
resource.stage.benchmark.node.<node-rank>.gpu.<index>.energy_joule
resource.stage.benchmark.node.<node-rank>.gpu.<index>.temperature_celsius.max
resource.stage.benchmark.node.<node-rank>.gpu.<index>.clock.sm_mhz.mean
resource.stage.benchmark.node.<node-rank>.gpu.<index>.clock.sm_mhz.min
resource.stage.benchmark.node.<node-rank>.gpu.<index>.clock.memory_mhz.mean
resource.stage.benchmark.node.<node-rank>.gpu.<index>.pstate.<normalized-state>.sample_count
```

消費電力量を示す `energy_joule` は、隣接する電力サンプル間のタイムスタンプ差に基づいて台形積分を行う。サンプリング欠損区間を勝手な補間で埋めることはせず、実際に有効な観測値が得られた区間のみを積算する。また、実際の有効観測時間も `resource.stage.benchmark.node.<node-rank>.gpu.<index>.sampled_duration_sec` として保存する。

パーセンタイル値の算出アルゴリズムおよび補間方法は実装コード内で一意に定義し、単体テストにおいて具体例を用いた検証を行う。

## 障害時の挙動

- `nvidia-smi` の起動に失敗した場合であっても、ベンチマーク本体の推論処理はそのまま実行する。
- 特定のクエリフィールドが `N/A` を返した場合は、0 に置き換えるのではなく空文字列のサンプルとして扱う。
- 有効なサンプルが1件も取得できなかった集約キーの値は空文字列とする。
- サンプラープロセスが異常終了した場合は標準エラー出力へ警告を記録するが、ベンチマーク本体の終了コードは変更しない。
- ベンチマークプロセス自体が異常終了した場合であっても、プロセス停止時点までに収集されたサンプルを集約して送信する。
- ノードのハードウェア障害や SIGKILL によってシャットダウン処理自体がスキップされた場合は、当該ノードの静的構成および実行時集約メトリクスが欠落する。この場合、`hardware.node_count.observed` と PJM 割り当てノード数との差異が欠損の明示的な目印となる。

## DCGMは第2段階

DCGM（Data Center GPU Manager）が計算ノード上で利用可能であり、必要な実行権限および同時取得可能なメトリクスグループが GitHub Actions 経由で確認できた場合に限り、第2段階として以下の指標を追加候補とする。

```text
resource.stage.benchmark.node.<node-rank>.gpu.<index>.engine_active_ratio.mean
resource.stage.benchmark.node.<node-rank>.gpu.<index>.tensor_active_ratio.mean
resource.stage.benchmark.node.<node-rank>.gpu.<index>.dram_active_ratio.mean
resource.stage.benchmark.node.<node-rank>.gpu.<index>.pcie.rx_bytes_per_sec.mean
resource.stage.benchmark.node.<node-rank>.gpu.<index>.pcie.tx_bytes_per_sec.mean
resource.stage.benchmark.node.<node-rank>.gpu.<index>.nvlink.bytes_per_sec.mean
resource.stage.benchmark.node.<node-rank>.gpu.<index>.xid_error_count
```

DCGM の詳細プロファイリングフィールドは、ハードウェア世代、ドライババージョン、およびハードウェアカウンタの排他競合によって同時取得できない場合がある。動作を推測した安易なフォールバック値は出力せず、未サポートの項目は空文字列とする。
なお、Nsight Systems や Nsight Compute は詳細プロファイリング成果物（artifact）を生成する別タスクの責務であり、通常実行を対象とする定常コレクターには組み込まない。
