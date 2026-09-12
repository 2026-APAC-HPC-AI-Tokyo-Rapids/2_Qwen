# Qwen / SGLang 実験メトリクス v2 キーカタログ

## 本文書の位置づけと凡例

この文書は、2026-08-10 時点の v2 計画（00-v2-plan.md）、観測マップ（01-v2-observation-map.md）、および契約スキーマ（02-v2-contract-schemas.md）に定義された収集予定値を、v1 のカテゴリ別ドット区切りキー（dotted key）規則に準拠させて整理した一覧である。v2 の JSON アーティファクト、生リレーション（raw relation）、およびコンシューマ列（consumer column）の間で名称が異なる項目については、検索・比較用の「キー」を標準名として統一し、対応する元フィールドやテーブルを「収集方法」列に記載している。

- `<phase>`、`<node>`、`<rank>`、`<gpu>`、`<name>` は、実行時に決定される実際の識別子や値へ置換する。
- `{a,b}` は、同一のプレフィックスを持つ全リーフ要素を列挙するための省略表記である。
- `[]` は、順序性を保持すべき長大リレーション（long relation）または配列要素を表し、単一のスカラー値へ圧縮してはならない。
- 「raw」はプロデューサーやコレクターが直接観測する値、「派生」はビューやリデューサーによって後から算出される値を示す。
- `schema_version`、ID、パス、argv、ハッシュ値などのリネージ・制御系フィールド（lineage/control field）は、末尾の独立した区分に分類する。
- v2 の正本ドキュメント（master docs）を内容の正とし、v1 関連文書はキー階層の命名規則を参照する目的のみに限定して使用する。
- 実行設定を一括する複合 ID や複合オブジェクトは作成しない。`nodes`、`mpi_processes`、`gpus_per_node`、テンソル並列数（TP）、リソースグループ、トランスポートポリシー、ポート番号、タイムアウト値などはそれぞれ独立したキーとして保持し、スプレッドシート上でも1キー1列として平坦に展開する。

## 1. 実行（Run）、ソース、成果物、キャッシュ

| キー | 意味 | 収集方法（コマンド／関数） | 適用・種別 |
|---|---|---|---|
| `run.identity.{run_id,run_purpose,repository_id}` | 実行 ID、実行目的、リポジトリ識別子 | ランナーが Git 管理下の run manifest から直接送信 | bench/profile、raw |
| `run.{started_at,finished_at}` | 実行の開始・終了 UTC 時刻 | スーパーバイザーのウォールクロック | 全 variant、raw |
| `run.elapsed_sec` | スーパーバイザー監視区間の経過秒数 | モノトニッククロック（monotonic clock）の差分 | 全 variant、raw |
| `run.exit_code` | 実行の終了コード | スーパーバイザーのパイプライン結果 | 全 variant、raw |
| `run.outcome` | 成否（`succeeded` / `failed`） | `run.exit_code == 0` からビュー側で導出 | 全 variant、派生（raw 保存は行わない） |
| `run.stage.<name>.launch.{argv,cwd,stderr_mode}` | 各ステージの実際の起動条件 | `StageRunner` がシェルを介さずプロセス生成（spawn）する直前に記録 | 対象 stage、raw |
| `run.stage.<name>.termination.{kind,code,signal}` | 各ステージの終了種別（`exit` / `signal` / `spawn_error`） | `StageRunner` の spawn/wait closed union。該当しない code/signal はキー欠損扱い | 対象 stage、raw |
| `perf.stage.<name>.wall_time_sec` | 各ステージの外側ウォールタイム | spawn 直前から子プロセスの待機完了までをモノトニック計時 | 対象 stage、raw |
| `collection_complete` | 必須プロデューサー・ファイルの完全性フラグ | ファイナライザが想定成果物セットと観測成果物セットを突合して判定 | bench/profile、raw bool（理由別の詳細フィールドは設けない） |
| `build.provenance` | SquashFS 本体、OCI マニフェスト、Buildah/Skopeo/Umoci の来歴 | 検証済みコンテナサイドカーおよび実ファイルから JSON を直接送信 | 全 variant、raw provenance |
| `input.model` | モデルリポジトリ、リビジョン、メタデータ | 検証済みモデルマニフェストを JSON で直接送信（重み全体の再ハッシュは行わない） | bench/profile、raw |
| `input.dataset` | データセットリポジトリ、リビジョン、パス、サイズ、SHA | 検証済みデータセットマニフェストを JSON で直接送信 | bench/profile、raw |
| `cache.node.<node>.input_manifest_sha256` | ステージング後のノードキャッシュ内容の同一性 | ノードラッパーが相対パスのバイト順にファイル・シンボリックリンク一覧を作成し、JCS SHA-256 を算出 | bench/profile、raw lineage |
| `source.track.head_commit` | 実行対象となった Qwen リポジトリのコミット | run manifest のコミットを Git オブジェクトと照合して直接送信 | 全 run、raw provenance |
| `source.container_build.qwen_commit` | SquashFS をビルドした際の Qwen コミット | コンテナサイドカー内の `source_commits.qwen` | 全 run、raw provenance |
| `source.sglang.commit` | SquashFS に含まれる SGLang ソースコミット | コンテナサイドカー内の `source_commits.sglang` | 全 run、raw provenance |

`<name>` に含まれるステージ集合は、少なくとも node launch、engine、benchmark、artifact finalize を網羅し、プロファイル実行（profile）では warmup/measured generation、GPU sampler、profile hardware、profiler capture を追加する。
generation の正常完了後は `network_imb_mpi1`、`network_nccl_all_reduce`、`network_nccl_all_gather`、`network_nccl_reduce_scatter` を追加し、それぞれ既存と同様の launch/termination/wall-time リーフを記録する。

## 2. スケジューラと実行環境

| キー | 意味 | 収集方法（コマンド／関数） | 適用・種別 |
|---|---|---|---|
| `scheduler.request.{nodes,mpi_processes,gpus_per_node,tensor_parallel_size}` | PJM、ノード、GPU、TP の個別要求値 | 投入ランナーが実行設定の各フィールドと実際の `pjsub` argv を記録・照合 | bench/profile、raw（各リーフは独立列） |
| `scheduler.request.{resource_group,transport_policy}` | リソースグループおよび通信ポリシーの個別要求値 | 同上のジョブ投入リクエスト | 同上、raw（各リーフは独立列） |
| `scheduler.request.{advertise_source,sglang_port,node_agent_port,rendezvous_timeout_sec,ready_timeout_sec}` | IP アドレス解決元、ポート、タイムアウトの個別設定値 | 実行設定から要求パラメータへ確定させて照合 | 同上、raw（各リーフは独立列） |
| `scheduler.request.subject_manifest_sha256` | ジョブ投入対象マニフェストのハッシュ | 投入前にファイルバイト列から SHA-256 を計算 | 同上、raw lineage |
| `scheduler.request.pjm_script.{repository_path,sha256}` | 使用した PJM スクリプトのパスとハッシュ | リポジトリ相対パスおよびファイルハッシュ | 同上、raw lineage |
| `scheduler.request.argv` | 実際のジョブ投入コマンド引数列 | 投入ランナーが実行した argv 配列 | 同上、raw control field |
| `scheduler.submit_job_id` | `pjsub` が返却したジョブ ID | ジョブ投入コマンド結果のパーサー | 同上、raw（欠損時はキー非存在） |
| `scheduler.job_entry_job_id` | ジョブ開始時に環境変数から観測されたジョブ ID | `PJM_JOBID` / `PJM_JOB_ID` のスナップショット | 同上、raw（欠損時はキー非存在） |
| `scheduler.accounting.{jid,st,sdt}` | PJM ジョブ ID、終了状態、開始時刻の生テキスト | `pjstat --history -S --choose jid,st,sdt,elp,ec,sn,pc,nnumu,mmszu,uctmut,sctmut --data --delimiter "\t" <job-id>` | 同上、raw |
| `scheduler.accounting.elp.{lexeme,seconds}` | 経過時間の生テキストおよび秒換算値 | 固定列パーサーによる抽出・変換 | 同上、raw nullable numeric |
| `scheduler.accounting.{ec,sn,pc,nnumu}.{lexeme,value}` | 終了コード、シグナル／ステータス、PJM コード、ノード数 | 同上の固定列パーサー | 同上、raw nullable numeric |
| `scheduler.accounting.mmszu.{lexeme,bytes}` | 最大 RSS の生テキストおよびバイト換算値 | 同上パーサー（括弧内のバイト数文字列をパース） | 同上、raw nullable numeric |
| `scheduler.accounting.{uctmut,sctmut}.{lexeme,seconds}` | ユーザー／システム CPU 時間の生テキストおよび秒換算値 | 同上パーサー（ミリ秒単位を秒へ変換） | 同上、raw nullable numeric |
| `environment.job_entry.<escaped_name>` | ジョブ起動直後の共通環境変数 | 変数名・値をバイト完全一致でスナップショット保存（変数名バイト順、値は Base64 エンコード） | bench/profile、raw dynamic family |
| `environment.runtime.<escaped_name>` | モジュールロード適用後の共通ランタイム環境変数 | `mpirun` 実行直前にランナープロセスから1回のみスナップショット取得 | bench/profile、raw dynamic family |
| `runtime.launchers` | 実際に使用された MPI / Enroot ランチャー情報 | 解決済みパス、実パス（realpath）、サイズ、SHA、バージョン出力を JSON で直接送信 | bench/profile、raw provenance |

## 3. SGLang 設定、ワーカー配置、プロセスライフサイクル

| キー | 意味 | 収集方法（コマンド／関数） | 適用・種別 |
|---|---|---|---|
| `sglang.node.<node>.bench_args_effective.<field>` | ノード上で実効化された `BenchArgs` の全 dataclass フィールド | CLI 引数パース直後、`Engine()` 初期化前の `bench_args_effective` ドメインイベント。JCS フィールド名バイト順でロスレスに展開 | bench/profile、raw dynamic family |
| `sglang.node.<node>.server_args_effective.<field>` | 起動時に確定した `ServerArgs` の全 dataclass フィールド | 自動パース、`_set_envs_and_config`、`check_server_args` 完了後、スケジューラ／デーモン起動前の `server_args_effective` イベント | 同上、raw dynamic family |
| `parallel.worker.<rank>.{tp_rank,gpu_id,gpu_uuid}` | グローバル TP ランクおよび GPU バインディング | 各ワーカープロセスの `worker_ready` イベント | bench/profile、raw |
| `parallel.worker.<rank>.{hostname,node_rank,local_rank}` | ワーカーのノードおよびランク配置 | 同上イベントおよびノードラッパーのコンテキスト | 同上、raw |
| `parallel.worker.<rank>.cpu_affinity` | ワーカープロセスの CPU アフィニティ | ワーカー内部で `os.sched_getaffinity(0)` を実行 | 同上、raw ordered set |
| `parallel.worker.<rank>.numa_mems` | ワーカープロセスの NUMA メモリポリシー | `/proc/self/status` の `Mems_allowed_list` | 同上、raw nullable |
| `parallel.worker.<rank>.runtime.{prefill_attention_backend,decode_attention_backend,torch_distributed_backend,transport_backend}` | ワーカーが実際に使用した各種バックエンド | `worker_ready` ペイロード（ServerArgs から推測しない） | 同上、raw |
| `parallel.node.<node>.{global_rank,local_rank,size,hostname}` | 外側 MPI ノード起動時の配置情報 | `node-topology/v1` の外側 MPI コンテキスト | 全 variant、raw |
| `parallel.node.<node>.{node_wrapper_argv,container_argv,sglang_argv}` | ノードラッパー、Enroot コンテナ、SGLang の実起動引数列 | ノードラッパーが起動直前に配列として記録 | 全 variant、raw control/configuration |
| `parallel.node.<node>.{cpu_affinity,numa_mems}` | ノードラッパーの CPU / NUMA バインディング | `os.sched_getaffinity(0)` および `/proc/self/status` | 全 variant、raw |
| `parallel.node.<node>.process.clock_id` | ライフサイクルタイムスタンプの基準クロック識別子 | Linux boot ID + `CLOCK_MONOTONIC` | 全 variant、raw |
| `parallel.node.<node>.process.{spawn_attempt_ns,child_started_ns,child_finished_ns}` | 子プロセスの spawn 試行、開始、終了時刻（ナノ秒） | `node_agent.py` 内の Python スーパーバイザー | 全 variant、raw（spawn 失敗時の子プロセス時刻は null） |
| `parallel.node.<node>.process.termination.kind` | プロセス終了種別（`waited` / `spawn_error`） | Python スーパーバイザーによるプロセス生成と `os.waitpid(child_pid, 0)` | 全 variant、raw |
| `parallel.node.<node>.process.termination.raw_wait_status` | POSIX 生待機ステータス | `os.waitpid()` の戻り値 | `kind=waited` の場合のみ、raw |
| `lifecycle.rendezvous.{rank0_hostname,address,port,created_at}` | ノードエージェントの制御ランデブー情報 | ランク0が `control/lifecycle/rendezvous.json` を生成 | bench/profile、raw control/configuration |
| `lifecycle.node.<node>.ready.{sequence,clock_id,received_ns,ack_ns}` | コーディネータが ready を受信・ACK した順序および時刻 | ノードエージェントのランデブーサマリー | 全 variant、raw |
| `lifecycle.node.<node>.shutdown.{sequence,invocation_id,clock_id,received_ns,ack_ns}` | シャットダウン対象の世代、順序、時刻 | 制御ストリームおよびライフサイクルサマリー | 全 variant、raw nullable |
| `lifecycle.node.<node>.termination.status` | 終了状態（`spawn_error` / `early_exit` / `early_signal` / `coordinated_exit` / `coordinated_signal`） | 生待機ステータス、子プロセス完了、シャットダウン受信状況をリデューサーで突合 | 参加ノードの長大リレーション、派生 |

## 4. ワークロード、測定結果、所要時間

| キー | 意味 | 収集方法（コマンド／関数） | 適用・種別 |
|---|---|---|---|
| `workload.<kind>.request.<ordinal>.source` | リクエストのデータセット行または生成ソース | リクエスト選定時に順序付きワークロードマニフェストへ記録 | `kind=warmup|measured`、raw |
| `workload.<kind>.request.<ordinal>.prompt.text` | 実際に生成エンジンへ渡されたテキストプロンプト | 順序付きワークロードライター | テキストプロンプト時のみ、artifact raw（BigQuery のスカラーメトリクスへ本文は展開しない） |
| `workload.<kind>.request.<ordinal>.prompt.token_ids` | 実際に渡されたトークン ID 配列 | 順序付きワークロードライター | トークン ID プロンプト時のみ、artifact raw |
| `workload.<kind>.request.<ordinal>.input_token_count` | テキストプロンプトの実測入力トークン数 | リクエスト選定時に対象と同一のトークナイザでエンコード | テキストプロンプト時のみ、raw |
| `workload.<kind>.request.<ordinal>.requested_output_token_count` | 要求出力トークン数 | 生成リクエストの入力オブジェクト | warmup/measured、raw |
| `workload.<kind>.request.<ordinal>.sampling_param.<name>` | 実サンプリングパラメータ | 生成処理へ渡されるマッピング情報をフィールド名バイト順で保存 | warmup/measured、raw dynamic family |
| `result.request.returned_count` | 検証に合格した正常レスポンス数 | プライマリリザルトライターがレスポンス配列をデコード・検証して計数 | bench/profile、raw |
| `result.request.error_count` | パースまたは検証エラーとなったレスポンス数 | 同上ライター（パース不能レスポンスもエラーに計上） | 全 result、raw |
| `result.token.input.total` | 実測入力トークン総数 | 結果アーティファクト内の `observed_input_tokens` を投影 | 確定可能な場合のみ、raw nullable |
| `result.token.output.total` | 検証済み完了トークン（completion tokens）総数 | レスポンス内の `completion_tokens` を集計（トークン ID の同一性検証可否とは独立） | 確定可能な場合、raw nullable |
| `result.token.output.ordered_token_ids_sha256` | レスポンス順序を含む出力トークン ID の同一性ハッシュ | 順序付きトークン ID 配列から JCS SHA-256 を算出 | 確定可能な場合のみ、raw lineage/validity |
| `perf.benchmark.generation.wall_time_sec` | 測定対象生成処理（measured generation）のウォールタイム | generate 呼び出しの開始・終了境界をリザルトライターが `generation_wall_sec` として記録 | bench/profile、raw |
| `perf.phase.{distributed_engine_init,input_prepare,warmup_generation,engine_shutdown}.wall_time_sec` | 各フェーズ（初期化、入力準備、ウォームアップ、シャットダウン）のウォールタイム | スーパーバイザーおよびランク0によるモノトニック計時 | 全 variant、raw scalar（profile ではタイムスタンプペアとの等価性を検証） |
| `perf.throughput.request_per_sec` | リクエストスループット | `returned_count / generation_wall_time_sec` | クエリ時にビュー側で算出（raw 保存は行わない） |
| `perf.throughput.input_token_per_sec` | 入力トークンスループット | `result.token.input.total / generation_wall_time_sec` | クエリ時にビュー側で算出（raw 保存は行わない） |
| `perf.throughput.output_token_per_sec` | 出力トークンスループット | `result.token.output.total / generation_wall_time_sec` | クエリ時にビュー側で算出（raw 保存は行わない） |
| `perf.throughput.total_token_throughput` | 合計トークンスループット | `(input + output tokens) / generation_wall_time_sec` | クエリ時にビュー側で算出 |
| `profile.phase.<invocation_id>.rank0.{clock_id,start_broadcast_ns,all_start_acks_received_ns,generation_start_ns,generation_end_ns,end_broadcast_ns,all_end_acks_received_ns}` | ランク0におけるフェーズ制御および生成境界タイムスタンプ | フェーズコントローラーサマリー | profile 専用、raw diagnostics |
| `profile.phase.<invocation_id>.node.<node>.{clock_id,start_sequence,end_sequence,start_received_ns,start_ack_ns,end_received_ns,end_ack_ns}` | ノードの GPU 観測ウィンドウ、シーケンス番号、ACK 境界 | ノード制御ストリームおよびフェーズサマリー | profile 専用、raw diagnostics |

## 5. MPI・NCCL ネットワーク性能指標

| キー | 意味 | 収集方法（コマンド／関数） | 適用・種別 |
|---|---|---|---|
| `network.imb_mpi1.process_count` | IMB MPI プロセス数 | generation 時と同一の値を引数 `-npmin` および実行セクションと照合 | bench/profile、raw scalar |
| `network.imb_mpi1.message_size.{min_bytes,max_bytes}` | IMB メッセージサイズ範囲 | 0 および 4194304 を独立したリーフに記録 | bench/profile、raw scalar |
| `network.imb_mpi1.iteration.{max_per_sample,time_limit_sec,policy}` | IMB 反復設定パラメータ | 引数の 1000、1、`dynamic` を独立したリーフに記録 | bench/profile、raw scalar |
| `software.imb_mpi1.version` | IMB-MPI1 のバージョン | 標準出力のバージョンマーカー文字列 | bench/profile、raw scalar nullable |
| `network.nccl_tests.mpi_process_count` | nccl-tests の MPI プロセス数 | generation 実行時と同一の値 | bench/profile、raw scalar |
| `network.nccl_tests.gpus_per_process` | 1プロセス当たりの GPU 数 | 引数 `-g` の指定値 | bench/profile、raw scalar |
| `network.nccl_tests.total_gpu_count` | コミュニケータ全体の総 GPU 数 | MPI プロセス数 × プロセス当たり GPU 数を実際の起動構成と照合 | bench/profile、raw scalar |
| `network.nccl_tests.message_size.{min_bytes,max_bytes,factor}` | NCCL メッセージサイズ範囲および倍率 | `-b 8 -e 1G -f 2` | bench/profile、raw scalar |
| `network.nccl_tests.{warmup_iterations,timed_iterations,check_iterations,rank_average_mode}` | ウォームアップ、計測、チェック回数、ランク平均モード | `-w 5 -n 20 -c 1 -a 1` | bench/profile、raw scalar |
| `software.nccl_tests.commit` | nccl-tests のソースコミット | コンテナ内の固定値 `717b68318278e93f371d8ffb46b076069d7c7851` | bench/profile、raw provenance |
| `software.nccl.version` | 実際にロードされた NCCL ライブラリのバージョン | ランタイムからのバージョンクエリ結果 | bench/profile、raw scalar |

`mpi_network_results_v2` は `(run_id, benchmark, process_count, message_bytes)` の粒度を持ち、
`iterations, time_min_us, time_avg_us, time_max_us, bandwidth_mbytes_per_sec, source_member` を保持する。
`nccl_network_results_v2` は `(run_id, collective, message_bytes, buffer_mode)` の粒度を持ち、
`element_count, data_type, reduction_op, mpi_process_count, gpu_count, time_us, algorithm_bandwidth_gbps, bus_bandwidth_gbps, error_count, source_member` を保持する。実際に得られた行のみを出力し、サマリーキーの事前作成は行わない。

## 6. CPU、メモリ、GPU トポロジー

| キー | 意味 | 収集方法（コマンド／関数） | 適用・種別 |
|---|---|---|---|
| `hardware.node.<node>.hostname` | ノードのホスト名 | 最小ハードウェアコレクターによる `gethostname()` | 全 variant、raw |
| `hardware.node.<node>.cpu.{architecture,model_name,vendor_id,cpu_family,model,stepping}` | CPU のハードウェア識別情報 | `lscpu_json` からの抽出・投影 | 全 variant、派生 nullable |
| `hardware.node.<node>.cpu.{physical_cores,numa_nodes}` | 物理コア数および NUMA ノード数 | `lscpu_json` からの導出 | 全 variant、派生 nullable |
| `hardware.node.<node>.kernel.release` | OS カーネルリリース番号 | `/proc/sys/kernel/osrelease` | 全 variant、raw nullable |
| `hardware.node.<node>.cpu.{online,isolated,nohz_full}` | CPU コア設定マスク | `/sys/devices/system/cpu/{online,isolated,nohz_full}` | 全 variant、raw nullable |
| `hardware.node.<node>.memory.{total_bytes,hugepage_size_bytes,hugepages_total}` | ノードのメモリ容量および HugePage 設定 | `/proc/meminfo` | 全 variant、raw nullable |
| `hardware.node.<node>.numa_balancing_enabled` | 自動 NUMA バランシングの有効化状態 | `/proc/sys/kernel/numa_balancing` | 全 variant、raw nullable |
| `hardware.node.<node>.cpu_boost.{source_path,raw,enabled}` | CPU ブースト機能の状態 | cpufreq の `boost`、または `intel_pstate/no_turbo` | 全 variant、raw nullable |
| `hardware.node.<node>.cpufreq_policy.<id>.{related_cpus,driver,governor,min_hz,max_hz}` | CPU 周波数制御ポリシー | `/sys/devices/system/cpu/cpufreq/policy*`（kHz を Hz へ換算） | 全 variant、raw nullable |
| `hardware.node.<node>.gpu.<gpu>.{index,name,uuid,pci_bus_id}` | GPU のハードウェア識別情報 | ホストおよびコンテナ内での `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` | 全 variant、raw |
| `hardware.node.<node>.gpu.<gpu>.memory.total_bytes` | GPU メモリ総容量 | 上記クエリの `memory.total` をバイトへ換算 | 全 variant、raw nullable |
| `hardware.node.<node>.gpu.<gpu>.driver_version` | NVIDIA ドライババージョン | 上記クエリ | 全 variant、raw nullable |
| `hardware.node.<node>.gpu.<gpu>.{persistence_mode,compute_mode,power_limit_watt}` | 永続化モード、コンピュートモード、電力上限（W） | 拡張 `nvidia-smi` クエリ | 全 variant、raw nullable |
| `hardware.node.<node>.gpu.<gpu>.clock.{sm_max_mhz,memory_max_mhz}` | 最大 SM クロックおよび最大メモリクロック | 拡張 `nvidia-smi` クエリ | 全 variant、raw nullable |
| `hardware.node.<node>.gpu.<gpu>.{mig_mode,ecc_mode}` | MIG モードおよび ECC モードの設定 | 拡張 `nvidia-smi` クエリ | 全 variant、raw nullable |
| `hardware.node.<node>.gpu.<gpu>.scope` | 観測スコープ（ホスト／コンテナ） | トポロジープロデューサーの実行コンテキスト | 全 variant、raw enum |
| `hardware.node.<node>.gpu_topology` | GPU、NIC、相互接続リンクのトポロジー | ホスト上での `nvidia-smi topo -m` 標準出力を構造化パース | 全 variant、raw topology relation |

## 7. プロファイル実行時の GPU サンプルと集約

| キー | 意味 | 収集方法（コマンド／関数） | 適用・種別 |
|---|---|---|---|
| `profile.gpu.node.<node>.sample.<sequence>.monotonic_ns` | サンプリング時のモノトニック時刻（ナノ秒） | ノードローカル GPU サンプラーが完全フレームごとに JSONL 追記 | profile 専用、raw |
| `profile.gpu.node.<node>.sample.<sequence>.wall_time_utc` | サンプリング時の UTC ウォールクロック時刻 | 同上サンプラーのウォールクロック | profile 専用、raw |
| `profile.gpu.node.<node>.sample.<sequence>.clock_id` | モノトニッククロックの基準識別子 | Linux boot ID + `CLOCK_MONOTONIC`（フェーズ制御ノードと完全一致が必須） | profile 専用、raw |
| `profile.gpu.node.<node>.sample.<sequence>.gpu.<gpu>.{uuid,logical_index}` | サンプルデータとトポロジーとの GPU 紐付け | 同上サンプラー（UUID に基づきトポロジーと結合） | profile 専用、raw |
| `profile.gpu.node.<node>.sample.<sequence>.gpu.<gpu>.utilization_percent` | GPU 使用率（%） | `nvidia-smi` の定期ポーリング | profile 専用、raw nullable |
| `profile.gpu.node.<node>.sample.<sequence>.gpu.<gpu>.memory.used_bytes` | GPU メモリ使用量（バイト） | 同上ポーリング結果をバイト換算 | profile 専用、raw nullable |
| `profile.gpu.node.<node>.sample.<sequence>.gpu.<gpu>.power_watt` | GPU 消費電力（W） | 同上ポーリング | profile 専用、raw nullable |
| `profile.gpu.node.<node>.sample.<sequence>.gpu.<gpu>.temperature_celsius` | GPU 温度（℃） | 同上ポーリング | profile 専用、raw nullable |
| `profile.gpu.node.<node>.sample.<sequence>.gpu.<gpu>.clock.sm_mhz` | SM クロック周波数（MHz） | 同上ポーリング | profile 専用、raw nullable |
| `resource.phase.measured_generation.node.<node>.gpu.<gpu>.duration_sec` | 測定生成区間の観測ウィンドウ長（秒） | `gpu-phase/v1` リデューサーがフェーズサマリーとサンプル区間の交差領域を算出 | profile 専用、派生 |
| `resource.phase.measured_generation.node.<node>.gpu.<gpu>.sample.valid_count` | 有効な完全サンプリングフレーム数 | 重複排除済み生サンプルリデューサー | profile 専用、派生 |
| `resource.phase.measured_generation.node.<node>.gpu.<gpu>.utilization.{valid_duration_sec,coverage_ratio,mean_percent,p95_percent}` | GPU 使用率の有効時間、カバレッジ比率、時間加重平均、95パーセンタイル値 | 区間 `[start, end)` の重なりとサンプリング間隔に基づく集約 | profile 専用、派生 nullable |
| `resource.phase.measured_generation.node.<node>.gpu.<gpu>.power.{valid_duration_sec,coverage_ratio,mean_watt}` | 消費電力の有効時間、カバレッジ比率、平均電力（W） | 重なり区間における電力サンプルの積分計算 | profile 専用、派生 nullable |
| `resource.phase.measured_generation.node.<node>.gpu.<gpu>.energy_joule` | 測定区間内の GPU 消費電力量（J） | 電力サンプルの台形積分（欠損区間の外挿補間は行わない） | profile 専用、派生 nullable |
| `resource.phase.measured_generation.node.<node>.gpu.<gpu>.memory.{valid_sample_count,max_used_bytes}` | メモリ使用量の有効サンプル数および最大使用量 | 非 null サンプルのリデューサー集約 | profile 専用、派生 nullable |
| `resource.phase.measured_generation.node.<node>.gpu.<gpu>.temperature.{valid_sample_count,max_celsius}` | GPU 温度の有効サンプル数および最高温度（℃） | 非 null サンプルのリデューサー集約 | profile 専用、派生 nullable |
| `resource.phase.measured_generation.node.<node>.gpu.<gpu>.clock.sm.{valid_sample_count,min_mhz}` | SM クロックの有効サンプル数および最低クロック（MHz） | 非 null サンプルのリデューサー集約 | profile 専用、派生 nullable |
| `resource.phase.measured_generation.gpu.{aggregate_present_count,utilization_valid_count,power_valid_count,memory_valid_count,temperature_valid_count,clock_valid_count,energy_included_count}` | GPU グリッド全体に対する有効 GPU 数 | 想定グリッド `(node, local_gpu_slot)` に個別 GPU 集約結果を外部結合して計数 | profile 専用、フリート派生 |
| `resource.phase.measured_generation.gpu.{utilization_coverage_min,power_coverage_min,valid_sample_count_min}` | フリート全体の最小カバレッジおよび最小サンプル数 | 個別 GPU 集約結果の最小値 | profile 専用、フリート派生 nullable |
| `resource.phase.measured_generation.gpu.{utilization_valid_duration_sec_min,power_valid_duration_sec_min}` | フリート全体の最小有効観測時間（秒） | 個別 GPU の有効時間の最小値 | profile 専用、フリート派生 nullable |
| `resource.phase.measured_generation.gpu.fleet_utilization_mean` | フリート全体での時間加重平均 GPU 使用率 | `sum(utilization_integral) / sum(valid_duration)`（GPU 平均値の単純算術平均は禁止） | profile 専用、フリート派生 nullable |
| `resource.phase.measured_generation.gpu.utilization_mean.{min,median,max}` | 個別 GPU の平均使用率の分布（最小／中央／最大） | 個別 GPU 集約結果の分布集計 | profile 専用、フリート派生 nullable |
| `resource.phase.measured_generation.gpu.utilization_p95.{min,median,max}` | 個別 GPU の95パーセンタイル使用率の分布 | 個別 GPU 集約結果の分布集計 | profile 専用、フリート派生 nullable |
| `resource.phase.measured_generation.gpu.energy_joule_sum` | 観測されたフリート全体の総消費電力量（J） | 有効な電力区間を持つ GPU の電力量のみを合算 | profile 専用、フリート派生 nullable |
| `resource.phase.measured_generation.gpu.{memory_max_bytes,temperature_max_celsius,sm_clock_min_mhz}` | フリート全体における最大メモリ、最高温度、最低 SM クロック | 個別 GPU 集約結果の最大／最大／最小値 | profile 専用、フリート派生 nullable |

## 8. プロファイル実行時の CPU / NUMA およびプロファイライベント

| キー | 意味 | 収集方法（コマンド／関数） | 適用・種別 |
|---|---|---|---|
| `profile.hardware.node.<node>.memory.available_bytes.postrun` | ベンチマーク終了直後の利用可能メモリ容量（MemAvailable） | プロファイルハードウェア取得モジュールが `/proc/meminfo` を読み取り | profile 専用、raw nullable |
| `profile.hardware.node.<node>.numa.<id>.cpulist` | 各 NUMA ノードに割り当てられた CPU コアリスト | `/sys/devices/system/node/node<id>/cpulist` | profile 専用、raw nullable |
| `profile.hardware.node.<node>.numa.<id>.memory.{total_bytes,free_bytes}` | NUMA ノードごとの総メモリおよび空きメモリ | ノード単位の `<id>/meminfo` パーサー | profile 専用、raw nullable |
| `profile.profiler.global_rank.<rank>.<activity>.<category>.<event_name>.call_count` | グローバル TP ランクごとのプロファイライベント呼び出し回数 | ランク別の gzip 圧縮 Chrome トレースを `(global_rank, activity, category, event_name)` でグルーピングして計数 | profile 専用、派生長大リレーション |
| `profile.profiler.global_rank.<rank>.<activity>.<category>.<event_name>.inclusive_duration_sum_ns` | グローバル TP ランクごとの累積所要時間（ナノ秒） | 同上トレースパーサーによる所要時間の合算 | profile 専用、派生長大リレーション |

## 9. コンシューマおよびビューでのみ算出する指標

| キー | 意味 | 算出ロジック |
|---|---|---|
| `validation.placement_valid` | 実行設定、スケジューラ要求、アカウンティング、トポロジー、SGLang 設定、ワーカー配置が完全に整合しているか | `qwen-placement/v1` リデューサー。ソース欠損時は NULL、不一致検出時は false |
| `validation.benchmark_valid` | 投入ワークロードと得られたプライマリリザルトが整合しているか | リクエスト数、トークン同一性、呼び出し世代、リザルトスキーマをリデューサーで照合 |
| `validation.affinity_valid` | ノードおよびワーカーのアフィニティ設定が計画ポリシーと合致しているか | ノード起動設定、ワーカーレディ情報、計画されたアフィニティポリシーを突合 |
| `eligibility.score_comparison` | ベンチマークスコアの比較対象として適格であるか | 収集完全性、配置妥当性、ベンチマーク妥当性、およびプロトコル要件をビュー側で総合評価 |
| `run.termination.benchmark.{kind,exit_code,signal}` | ベンチマークステージの終了分類 | `run.stage.benchmark.termination` の内容をコンシューマ列へ展開 |
| `run.termination.engine.{kind,exit_code,signal}` | ノード0上の推論エンジンの終了分類 | POSIX 生待機ステータスから `WIFEXITED/WEXITSTATUS/WIFSIGNALED/WTERMSIG` を用いて導出 |
| `resource.job.memory.max_rss_bytes` | ジョブ全体の最大 RSS（バイト） | `scheduler.accounting.mmszu.bytes` を投影 |
| `resource.job.cpu.{user_time_sec,system_time_sec}` | ジョブ全体のユーザー／システム CPU 時間（秒） | `uctmut.seconds` / `sctmut.seconds` を投影 |
| `scheduler.terminal.{state,exit_code}` | スケジューラの最終終了状態および終了コード | スケジューラ終了状態・アカウンティング情報を投影 |
| `artifact.{container_artifact_sha256,runtime_launcher_identity}` | SquashFS イメージのハッシュおよび実ランチャーの識別情報 | コンテナサイドカーおよびランタイムランチャーの検証結果 |
| `artifact.{model_manifest_identity,dataset_manifest_identity}` | モデルおよびデータセットのマニフェスト識別子 | 親ジョブが管理するマニフェスト情報を子ジョブ実行へ結合 |
| `workload.{measured_manifest_identity,warmup_manifest_identity}` | 測定用およびウォームアップ用ワークロードの識別子 | ワークロードマニフェストからの投影 |
| `cache.manifest_identity` | ノード順序を保持したキャッシュ同一性の集合 | `[{node_rank, manifest_sha256}, ...]` を `cache_nodes/v1` で集約（再ハッシュは行わない） |
| `workload.measured.{selected_count,planned_input_tokens,requested_output_tokens}` | 測定ワークロードのリクエスト総数、予定入力トークン数、要求出力トークン数 | 順序付きワークロードリクエストリレーションを集計（成功リクエスト数とは区別） |
| `workload.warmup.mode` | ウォームアップワークロードの構成モード | ウォームアップマニフェストの種別・リクエスト形状から投影 |
| `result.request.missing_response_count` | 予定されたリクエストのうちレスポンスが得られなかった欠損件数 | `measured_selected_count - returned_count - error_count` を妥当性リデューサーで算出 |
| `execution.{nodes,mpi_processes,gpus_per_node,tensor_parallel_size}` | 予定されたノード数、MPI プロセス数、GPU 数、TP 数 | 実行設定とスケジューラ要求の一致確認済み投影値（各リーフを独立列に展開） |
| `execution.{resource_group,transport_policy,advertise_source}` | リソースグループ、トランスポートポリシー、アドレス解決元 | 同上（各リーフを独立列に展開） |
| `execution.{sglang_port,node_agent_port,rendezvous_timeout_sec,ready_timeout_sec}` | ポート番号およびタイムアウト設定値 | 同上（各リーフを独立列に展開） |
| `parallel.observed.{node_count,distinct_host_count,mpi_process_count,global_rank_count}` | 実際に観測されたノード数、ホスト数、MPI プロセス数、グローバルランク数 | ノード起動情報およびワーカーレディリレーションから計数 |
| `hardware.gpu.{observed_record_count,unique_uuid_count}` | 観測された GPU トポロジーレコード数および一意な GPU UUID 数 | コンテナトポロジーリレーションから計数 |
| `hardware.gpu.per_node.{min,max}` | ノード当たりに観測された GPU 数の最小／最大値 | コンテナトポロジーをノード単位に計数 |
| `parallel.effective.{tp,nnodes}` | SGLang の実効テンソル並列数（TP）およびノード数 | ノード設定の合意値（consensus）である `tp_size` / `nnodes` |
| `sglang.config.{benchmark_args_identity,server_args_identity}` | 全フィールドを含む BenchArgs / ServerArgs のノード間合意識別子 | 設定スナップショットの JCS 識別子およびノード間合意の検証 |
| `sglang.runtime.{backend_consensus,transport_consensus}` | ワーカーのバックエンドおよびトランスポートの合意値 | ワーカーレディイベントの4つのバックエンドフィールドを集約判定 |
| `hardware.consensus.gpu.{model,memory_bytes,driver,power_limit_watt,max_sm_clock_mhz,compute_mode,mig_mode}` | GPU ハードウェアおよび設定のノード／GPU 間合意値 | トポロジー行を合意リデューサーで集約（欠損や不一致検出時は NULL） |
| `hardware.consensus.cpu.{model,kernel_release,numa_node_count,memory_bytes,governor,boost_enabled}` | CPU、カーネル、メモリ、周波数制御設定のノード間合意値 | 最小ハードウェア行を合意リデューサーで集約（欠損や不一致検出時は NULL） |
| `profile.numa.cpu_layout_identity` | プロファイル NUMA CPU 配置の同一性識別子 | NUMA cpulist リレーションを正規化 JSON（canonical JSON）化 |
| `profile.hardware.memory.available_bytes.postrun.min` | ベンチマーク終了後におけるノード間の最小 MemAvailable 容量 | ノード別サマリーの最小値 |
| `hardware.gpu.binding_valid` | ワーカーの GPU バインディングとトポロジーとの完全整合性 | ワーカー UUID / インデックス、トポロジー、プロトコル上の GPU 数を完全突合 |

## 10. リネージ・制御系フィールドおよび収集対象外項目

| 区分 | 対象フィールド／キー | 取り扱い方針 |
|---|---|---|
| リネージ／制御 | `schema_version`、`run_id`、`invocation_id`、レコード OID/ハッシュ、ソースファイル名/パス/ハッシュ、argv、stdout/stderr、バンドルマニフェスト | 成果物の検証・再現性確保・テーブル結合のために保持するが、科学的な実験メトリクスのリーフキーとしては計上しない |
| 収集対象外 | ネットワーク用 `lane`、`qualification*`、`block_id`、`block_order`、`replicate_index`、`pair_id`、契約・プロトコル ID / ハッシュ | 各実行を完全に独立した試行として扱い、通信設定は独立したスカラーキーへ展開する |
| 収集対象外 | MPI / NCCL サマリー行、`ib_*_bw`、`ib_*_lat` | メッセージサイズ別の長大リレーションからクエリ実行時に必要に応じて導出する |
| アーティファクト限定 | プロンプト本文、プロンプトトークン ID 配列、順序付きワークロードリクエスト全体 | 再現性確保用アーティファクトファイルにのみ保持し、BigQuery のスカラーメトリクスへテキスト本文は展開しない |
| 収集対象外 | `result.benchmark.successful_requests` | v2 では検証済みレスポンス数（`result.request.returned_count`）およびエラー数（`error_count`）に厳密に置き換える |
| 収集対象外 | 生スループット値、`last_gen_throughput` | 基本計数値と生成所要時間からビュー側で厳密に計算するため、生値としての永続化は行わない |
| 収集対象外 | ベンチマークバックエンドの生リザルトフィールド、予定リクエスト数を成功数として流用するフィールド | バックエンド情報はワーカーランタイム設定が管理し、予定件数はワークロード側で独立管理する |
| 収集対象外 | 生成テキスト本文、リクエストごとの出力ハッシュ値、生プロンプトの BigQuery 行 | 検索テーブルへ巨大なテキスト本文を複製しない。出力の同一性検証に必要な順序付きトークン ID のハッシュのみを結果成果物に記録する |
| 収集対象外 | フィールドごとの missing/reason/status フラグ、プレースホルダー値、再試行カウンター、カバレッジ判定フラグ | 欠損値は生キーの非存在（absent）として扱い、固定幅テーブルでは NULL を格納する。実行全体の成否は `collection_complete` で一括管理する |
| 収集対象外 | 通常ベンチマーク実行（bench）におけるダミーの GPU サンプルやプロファイル集約プレースホルダー | GPU テレメトリは profile 専用とする。非該当の実行では行を作成せず、コンシューマ列には NULL を設定する |
| 収集対象外 | 参加ノード終了ステータスの配列・集約・ワースト値 | `(run, node_rank)` の長大リレーションのみを出力する |
| 第2段階候補 | DCGM のエンジン／Tensor Core／DRAM 稼働率、PCIe / NVLink 帯域幅、XID エラーカウンター | GitHub Actions 上での機能検証（capability probe）で動作を確認できた場合に限り追加する。v2 の必須要件には含めない |
| 動的ファミリー | `sglang.node.<node>.{bench_args_effective,server_args_effective}.<field>`、`profile.profiler.global_rank.<rank>.<activity>.<category>.<event_name>.*` | リーフ構造は定義済み。引数フィールド名は固定版の SGLang 定義に従い、プロファイライベント名はトレース内の文字列を単射エスケープして使用する |
| アーティファクト限定 | `hardware.node.<node>.lscpu_json` | ハードウェア検証用アーティファクトにのみ生 JSON を保持し、クエリ可能なドット区切りメトリクスとしては主要な CPU・設定項目のみを抽出する |

## 11. 実装状況

本カタログは「v2 において収集を予定している契約仕様」を定めたものであり、現行の `qwen-experiment-metrics` パッケージにおける実装完了済み一覧を示すものではない。現行の単体テスト（18件）はすべて通過しているものの、その実装内容は主に v1 相当の即時スカラー送信にとどまっている。v2 の本格運用に不可欠なランナーマニフェスト、スコープ別環境変数、ノード／ランク単位のイベント、検証済みプライマリリザルト、生のプロファイル用 GPU サンプル、フェーズリデューサ、プロファイル用ハードウェア／プロファイラ連携、ノードエージェントのライフサイクル管理、ファイナライザ／パッケージング、およびランナー限定パブリッシャなどの各機能は未実装である。

## 参照元

- `research/experiment-metrics/00-v2-plan.md`
- `research/experiment-metrics/01-v2-observation-map.md`
- `research/experiment-metrics/02-v2-contract-schemas.md`
- `research/experiment-metrics.md` および `research/experiment-metrics/01-*.md`〜`07-*.md`（キー階層規則の参照元）
