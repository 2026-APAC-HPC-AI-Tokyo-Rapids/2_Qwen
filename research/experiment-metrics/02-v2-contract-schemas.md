# Qwen/SGLang v2 コントラクトスキーマ仕様書

## 役割

本書は、[概要計画](./00-v2-plan.md)および[観測マップ](./01-v2-observation-map.md)の設計内容を `qwen-v2.toml` および JSON Schema へ転記するための、フィールドレベルの正本仕様書である。値のソース、フィールド名、データ型、出現順序、リデューサ（reducer）の意味論について、実装者が独自の解釈を追加してはならない。生の値（raw value）を別名で複製せず、ソースアーティファクト、`run`/`key`/`value` ファクト、およびクエリ実行時コンシューマの対応関係のみを厳密に定義する。

本プロジェクトの障害モデル（failure model）は単純明快に維持する。取得できなかった値や実行バリアントにおいて非適用となる値は、コンシューマ側で `NULL` として扱い、フィールド別の理由（reason）・ステータス・欠損記録用プレースホルダは設けない。実行レベルの状態管理は `collection_complete` のみとする。

以下に定義する closed object は、特記のない限り全フィールドが必須（required）である。フィールド名末尾の `?` は省略可能（optional、非該当時フィールド自体が存在しない（absent））を示し、値型の `|null` のみが null 許容（nullable）を示す。optional と nullable を同一の記号で混同して表現してはならない。
全オブジェクトにおいて `additionalProperties=false` を原則とする。integer は 64 ビット符号付き整数範囲、`uint` は非負整数、decimal は有限確定値、hash は小文字 64 桁の 16 進数 SHA-256、array は記載された要素順序をセマンティクスとして保持する。

## 正準識別子（Canonical Identity）

| 項目 | v1 定義 |
|---|---|
| リポジトリ ID | 固定文字列 `apac2026-qwen`。ランタイムの Git リモート URL は識別子として使用しない |
| 実行 ID（run ID） | `r-` ＋ OS の暗号論的擬似乱数生成器（CSPRNG）による 16 バイト値の小文字 16 進数 32 文字 |
| JSON 正準バイト列 | RFC 8785 (JCS) 準拠、UTF-8 エンコーディング。重複キー、`NaN`、`Infinity` は禁止 |
| コミット済み TOML 識別子 | チェックアウトされたファイルの実バイト列の SHA-256 ハッシュ |
| Git オブジェクト識別子 | `{object_hash_algorithm: enum "sha1"|"sha256", oid}` |
| アーティファクト内パス | アーカイブのルートからの相対 POSIX パス。絶対パス、`..`、空コンポーネントは禁止 |
| メンバー参照 | closed な `{path, size_bytes, sha256}` |
| 環境変数スナップショット | 変数名のバイト昇順に並べた `name\0value\0` の連続形式（終端も NUL） |

マニフェスト自身のハッシュ値はマニフェスト本文内には含めず、参照元または外側のマニフェスト（outer manifest）に保持する。

## source track v1

`control/run-manifest.json` には、全フィールド必須かつ `additionalProperties=false` の `source_track` を埋め込む。

```text
schema_version: const "source-track/v1"
object_hash_algorithm: enum "sha1"|"sha256"
base_role: enum "score_reference"|"profile_base"
base_commit: algorithm-constrained full OID
head_commit: algorithm-constrained full OID
```

bench（ベンチマーク）では `score_reference`、profile（プロファイル）では `profile_base` を選択する。バリデータは、同一のリポジトリ／オブジェクトアルゴリズム、クリーンなチェックアウト状態、完全長 OID の実在性、および `base` が `head` の先祖コミット（ancestor）であることをジョブ投入前に検証する。ブランチ名は識別子として使用しない。
プロジェクション（projection）は、run ID、実行目的（purpose）、リポジトリ識別子、base/head コミット、マニフェスト／メンバーハッシュを一度だけ公開する。

## 実行マニフェストと実行設定

`control/run-manifest.json` は通常実行のみを表現し、以下のフィールドを保持する。

```text
schema_version: const "run-manifest/v1"
run_id: run ID
run_purpose: enum "bench"|"profile"
repository_id: const "apac2026-qwen"
source_track: source-track/v1
observed_control_plane_sha256
instrumentation_bundle_sha256
```

ノード数、MPI プロセス数、GPU/ノード数、テンソル並列数（TP）、リソースグループ、転送方式、アドレスソース、ポート番号などの実行設定は、実行設定ドキュメントの独立した個別フィールドであり、それらを単一の ID やオブジェクトに集約してはならない。

実行設定ドキュメントにおける `run_purpose` は `bench | profile` のいずれかのみである。共通セクションには、ソース参照、個別実行設定、ワークロード、アーティファクトポリシー、バンドルパス／ハッシュを保持する。profile の場合はさらにプロファイラ設定を保持する。用途に合致しないフィールドは存在させてはならない（absent）。

実際の制御プレーンハッシュ（observed_control_plane_sha256）は、ソーストラック、バンドル、および実行設定ドキュメントが列挙する制御ファイル群の実際の BLOB OID から計算する。設定ドキュメントに記載された期待値を観測値として転記してはならない。

## インスツルメンテーションバンドル

バンドルは、測定目的に中立な意味的定義（semantic definition）であり、フック、イベント、サンプル、ラッパー、ノードエージェント、環境変数スコープ、ソースパスの許可リスト、argv テンプレート、シンク先ソース ID を closed なスキーマとして保持する。
ランナーは `score -> bench`、`profile -> profile` の固定マッピングによって 1 つのバンドルを選択し、マニフェストにバンドルの SHA-256 ハッシュを記録する。
実際の argv およびカレントワーキングディレクトリ（cwd）は StageRunner ファクトが、シンクパステンプレートはソースレジストリが、実際のバイト列を含むファイル識別子はアーティファクトがそれぞれ管理・所有する。

experiment-metrics-v2 の実装中に実施する「計測なし／あり」の比較測定は、一時的な非回帰（非擾乱性）確認である。検証完了後に測定結果は直ちに破棄し、専用の実行目的、アーム（arm）、ペア、割り当て、マニフェスト、アーティファクト、メトリクスキー、リレーション、スプレッドシート列、適格性判定、意思決定レコードは一切定義しない。

## container sidecar v1

`source/container-sidecar.json` は、Buildah、Skopeo、Umoci によるコンテナビルド完了後に生成される closed object である。

```text
schema_version: const "container-sidecar/v1"
container_artifact: {artifact_relative_path: const "sglang.sqsh", size_bytes: uint, sha256: sha256}
oci_manifest: {sha256: sha256}
source_commits: {qwen: full Git OID, sglang: full Git OID}
toolchain: {buildah: nonempty version string, skopeo: nonempty version string, umoci: nonempty version string}
```

プロビジョニング時には、実際に起動する SquashFS バイト列のサイズと SHA-256 ハッシュをサイドカーと照合する。検証済みの値は `build.provenance`、`source.container_build.qwen_commit`、`source.sglang.commit` として実行ファクトに記録する。
モデルおよびデータセットは、それぞれ `source/model-metadata.json` と `source/dataset-manifest.json` を正本とし、重みなしダミーロードやランダムデータセットの実引数を確定させる。ワークフローの実行／試行情報は、運用リレーション（operational relation）のみが管理する。

## ランタイムランチャー検証 v1

実行バンドルは `source/runtime-launchers.json` を一意に所有する。closed なソースレコードは以下の構造を持つ。

```text
{
  schema_version: const "runtime-launcher-validation/v1",
  executables: exactly 2 [
    {role: enum "mpirun"|"enroot", requested_path: string, resolved_realpath: absolute POSIX path, size_bytes: uint, sha256: sha256}
  ],
  versions: exactly 2 [
    {role: enum "mpirun"|"enroot", argv: exact [same-role requested path, "--version"|"version"], outcome: {exit_code: uint}, stdout: member, stderr: member}
  ]
}
```

両配列は定義されたロール順序を維持し、各ロールは厳密に 1 回ずつ出現する。外側の MPI ステージおよびノードラッパーは、この解決された絶対パスを使用し、PATH 検索に依存してはならない。

パブリッシャー側のプロジェクションは、全フィールド必須の以下のスキーマを持つ。

```text
{
  schema_version: const "runtime-launcher-projection/v1",
  run_id: string,
  executables: [{role, requested_path, resolved_realpath, size_bytes, sha256}, ...],
  versions: [
    {role, argv, outcome, stdout_member: relative POSIX path, stdout_b64: base64, stderr_member: relative POSIX path, stderr_b64: base64}, ...
  ],
  source_member: relative POSIX path,
  source_sha256: sha256
}
```

`runtime-launcher-semantic/v1` は、パスやメンバーロケータを除いた順序付きオブジェクトとして以下のように定義される。

```text
{
  schema_version: const "runtime-launcher-semantic/v1",
  launchers: [{role, sha256, outcome, stdout_b64, stderr_b64}, ...]
}
```

コンシューマ側の識別子は `SHA-256(JCS(object))` とする。

## モデル、データセット、キャッシュ

### モデルマニフェスト v1

`source/model-manifest.json` の `source` は closed な直和型（tagged union）である。

- `{kind="hub_revision", repository: string, revision: full immutable hex commit}`
- `{kind="prebuilt_manifest", shards: [{path: relative path, size_bytes: uint, sha256}, ...]}`
  shards はパスのバイト昇順でソートし、必要となる意味的識別子はビュー側で `SHA-256(JCS(shards))` として動的に導出する。同レコード内に冗長に複製してはならない。

実行レコードは、上記の source に加え、以下の小規模メタデータ許可リスト（small metadata allowlist）に合致する実在メンバーのみをパスのバイト昇順で保持する。

```text
config.json
model.safetensors.index.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
generation_config.json
chat_template.json（または tokenizer_config 内の chat template ソースレコード）
```

トップレベルは `schema_version="model-manifest/v1"`, `source`, `small_metadata` から構成され、各メタデータメンバーは `role=configuration|index|tokenizer|tokenizer_config|special_tokens|generation_config|chat_template`、`path`, `size_bytes`, `sha256` を保持する。index ファイルが参照するシャード集合は、index をパースして事前ビルドマニフェストと照合するが、実行マニフェストへは再掲しない。1 回の実行ごとに約 400 GB に達するシャード本文の再ハッシュ計算を行ってはならない。

### データセットマニフェスト v1

`source/dataset-manifest.json` は、`schema_version="dataset-manifest/v1"`, `repository: string`, `full_revision: string`, `relative_path: relative path`, `size_bytes: uint`, `sha256: sha256` を保持する。
ダウンロードコマンドはこれらの情報から完全に再現できるため保存しない。変動する `main` ブランチ参照や共有ファイルシステムの絶対パスを識別子として用いてはならない。

### キャッシュマニフェスト v1

`source/cache/node-<node-rank>-input-manifest.json` は、`schema_version="cache-manifest/v1"`, `run_id`, `node_rank: uint`, `members` を保持し、キャッシュルートからの相対パスのバイト昇順でメンバーを列挙する。
通常ファイルは `{type="file", path, size_bytes: uint, sha256}`、シンボリックリンクは `{type="symlink", path, target_b64: base64}` を保持する。
更新時刻（mtime）、UID、GID、ステージング先の絶対パスは識別子に含めない。ファクトの識別子は、個別実行のバインディングを除外した `{schema_version: "cache-members/v1", members: <ソート済みメンバー一覧>}` の JCS SHA-256 とし、キャッシュ内容が同一であれば実行を跨いでもハッシュ値が完全に一致するようにする。
ソースファイルのバイトハッシュはアーティファクトの来歴追跡のみに使用する。意味的識別子のみを `cache.node.<node_rank>.input_manifest_sha256` としてインデックス化し、ファイル数、合計サイズ、コールド／ウォームの判定ラベルを生テーブルへ重複記録してはならない。

## 環境変数、トポロジ、ランタイム設定

すべての `.snapshot` ファイルは、変数名のバイト昇順に整列された `name\0value\0` の連続形式（終端も NUL）とする。パーサは変数名および値を一切変更せず、BigQuery 上では変数名を単射エスケープし、値を RFC 4648 Base64 文字列として格納する。スコープは `job_entry | host_exec_node | sglang_launch_node | worker_global_rank` のいずれかであり、完全なスナップショット（full snapshot）を保持する。共通部と差分（common/delta）に分割するような第 2 の生表現は設けない。

最小 CPU ハードウェア情報は、ノードラッパーが各ノード上で一度だけ `LC_ALL=C lscpu --json` を実行して採取する。`source/hardware/node-<node-rank>.json` は `hardware-node/v1` に準拠し、以下の closed object とする。

```text
schema_version: const "hardware-node/v1"
node_rank: uint
hostname: string
commands: {
  lscpu: {argv: ["lscpu", "--json"], exit_code: int, stderr_member: relative POSIX path}
}
lscpu_json: object|null
platform_settings: {
  kernel_release: string|null,
  online_cpus: uint[]|null,
  isolated_cpus: uint[]|null,
  nohz_full_cpus: uint[]|null,
  memory: {total_bytes: uint|null, hugepage_size_bytes: uint|null, hugepages_total: uint|null},
  numa_balancing_enabled: bool|null,
  boost: {source_path: string|null, raw: string|null, enabled: bool|null},
  cpufreq_policies: [
    {policy_id: uint, related_cpus: uint[], driver: string|null, governor: string|null, min_hz: uint|null, max_hz: uint|null},
    ...
  ]|null
}
```

同一のソースレコード内へ `lscpu` から導出される CPU 識別情報を重複記録してはならない。
カーネルバージョンは `/proc/sys/kernel/osrelease`、CPU 集合は `/sys/devices/system/cpu/{online,isolated,nohz_full}`、メモリおよび HugePage 情報は `/proc/meminfo` を読み取り、単位 kB を 1024 倍してバイト換算する。
NUMA バランシングは `/proc/sys/kernel/numa_balancing` から取得し、boost 状態は cpufreq の `boost` を優先参照し、存在しない場合は intel_pstate の `no_turbo` を論理反転して取得する。cpufreq 設定は数値ポリシー ID 昇順に関連 CPU、ドライバ、ガバナ、最小／最大周波数を読み取り、単位 kHz を 1000 倍して Hz 換算する。
CPU リストファイルが存在して空の場合は `[]`、インターフェースが存在しないか読み取り不能な場合は null とする。また cpufreq ルートディレクトリが存在してポリシーが 0 件の場合は `[]`、ルート自体が存在しないか列挙不能な場合は null とする。BigQuery 上ではノードランク昇順に投影し、ノード間の合意（consensus）はビュー側で動的に導出する。
cpufreq の `related_cpus` は ASCII 空白区切りの非負整数列を狭義単調増加な一意配列へ変換し、`nohz_full` については空白トリム後の文字列リテラル `(null)` も `[]` へ正準化する。

`minimal-hardware-projection/v1` は、全フィールド必須の以下のスキーマとする。

```text
{
  schema_version: const "minimal-hardware-projection/v1",
  run_id: string,
  node_rank: uint,
  hostname: string,
  cpu_identity: {
    architecture: string,
    model_name: string,
    vendor_id: string,
    cpu_family: string,
    model: string,
    stepping: string|null,
    physical_cores: uint|null,
    numa_nodes: uint|null
  },
  platform_settings: <hardware-node/v1 の platform_settings>,
  source_member: relative POSIX path,
  source_sha256: sha256
}
```

CPU 識別情報は、`lscpu` の出力ラベル（`Architecture`, `Model name`, `Vendor ID`, `CPU family`, `Model`, `Stepping`, `Socket(s)`, `Core(s) per socket`, `NUMA node(s)`）の末尾コロンを厳密に 1 つ除去した上で照合し、物理コア数は「ソケット数 × ソケットあたりコア数」として算出する。

`source/topology/node-<node-rank>.json` は、以下の closed なソースレコードを保持する。

- `schema_version="node-topology/v1"`, `run_id`, `node_rank: uint`、外側 MPI の `global_rank: uint`, `local_rank: uint`, `size: uint`, `hostname: string`
- ホスト側およびコンテナ側それぞれの厳密な `nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version,persistence_mode,compute_mode,power.limit,clocks.max.sm,clocks.max.memory,mig.mode.current,ecc.mode.current --format=csv,noheader,nounits` コマンド実行記録（`{argv: string[], exit_code: int, stdout_member, stdout_size_bytes: uint, stdout_sha256: sha256, stderr_member, stderr_size_bytes: uint, stderr_sha256: sha256}`）
- ホスト側の `nvidia-smi topo -m` に関する同一形式のハッシュ封印コマンド記録
- ノードごとの実際の `node_wrapper_argv`, `container_argv`, `sglang_argv` の文字列配列、ホスト側 CPU アフィニティの uint 配列、NUMA mems 文字列

GPU 配列は、生の標準出力から `topology/v1` リデューサが一度だけパースし、ノードランク／GPU インデックス昇順の `{node_rank, index, name, uuid, pci_bus_id, memory_total_bytes, driver_version, persistence_mode, compute_mode, power_limit_w, max_sm_clock_mhz, max_memory_clock_mhz, mig_mode, ecc_mode, scope=host|container}` として BigQuery へ投影する。ソース JSON 内にパース済みコピーを保持せず、ノード argv や `hardware.gpu.nodes` などの第 2 の生ファクトも新設しない。
`node-launch-projection/v1` は、`schema_version: const "node-launch-projection/v1"`, `run_id`, `node_rank`, `global_rank`, `local_rank`, `size`, `hostname`, `node_wrapper_argv`, `container_argv`, `sglang_argv`, `cpu_affinity`, `numa_mems`, `source_member`, `source_sha256` を保持する。
`topology-projection/v1` は、上記の GPU レコードに `schema_version: const "topology-projection/v1"`, `run_id`, `source_member`, `source_sha256` を付加した構造とする。

BenchArgs および ServerArgs のイベントペイロードは `config-snapshot/v1` スキーマとし、`config_type=BenchArgs|ServerArgs`、`definition={source_path: repository-relative POSIX path, class_name: nonempty string, source_blob_oid: algorithm-constrained full OID}`、およびフィールド名のバイト昇順にソートされた `fields=[{name, declared_type, value}, ...]` を保持する。`value` は実際の dataclass 値を JCS 表現したものであり、すべてのフィールドを欠落なく保存する。この closed なメタ構造により、SGLang のコミット進行に伴うフィールド増減に柔軟に対応でき、手作業の許可リスト保守を不要にする。横持ちの Runs テーブル列は、コントラクトで指定されたスカラーリーフ値をクエリ実行時に投影する。イベント種別はそれぞれ `bench_args_effective`、`server_args_effective` に固定する。
BLOB OID のアルゴリズムは、SIF のプロベナンスにおける `source_commits[commit_role="sglang_source"]` から継承し、該当コミットのツリー下で `source_path` が指す BLOB と厳密に一致させる。SGLang のコミットハッシュはビルドプロジェクションのみが所有し、イベントペイロードへ重複記録してはならない。

ワーカー準備完了（worker-ready）エンベロープは、以下の closed フィールドを保持する。

```text
schema_version: const "qwen-domain-event/v1"
run_id
producer_id
node_rank
global_rank
sequence
event_type: "worker_ready"
payload: {
  tp_rank,
  gpu_id,
  gpu_uuid,
  cpu_affinity,
  numa_mems,
  prefill_attention_backend,
  decode_attention_backend,
  torch_distributed_backend,
  transport_backend
}
```

期待されるグローバルランクは `0..protocol.tensor_parallel_size-1`、ノードランクは `0..protocol.nodes-1` である。これらの値を ServerArgs から安易に推測してはならない。

`qwen-placement/v1` リデューサは、新たな生テーブルや理由フィールドを作成せず、プロトコルの個別実行設定、スケジューラ要求、全ノード起動情報、コンテナ GPU トポロジを一度だけ照合し、bench/profile の各ノードにおける `server_args_effective` および全 worker-ready イベントを追加照合する。
スケジューラ要求のノード数、MPI プロセス数、リソースグループ、転送ポリシーは、プロトコルの各フィールドと厳密に一致しなければならない。ノード起動情報では、ノードランクと外側 MPI のグローバルランクがともに厳密に `0..protocol.nodes-1` であり、各レコードにおいて `global_rank=node_rank`, `local_rank=0`, `size=protocol.mpi_processes` かつホスト名が全ノード間で重複なく一意（distinct）でなければならない。
共通のコンテナトポロジは、期待される各ノードにおいて厳密に `protocol.gpus_per_node` 件とインデックス集合 `0..protocol.gpus_per_node-1` を持ち、全ノード全体で厳密に `protocol.tensor_parallel_size` 件のグローバルに一意な UUID を持たなければならない。
各 ServerArgs は `tp_size`, `nnodes`, `node_rank` がプロトコルの TP、ノード数、当該ノードランクとそれぞれ一致し、分散アドレスはプロトコルの rank 0 アドバタイズ／ポートコントラクトと一致しなければならない。
worker エンベロープは、グローバルランクと `payload.tp_rank` がともに厳密に `0..protocol.tensor_parallel_size-1` で 1 対 1 に対応し、各ノードに厳密に `protocol.gpus_per_node` 件が存在しなければならない。各ワーカーの `gpu_id`/`gpu_uuid` は同一ノードのコンテナトポロジ `(index, uuid)` と全単射（bijection）をなし、全ノードのワーカー UUID も一意でなければならない。
必要なソースが欠落している場合、既存の `placement_valid` は NULL とし、すべてのソースが揃っているものの 1 項目でも不一致がある場合は false、それ以外の場合のみ true とする。`benchmark_valid` は測定結果の整合性のみを表す既存の定義を厳守する。bench における `score_comparison_eligible` のみが `placement_valid=true` を必須要件とする。profile では検証結果を表示するもののスコア統計には算入せず、原因列や検証専用アーティファクトは作成しない。

## イベント、ライフサイクル、フェーズ

すべてのドメインイベント（domain event）に共通するエンベロープは、`schema_version: const "qwen-domain-event/v1"`, `run_id`, `producer_id`, `node_rank`, `global_rank?`, `sequence`, `event_type`, `payload` である。
同一 producer 内の `sequence` は 0 から連続的にインクリメントする。クロック ID は Linux のブート ID とクロック名を組み合わせた `<boot_id>:CLOCK_MONOTONIC` 形式とし、別ノードの単調増加クロック値との間で直接の差分計算を行ってはならない。

`control/lifecycle/rendezvous.json` は `schema_version: const "lifecycle-rendezvous/v1"`, `run_id`, `rank0_hostname`, `address`, `port=protocol.node_agent_port`, `created_at` を保持し、サマリファイル `control/lifecycle/summary.json` は `schema_version="lifecycle-summary/v1"`, `run_id`, `nodes` を保持する。各ノードエントリは `node_rank: uint`, `ready: {sequence: uint, clock_id: string, received_monotonic_ns: uint, ack_monotonic_ns: uint}`, `shutdown: {sequence: uint, invocation_id: string, clock_id: string, received_monotonic_ns: uint, ack_monotonic_ns: uint}|null` である。
`nodes` 配列はノードランク昇順に並べた厳密に `protocol.nodes` 件であり、ランクの集合は `0..protocol.nodes-1` と完全に一致する。各エントリにおいて `ready.received_monotonic_ns <= ready.ack_monotonic_ns` を満たす必要がある。shutdown が non-null の場合は ready と同一のクロック ID であり、`ready.ack_monotonic_ns <= shutdown.received_monotonic_ns <= shutdown.ack_monotonic_ns` を満たさなければならない。
ノードエージェントのサマリがシャットダウン受信の唯一のソースであり、ラッパーのライフサイクルが子プロセスの終了検知の唯一のソースである。固定ポートの衝突やタイムアウトは実行失敗（run 失敗）として扱い、自動リトライは行わない。
`lifecycle-summary-projection/v1` は、この closed なサマリに `source_member: relative POSIX path`, `source_sha256: sha256` を追加した全フィールド必須のオブジェクトである。

ラッパーがファクト `parallel.node.<node_rank>.process_lifecycle` に出力する `node-lifecycle/v1` も closed object とし、以下の構造に固定する。

```text
schema_version: const "node-lifecycle/v1"
clock_id: nonempty string
spawn_attempt_monotonic_ns: uint
child_started_monotonic_ns: uint|null
child_finished_monotonic_ns: uint|null
termination: one of
  {kind: const "waited", raw_wait_status: uint}
  {kind: const "spawn_error"}
```

プロセスの生成（spawn）に成功した場合のみ両方の子プロセス用タイムスタンプを保持し、`attempt <= started <= finished` の順序関係を満たすものとする。`spawn_error` の場合は双方を null とし、`waited` の場合は双方を non-null かつ生の終了ステータスが POSIX の `WIFEXITED xor WIFSIGNALED` を満たすものとする。正常終了／シグナル終了の分類および終了コード／シグナル番号は、コンシューマが生ステータスから一度だけ導出する。ラッパー側でそれらの分類情報やシャットダウンレコード、早期終了／協調終了の種別を冗長に記録してはならない。
外側の Fact に含まれる `run` やキー内の `node_rank` を値（value）内に再掲してはならない。ノードの子プロセスは `node_agent.py` 内の軽量な Python スーパーバイザが `os.waitpid(child_pid, 0)` により一度だけ回収し、その生ステータスを保存する。Bash の `wait` による `$?` はこのソースには使用しない。

`control/phase/summary.json` は `phase-summary/v1` に準拠し、profile 実行時のみ生成される。呼び出しごとに `invocation_id`, `kind=warmup|measured`、rank 0 側の `{clock_id, start_broadcast_ns, all_start_acks_received_ns, generation_start_ns, generation_end_ns, end_broadcast_ns, all_end_acks_received_ns}`、およびノードランク昇順の `{node_rank, clock_id, start_sequence, end_sequence, start_received_ns, start_ack_ns, end_received_ns, end_ack_ns}` を保持する。
ノード配列はノードランク昇順に厳密に `protocol.nodes` 件並び、ランク集合は `0..protocol.nodes-1` と一致する。各エントリは単一のクロック ID 上において `start_received_ns <= start_ack_ns <= end_received_ns <= end_ack_ns` を満たす必要がある。
rank 0 における 6 つのタイムスタンプは記載順に単調非減少（nondecreasing）であり、同一のクロック ID を使用する。開始／終了シーケンス番号は同一ノードの制御ストリーム上で狭義単調増加であり、GPU サンプルとノードウィンドウのクロック ID は厳密に一致していなければならない。
GPU フェーズウィンドウは同一ノードにおける区間 `[start_received_ns, end_received_ns)` であり、ACK 時刻は開始・終了の遅延（lead/trail）診断のみに使用する。
warmup および measured における rank 0 のテキスト生成境界は、通常の結果ライターが用いる同一の単調増加ナノ秒クロック値であり、統一的な `perf.phase.warmup_generation.wall_time_sec` または `perf.benchmark.generation.wall_time_sec` は `(generation_end_ns - generation_start_ns) / 1_000_000_000` の厳密な有限小数値として確定する。profile では同一の呼び出し種別（invocation kind）において等価性を検証する。スカラー値は全バリアント共通の結果スキーマとして、タイムスタンプのペアはプロファイルウィンドウの内包判定にのみ必要とされるため併存させる。
GPU サンプリング周期の初期値は 1.0 秒とする。カバレッジの閾値を根拠なく固定値で足切りせず、実測カバレッジと有効測定期間をそのまま表示する。

`phase-summary-projection/v1` は、全フィールド必須の以下の構造とする。

```text
{
  schema_version: const "phase-summary-projection/v1",
  run_id: string,
  invocation_id: string,
  kind: string,
  rank0: {
    clock_id: string,
    start_broadcast_ns: uint,
    all_start_acks_received_ns: uint,
    generation_start_ns: uint,
    generation_end_ns: uint,
    end_broadcast_ns: uint,
    all_end_acks_received_ns: uint
  },
  nodes: [
    {node_rank, clock_id, start_sequence, end_sequence, start_received_ns, start_ack_ns, end_received_ns, end_ack_ns},
    ...
  ],
  source_member: relative POSIX path,
  source_sha256: sha256
}
```

上記のデータ型および順序制約をそのまま適用する。

GPU サンプルは、以下の closed なスキーマとして定義される。

```text
schema_version: const "gpu-sample/v1"
run_id: string
node_rank: uint
sequence: uint
clock_id: string
monotonic_ns: uint
wall_time_utc: RFC3339 UTC string
gpu_uuid: string
logical_index: uint
utilization_percent: decimal|null
memory_used_bytes: uint|null
power_w: decimal|null
temperature_c: decimal|null
sm_clock_mhz: decimal|null
```

`sequence` はノードごとのポーリングフレーム順序番号（ordinal）であり、同一のポーリングから得られたすべての期待される GPU 行は同一のシーケンス値を持つ。フレームは、期待される GPU UUID のレコードが厳密に 1 行ずつ揃った場合のみ完全（complete）と見なし、途中の不完全なフレームはアーティファクト内に残すことがあってもプロジェクションには含めない。完全なフレーム間におけるシーケンスの欠落（gap）は許容するが、補間のための架空サンプルを生成してはならない。使用率は 0〜100、その他の非 null の測定値はすべて非負値とする。取得できなかったフィールドは null とし、ポーリングステータスや理由を記録する専用フィールドは設けない。
パブリッシャーは profile raw tar をパースし、同一のサンプルフィールドに `source_member: relative POSIX path`, `source_sha256: sha256` を追加したプロジェクションへ転記する。トポロジおよびプロファイルハードウェアについても同様に、厳密なソースレコードを `qwen_topology_raw_v2` および `qwen_profile_hardware_raw_v2` へ転記し、ハッシュのみで値を不可視化してはならない。
このフィールド追加後の closed object を `gpu-sample-projection/v1` と定義し、スキーマバージョンを同名の定数に置き換える。

プロファイル専用の CPU／NUMA 生データは、ノードランクごとに `profile/raw/hardware/node-<node-rank>/{proc-meminfo.txt, numa/node-<id>-cpulist.txt, numa/node-<id>-meminfo.txt, capture.json}` の固定パスに配置する。`capture.json` は各コマンド実行に関する全フィールド必須かつ `additionalProperties=false` の closed object である `profile-hardware-capture/v1` を保持する。

```text
{
  schema_version: const "profile-hardware-capture/v1",
  run_id: run ID,
  node_rank: uint,
  hostname: nonempty string,
  members: [
    {role: enum "proc_meminfo"|"numa_cpulist"|"numa_meminfo", numa_node_id: uint|null, path: relative POSIX path, size_bytes: uint, sha256: sha256}
  ]
}
```

`members` 配列は `role`/`node`/`path` 順に一意であり、`proc_meminfo` はノード ID が null で厳密に 1 件、NUMA の 2 つのロールは検出された NUMA ノードごとに同一 ID を 1 件ずつ保持する。読み取り元はロールおよびノード ID から `/proc/meminfo`、`/sys/devices/system/node/node<id>/cpulist`、同 `meminfo` へ固定的に導出され、同一の `source_path` をレコード内へ重複記録してはならない。読み取り境界はプロファイルノードエージェントの既存ライフサイクルが一度だけ所有し、ノード別キャプチャへ argv や実行結果（outcome）を複写しない。

パブリッシャー側のレコード行は、全フィールド必須かつ `additionalProperties=false` の以下のスキーマとする。

```text
{
  schema_version: const "profile-hardware-projection/v1",
  run_id: string,
  node_rank: uint,
  hostname: nonempty string,
  record_kind: enum "node_summary"|"numa_node",
  numa_node_id: uint|null,
  postrun_memory_available_bytes: uint|null,
  cpulist: string|null,
  mem_total_bytes: uint|null,
  mem_free_bytes: uint|null,
  source_members: [
    {role: enum "capture"|"proc_meminfo"|"numa_cpulist"|"numa_meminfo", path: relative POSIX path, sha256: sha256},
    ...
  ]
}
```

すべての行に厳密に 1 件の `capture` を持たせ、ノードサマリ行では NUMA 関連フィールドを null、NUMA 行では実行後メモリフィールドを null とし、NUMA 行のソースメンバーは capture, cpulist, meminfo を独立したエントリとして保持する。キャプチャは `gethostname()` を一度だけ記録し、同一実行の最小ハードウェア情報とノードランク集合・ホスト名集合および各 `(node_rank, hostname)` の対応関係を厳密に一致させる。`/proc/meminfo` は ASCII の `Key: uint kB` 形式から `MemAvailable` を抽出し、NUMA の meminfo は `^Node <id> (MemTotal|MemFree):[ \t]+([0-9]+)[ \t]+kB$` をそれぞれ厳密に 1 回ずつ選択して、別ノード ID プレフィックスを拒絶した上で kB を 1024 倍してバイト換算する。重複キー、非整数値、オーバーフロー、単位不一致が検知された場合は該当キャプチャを投影しない。cpulist は ASCII 空白をトリムし、カーネルのリスト文法から狭義単調増加の一意な CPU 集合を最小レンジ表記に正準化する。空ファイルは空文字列とし、読み取り不能またはパース不能な値のみを NULL とする。最小ハードウェア情報にすでに記録されている CPU 識別情報や設定値を再収録してはならない。

プロファイル時の PyTorch Profiler は、グローバル TP ランクごとに `profile/raw/torch/qwen-profile-TP-<global-rank>.trace.json.gz` と同一ステム（stem）の `.capture.json` を厳密に 1 件ずつ生成する。キャプチャは全フィールド必須かつ `additionalProperties=false` の closed object である `profiler-capture/v1` とする。

```text
{
  schema_version: const "profiler-capture/v1",
  run_id: run ID,
  global_rank: uint,
  invocation_id: nonempty string,
  source_member: relative POSIX path,
  size_bytes: uint,
  sha256: sha256
}
```

`global_rank` はファイル名、トレースメタデータ、およびプロトコルの `0..tensor_parallel_size-1` と一致させ、source_member は対応する gzip 圧縮トレースを指す。キャプチャ自体を独立した生リレーションへ投影することはせず、トレースとの来歴追跡にのみ使用する。全ランクのトレースが完全に揃わない場合は、既存の `collection_complete=false` を設定し、欠損ランク用のプレースホルダやプロファイラ専用ステータス／理由コード、リトライ処理は作成しない。

パブリッシャーは実在するトレースを一度だけパースし、完全な継続時間イベント（duration event）を `(run, global_rank, activity, event_category, event_name)` でグループ化した closed object である `profiler-event-projection/v1` として抽出し、`qwen_profiler_events_raw_v2` へ転記する。

```text
{
  schema_version: const "profiler-event-projection/v1",
  run_id: run ID,
  global_rank: uint,
  activity: enum "CPU"|"GPU",
  event_category: string,
  event_name: string,
  call_count: positive uint,
  inclusive_duration_sum_ns: nonnegative decimal,
  source_member: relative POSIX path,
  source_sha256: sha256
}
```

平均値、構成比、ランク／実行合計値、Top-N、分類サマリは生データとしては保存せず、クエリ実行時に導出する。bench 実行ではこのリレーションに行を生成せず、固定幅コンシューマでは NULL として表現する。

GPU 集約指標は生ファクト（raw fact）には保存しない。`gpu-phase/v1` リデューサおよびビューは、`phase, node_rank, gpu_uuid` の粒度で以下の closed な列集合を返す。

```text
phase_duration_sec: decimal,
valid_sample_count: uint,
valid_utilization_duration_sec: decimal, utilization_coverage: decimal,
utilization_mean: decimal|null, utilization_p95: decimal|null,
valid_power_duration_sec: decimal, power_coverage: decimal,
energy_j: decimal|null, power_mean: decimal|null,
valid_memory_sample_count: uint, memory_max_bytes: uint|null,
valid_temperature_sample_count: uint, temperature_max_c: decimal|null,
valid_sm_clock_sample_count: uint, sm_clock_min_mhz: decimal|null
```

`valid_sample_count` は、スキーマ妥当な完全フレームに属し、ステップインターバルがフェーズウィンドウと正の期間で交差する当該 GPU の行数であり、すべての測定値が null であってもカウントする。メトリクス別のカウントおよびメモリ／温度／SM クロックの最大値／最小値は、同一の交差条件を満たし、かつ該当測定値が non-null である行のみを対象とする。したがって開始前（pre-start）の行であっても、その有効区間がフェーズ内に重なる場合はカウントおよび集約値の双方に反映される。各サンプル値は、次のサンプル時刻、フェーズ終了時刻、または `sample_time + 2 * nominal_interval` のいずれか最小の時点まで一定値として保持される（ステップ関数）。所要時間（duration）系の指標は、このステップインターバルとフェーズウィンドウの重複期間のみを合算し、カバレッジは `valid_duration / phase_duration` として算出する。積算電力量（energy_j）は有効電力区間をステップ積分して算出し、欠損区間への外挿を行ってはならない。利用率の p95 は重複期間で重み付けした最近傍ランク法により求め、平均電力は `power_mean = energy_j / valid_power_duration_sec` とする。フリート全体のプロジェクションは、期待される GPU グリッドに対して外部結合（LEFT JOIN）し、対象／有効 GPU 数、最小カバレッジ、GPU 統計値の最小／中央値／最大、総電力量をクエリ実行時に動的に導出する。完了フラグやステータスを表す真偽値列は新設せず、必要な入力が不足しているプロジェクション列は NULL とする。

## ワークロードと実行結果

### 順序付きワークロード v1

`source/workload-{warmup,measured}.json` は、`schema_version: const "ordered-workload/v1"`, `kind`, `requests` を保持する。`kind` は `warmup | measured` であり、warmup のみ `mode` 列挙型（`random | dataset`）を追加する。実行 ID（run ID）は意味的なワークロードバイト列には含めず、実行バンドルマニフェストおよびパブリッシャープロジェクションのみがそのバインディングを所有する。これにより、同一の実際のワークロードは実行を跨いでも同一のソース SHA-256 ハッシュを維持する。
`requests` 配列は実際の生成順序（generate 順）であり、各要素は以下の構造を持つ。

```text
source: {kind: "dataset_row", array_index: uint} | {kind: "generated", generator_id: string, seed: uint, generator_ordinal: uint},
prompt_payload: {kind: "text", text: string, input_token_count: uint} | {kind: "token_ids", token_ids: [int, ...]},
requested_output_token_count: uint,
sampling_params: [{name: string, declared_type: string, value: JCS value}, ...]
```

`prompt_payload` は現行のテキスト経路を無断でトークン ID 経路へ変更せず、実際の generate 呼び出しへ渡す値をタグに従って忠実に保存する。text 形式の `input_token_count` はリクエスト選択時にトークナイザが実測した値であり、token_ids 形式では配列長から自明に導出できるため保存しない。
サンプリングパラメータは、実際の生成呼び出しに渡されたすべてのフィールドを変数名のバイト昇順で保持する。マニフェストの識別子は、リクエスト配列の順序および厳密なプロンプトペイロードを含む。生成されたプロンプト本文そのものは保存しない。
パブリッシャーはマニフェストのトップレベルを `qwen_workload_manifests_raw_v2` へ、各リクエストレコードを `qwen_workload_requests_raw_v2` へ転記する。ソースメンバーパスおよびハッシュはマニフェストリレーションのみが所有し、リクエストリレーションはマニフェスト識別子をキーとして結合する。

`workload-manifest-projection/v1` は、全フィールド必須かつ `additionalProperties=false` の以下のスキーマとする。

```text
{
  schema_version: const "workload-manifest-projection/v1",
  run_id: string,
  workload_kind: enum "warmup"|"measured",
  warmup_mode: enum "random"|"dataset"|null,
  source_member: relative POSIX path,
  source_sha256: sha256
}
```

measured 実行における `warmup_mode` は null とし、`source_sha256` が実行バインディングを除外した正準ワークロードファイルの意味的マニフェスト識別子を兼ねる。
`workload-request-projection/v1` は全フィールド必須の `{schema_version: const "workload-request-projection/v1", run_id, workload_kind, ordinal: uint, request: <上記 request closed object>}` を保持し、親マニフェストプロジェクションと `run_id/workload_kind` で結合する。マニフェストハッシュやソース来歴情報を各リクエスト行へ重複記録してはならない。
`ordinal` は順序付き `requests` 配列の 0 起点オフセットからパブリッシャーが正規化時に一度だけ付与し、ソースアイテム内へは複写しない。

マニフェストのみが順序付きリクエストの生データの正本である。選択件数、計画された入力／出力トークン総数、ウォームアップモードなどを表す個別の生キーは新設せず、コンシューマが配列から直接導出する。

### 主要測定結果 v1

`source/primary-result.json` は、rank 0 が generate 呼び出しから復帰した直後にアトミックに書き込み／リネームし、以下の構造を持つ。

```text
schema_version: const "primary-result/v1"
run_id: nonempty string
invocation_id: nonempty string
generation_wall_sec: positive finite decimal
returned_count: uint
error_count: uint
observed_input_tokens: uint|null
observed_output_tokens: uint|null
ordered_output_token_ids_sha256: sha256|null
```

パース不能なレスポンスは `error_count` に計上し、取得できなかったトークン総数を 0 で補完してはならない。生成例外の発生時に主要結果を捏造して記録してはならない。
出力識別子（ordered_output_token_ids_sha256）は、レスポンス配列の順序を厳密に保持した `[[uint token_id, ...], ...]` の JCS SHA-256 であり、すべてのレスポンスの `output_ids` が非負整数配列として正しく読み取れ、各配列長が同一レスポンスの `completion_tokens` と一致し、かつ全長の和が `observed_output_tokens` と一致する場合のみ non-null とする。生成テキスト本文、リクエスト単位のハッシュ、同一ハッシュの別名ファクトは作成しない。
`benchmark_valid` は、error=0、returned がワークロードリクエスト長と一致、observed_input がワークロード由来の計画総数と一致、generation_wall が正の有限値、かつ `ignore_eos` 固定時は observed_output が要求値と一致していることのみから論理的に導出する。

ビュー側の計算式は以下のように定義され、スループット値を生テーブルへ保存してはならない。

```text
request_throughput = returned_count / generation_wall_sec
input_token_throughput = observed_input_tokens / generation_wall_sec
output_token_throughput = observed_output_tokens / generation_wall_sec
total_token_throughput = (observed_input_tokens + observed_output_tokens) / generation_wall_sec
```

## ネットワーク測定キャプチャとプロジェクション v1

```text
mpi-network-result/v2 = {
  run_id: string,
  benchmark: enum "PingPong"|"Sendrecv"|"Allreduce",
  process_count: uint > 0,
  message_bytes: uint,
  iterations: uint > 0,
  time_min_us: finite number|null,
  time_avg_us: finite number|null,
  time_max_us: finite number|null,
  bandwidth_mbytes_per_sec: finite number >= 0|null,
  source_member: relative POSIX path
}

nccl-network-result/v2 = {
  run_id: string,
  collective: enum "all_reduce"|"all_gather"|"reduce_scatter",
  message_bytes: uint > 0,
  element_count: uint > 0,
  data_type: const "float",
  reduction_op: enum "sum"|null,
  buffer_mode: enum "out_of_place"|"in_place",
  mpi_process_count: uint > 0,
  gpu_count: uint > 0,
  time_us: finite number >= 0,
  algorithm_bandwidth_gbps: finite number >= 0,
  bus_bandwidth_gbps: finite number >= 0,
  error_count: uint,
  source_member: relative POSIX path
}
```

IMB のソースメンバー構成は `source/network/imb-mpi1/{argv.json, stdout.txt, stderr.txt}`、NCCL は集約通信ごとの `{argv.json, result.json, stdout.txt, stderr.txt}` に固定する。IMB パーサは完全に終了したセクションのみを受理し、NCCL パーサは `-J` による構造化 JSON のみを正本とする純粋関数（pure function）である。途中で切断されたソースやパース不能なソースから中途半端な部分行を生成してはならない。

`mpi_network_results_v2` の粒度は `(run_id, benchmark, process_count, message_bytes)`、`nccl_network_results_v2` の粒度は `(run_id, collective, message_bytes, buffer_mode)` である。各レコード行はソースメンバーパスのみを保持し、アーティファクトハッシュはマニフェストおよびプロジェクションの来歴追跡のみが所有する。algbw や busbw を再計算せず、サマリ行やサマリキーを追加しない。
`all_reduce` および `reduce_scatter` における `reduction_op` は `sum`、`all_gather` では NULL に固定する。JSON 内のサイズ、要素数、データ型、out-of-place/in-place、所要時間、algbw、busbw、エラー数を不可逆な変換なく転記し、8 バイトから 1 GiB までの 2 の冪乗で実在するメッセージサイズのみを受理する。
コンテナビルドにおいては、11 個の NCCL テストバイナリ、JSON 出力用 `-J` オプション、主要 3 集約通信の正常起動可能性を Buildah の作業コンテナ内で検証し、検証に合格しない限り SquashFS イメージを公開してはならない。

## 制御、マージ、パッケージング

PJM エントリは通常実行のみを取り扱う。

```text
job-entry/v1:
  schema_version, run_id, pj_job_id, entered_at
bundle-manifest/v1:
  schema_version, run_id, members[]
```

ランナーはジョブ投入前後の厳密な argv, stdout, stderr, レスポンス、PJM 側のジョブ ID、終了時のアカウンティング情報を確実に保存する。マニフェストの run ID と job-entry の run ID、および投入・job-entry・アカウンティングにおけるジョブ ID が完全に一致しない限りパブリッシュを実行してはならない。
通常実行のバンドルは `control/`, `source/`, `logs/`, `metrics.jsonl`、プロファイル時のみ `profile/raw/` を保持する。アーティファクトファイルはロール順序に従って一意に配置し、外側マニフェスト自身をメンバー一覧に含めてはならない。

マージ処理は、実装と測定目的から期待される producer およびファイルの集合を一度だけ導出する。完成したファクト断片のみを読み込み、同一所有者によるバイト単位で完全に同一なファクトのみを重複排除（dedupe）し、値の不一致が検知された場合は衝突（conflict）として run を不完全（incomplete）状態にする。
値が取得できなかった場合や非適用である場合は、コンシューマ側で NULL として扱い、欠損理由コードやステータス、プレースホルダ用のキーは作成しない。

## アーティファクトプロジェクションレジストリ v1

ソースインスタンスの所有スコープは `run` または `build` のいずれかのみである。run 所有者は run ID、build 所有者はビルド識別子を持ち、1 つのソースインスタンスを厳密に 1 つのプロジェクションパーサにのみルーティングする。

主要なプロジェクションリレーションは以下のとおりである。

| ソース／リレーション | 粒度（grain） | 適用対象（applicability） |
|---|---|---|
| source track | run ID | bench / profile |
| runtime launcher | run ID, role | bench / profile |
| container provenance | SquashFS SHA-256 | bench / profile |
| model/dataset/cache manifest | run ID, kind | bench / profile |
| workload manifest/request | run ID, kind, ordinal | bench / profile |
| minimal hardware | run ID, node rank | bench / profile |
| topology/placement | run ID, node/global rank | bench / profile |
| environment | run ID, scope, name | bench / profile |
| lifecycle/result | run ID または invocation ID | bench / profile |
| GPU samples/phases | run ID, node rank, sequence | profile only |
| artifact index | run ID, role | bench / profile |
| IMB network | run ID, benchmark, process count, message bytes | bench / profile（テキスト生成成功時のみ） |
| NCCL network | run ID, collective, message bytes, buffer mode | bench / profile（テキスト生成成功時のみ） |

メトリクスファミリやプロジェクションインスタンスを、専用の非擾乱確認用名前空間へ複製してはならない。

## ファクトとコンシューマルーティング

スカラーおよび複合ファクトは `run/key/value` の生テーブルにのみ出力する。完全なレコードをアーティファクトまたはプロジェクションが所有している場合、初回値／最終値／件数／平均値、構成比、ランク／実行合計値、Top-N などの部分集合キーやサマリキーは保存せず、クエリ実行時に動的に導出する。ソースアーティファクトとファクトの間における二重所有（dual ownership）を禁止する。

コンシューマはキーと実行目的に基づいて適切なプロジェクションを選択する。用途別の適用性を評価した上で期待されるロケータやグリッドを生成し、プロファイル専用列はベンチマーク実行において正当な理由記述なしの純粋な NULL とする。ベンチマークの比較適格性は、通常実行のソース、ワークロード、結果、配置、完全性のみから導出し、開発時の非擾乱確認結果を結合してはならない。

実装中の「計測なし／あり」の比較測定は公開コントラクトの外部で一時的に実施し、検証完了後に出力を破棄する。したがって、本コントラクトには専用のスキーマ、ソース、ライター、パーサ、プロジェクション、リレーション、アーティファクトロール、メトリクスキーを一切追加しない。
