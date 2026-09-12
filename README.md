# Qwen3-235B PD inference on Firmus

公式Qwen競技の `Qwen/Qwen3-235B-A22B-Instruct-2507` FP8 weightsを、Firmusの2ノード16基のH200を用いてPrefill–Decode分離（PD分離）構成で評価するリポジトリです。実行エントリポイントは [Qwen workflow](.github/workflows/qwen.yml) です。玄界およびASPIRE 2A+向けのワークフローは、モデルや実行環境が整備されるまでの間、既存の`sglang.check_env`による環境検査のみを実行します。

Firmusにおける初期構成は、node 0のPrefillワーカー4基（TP2）およびnode 1のDecodeワーカー2基（TP4/EP4）です。各ワーカーはNIXLを使用し、計算ノード上で状態が`ACTIVE`かつリンク層が`InfiniBand`であるHCAだけを`UCX_NET_DEVICES`へ渡します。複数NIC環境でもNIXLの制御経路とデータ経路が同じ到達可能なアドレスを使うように、各ノードのIPv4アドレスをrouter URL、ヘルスチェック、およびSGLangの`SGLANG_HOST_IP`へ明示します。計6基のワーカーの健全性（ヘルスチェック）を確認したのち、`sglang-router`の単一ゲートウェイを起動し、AIPerfはその`/v1/chat/completions`エンドポイントにのみ接続します。サービスの起動、ヘルスチェック、AIPerfの実行、および公式結果判定は、同一の [Slurm job](platforms/firmus/run-qwen-pd.slurm) 内で一貫して実行されます。

[Firmus run config](configs/firmus-pd.json) は現在、短時間の探索ラン（concurrency 16）向けに設定されています。`mode`を`official`に変更すると、AIPerfは公式規定（warmup 60秒、benchmark 120秒、grace 30秒、ISL/OSL 4000/200、seed 2026、`first_turn_prefix`によるキャッシュバスト、TTFT 3000 msかつITL 20 ms）の固定コマンドを実行します。open-loopの`--request-rate`は使用しません。[result parser](scripts/parse-aiperf-result.py) は`profile_export_aiperf.json`から`goodput`と`good_request_fraction`を読み取り、fractionが0.90以上の場合にのみ、`goodput × 200`をoutput token/sのスコアとして出力します。

通常ランでは、ワーカー／ルーター／AIPerfのログ、AIPerfの成果物（raw artifacts）、Prometheusスナップショット、およびソース／コンテナ／モデル／トークナイザーの来歴を回収します。コミットメッセージに`[profile]`を付与した実行に限り、各GPUの使用率、メモリ、消費電力、温度、クロックに加え、CPU時間、NICの送受信量、InfiniBand counterを1秒間隔でサンプリングし、`profile`成果物へ分離して記録します。プロファイリングのオーバーヘッドを含むため、通常のベンチマーク結果と`[profile]`付き実行の結果を直接比較してはなりません。

コンテナは、AWSシンガポールリージョン上に一時起動する`m7i.2xlarge`インスタンスにおいて、Buildahの`chroot` isolationを用いて構築します。生成したOCI layoutをSkopeoおよびUmociで検証した上でSquashFS化し、SGLang mirrorの特定コミット、AIPerf 0.12.0、sglang-router 0.3.2を固定します。各クラスタは自身のローカルアーカイブのみを検索し、クラスタ間でのキャッシュ複製は行いません。ソースコード、ツールチェーン、依存関係を変更したコミットには、メッセージの先頭に`[require-build]`を付与してください。

push契機で計算ジョブを起動する場合は、コミットメッセージに`[cluster:firmus] [nodes:2]`を付与します。新規コンテナビルドを伴うコミットメッセージの例は次の通りです。

```text
[require-build] [cluster:firmus] [nodes:2] Run official Qwen3 PD baseline
```

`[cluster:genkai]`および`[cluster:aspire2ap]`は、各クラスタにおける環境検査ジョブを起動します。クラスタタグのないpushでは静的検証のみが実行され、`[skip ci]`を先頭に付けたpushではGitHub Actions自体が起動しません。実行成果物（run artifacts）は、一旦GitHub ActionsのArtifactとして回収された後、home-labの`/hdd_raid/samba/net_share/qwen-artifacts/<run-id>/<attempt>`へSHA-256マニフェスト付きで退避・保管されます。

公式仕様、固定コマンド、モデル配置の検証結果、および実装判断の詳細は [公式Qwen3 PD実装記録](research/official-qwen3-pd.md) を参照してください。AWSリソースと認証情報の分離境界については [AWS SquashFSビルド](research/aws-container-build.md)、Enrootへの移行経緯と設計理由は [OCI / Enroot移行記録](research/container-runtime-migration.md) に記録されています。
