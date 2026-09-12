# Qwen3-235B PD inference on Firmus

公式Qwen競技の `Qwen/Qwen3-235B-A22B-Instruct-2507` FP8 weightsを、Firmusの2ノード16 H200でPrefill–Decode分離して評価するリポジトリです。実行入口は [Qwen workflow](.github/workflows/qwen.yml) です。玄界とASPIRE 2A+のworkflowは、モデルや実行環境を整備するまで既存の`sglang.check_env`検査だけを行います。

Firmusの開始構成は、node 0の4×Prefill TP2とnode 1の2×Decode TP4/EP4です。各workerはNIXLと計算ノードで検出したInfiniBand deviceを使います。6 workerのhealth確認後、`sglang-router`の単一gatewayを起動し、AIPerfはその`/v1/chat/completions`だけへ接続します。serviceの起動、health確認、AIPerf、公式結果判定は同じ [Slurm job](platforms/firmus/run-qwen-pd.slurm) 内で連続して実行されます。

[Firmus run config](configs/firmus-pd.json) は現在、短時間の探索runをconcurrency 16で行う設定です。`mode`を`official`へ変更すると、AIPerfは公式のwarmup 60秒、benchmark 120秒、grace 30秒、ISL/OSL 4000/200、seed 2026、`first_turn_prefix` cache bust、TTFT 3000 msかつITL 20 msの固定commandを実行します。open-loopの`--request-rate`は使用しません。[result parser](scripts/parse-aiperf-result.py) は`profile_export_aiperf.json`の`goodput`と`good_request_fraction`を読み、fraction 0.90以上のときだけ`goodput × 200`をoutput token/sのscoreとして出力します。

通常runではworker、router、AIPerfのログ、AIPerf raw artifacts、Prometheus snapshot、source/container/model/tokenizerの来歴を回収します。コミットに`[profile]`を付けたrunだけ、各GPUのutilization、memory、power、temperature、clockを1秒ごとに別artifactへ記録します。通常のbenchmark値とprofile付きrunは直接比較しません。

コンテナはAWS Singaporeリージョンの一時的な`m7i.2xlarge`から、Buildahの`chroot` isolationで構築します。OCI layoutをSkopeoとUmociで検証してSquashFS化し、SGLang mirrorのcommit、AIPerf 0.12.0、sglang-router 0.3.2を固定します。各クラスタは自分のローカルarchiveだけを検索し、クラスタ間でcacheをコピーしません。source、toolchain、依存関係を変えたコミットは`[require-build]`を先頭に付けます。

pushで計算を起動する場合は`[cluster:firmus] [nodes:2]`をコミットメッセージへ付けます。新規コンテナを伴う例は次の通りです。

```text
[require-build] [cluster:firmus] [nodes:2] Run official Qwen3 PD baseline
```

`[cluster:genkai]`と`[cluster:aspire2ap]`は各クラスタの環境検査を起動します。クラスタタグのないpushは検証だけを実行し、`[skip ci]`を先頭に付けたpushではActionsを起動しません。run artifactsは一度GitHub Artifactへ回収した後、home-labの`/hdd_raid/samba/net_share/qwen-artifacts/<run-id>/<attempt>`へSHA-256 manifest付きで保管されます。

公式仕様、固定command、モデル配置の検証結果、実装判断は [公式Qwen3 PD実装記録](research/official-qwen3-pd.md) を参照してください。AWS資源と資格情報境界は [AWS SquashFSビルド](research/aws-container-build.md)、既存のEnroot移行理由は [OCI / Enroot移行記録](research/container-runtime-migration.md) に記録しています。
