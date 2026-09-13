# Qwen3-235B PD inference on Firmus

公式Qwen競技の `Qwen/Qwen3-235B-A22B-Instruct-2507` FP8 weightsを、Firmusの2ノード16基のH200を用いてPrefill–Decode分離（PD分離）構成で評価するリポジトリです。実行エントリポイントは [Qwen workflow](.github/workflows/qwen.yml) です。玄界およびASPIRE 2A+には同じ固定revisionの全24モデルシャードと検証済みSquashFSが配置されており、現在の各ワークフローは要求した全ノードで`sglang.check_env`とNIXLバージョンを検証します。

Firmusにおける初期構成は、node 0のPrefillワーカー4基（TP2）およびnode 1のDecodeワーカー2基（TP4/EP4）です。各ワーカーはNIXLを使用し、計算ノード上で状態が`ACTIVE`かつリンク層が`InfiniBand`であるHCAだけを`UCX_NET_DEVICES`へ渡します。複数NIC環境でもNIXLの制御経路とデータ経路が同じ到達可能なアドレスを使うように、各ノードのIPv4アドレスをrouter URL、ヘルスチェック、およびSGLangの`SGLANG_HOST_IP`へ明示します。計6基のワーカーの健全性（ヘルスチェック）を確認したのち、`sglang-router`の単一ゲートウェイを起動し、AIPerfはその`/v1/chat/completions`エンドポイントにのみ接続します。サービスの起動、ヘルスチェック、AIPerfの実行、および公式結果判定は、同一の [Slurm job](platforms/firmus/run-qwen-pd.slurm) 内で一貫して実行されます。

[Firmus run config](configs/firmus-pd.json) は探索用ではなく、公式の並行数16構成です。ウォームアップ60秒、ベンチマーク120秒、グレース期間30秒で固定実行し、ISL/OSLは4000/200、シード値は2026、キャッシュ無効化には`first_turn_prefix`を用います。オープンループの`request-rate`は設定せず、SLO基準値はTTFT 3000 msおよびITL 20 msとします。[result parser](scripts/parse-aiperf-result.py) はgood request率が0.90以上の場合にのみ`goodput × 200`を算出します。

公式判定には実行した`aiperf-command.txt`と`aiperf-provenance.txt`を渡し、固定コマンド、AIPerf 0.12.0、tokenizer revision、concurrency、gatewayの一致も検証します。合成 ISL の設定値は `4000` ですが、AIPerf 0.12.0 のキャッシュバスト処理に伴うマーカートークン化の変動（減算推定値 `14` に対し実際は `12`〜`16` トークン）により、クライアント側で観測される有効な ISL は `3984`〜`3988` の範囲に分布します。そのためパーサーは、従来の固定値（`4000` または `3986`）の受容を維持しつつ、`min`/`max` が `3984`〜`3988` に収まり、`count`、整数 `sum`、および `average == sum / count` の整合性が確認された分布統計を正当な結果として受け入れます。なお、分布の裏付けを伴わない近傍の非整数平均値は拒否されます。観測値は変更せず、設定値と判定結果を`score.json`の`official_validation`へ記録します。[補正値の根拠と判定範囲](research/aiperf-isl-validation.md)を参照してください。

公式の `AIPerf` 引数では `min_tokens=200`、`max_tokens=200`、`ignore_eos=true` を指定して要求 OSL を `200` とする。検証処理では観測された `min`、`max`、`count`、`sum` を保持する。`AIPerf 0.12.0` の定義では要求 OSL の `5%` または `50` トークンの小さい方を超える絶対差を不一致（mismatch）と判定するため、要求値 `200` に対する許容観測範囲は `190` 以上 `210` 以下（両端含む）となる。検証要件として、`count` が `request_count` と等しいこと、`average` が `sum` / `count` と等しいこと、`sum` が整数であること、および `osl_mismatch_count` が `0` であることを義務付ける。スコア算出は goodput に設定値の `200` を乗じる仕様を維持する。

通常ランでは、ワーカー／ルーター／AIPerfのログ、AIPerfの成果物（raw artifacts）、Prometheusスナップショット、およびソース／コンテナ／モデル／トークナイザーの来歴を回収します。コミットメッセージに`[profile]`を付与した実行に限り、各GPUの使用率、メモリ、消費電力、温度、クロックに加え、CPU時間、NICの送受信量、InfiniBand counterを1秒間隔でサンプリングし、`profile`成果物へ分離して記録します。プロファイリングのオーバーヘッドを含むため、通常のベンチマーク結果と`[profile]`付き実行の結果を直接比較してはなりません。

コンテナは、AWSシンガポールリージョン上に一時起動する`m7i.2xlarge`インスタンスにおいて、Buildahの`chroot` isolationを用いて構築します。生成したOCI layoutをSkopeoおよびUmociで検証した上でSquashFS化し、SGLang mirrorの特定コミット、AIPerf 0.12.0、sglang-router 0.3.2を固定します。各クラスタは自身のローカルアーカイブのみを検索し、クラスタ間でのキャッシュ複製は行いません。ソースコード、ツールチェーン、依存関係を変更したコミットには、メッセージの先頭に`[require-build]`を付与してください。

push契機で計算ジョブを起動する場合は、コミットメッセージに`[cluster:firmus] [nodes:2]`を付与します。新規コンテナビルドを伴うコミットメッセージの例は次の通りです。

```text
[require-build] [cluster:firmus] [nodes:2] Run official Qwen3 PD baseline
```

`[cluster:genkai]`および`[cluster:aspire2ap]`は、各クラスタにおける環境検査ジョブを起動します。`[nodes:N]`で要求した各ノードへ1 rankを配置し、全rankの成功と相異なるhostnameを確認します。クラスタタグのないpushでは静的検証のみが実行され、`[skip ci]`を先頭に付けたpushではGitHub Actions自体が起動しません。実行成果物（run artifacts）は、一旦GitHub ActionsのArtifactとして回収された後、home-labの`/hdd_raid/samba/net_share/qwen-artifacts/<run-id>/<attempt>`へSHA-256マニフェスト付きで退避・保管されます。

公式仕様、固定コマンド、モデル配置の検証結果、および実装判断の詳細は [公式Qwen3 PD実装記録](research/official-qwen3-pd.md) を参照してください。AWSリソースと認証情報の分離境界については [AWS SquashFSビルド](research/aws-container-build.md)、Enrootへの移行経緯と設計理由は [OCI / Enroot移行記録](research/container-runtime-migration.md) に記録されています。
