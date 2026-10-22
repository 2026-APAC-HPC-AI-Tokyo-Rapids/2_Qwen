# Qwen3-235B PD inference on Firmus, Genkai, and ASPIRE 2A+

vLLM用イメージは、GitHub Actionsの **AWSでvLLMイメージをビルド** から手動で生成できます。Docker用アーカイブとFirmus用SquashFSを生成し、後者をFirmusへ配布します。実行方法と成果物の場所は [vLLM image build on AWS](research/vllm-container-build.md) を参照してください。

公式Qwen競技の `Qwen/Qwen3-235B-A22B-Instruct-2507` FP8 weightsをPrefill–Decode分離（PD分離）構成で評価するリポジトリです。実行エントリポイントは [Qwen workflow](.github/workflows/qwen.yml) です。Firmus、玄界、ASPIRE 2A+のそれぞれに同じ固定revisionの全24モデルシャードと検証済みSquashFSを配置し、各クラスタのSlurm、PJM、PBSジョブ内でモデル検証、SGLang workerとPD routerの起動、公式AIPerf、ローカル結果検証までを連続して実行します。

The ASPIRE candidate ports the Firmus **8366.32 output tok/s** configuration ([39550627795](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39550627795), `fd122bc`) onto ASPIRE's measured **5076.43 output tok/s** baseline ([39098592938](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39098592938)). It retains the H100 P2/D2 topology, TP4/DP4/EP4, concurrency 104, EAGLE3 steps3/draft4, FP8 KV, routing, KV pools, Decode cap 80 and graph buckets through 20. The port uses the source run's exact base image (`652aaeef…`, build 39379223361) and six ordered package overlays: fused Prefill norm/quant, contiguous-KV attention, hot-4 expert replication, FC2 per-128 scales, FC2 staging swizzle and two math warpgroups. Prefill loads 140 physical experts for the unchanged 128 logical experts; Decode keeps its original placement. Request-aligned Prefill chunks, optimistic Prefill and asynchronous Decode admission use the source run's switches.

Provisioning verifies the pinned image and host extension, applies the same overlay helper used by Firmus, and archives the patch hashes. Before PBS submission, a CPU Enroot check verifies the actual package mounts, all 94 layers of the replicated expert map, and the prepared draft files. After execution, every DP rank's placement, replica count, scheduler settings and KV capacity are checked. The migration record is [research/runtime-performance/2026-10-22-aspire-firmus8366-port.json](research/runtime-performance/2026-10-22-aspire-firmus8366-port.json). This candidate has CPU validation only; its H100 memory peak and throughput have not been measured.

Firmus では、最終トポロジーとして node 0 に TP2 の Prefill ワーカー 4 基、node 1 に TP4/EP4 の Decode ワーカー 2 基を使用します。各ワーカーは NIXL を使用し、ポート状態が `ACTIVE` であるすべての HCA を `UCX_NET_DEVICES` に渡します。検証済みノードでは、これは 8 基の InfiniBand HCA と 2 基の Ethernet HCA から構成される計 10 個のデバイスとなります。なお、本実装では意図的に `link_layer` によるフィルタリングを行っていません。複数NIC環境でもNIXLの制御経路とデータ経路が同じ到達可能なアドレスを使うように、各ノードのIPv4アドレスをrouter URL、ヘルスチェック、およびSGLangの`SGLANG_HOST_IP`へ明示します。計6基のワーカーの健全性（ヘルスチェック）を確認したのち、`sglang-router`の単一ゲートウェイを起動し、AIPerfはその`/v1/chat/completions`エンドポイントにのみ接続します。サービスの起動、ヘルスチェック、AIPerfの実行、および固定された公式ワークロードのローカル検証は、同一のスタンドアロンSlurmジョブ [submission/run/submit.sbatch](submission/run/submit.sbatch) 内で一貫して実行されます。

`submission/run/submit.sbatch` は、Firmus 公式の単一かつ正準な実行規約です。本スクリプトにより、並行度 16、ウォームアップ・ベンチマーク・グレースの各フェーズ（60秒／120秒／30秒）、ISL 4000、OSL 200、シード値 2026、`first_turn_prefix` によるキャッシュバースト、リクエストレートなしのクローズドループ、4基の TP2 prefill ワーカー、2基の TP4 EP4 decode ワーカー、およびローカル結果バリデータが固定されます。なお、`scripts/parse-aiperf-result.py` とスタンドアロンスクリプト内に組み込まれたパーサーはバイト単位で同一であり、主催者提供のソフトウェアではなくローカルバリデータとして位置付けられています。

最終的な並行数16の測定結果および対照となる並行数8のベースラインは、それぞれ`submission/results/logs/17021`および`submission/results/logs/17023`に格納されている。Goodputは並行数8の2.61749910791334 requests/sに対し、並行数16では4.548422736312085 requests/sとなり、73.76979126988392%高かった。本結果は各条件1回の測定でばらつきの推定値はなく、宣言された実行影響条件の差は並行数のみであるため、HCAやその他の要因に対する因果関係を主張するものではない。

ローカル検証器は `aiperf-command.txt` および `aiperf-provenance.txt` を受け取り、実行コマンドの完全一致、AIPerf 0.12.0、トークナイザリビジョン、並行性、ゲートウェイ、エクスポート各フェーズの所要時間、シード値、キャンセル／エラー状態、およびリクエスト整合性を検証します。設定上の合成ISLは4000ですが、クライアント側で推定14トークンが減算される一方で実際のマーカートークン化が12〜16トークンで変動するため、`first_turn_prefix` により有効な観測長は3984〜3988の範囲に分布します。合格判定には、最小値および最大値が同範囲内にあること、件数が `request_count` と一致すること、合計値が整数であること、平均値が合計／件数と厳密に一致することが常に必須であり、分布の証拠データを欠く結果は棄却されます。なお、生の値は変更されず、ローカル検証結果は `score.json` の `official_validation` に格納されます。[補正値の根拠と判定範囲](research/aiperf-isl-validation.md)を参照してください。

公式の `AIPerf` 引数では `min_tokens=200`、`max_tokens=200`、`ignore_eos=true` を指定して要求 OSL を `200` とする。検証処理では観測された `min`、`max`、`count`、`sum` を保持する。`AIPerf 0.12.0` の定義では要求 OSL の `5%` または `50` トークンの小さい方を超える絶対差を不一致（mismatch）と判定するため、要求値 `200` に対する許容観測範囲は `190` 以上 `210` 以下（両端含む）となる。検証要件として、`count` が `request_count` と等しいこと、`average` が `sum` / `count` と等しいこと、`sum` が整数であること、および `osl_mismatch_count` が `0` であることを義務付ける。スコア算出は goodput に設定値の `200` を乗じる仕様を維持する。

通常実行では、6基のワーカーログ、ルーターログ、AIPerf の生エクスポート、スコア、スケジューラアカウンティング、ならびにソースコンテナ・モデル・トークナイザの来歴情報（provenance）が収集されます。Prometheus ワーカーのスナップショットは収集されません。プロファイル実行時のみ、各SGLang workerは共通の `scripts/exec-profiler.sh` 実行境界を通り、そのargv証跡とGPU、CPU、NIC、およびInfiniBandの1秒間隔テレメトリが個別のprofileディレクトリに収集されます。全workerのargv証跡が揃わないprofile実行は成功扱いにしません。プロファイル時の測定値は通常ベンチマークの結果と直接比較してはなりません。

コンテナは、AWSシンガポールリージョン上に一時起動する`m7i.2xlarge`インスタンスにおいて、Buildahの`chroot` isolationを用いて構築します。生成したOCI layoutをSkopeoおよびUmociで検証した上でSquashFS化し、SGLang mirrorの特定コミット、AIPerf 0.12.0、sglang-router 0.3.2を固定します。各クラスタは自身のローカルアーカイブのみを検索し、クラスタ間でのキャッシュ複製は行いません。ソースコード、ツールチェーン、依存関係を変更したコミットには、メッセージの先頭に`[require-build]`を付与してください。

push契機で計算ジョブを起動する場合は、コミットメッセージに`[cluster:firmus] [nodes:2]`を付与します。新規コンテナビルドを伴うコミットメッセージの例は次の通りです。

```text
[require-build] [cluster:firmus] [nodes:2] Run official Qwen3 PD baseline
```

`[cluster:firmus]`と`[cluster:aspire2ap]`は`[nodes:2]`、`[cluster:genkai]`は`[nodes:2]`または`[nodes:4]`でPDベンチマークを起動します。複数のクラスタタグを指定した場合は並行実行します。クラスタタグのないpushでは静的検証のみが実行され、`[skip ci]`を先頭に付けたpushではGitHub Actions自体が起動しません。全クラスタのログ、AIPerf raw export、score、profile、来歴は`qwen-<cluster>-pd-*`のGitHub Actions Artifactとして回収し、その後home-labの`/srv/qwen-artifacts/<run-id>/<attempt>`へSHA-256マニフェスト付きで退避・保管します。

公式仕様、固定コマンド、モデル配置の検証結果、および実装判断の詳細は [公式Qwen3 PD実装記録](research/official-qwen3-pd.md) を参照してください。AWSリソースと認証情報の分離境界については [AWS SquashFSビルド](research/aws-container-build.md)、Enrootへの移行経緯と設計理由は [OCI / Enroot移行記録](research/container-runtime-migration.md) に記録されています。
