# 公式Qwen3 PD実装記録

This document records the earlier generic launcher and its historical Firmus runs. The current submission (10054.49 good output tok/s experimental result, with a separate unmodified Basic) is documented in [submission/README.md](../submission/README.md); the unchanged generic launcher now lives at `platforms/shared/qwen-pd.sbatch`.

## 正本

- 公式Task Rules: <https://github.com/hpcac/2026-APAC-HPC-AI/blob/main/task/1.%20Qwen3%20AI%20Task%20Rules.md>
- Firmus Application Notes: <https://github.com/hpcac/2026-APAC-HPC-AI/blob/main/task/3.1%20Qwen3%20Application%20Notes%20-%20Firmus%20AI%20Cloud.md>
- AIPerf v0.12.0: <https://github.com/ai-dynamo/aiperf/releases/tag/v0.12.0>
- 実装追跡: <https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/issues/13>

2026-09-12にTask Rules、Application Notes、および公式Slackのアナウンスを再照合した。公式のベンチマーク実行では、Qwen3-235B-A22B-Instruct-2507のFP8重み（weights）、2ノード16基のGPU、Prefill–Decode分離（PD分離）、および単一のOpenAI互換ゲートウェイの構成が必須要件である。AIPerfはclosed-loop形式で実行し、固定入出力長4000/200トークン、ウォームアップ60秒、ベンチマーク計測120秒、グレイス期間30秒、ランダムシード2026を使用する。TTFT（Time to First Token）3000 ms以内かつ平均ITL（Inter-Token Latency）20 ms以内を同時に満たすリクエストの割合（good_request_fraction）が0.90以上のランのみが採点対象となり、最終スコアは`goodput (request/s) × 200`で算出される。

## 固定した来歴

| 対象 | 固定値 |
|---|---|
| Model repository | `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` |
| Model revision | `e156cb4efae43fbee1a1ab073f946a1377e6b969` |
| Tokenizer repository | `Qwen/Qwen3-235B-A22B-Instruct-2507` |
| Tokenizer revision | `ac9c66cc9b46af7306746a9250f23d47083d689e` |
| AIPerf | `v0.12.0`, commit `be53bf2953d30e46c500e6a80fc1f8b6f84bc718` |
| sglang-router | `0.3.2` |

Firmus上の`/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8`には24個のsafetensorsシャードが配置されており、Hugging Faceダウンロードメタデータのコミットリビジョンも上表のmodel revisionと完全に一致することを確認した。玄界にも221 GiBのモデル重みが配置されている。ASPIRE 2A+では、その後モデル配置が完了し、固定revision `e156cb4efae43fbee1a1ab073f946a1377e6b969`、24/24シャード、シャード総容量236,426,193,880 bytesを確認した。

## 玄界・ASPIRE 2A+での実機検証

main統合コミット`ae9ac10`のActions run 34700473422から両クラスタを同時に実行し、配備、schedulerジョブ、成果物回収、home-lab退避を含むワークフロー全体が成功した。両クラスタともEnroot 4.2.0、CUDA 13.0、PyTorch 2.11.0+cu130、SGLang、NIXL/nixl-cu13 1.4.1を確認し、共通のSquashFS（12,106,153,984 bytes、SHA-256 `6b4d81eb6400fa33400e5725a2842f63565b700d579f9be812a7c229a80da3b5`）を使用した。

玄界のPJM job 6769520は37秒、終了コード0で完了した。rank 0は`b0003`、rank 1は`b0004`に配置され、各ノードで4基のH100を確認した。成果物10300281428（6,220 bytes、SHA-256 `8180f7026d604c1b73c23d729a2fa9d6b876fb785d26dc580a81e2e27f6ffc3f`）を退避した。先行run 34693425543ではPJM job 6767065自体が成功した後、開始待機中のSSH接続が2時間13分無出力となり切断された。この実測事象に対し、PJM待機状態を15秒ごとに出力する修正を加え、単独再試行run 34700290250と統合runの双方でActions全体の成功を確認した。

ASPIRE 2A+のPBS job 208754.pbs111は34秒、`Exit_status=0`で完了した。rank 0は`a2ap-dgx035`、rank 1は`a2ap-dgx036`に配置され、各rankに1基のH100 80 GB HBM3を割り当てた。成果物10300421905（7,681 bytes、SHA-256 `c05896524527d41c8b78ada7bbb4b109352c9d4aa299adbc7a067d0f3aaba523`）を退避した。両クラスタの全ノード検証は環境と配備経路を対象とし、Firmus向け公式PD性能値の代替とはしない。

## 実行と判定

最終候補トポロジは、node 0に4基のTP2 Prefillワーカー、node 1に2基のTP4/EP4 Decodeワーカーを配置し、Decode側の`max-running-requests`を256、全ワーカーの`max-total-tokens`を131072とする。この容量は、公式の16リクエスト（入力4000＋出力200トークン）がすべて単一ワーカーに集中した場合に必要となる67200トークンを上回りつつ、自動推定容量1531197トークン全体に対するNIXLのeagerなprepared descriptor構築を防止する。最終トポロジではNIXLを維持し、8基のInfiniBandおよび2基のEthernet HCAを含むFirmus上のすべてのアクティブなHCAをUCX_NET_DEVICESに設定するとともに、各ワーカーに対して個別のAPI/ブートストラップポートを維持する。

現在の Firmus の実行パスは `platforms/shared/qwen-pd.sbatch` のみに一本化されており、ワーカーおよびルーターを起動した後、単一の v1 chat completions ゲートウェイ経由で固定の公式 AIPerf コマンドを実行します。旧来のモジュール式実行スクリプト群や個別の JSON 実行設定は、スタンドアロン契約との重複を排除するため、最終成功ランの完了後に削除されました。なお、リポジトリ内のパーサーと埋め込みスタンドアロンパーサーはバイト単位で同一なローカル検証器として維持されており、オープンループのリクエストレートは一切使用されません。

Actions `34728353522` / Firmus Slurm `17007` では、`g1`/`g4` 上で同一の bounded-KV 構成を使用しました。全 16 ランクで CUDA graph キャプチャおよび内部 PD warmup を完了し、OOM、UCX、NIXL エラーは発生しませんでしたが、選択された `prefill-3` から `decode-0` への外部 prewarm が、`bootstrap-req` が 1 のまま推移する中でクライアント側の 120 秒の `curl` 制限によりアボートされました。7分50秒後に `FAILED 28:0` で終了したものの、アーティファクト `10308936949` は回収され、ランタイムはクリーンアップされて残留ジョブはなく、AIPerf には到達しませんでした。

GitHub Actions 34729264020 / Firmus Slurm 17011 では、g1/g4 上で bounded-KV 構成を使用した。すべての PD サーバーは内部ウォームアップを完了した。外部の two-token prewarm が 01:06:17Z に開始されたが、180 秒以上にわたり応答は 0 バイトのままで、関連ログの進行が停止し、GPU 使用率も 0% であった。これをフリーズと判定し、有限なクレジットを温存するため自ジョブのみをキャンセルした。Slurm は 6 分 44 秒後に CANCELLED by 3014 を記録した。Artifact 10308133762 は回収され、ランタイムのクリーンアップが完了し、残留ジョブは存在しない。AIPerf には到達しなかった。

最終実行パスから追加の外部 prewarm を削除した。Slurm 17003 および 17005 により、制限された KV 容量でも所定の公式 AIPerf ワークロードを完了できることが示された一方、Slurm 17007 および 17011 では、1 回の余分な prewarm によって AIPerf 前にストールが発生し、本来実行可能なアロケーションが拒否され得ることが示された。公式 AIPerf コマンドには必要な 60 秒間のウォームアップが既に含まれているため、最終パスでは worker および router の準備完了直後に同コマンドを開始する。これにより、公式の 60/120/30 秒ワークロード、concurrency 16、およびスコアリング入力を変更することなく、観測された障害点が排除される。

Slurm 17005 は、558 リクエスト、エラー 0 件、good request fraction 1.0、goodput 4.536230620411887 requests/s、スコア 907.2461240823774 output tokens/s で固定の公式 AIPerf ワークロードを完了した。job exit が 1 となったのは、当時のラッパーが観測された平均 OSL に対して厳密に 200 であることを誤って要求したことのみが原因である。バージョン固定された AIPerf 0.12.0 は、観測された 200..201 の分布を osl_mismatch_count 0 として許容する。修正済みパーサーを用いて不変の raw export をリプレイすると、official_validation.valid は true となる。リポジトリの submission/results/17005 配下には、raw export、すべてのサービスログ、当初のリジェクト内容、修正後スコア、provenance、および SHA-256 マニフェストが格納されている。

GitHub Actions `34732532321` において、コミット `909dae3f198593da1ea704f6d31d570629da4773` がスタンドアロンの `platforms/shared/qwen-pd.sbatch` 経由で実行され、`g1` および `g4` 上の `16` 基の `H200` GPU を用いた Firmus Slurm ジョブ `17021` として処理された。本ジョブは所要時間 `00:11:16`、ステータス `COMPLETED`、ExitCode `0:0` で正常に完了した。固定された AIPerf `0.12.0` のフェーズ `60/120/30`、並行度 `16` の条件下で計 `558` リクエストをエラー `0` 件で処理し、good request fraction は `1.0`、goodput は `4.548422736312085` requests/s を記録したほか、設定された OSL `200` を乗じたローカル正規化スコア（goodput × OSL）は `909.6845472624169` output tokens/s、raw AIPerf 出力トークンスループットは `909.6763959671906` tokens/s であった。なお、ローカル検証の `official_validation.valid` は `true` であったが、これは主催者側のソフトウェアではなくローカルのバリデータによるものである。アーカイブパスは `/srv/qwen-artifacts/34732532321/1` であり、クリーンアップも完了して残存する Slurm ジョブは存在しない。本実行結果は、履歴として `17005` を保持しつつ、それを置き換える最終エビデンスとなる。

並行数8（Actions `34734863213` / Slurm `17023` / commit `b2ae586`）との制御比較では、320リクエスト、エラー0件、good request fraction 1.0、goodput 2.61749910791334 requests/s、ローカル正規化スコア523.499821582668 output tokens/s、生のAIPerf出力トークンスループット523.4916418979558 output tokens/sを記録し、`official_validation.valid`は`true`であった。並行数16のgoodputは並行数8より73.76979126988392%高かった。ローカル正規化スコアはgoodputに設定OSL 200を乗じた値であり、生のAIPerf出力トークンスループットとは区別する。

各条件の測定は1回のみで、ばらつきの推定値はない。両実行で宣言された実行影響条件の差は並行数8対16のみであり、この結果をHCAなど他の要因へ帰属させない。選定証跡は`submission/results/logs/17023`および`submission/results/logs/17021`に収録した。

`scripts/parse-aiperf-result.py` は AIPerf 0.12.0 の生エクスポートを読み取り、goodput、good request fraction、リクエスト／エラー数、実測ISL/OSL、0.90ゲート、および goodput の200倍の値を記録します。本スクリプトは前述と同様に完全なISL分布を検証し、平均値がちょうど4000や3986であることに対するレガシーな例外措置は一切設けられていません。さらに、エクスポートの同一性を argv、バージョン、トークナイザリビジョン、モード、並行性、ゲートウェイ、フェーズ所要時間、シード値、キャンセル／エラー状態、および内部で整合した集計情報へとバインドします。本ツールは公式パーサーではなく、固定された公式ワークロード向けのローカル検証器と位置付けられます。補正の実装根拠と判定範囲は[`aiperf-isl-validation.md`](aiperf-isl-validation.md)に記録した。

通常のランでは、サービスログ、AIPerf 生アーティファクト、スケジューラのアカウンティング、およびソースコンテナ・モデル・トークナイザーの来歴情報（provenance）を収集しますが、ワーカーの Prometheus スナップショットは収集しません。別ディレクトリへの1秒間隔の GPU、CPU、NIC、および InfiniBand テレメトリ収集はプロファイルタグ付きランでのみ実施され、これらのプロファイル結果はベンチマークスコアリングから除外されます。

## FirmusでのE2E起動検証

fresh container buildはActions run 34681001728で成功した。生成したSquashFSは12,106,153,984 bytesで、SHA-256は`6b4d81eb6400fa33400e5725a2842f63565b700d579f9be812a7c229a80da3b5`である。コンテナ内でSGLang、PyTorch 2.11.0+cu130、AIPerf 0.12.0、sglang-router 0.3.2、NIXL 1.4.1、およびNCCL testsを検証した。ビルドログとsidecarは`submission/build`にも収録している。

最初のE2Eでは6基のワーカーとルーターの起動後、AIPerfのオフラインtokenizer解決に失敗した。Hugging Face標準キャッシュの`refs/main`を固定revisionへ対応付け、6個のtokenizerファイルを1回の`hf download`で取得して全ファイルを検査することで解消した。次のE2EではAIPerfのtokenizer初期化と12,800件のsynthetic dataset生成まで進んだが、ルーターのコンテナに`/model`がマウントされておらず、ルーター自身のtokenizer登録が失敗した。ワーカーと同じモデルパスをルーターへ読み取り用に渡し、Hugging FaceおよびTransformersをオフラインモードに固定した。

これらの失敗ランはワーカー起動、NIXL初期化、単一gatewayのヘルスチェック、AIPerf入力生成という段階ごとに原因を分離するための記録であり、性能値としては採用しない。最終結果には、ルーター修正後に完走した通常ランのraw exportのみを用いる。

## 提出用バンドル

`platforms/shared/qwen-pd.sbatch`は、コンテナパスとモデルパス以外の実行構成を1つのSlurmスクリプトに固定している。2ノードのプロセス起動、6ワーカーのヘルスチェック、PDルーター、公式AIPerf、raw exportの探索、good request fractionの適格判定、およびスコア生成を同一allocation内で実行する。固定された公式 `argv` および来歴の保持を必須とし、OSL `200`、`good_request_fraction >= 0.90`、ならびに整合するISL分布を確認できた実行だけを有効とする。設定値、観測値、判定結果は`score.json`へ分けて保存する。
