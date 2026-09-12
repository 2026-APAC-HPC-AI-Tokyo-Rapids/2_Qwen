# 公式Qwen3 PD実装記録

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

Firmusで公式に検証された初期構成を採用し、node 0にPrefillワーカー4基をTP2で、node 1にDecodeワーカー2基をTP4/EP4で配置する。Decode側の`max-running-requests`は256とする。全ワーカーで通信バックエンドにNIXLを選択し、計算ノードごとにアクティブなInfiniBandデバイスを検出して`UCX_NET_DEVICES`へと引き渡す。各ワーカーのAPIポート番号および各Prefillワーカーのブートストラップポート番号は重複しないよう一意に割り当てる。

`scripts/run-qwen-pd-job.sh`は、ワーカー群、ルーター、AIPerfの順序でプロセスを起動する。AIPerfから参照されるエンドポイントは、ルーターが提供する`/v1/chat/completions`のみである。`scripts/run-aiperf.sh`は公式モード（official）と探索モード（exploratory）を明示的に切り分け、公式モードでは規定のワークロード引数を厳密に固定する。探索モードでのみ、`configs/firmus-pd.json`に定義された短時間実行の設定を適用できる。なお、いずれのモードにおいてもopen-loop用の`--request-rate`は指定しない。

`scripts/parse-aiperf-result.py`は、AIPerf 0.12.0が出力する`profile_export_aiperf.json`からgoodput、good request fraction、リクエスト数／エラー数、および実測されたISL/OSLを読み取り、公式の0.90合格判定（gate）と算出スコアをJSONファイルへ保存する。公式モードでは、保存したargv全体、AIPerf version、tokenizer revision、mode、concurrency、gatewayが固定ワークロードと一致することも検証する。この来歴が揃う場合だけ、設定ISL 4000に加えてAIPerfの`first_turn_prefix`補正後にclient exportへ記録される3986を厳密に受理する。raw観測値と`goodput × 200`の計算は変更しない。公式モードにおいて合格基準または来歴検証を満たさない場合はジョブを失敗（fail）扱いとする。探索モードでは、基準未達のデータも分析用成果物としてそのまま保持する。補正の実装根拠と判定範囲は[`aiperf-isl-validation.md`](aiperf-isl-validation.md)に記録した。

通常ランにおける観測対象は、各種サービスログ、AIPerf成果物、およびワーカーのPrometheusスナップショットである。コミットメッセージに`[profile]`を付与したランに限り、1秒間隔のGPUテレメトリも併せて取得する。プロファイリングに伴う計測負荷が含まれるため、`[profile]`付きランの数値を通常ランの公式スコアとして扱ってはならない。

## FirmusでのE2E起動検証

fresh container buildはActions run 34681001728で成功した。生成したSquashFSは12,106,153,984 bytesで、SHA-256は`6b4d81eb6400fa33400e5725a2842f63565b700d579f9be812a7c229a80da3b5`である。コンテナ内でSGLang、PyTorch 2.11.0+cu130、AIPerf 0.12.0、sglang-router 0.3.2、NIXL 1.4.1、およびNCCL testsを検証した。ビルドログとsidecarは`submission/build`にも収録している。

最初のE2Eでは6基のワーカーとルーターの起動後、AIPerfのオフラインtokenizer解決に失敗した。Hugging Face標準キャッシュの`refs/main`を固定revisionへ対応付け、6個のtokenizerファイルを1回の`hf download`で取得して全ファイルを検査することで解消した。次のE2EではAIPerfのtokenizer初期化と12,800件のsynthetic dataset生成まで進んだが、ルーターのコンテナに`/model`がマウントされておらず、ルーター自身のtokenizer登録が失敗した。ワーカーと同じモデルパスをルーターへ読み取り用に渡し、Hugging FaceおよびTransformersをオフラインモードに固定した。

これらの失敗ランはワーカー起動、NIXL初期化、単一gatewayのヘルスチェック、AIPerf入力生成という段階ごとに原因を分離するための記録であり、性能値としては採用しない。最終結果には、ルーター修正後に完走した通常ランのraw exportのみを用いる。

## 提出用バンドル

`submission/run/submit.sbatch`は、コンテナパスとモデルパス以外の実行構成を1つのSlurmスクリプトに固定している。2ノードのプロセス起動、6ワーカーのヘルスチェック、PDルーター、公式AIPerf、raw exportの探索、good request fractionの適格判定、およびスコア生成を同一allocation内で実行する。固定コマンドと来歴が一致する場合に限り平均ISL 4000またはprefix補正後のclient観測値3986を受理し、平均OSL 200から外れる結果は失敗させる。設定値、観測値、判定結果は`score.json`へ分けて保存する。
