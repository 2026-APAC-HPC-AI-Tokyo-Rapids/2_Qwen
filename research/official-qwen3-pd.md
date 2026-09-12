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

Firmus上の`/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8`には24個のsafetensorsシャードが配置されており、Hugging Faceダウンロードメタデータのコミットリビジョンも上表のmodel revisionと完全に一致することを確認した。玄界にも221 GiBのモデル重みが配置されている。一方、ASPIRE 2A+は確認時点で77 GiBのダウンロードキャッシュのみが存在し、model rootにsafetensorsファイルが存在しないため、現時点での実行対象からは除外している。

## 実行と判定

Firmusで公式に検証された初期構成を採用し、node 0にPrefillワーカー4基をTP2で、node 1にDecodeワーカー2基をTP4/EP4で配置する。Decode側の`max-running-requests`は256とする。全ワーカーで通信バックエンドにNIXLを選択し、計算ノードごとにアクティブなInfiniBandデバイスを検出して`UCX_NET_DEVICES`へと引き渡す。各ワーカーのAPIポート番号および各Prefillワーカーのブートストラップポート番号は重複しないよう一意に割り当てる。

`scripts/run-qwen-pd-job.sh`は、ワーカー群、ルーター、AIPerfの順序でプロセスを起動する。AIPerfから参照されるエンドポイントは、ルーターが提供する`/v1/chat/completions`のみである。`scripts/run-aiperf.sh`は公式モード（official）と探索モード（exploratory）を明示的に切り分け、公式モードでは規定のワークロード引数を厳密に固定する。探索モードでのみ、`configs/firmus-pd.json`に定義された短時間実行の設定を適用できる。なお、いずれのモードにおいてもopen-loop用の`--request-rate`は指定しない。

`scripts/parse-aiperf-result.py`は、AIPerf 0.12.0が出力する`profile_export_aiperf.json`からgoodput、good request fraction、リクエスト数／エラー数、および実測されたISL/OSLを読み取り、公式の0.90合格判定（gate）と算出スコアをJSONファイルへ保存する。公式モードにおいて合格基準に達しなかった場合はジョブを失敗（fail）扱いとする。探索モードでは、基準未達のデータも分析用成果物としてそのまま保持する。

通常ランにおける観測対象は、各種サービスログ、AIPerf成果物、およびワーカーのPrometheusスナップショットである。コミットメッセージに`[profile]`を付与したランに限り、1秒間隔のGPUテレメトリも併せて取得する。プロファイリングに伴う計測負荷が含まれるため、`[profile]`付きランの数値を通常ランの公式スコアとして扱ってはならない。

## FirmusでのE2E起動検証

fresh container buildはActions run 34681001728で成功した。生成したSquashFSは12,106,153,984 bytesで、SHA-256は`6b4d81eb6400fa33400e5725a2842f63565b700d579f9be812a7c229a80da3b5`である。コンテナ内でSGLang、PyTorch 2.11.0+cu130、AIPerf 0.12.0、sglang-router 0.3.2、NIXL 1.4.1、およびNCCL testsを検証した。ビルドログとsidecarは`submission/build`にも収録している。

最初のE2Eでは6基のワーカーとルーターの起動後、AIPerfのオフラインtokenizer解決に失敗した。Hugging Face標準キャッシュの`refs/main`を固定revisionへ対応付け、6個のtokenizerファイルを1回の`hf download`で取得して全ファイルを検査することで解消した。次のE2EではAIPerfのtokenizer初期化と12,800件のsynthetic dataset生成まで進んだが、ルーターのコンテナに`/model`がマウントされておらず、ルーター自身のtokenizer登録が失敗した。ワーカーと同じモデルパスをルーターへ読み取り用に渡し、Hugging FaceおよびTransformersをオフラインモードに固定した。

これらの失敗ランはワーカー起動、NIXL初期化、単一gatewayのヘルスチェック、AIPerf入力生成という段階ごとに原因を分離するための記録であり、性能値としては採用しない。最終結果には、ルーター修正後に完走した通常ランのraw exportのみを用いる。

## 提出用バンドル

`submission/run/submit.sbatch`は、コンテナパスとモデルパス以外の実行構成を1つのSlurmスクリプトに固定している。2ノードのプロセス起動、6ワーカーのヘルスチェック、PDルーター、公式AIPerf、raw exportの探索、good request fractionの適格判定、およびスコア生成を同一allocation内で実行する。実測された平均ISL/OSLが4000/200から外れる場合も公式結果として失敗させる。
