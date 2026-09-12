# 旧Qwen 3.5 ランタイム構成

この文書は、OCI / Enroot への移行前に正本として運用されていた構成を履歴として保存したものである。現在の構成については [OCI / Enroot移行記録](../container-runtime-migration.md) を参照する。

ここでいう「現在」は、当時の `main` ブランチにおける [Genkai workflow](../../.github/workflows/genkai.yml) および [PJM script](../../scripts/run-qwen35.pjm) の仕様を指す。

## プロビジョニング

ワークフロー全体で使用するモデルおよびデータセットの共有ファイルシステム上のパスを以下のように定義する。

```text
MODEL_PATH=/fast/PROJECT/Qwen/Qwen3.5-397B-A17B-FP8
DATASET_PATH=/fast/PROJECT/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json
```

`provision` ジョブは GPU を使用せず、ホスト環境の `jq` コマンドを用いてモデル設定（config）が Qwen 3.5 であることを確認する。さらに safetensors のインデックスファイルを読み込み、列挙された全シャードが存在しかつ空ファイルでないことを検証する。なお、巨大なチェックポイントファイル全体の再読み込みを伴うチェックサム検証は実施しない。

ShareGPT データセットは、`hf download` コマンドにより上記のパスへ事前に取得しておく。モデルおよびデータセットは SIF 内には同梱しない。

検証の完了後、`/fast/PROJECT/sglang-runs` 配下に実行単位の一時ランタイムディレクトリを作成し、以下の資材を配置する。

```text
sglang.sif
sif-source-run.txt
scripts/
logs/
results/
```

SIF は別ファイルシステム上のアーカイブを指すシンボリックリンクであり、スクリプト類はそのワークフロー実行時のチェックアウトからコピーして配置する。PJM スクリプトの内容および結果の出力先は、リポジトリのチェックアウトやビルドディレクトリには依存しない。ただし後述のとおり、当時のジョブ投入時におけるカレントワーキングディレクトリ（cwd）にはチェックアウトへの依存が残存していた。

## PJM ジョブのライフサイクル

ワークフローは `pjsub -x` を用いて `RUNTIME_DIR`、`MODEL_PATH`、`DATASET_PATH`、およびワークフローのメタデータを環境変数として引き渡し、Node Group B の2ノード（各ノード H100 4基、計8 GPU）、実行制限時間10分の PJM ジョブを投入する。標準の `pjwait` でジョブの開始と完了を待機し、実行中のログを GitHub Actions へストリーミングする。

ベンチマークの終了コードは、プロファイララッパー、PJM スクリプト、およびワークフローへと順次伝播する。正常終了時のみ `results.jsonl` の存在を確認した上で `run-success` マーカーファイルを作成し、ワークフロー側でジョブ完了後にこのマーカーの有無を検証する。

ジョブの成功・失敗・キャンセルのいずれにおいても、ログおよび結果ファイルの回収を試行する。その後、残存する PJM ジョブを確実に停止させてから、一時ランタイムディレクトリを削除する。

ただし、当時のワークフローは `pjsub` 実行前に共有ファイルシステム上の `RUNTIME_DIR` へカレントディレクトリを移動していなかった。そのため、キュー待ちの間にエフェメラルランナーのチェックアウトが消失すると、PJM ジョブが `NO CURRENT DIR` エラーによって実行開始前に異常終了する事象が確認されている。この経緯および必要な対処については [trial-record.md](trial-record.md#7-ephemeral-runner-と-no-current-dir) を参照する。

## 計算ノード上の実行環境

PJM スクリプトは NVIDIA HPC SDK 25.11、HPC-X 2.25.1、CUDA 13.2.0、Singularity CE 4.1.3 をロードする。Open MPI は CPU バインディングを無効化した上で各ノードに1プロセスを配置し、各コンテナインスタンスがノード内の4 GPUを担当する。SGLang は `tp_size=8`、`nnodes=2` としてノード間テンソル並列（Tensor Parallelism）を構成する。Genkai の計算ノードでは通常のコンテナプロセス生成権限が制限されているため、コンテナは `--userns --nv --cleanenv --containall` フラグを付与して起動する。FUSE マウントが許可されない環境では、Singularity の一時サンドボックス展開へとフォールバックする。バインドマウントは必要最小限とし、以下のパスのみを明示的にマウントする。

- モデル: `/model:ro`
- データセットの単一 JSON ファイル: `/dataset/ShareGPT_V3_unfiltered_cleaned_split.json:ro`
- ノードローカルディスク上の書き込み可能なホームおよび JIT キャッシュ領域: `/runtime-home`、`/jit-cache`
- 実行ごとの結果出力ディレクトリ: `/output`

Hugging Face、Transformers、Datasets の各ライブラリはオフラインモードに設定する。GPU アーキテクチャやデバイスを必要とする Triton や DeepGEMM などの JIT コンパイルは、計算ノード側でのみ実行する。推論実行時の引数は環境変数ではなく、PJM スクリプト内のコマンドライン引数としてバージョン管理する。

## JIT キャッシュ

永続キャッシュは共有ファイルシステム上の `${HOME}/.cache/sglang-jit` に保持する。ただし、ベンチマーク実行中に共有ファイルシステムへ直接アクセスすることは避け、各計算ノードのローカルディスク（`/tmp/sglang-jit.*`）へスナップショットをステージングした上でバインドマウントする。

ステージング処理は共有ロック、正常終了後の `rsync` による共有ファイルシステムへの書き戻しは排他ロックで保護する。ベンチマークが失敗またはキャンセルされた場合、ローカルキャッシュの書き戻しは行わない。ノードローカルの JIT キャッシュおよび Singularity の一時ディレクトリは、ジョブ終了時に確実に削除する。

GPU 構成情報および1秒間隔のテレメトリは各ノードで収集し、生サンプルをディスクへ書き出さずにメモリ上で GPU ごとのスカラー値へ集約する。ランク0（Rank 0）のコレクターが2ノード分のデータを BigQuery へ送信し、`hardware.gpu.count` には各ノードの実測 GPU 数の総和を記録する。

## 固定ベンチマークコマンド

当時の構成では、SGLang の `sglang.benchmark.offline_throughput` を以下の主要パラメータで実行していた。

| 項目 | 値 |
|---|---|
| ロード形式（load format） | `auto` |
| テンソル並列サイズ（TP size） | 8 |
| データセット | ShareGPT |
| プロンプト数 | 32 |
| 出力トークン長（output length） | 256 |
| コンテキスト長（context length） | 262144 |
| 乱数シード（seed） | 42 |
| 静的メモリ確保比率（static memory fraction） | 0.85 |

ベンチマーク結果は `/output/results.jsonl` に、実行コマンドおよびコンソール標準出力は `/output/throughput.log` にそれぞれ保存される。
