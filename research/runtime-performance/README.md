# GenkaiにおけるQwen 3.5 benchmarkの調査記録（旧構成）

この記録は実モデルとSIFを使った従来経路の履歴である。現行の両クラスタdummy経路は [OCI / Enroot移行記録](../container-runtime-migration.md) を参照する。

このディレクトリは、Genkai の GPU 計算ノードで Qwen 3.5 を SGLang により実行し、変更の影響を比較するための固定条件と運用方法を記録する。

workflow は実行ごとに SIF、script、log、result をまとめた一時 runtime directory を作り、PJM job の終了後に成果物を Actions artifact へ回収する。

現行構成は [current-runtime.md](current-runtime.md)、実行時の失敗と設計判断は [trial-record.md](trial-record.md) を参照する。

## 現在の比較対象

- model: `Qwen/Qwen3.5-397B-A17B-FP8`
- workload: SGLang `offline_throughput`
- dataset: `anon8231489123/ShareGPT_Vicuna_unfiltered`
- node: Genkai Node Group B x 2
- GPU: 8
- tensor parallel size: 8
- prompt 数: 32
- output length: 256
- seed: 42

この benchmark は SGLang の変更を同条件で比較するためのものである。model、dataset、benchmark 引数を変更した結果は、従来条件の結果と直接比較しない。

## 成果物

`results/` には次を出力する。

| ファイル | 内容 |
|---|---|
| `environment.log` | workflow、PJM、commit、SIF、module、GPU、container、model と dataset の情報 |
| `throughput.log` | 実行コマンド、benchmark の標準出力・標準エラー、終了 code |
| `results.jsonl` | SGLang benchmark の構造化結果 |

PJM の標準出力・標準エラーは `logs/<job-id>.log` に保存する。workflow は logs と results を別々の Actions artifact として回収する。

現時点の [profiler wrapper](../../scripts/exec-profiler.sh) は受け取ったコマンドをそのまま実行する。profiler を導入するときは、この wrapper に起動設定を追加し、benchmark 本体と workflow の変更を避ける。
