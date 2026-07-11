# SGLang + Qwen 3.5 experiments on Genkai

Qwen 3.5 の SGLang 推論性能を再現可能に比較するための実験リポジトリです。主指標は `sglang.benchmark.offline_throughput` の最大スループットです。実ウェイトが揃う前は `LOAD_FORMAT=dummy` を使います。

## 実行モード

| mode | 用途 | 出力 |
|---|---|---|
| `throughput` | ウォームアップ後に複数回測定 | 環境メタデータ、各 trial のログ |
| `torch-profile` | SGLang 内蔵 PyTorch profiler | trace とログ |
| `nsys` | Nsight Systems による CUDA/OS trace | `.nsys-rep` とログ |
| `smoke` | 小さい入力で起動確認 | ログ |

結果は `results/<UTC日時>-<git SHA>-<mode>/` に分離され、Git SHA、差分有無、GPU、Python、SGLang、引数を保存します。

## 使い方

モデルの `config.json` と tokenizer があるローカルディレクトリを指定します。外部ダウンロードは既定で禁止しています。

```bash
MODEL_PATH=/path/to/Qwen3.5-27B LOAD_FORMAT=dummy scripts/run-experiment.sh smoke
MODEL_PATH=/path/to/Qwen3.5-27B LOAD_FORMAT=dummy scripts/run-experiment.sh throughput
MODEL_PATH=/path/to/Qwen3.5-27B LOAD_FORMAT=dummy scripts/run-experiment.sh torch-profile
MODEL_PATH=/path/to/Qwen3.5-27B LOAD_FORMAT=dummy scripts/run-experiment.sh nsys
```

既定値は [configs/default.env](configs/default.env) にあり、環境変数で上書きできます。`EXTRA_ARGS` で SGLang の最適化フラグを追加できます。

## Genkai

[jobs/genkai-qwen.pjm](jobs/genkai-qwen.pjm) は H100 1基の `b-batch` 用です。ログインノードでは実行しません。

```bash
REPO_DIR=$PWD MODEL_PATH=/path/to/local/model pjsub jobs/genkai-qwen.pjm
```

GitHub Actions 自体は push ごとに Genkai の self-hosted runner で起動し、スクリプトを検証します。コミットメッセージに `[run-qwen]` がある場合だけ投入準備ステップへ進みます。ただし安全のため `pjsub` はコメントアウトしてあり、現在は予定コマンドの表示だけです。手動 dispatch も `submit=true` の明示確認が必要です。

## 推奨手順

1. `smoke`（dummy）で互換性とメモリ容量を確認する。
2. `throughput` を最低3 trial 実行し、中央値で比較する。
3. 差が出た変更だけ `torch-profile`、カーネル/同期の調査は `nsys` を使う。
4. 実ウェイト到着後、`LOAD_FORMAT=auto` に変えて dummy 結果との差を記録する。
