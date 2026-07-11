# SGLang Qwen experiments on Genkai

このリポジトリは、Qwen 3.5のSGLang推論をGenkai上で高速化するための実験基盤です。
性能値だけでなく、実験設定、Git commit、ジョブスクリプト、GPU環境、SGLang出力を同じ結果ディレクトリへ保存します。

## 実験の流れ

実験条件は `experiments/*.env` に置きます。
最初は既存のbaselineを複製し、一つの最適化だけを変更します。

```bash
cp experiments/qwen35-fp8-baseline.env experiments/qwen35-fp8-cuda-graph.env
$EDITOR experiments/qwen35-fp8-cuda-graph.env
```

ジョブ生成はローカルでも実行できます。
このコマンドは `pjsub` を呼びません。

```bash
python3 scripts/submit.py experiments/qwen35-fp8-baseline.env
bash -n build/jobs/qwen35-fp8-baseline.pjm
```

Genkaiでの投入には明示的な `--submit` と、実験名の再入力が必要です。

```bash
python3 scripts/submit.py experiments/qwen35-fp8-baseline.env --submit
```

## 実行モード

| mode | 測定対象 | 主な出力 |
|---|---|---|
| `smoke` | 起動、モデル互換性、OOM | `smoke.log` |
| `throughput` | HTTPを介さない最大throughput | `results.jsonl`、trial log |
| `torch-profile` | PyTorch演算とCUDA kernel | profiler trace |
| `nsys` | kernel、同期、CPU/GPU並行性 | `.nsys-rep` |

throughput測定には `python -m sglang.benchmark.offline_throughput` を使います。
モデル取得を測定へ混ぜないため、Hugging Faceのオンラインアクセスは無効です。

## 結果の比較

結果ディレクトリから中央値をCSVへ集約します。

```bash
python3 scripts/summarize.py \
  /fast/pj26000080/qwen-results/<baseline> \
  /fast/pj26000080/qwen-results/<candidate> \
  --output summary.csv

python3 scripts/compare.py summary.csv \
  --baseline baseline \
  --candidate candidate
```

判定方法は [実験の判定基準](docs/experiment-policy.md) に定めています。

## GitHub Actions

pushごとにGenkaiのself-hosted runnerでスクリプトを検証します。
コミットメッセージに `[run-qwen]` がある場合だけ投入準備へ進みます。
現在のworkflowでは `pjsub` をコメントアウトしているため、Actionsからジョブは投入されません。

## 構成

```text
experiments/        比較対象となる実験設定
jobs/templates/     設定からPJMを生成するtemplate
scripts/            実行、生成、投入、集約、比較
docs/               測定と採用の規則
results/            ローカル結果（Git管理外）
tests/              shellと集約処理の回帰テスト
```
