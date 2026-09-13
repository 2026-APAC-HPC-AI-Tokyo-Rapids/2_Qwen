# Firmus Slurm 17023 並行数8の対照実行結果

## 実行概要

- **Slurm Job ID**: `17023`
- **GitHub Actions Run ID**: `34734863213`
- **コミットハッシュ**: `b2ae58604edbd7590889db8f47bf62e769ed3467`
- **投入スクリプト**: `submission/run/submit.sbatch`
- **クラスタ・ノード構成**: Firmus 2ノード / 16 × H200
- **実行ステータス / 経過時間**: `COMPLETED 0:0`（実行時間: `00:11:17`）
- **コンテナSHA-256**: `6b4d81eb6400fa33400e5725a2842f63565b700d579f9be812a7c229a80da3b5`
- **コンテナサイズ**: 12,106,153,984 bytes

## ワークロード設定およびメトリクス

AIPerf 0.12.0の固定公式ワークロードを、並行数8、ウォームアップ60秒、ベンチマーク120秒、グレース期間30秒で実行した。

| 項目 | 測定値 |
| --- | ---: |
| 総リクエスト数 | 320 |
| Good requests | 320 |
| エラー数 | 0 |
| Good request fraction | 1.0 |
| Goodput | 2.61749910791334 requests/s |
| ローカル正規化スコア（Goodput × 設定OSL 200） | 523.499821582668 output tokens/s |
| AIPerf生出力トークンスループット | 523.4916418979558 output tokens/s |
| TTFT（avg / p99 / max） | 442.186383334375 / 833.8151279900001 / 1002.226177 ms |
| ITL（avg / p99 / max） | 12.998010956415996 / 14.948854403618089 / 15.011618633165828 ms |
| ISL（min / max / count / sum / avg） | 3986 / 3987 / 320 / 1275521 / 3986.003125 |
| OSL（min / max / count / sum / avg） | 199 / 200 / 320 / 63999 / 199.996875 |
| `osl_mismatch_count` | 0 |
| `official_validation.valid` | `true` |

ローカル正規化スコアと生出力トークンスループットの微小な差は、観測されたOSLの合計が64,000ではなく63,999であったことによる。本パーサーは公式パーサーではなく、固定された公式ワークロード向けのローカル検証器である。

## アーティファクト

- **完全アーティファクトID**: `10310578086`
- **SHA-256**: `e1c56c0aa1cf9af2120d422c59cb93f9dfdd382be4c7893e497a305341c51dc2`
- **サイズ**: 86,550,407 bytes
- `/srv/qwen-artifacts/34734863213/1`へ転送・保存後、GitHub上の一時コピーを削除した。
- 選定バンドルでは大容量のAIPerf補助アーティファクトを除外し、生のエクスポート、実行コマンドと来歴、全6ワーカー、router、AIPerf、Slurmの各ログ、モデル・コンテナ・ソースの来歴、スコア、`run-success`、`SHA256SUMS`を収録した。

## リプレイ・検証コマンド

```bash
uv run --project qwen-experiment-metrics python scripts/parse-aiperf-result.py submission/results/logs/17023/profile_export_aiperf.json /tmp/score-17023.json --concurrency 8 --require-eligible --command submission/results/logs/17023/aiperf-command.txt --provenance submission/results/logs/17023/aiperf-provenance.txt
```

## 並行数16との制御比較

| 項目 | 並行数8（17023） | 並行数16（17021） |
| --- | ---: | ---: |
| Actions Run ID | 34734863213 | 34732532321 |
| リクエスト数 | 320 | 558 |
| エラー数 | 0 | 0 |
| `official_validation.valid` | `true` | `true` |
| Goodput（requests/s） | 2.61749910791334 | 4.548422736312085 |
| ローカル正規化スコア（output tokens/s） | 523.499821582668 | 909.6845472624169 |
| AIPerf生出力トークンスループット（tokens/s） | 523.4916418979558 | 909.6763959671906 |

並行数16のGoodputは並行数8より73.76979126988392%高かった。各条件の試行数はそれぞれ1回であり、ばらつきの推定値はない。実行結果に影響する宣言済みの条件差は並行数8対16のみであるため、この差をHCAやその他の要因へ帰属させない。
