# Firmus Slurm 17021 最終実行結果

本レポートは、現行コードベースによるスタンドアロンのエンドツーエンド（E2E）実行において、初めて最終的な成功（terminal success）を実証したエビデンス記録である。実行後のランタイムディレクトリは削除されており、クラスタ上に残存ジョブ（residual job）は存在しない。

## 実行環境および来歴

- **GitHub Actions Run ID**: `34732532321`
- **コミットハッシュ**: `909dae3f198593da1ea704f6d31d570629da4773`
- **スタンドアロンエントリーポイント**: `submission/run/submit.sbatch`
- **計算機環境**: Firmus 2ノード（16 × H200 GPU）
- **Slurm実行ステータス**: `COMPLETED`（終了コード `0:0`、経過時間 `00:11:16`）
- **コンテナSHA-256**: `6b4d81eb6400fa33400e5725a2842f63565b700d579f9be812a7c229a80da3b5`
- **コンテナサイズ**: 12,106,153,984 bytes

## ベンチマークワークロードおよび主要メトリクス

AIPerf 0.12.0の固定公式ワークロード（並行数16、フェーズ設定60秒／120秒／30秒）を実行した。

| 項目 | 測定値 |
| --- | ---: |
| 総リクエスト数 | 558 |
| エラー数 | 0 |
| Good requests | 558 |
| Good request fraction | 1.0 |
| Goodput | 4.548422736312085 requests/s |
| ローカル正規化スコア（Goodput × 設定OSL 200） | **909.6845472624169 output tokens/s** |
| AIPerf生出力トークンスループット | **909.6763959671906 output tokens/s** |
| TTFT（avg / p99 / max） | 513.7301314480287 / 1414.8025531499998 / 1570.409445 ms |
| ITL（avg / p99 / max） | 14.941216278695165 / 16.697356633115575 / 16.739300150753767 ms |
| ISL（min / max / count / sum / avg） | 3986 / 3987 / 558 / 2224189 / 3986.0017921146955 |
| OSL（min / max / count / sum / avg） | 199 / 200 / 558 / 111599 / 199.99820788530465 |
| `osl_mismatch_count` | 0 |
| ローカル検証結果（`official_validation.valid`） | `true` |

## アーティファクトとエビデンス構成

- **GitHub artifact ID**: `10309459843`
  - SHA-256: `b71ff85fee6634b902980da2d1e9154ec65e20f903ed96537a62f07b050fd428`
  - `/hdd_raid/samba/net_share/qwen-artifacts/34732532321/1`へ永続保存後、GitHub上の一時コピーを削除した。
  - 完全なアーティファクトにはAIPerfが生成した補助ファイルも含まれる。
- この選定バンドルには、raw export、実行コマンドと来歴、全6ワーカー、router、AIPerf、Slurmの各ログ、source provenance、スコア、`run-success`、`SHA256SUMS`を収録した。

## リプレイ・検証コマンド

本パーサーは公式パーサーではなく、固定された公式ワークロード向けのローカル検証器である。

```bash
uv run --project qwen-experiment-metrics python scripts/parse-aiperf-result.py submission/results/logs/17021/profile_export_aiperf.json /tmp/score-17021.json --concurrency 16 --require-eligible --command submission/results/logs/17021/aiperf-command.txt --provenance submission/results/logs/17021/aiperf-provenance.txt
```
