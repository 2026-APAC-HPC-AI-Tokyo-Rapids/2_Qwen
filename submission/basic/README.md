# Basic — 公開版SGLang、source変更なし

既存の[Bonus提出物](../README.md)と分けて、公式SGLang 0.5.23の起動オプション・環境変数・配置だけを調整する。Firmus 2 node / 16 H200 GPUを使用する。現時点のconfigは比較中の候補であり、適格な最適点の選定は未完了。初回c160の測定結果、進行中の比較と採否基準は[実験記録](../../research/runtime-performance/2026-10-30-basic-stock-sglang.md)を参照。

## 再実行

このディレクトリ単体ではなく、リポジトリ全体を取得する。依存する制御スクリプトはリポジトリ内にあり、SGLangのsource patchではない。

1. [ビルド・配備手順](../../README.md#basic-and-bonus-submissions-2026-10-30-policy)に従い、公式imageから検証済みSquashFSとsidecarを用意する。比較の再現にはconfigに固定した既存imageを使う。独立再ビルド時は生成したsidecarのSHA・size・source commit・build IDをコピーしたconfigへ反映する。
2. 公式target `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` revision `e156cb4efae43fbee1a1ab073f946a1377e6b969` の全24 shardとmetadataを共有領域へ置く。固定tokenizer `Qwen/Qwen3-235B-A22B-Instruct-2507` revision `ac9c66cc9b46af7306746a9250f23d47083d689e` のconfig.json、generation_config.json、merges.txt、tokenizer.json、tokenizer_config.json、vocab.jsonも用意する。
3. login nodeで次のprepareを行う。public Nebius draftはこの段階で取得し、重みSHAとBF16を検査する。

```bash
bash submission/basic/prepare.sh /scratch/home/ACCOUNT/qwen-basic-run /shared/BUILD.sqsh /shared/Qwen3-235B-A22B-Instruct-2507-FP8 /shared/tokenizer/ac9c66cc9b46af7306746a9250f23d47083d689e configs/firmus-basic.json
cd /scratch/home/ACCOUNT/qwen-basic-run
sbatch --export=PATH,HOME,USER,LOGNAME scripts/submit.sbatch
```

Slurm script内のpartition/resource指定はFirmus向け。prepareは空のrun directoryを新規作成する。既存runへ上書きしない。scheduler allocation内でP3/D1、公開gateway、readiness、実設定検査、接続sanity、固定60/120/30秒の公式AIPerfを実行する。AIPerfは隔離venvに置き、serving依存packageは更新しない。

## 結果の確認

ジョブのterminal状態とexit codeを確認し、`results/score.json` の `official_validation.valid`、`eligible`、good fraction、request errorsを確認する。source/version/topologyと要求・実KV poolのreadback一致も必須。`run-success` はそれらが通った場合のみ生成される。sanity応答は主催者による全件accuracy検査の代替にはならない。

run directoryの`source/`、`logs/`、`results/`を回収してchecksum付きで保管した後にrun directoryを削除する。SquashFSは後続実験で再利用する。Actions経由ではこの回収・保管・削除まで自動化し、資格情報は計算nodeへ渡さない。

Actionsはarchiveのchecksumとterminal状態を検査してから、既存のBigQuery senderで設定・来歴・公式score・AIPerf指標を送信する。派生JSONLはraw archiveの外側に保存する。過去の結果は `gh workflow run qwen-basic-metrics.yml --ref main -f measurement=RUN_ID/ATTEMPT` で指標だけを送信できる。この操作ではGPUジョブを投入しない。送信失敗は測定結果を無効にせず、同workflowのログに残す。
