# ASPIRE 2A+向けAISimulate実現性検証

2026-09-26。対象は `aspire2p.nscc.sg`、2ノード・16 H100。これはCPU上の事前評価であり、公式性能結果、実機での起動成功、accuracy合格を意味しない。

## 再現

リポジトリrootから実行する。GPU、モデル重み、Dynamoサービスは不要。

```bash
uv venv --python 3.12.12 /tmp/qwen-simulation-env
uv pip sync --python /tmp/qwen-simulation-env/bin/python research/simulation/requirements.lock
/tmp/qwen-simulation-env/bin/python research/simulation/run.py /tmp/qwen-simulation-output
```

出力先には未作成のディレクトリを指定する。`prediction.yaml` がこのコミットの唯一の実験条件。条件比較はこのファイルを変更した個別commitをpushする。matrix/dispatch/sweepで条件を切り替えない。CPU専用workflow `simulation.yml` が診断・予測・raw request・source/memory/time details・来歴をartifactへ保存する。クラスタタグは付けず、PBS/GPU計算は起動しない。依存関係は今回検証した環境を全件固定した。

`run.py` はroleごとにop-level estimatorを検査し、未対応ならexit 2と診断を保存する。固定時間モデルや未学習のregressionへ切り替えて成功を装わない。予測結果には `official_result: false` を付ける。SLAの参考判定はrequest単位のTTFTと平均ITLの同時条件を使い、集計平均の合格だけでは判定しない。

## 入力の来歴

- 開始時のmain: `aa28a7d`。ASPIREは1×TP8 prefill / 1×TP8 EP8 decode、concurrency 16。`submission/run/submit.sbatch` が正本。
- AISimulate wheel: `0.13.0.dev202609240000000055`、Python 3.12.12。参照したupstream source: `992dddc7ee2bad8fd523e7db99eea4ac64254766`（wheelとの同一性を主張しない）。実行APIはインストールしたwheelでも確認した。
- `model/config.json`: [公式FP8 checkpointの固定revision](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8/blob/e156cb4efae43fbee1a1ab073f946a1377e6b969/config.json)。SHA-256 `f056921e486d3e22aed125b5218b87146ea2ac838ccf1521dea5b08f809e8d59`。重みは同梱しない。ローカルconfigによりHF mainの変化を避ける。
- 性能データ: H100 SXM / SGLang `0.5.14`。実配備のSGLang source `fc275674a97d78553b22597ee72c8fd48f5420f4` と同一ではない。
- ASPIRE loginへの読み取りで `a2ap-login01`、共有scratch=Lustre、loginの/tmp=XFSを確認。compute nodeのlocal filesystemや実効NIC帯域は今回測定していない。

## 現行構成で見つかった制約

現行PrefillのTP8、MoE TP8を入力すると、AISimulateが次の理由で拒否した。

```text
Invalid quantized MoE configuration:
(moe_intermediate_size=1536 / moe_tp_size=8) % weight_block_size=128 != 0
```

1536/8=192が128の倍数でない。最初のCLIエラーは `regression estimator is not ready` だったが、SDK diagnosticsで上の根本理由を確認した。これはsimulatorの構成制約であり、実配備SGLangのpadding対応を調べずに実機失敗と断定しない。

以前のActions [35108498091](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/35108498091) ではASPIREのPBS `209623.pbs111` が失敗し、`run-success` が欠けていた。Actionsログだけでは最初のPBS致命エラーは分からず、今回のMoE制約と同じ原因とは認定しない。Firmusの成功値もH100校正に流用しない。過去のresearchと2026-09-16のCodex会話記録を照合した。

## CPU実行で確認した候補

現行形状を保存したcommitは `e61bb78`。その[CPU Actions](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/36220794741)は上記のTP8不整合を検出して失敗、artifact回収まで完了した。同commitの通常Qwen workflowはクラスタタグなしの静的検証に成功した。

この後の `prediction.yaml` はnode 0に2×TP4 prefill、node 1に1×TP8/EP8 decodeを置く候補。BF16 KVとBF16 attentionを明示する。simulatorはFP8 checkpointからKV/attentionもFP8と推論するため、省略時の予測を実機のauto/BF16設定と比較しない。KVのみBF16に変更するとFP8 attention/BF16 KVの性能データがなく失敗した。両方を明示して予測を完走した。最初のTP8拒否は精度既定値の補正前であり、性能の対照実験ではない。

ローカルCPU実行（160 requests、N=16、仮定50 GB/s）では、予測throughput 4.492 requests/s、平均TTFT 275.6 ms、p99 TTFT 1167.8 ms、平均ITL 16.22 ms、request単位の同時SLA達成160/160。仮想時間35.62秒、replay部分のCPU wall time約1.24秒。初期化や依存導入時間を含まない。これは感度検証の開始点であり、実機の最適構成・性能改善を示す結果ではない。

候補commit `bc3fda9` の[CPU Actions 36220906565](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/36220906565)は成功。artifact `10899022184`（86,740 bytes）をダウンロードし、raw request、予測、来歴、診断を確認した。ローカルとthroughput・SLA件数・仮想時間が一致し、TTFT平均の差は浮動小数点丸めの範囲だった。Actionsのreplay wall timeは約0.98秒。基準の失敗artifactは `10898452544`。採用候補を保証する性能比較やASPIRE GPU実行は行っていない。

## 適用範囲と限界

DynoSimはDynamo Router/Plannerを含むシミュレーションの入口で、AISimulateは共通の予測エンジン。現在の実装はSGLang + sglang-router 0.3.2であり、まず `--stack engine` を使用する。Dynamoへの移行やDynoSimのruntime統合は今回実行していない。

事前評価に向くのはTP/EP、replica/P:D比、concurrency、バッチ容量、KV容量の候補削減。ノード内8 GPUへの配置と2ノード・16 GPUの合計を守る。H100 80 GiBでは236 GBの重みをTP2に収容できないため、Firmusの4×TP2 prefillをそのまま移植しない。

今回の仮定・差分:

- 4000/200 tokens、concurrency 16、160 requests。Shakespeare本文・tokenizer/chat template・seed 2026・AIPerf 60/120/30秒は再現していない。prefix cacheを無効化してcache-bustの効果を近似した。公式の有効ISL 3984–3988との差も残る。
- context length 8192、memory fraction 0.90。ただし実機の `max-total-tokens=131072`、chunked prefill、CUDA graph予約量との完全一致は未検証。simulatorは既定のHBM容量推定を使う。
- weight FP8は公式configから推論。KVとattentionはBF16を明示。転送は全層の非重複KV `2×94×4×128×2=192512 bytes/token` と仮定する。TP>KV head数での複製、分割方式、NIC競合やNIXL接続確立は再現しない。
- KV転送に50 GB/sを仮定する。実測値ではなく、H100 system profileのper-GPU帯域からnode全体への一致を主張しない。既定の無遅延を避けるための感度検証の起点。
- 汎用H100のNCCL/演算データには実測、補間、経験モデルが混在する。ASPIREのNIC/クロック/ライブラリに固有の校正値ではない。
- 合成closed-loopの終了までの予測であり、120秒の公式goodputや公式スコアを算出・提出しない。simulatorはmodel outputを生成せずaccuracyも判定できない。

## 実機前後の進め方

1. 条件ごとにcommitし、CPU Actionsでunsupported/OOMの候補を除外する。主要因は一度に一つ（最初はprefillのTP/replica構成）。
2. 基準候補のN=8/16/32、次いで転送帯域の仮定を個別commitで評価する。平均性能だけでなくrequest単位の同時SLAを確認する。今回そのsweep全体は実行していない。
3. 2～3候補に絞って、ASPIRE上のActions→PBS→Enroot→SGLang/NIXL→単一gateway→AIPerf→artifact回収で実測する。少数の実測を使って順位・予測誤差を校正し、同一条件を繰り返す。profileは通常benchから分ける。
4. 最終候補は公式AIPerfの固定60/120/30秒で確認する。ローカル `scripts/parse-aiperf-result.py` は主催者の公式parserではない。主催者の再検証・accuracy判定が最終基準。

## 出典

- 指定された統合要綱（2026-09-26）
- [公式Qwen rulesと9/25 Addendum](https://github.com/hpcac/2026-APAC-HPC-AI/blob/main/1.%20Qwen3%20AI%20Task%20Rules.md)
- [DynoSimの位置付け](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/simulation/overview)
- [AISimulate CLI](https://github.com/ai-dynamo/aisimulate/blob/992dddc7ee2bad8fd523e7db99eea4ac64254766/docs/cli/user-guide.md)

校正データの収集と実測の研究は [Issue #15の記録](CALIBRATION.md) を参照。
