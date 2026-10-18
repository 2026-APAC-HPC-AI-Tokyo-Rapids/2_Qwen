# 実行性能の調査記録

現行Qwen3-235B-A22B・公式4000/200タスクの調査:

- [2026-10-18: Firmus で Decode MegaMoE kernel 3 patch と Prefill 静的 expert 配置を束ねた構成が 7251.58 / 7185.18（参照 6521.98 / 6485.02 比 +11%、ITL 不合格 82–106 → 7–9 件、Decode MegaMoE −13%、Prefill −16%）、ブランチ止め・merge 判断待ち](2026-10-18-firmus-decode-kernel-placement-bundle.md)
- [2026-10-18: SM90 MegaMoE Decode の FC1 量子化器（bank conflict 解消、epilogue 21→12.5 µs/CTA）と dispatch の remote load 重ねを hoist に追加（Decode −0.5〜2.4%、image 比 −3.6〜12.5%、出力 bit 一致、ブランチ止め）。arrival の後回しは不採用](2026-10-18-megamoe-decode-epilogue-dispatch.md)
- [2026-10-18: SM90 MegaMoE Decode mainloop の software pipeline 案（不採用）と N_SWAP hoist（kernel 単体で Decode −2〜11%、出力 bit 一致、ブランチ止め）](2026-10-18-megamoe-decode-mainloop-pipeline.md)
- [2026-10-18: Prefill MegaMoE の EP rank 負荷不均衡（trivial配置 max/mean 1.338）と静的expert置換のbench（kernel −8%、score は単一runで判定不能・未採用）](2026-10-18-megamoe-expert-placement.md)
- [2026-10-18: SM90 MegaMoE の kernel 単体 bench（`[kernel-bench]`）と kernel 内 timeline、Decode-128／Prefill-4096 の段階別内訳](2026-10-18-megamoe-kernel-timeline.md)
- [2026-10-17: ブランチ統合時の監査（統合した基点、取り込んだ修正、未取り込みの候補、残すブランチ）](2026-10-17-branch-integration-audit.md)
- Firmus vLLM比較の記録: [初期プロファイル](../firmus-framework-profile-20261009.md)、[EAGLE3 c64](../firmus-vllm-eagle-c64.md)、[EAGLE3 c104](../firmus-vllm-eagle-c104.md)、[c104最適化](../firmus-vllm-c104-optimization.md)
- [2026-10-13: 10/09〜10/13改善キャンペーンのまとめ（最良値、効いた手法・効かなかった手法、残る手段）](2026-10-13-campaign-summary.md)
- [2026-10-09〜13: Prefill供給律速とP/D再配分、Decode律速、XQA・twoshot・page64・TensorRT-LLMの検証](2026-10-09-prefill-supply-rebalance.md)
- [2026-10-10: DP attention＋DeepEPへの構成転換の計画と結果（不採用）](2026-10-10-dp-attention-deepep-plan.md)
- [2026-10-09: FirmusのEP2 / EP8共有expert検証](2026-10-09-firmus-expert-sharing.md)
- [2026-10-09: AISimulateの評価範囲、ベンチの妥当性、DP/EP/PP/TP/CPの再評価](2026-10-09-simulation-parallelism.md)
- [2026-10-09: ASPIRE H100 / Firmus H200の容量・負荷検証](2026-10-09-capacity-load.md)
- [並列構成の容量・計算量・通信量の概算](2026-10-09-parallelism-bounds.json)

## GenkaiにおけるQwen 3.5 benchmarkの調査記録（旧構成）

以下は、実モデルとSIFを用いていた従来経路の履歴であり、現行の性能条件ではない。その後の移行経緯は [OCI / Enroot移行記録](../container-runtime-migration.md) を参照のこと。

本ディレクトリは、玄界のGPU計算ノード上でQwen 3.5をSGLangにより実行し、各種変更が性能に与える影響を厳密に比較するための固定条件および運用手順を記録している。

ワークフローは実行ごとにSIF、スクリプト、ログ、および実行結果をまとめた一時runtimeディレクトリを作成し、PJMジョブ終了後に生成成果物をGitHub ActionsのArtifactとして回収する。

当時の構成詳細は [current-runtime.md](current-runtime.md)、実行時の障害と設計判断の記録は [trial-record.md](trial-record.md) を参照のこと。

## 当時の比較基準条件

- model: `Qwen/Qwen3.5-397B-A17B-FP8`
- workload: SGLang `offline_throughput`
- dataset: `anon8231489123/ShareGPT_Vicuna_unfiltered`
- node: Genkai Node Group B × 2ノード
- GPU: 8基
- tensor parallel size: 8
- prompt数: 32
- output length: 256
- seed: 42

本ベンチマークは、SGLangの変更影響を同一条件下で比較することを主眼としている。モデル、データセット、またはベンチマーク引数を変更した結果は、従来条件での測定値と直接比較してはならない。

## 成果物

`results/` ディレクトリには以下のファイルを出力する。

| ファイル | 内容 |
|---|---|
| `environment.log` | ワークフロー、PJM、コミット、SIF、モジュール、GPU、コンテナ、モデルおよびデータセットの情報 |
| `throughput.log` | 実行コマンド、ベンチマークプロセスの標準出力・標準エラー出力、終了コード |
| `results.jsonl` | SGLangベンチマークの構造化実行結果 |

PJMジョブの標準出力・標準エラー出力は `logs/<job-id>.log` に保存する。ワークフローは `logs` と `results` をそれぞれ独立したGitHub Actions Artifactとして回収する。

当時の [profiler wrapper](../../scripts/exec-profiler.sh) は受け取ったコマンドをそのまま透過的に実行するラッパーである。プロファイラーを導入する際は本ラッパー側に起動設定を追加し、ベンチマーク本体およびワークフロー側の変更を回避する設計としている。
