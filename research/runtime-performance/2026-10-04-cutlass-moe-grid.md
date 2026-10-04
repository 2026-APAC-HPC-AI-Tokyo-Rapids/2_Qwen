# CUTLASS MoEの有限cluster/stage探索

更新日: 2026-10-04 JST。ユーザーの実装・探索・E2E確認指示に基づく継続実験。大域最適性の保証ではない。

固定SGLang fc275674a / CUTLASS 57e3cfbでは、M128 N128 K128 Cooperativeのcluster 1x1/1x2/2x1/2x2とAuto/2/3/4段を最大16構成で比較する。FP8入力・scale128・FP32蓄積・BF16出力・top8を保持する。固定headerはStageCountAutoCarveoutだけに対応するためreserveで段数を制限し、resolved段数と実SharedStorage容量をcompile-time検査・manifestへ保存する。同一resolved構成はprobeで重複除外する。過去に遅かったPingpongと未成立N256 layoutは今回の範囲外。

16構成を4 TUへ分割、MAX_JOBS=1で一buildにまとめる。各trialの再compile・全E2Eは行わない。up/downは相方stockのwholeMoE graph時間で別探索し、8192 balanced/skewで全5 paired rounds改善・精度通過したrole上位2つだけjoint最大4組をheldoutで検証する。合成routingであり実expert分布は未取得。小さい離散空間なのでまずgridを使い、Optuna/runtime依存は追加しない。局所winnerを自動採用せず有望幅を確認してから同newimageのshuffleon/tileoff対照と固定candidate E2Eへ進む。

9e207a7 / Actions37193147613はclusterタグ無しでvalidationのみ成功し、native compileは未実施。新しい[build-only]は[require-build][cluster:aspire2ap]を伴いimageだけ作り、通常PDを抑止する。[cutlass-grid-probe][nodes:1][cluster:aspire2ap]はそのimageを再利用する専用経路で、両特殊タグの併用は拒否する。通常clusterタグのPD経路を維持する。

microprobeはnormal routeのshared 1GPU、14 CPU/235GB、place=free、wall15分・実処理timeout720秒。モデルstagingや公式score生成は行わず、schedulerが割り当てたGPU UUIDだけをEnrootへ渡しtorch device_count=1を確認する。固定026/030の既存反復on PBS216140は健全なQを維持、取消・配置変更なし。1GPU結果は最終固定pair16GPUの公式E2E比較を代替しない。sourceimage hash、module/source hash、production ID、runtime/probe hashを分離して保存する。

## ビルド成立と1GPU測定開始

[89195b0 / build37194029820](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37194029820)はnative compile/link/importを含め成功。16構成を4 geometry objectとbindingの5 TUで一度のimagebuildへ収録、MAX_JOBS1。image SHA256 `66925d036b288267adaeaafba65397b1a08313ab03b3e7efb5f05fec9bbb2dab`、サイズ12,232,060,928 bytes、module `f680e7e424536544b02b47d9d1e1e4bd1d615db28e5eaf2cd9a2922696fd1737`、source `01b81e1d034e93aab8f02f203e808322a069d2caf8b5b74b1f27cf794ba89d00`。Auto段数6、固定2/3/4が全clusterで成立し16unique、stage重複なし。SharedStorageは218624/84992/118272/152064 bytes、GEMM register168/helper30。Carveoutは実SMEM増量ではなくstage制約であり、register削減は確認されない。GPU can_implement/精度・速度はこの時点では未確認。

[9de0957 / finite1GPU37195511116](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37195511116)を同image再利用で投入。最大16config、各train最大34variant×2shape×5round、heldout最大6variant×3shape×5round＝430 graph測定・43000 timed replays。各測定3 eager warmup＋3 graph warmup、720秒timeout/15分wall。probe source/hash・native/productionID照合、1visibleGPU・精度gateを含む19CPUtests/Ruff/独立review通過。合成wholeMoE測定であり、公式採用評価・候補自動採用は行わない。結果待ち、旧repeat-on216140 Qを維持する。
