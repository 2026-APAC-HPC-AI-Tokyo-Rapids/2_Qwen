# CUTLASS MoEの有限cluster/stage探索

更新日: 2026-10-04 JST。ユーザーの実装・探索・E2E確認指示に基づく継続実験。大域最適性の保証ではない。

固定SGLang fc275674a / CUTLASS 57e3cfbでは、M128 N128 K128 Cooperativeのcluster 1x1/1x2/2x1/2x2とAuto/2/3/4段を最大16構成で比較する。FP8入力・scale128・FP32蓄積・BF16出力・top8を保持する。固定headerはStageCountAutoCarveoutだけに対応するためreserveで段数を制限し、resolved段数と実SharedStorage容量をcompile-time検査・manifestへ保存する。同一resolved構成はprobeで重複除外する。過去に遅かったPingpongと未成立N256 layoutは今回の範囲外。

16構成を4 TUへ分割、MAX_JOBS=1で一buildにまとめる。各trialの再compile・全E2Eは行わない。up/downは相方stockのwholeMoE graph時間で別探索し、8192 balanced/skewで全5 paired rounds改善・精度通過したrole上位2つだけjoint最大4組をheldoutで検証する。合成routingであり実expert分布は未取得。小さい離散空間なのでまずgridを使い、Optuna/runtime依存は追加しない。局所winnerを自動採用せず有望幅を確認してから同newimageのshuffleon/tileoff対照と固定candidate E2Eへ進む。

9e207a7 / Actions37193147613はclusterタグ無しでvalidationのみ成功し、native compileは未実施。新しい[build-only]は[require-build][cluster:aspire2ap]を伴いimageだけ作り、通常PDを抑止する。[cutlass-grid-probe][nodes:1][cluster:aspire2ap]はそのimageを再利用する専用経路で、両特殊タグの併用は拒否する。通常clusterタグのPD経路を維持する。

microprobeはnormal routeのshared 1GPU、14 CPU/235GB、place=free、wall15分・実処理timeout720秒。モデルstagingや公式score生成は行わず、schedulerが割り当てたGPU UUIDだけをEnrootへ渡しtorch device_count=1を確認する。固定026/030の既存反復on PBS216140は健全なQを維持、取消・配置変更なし。1GPU結果は最終固定pair16GPUの公式E2E比較を代替しない。sourceimage hash、module/source hash、production ID、runtime/probe hashを分離して保存する。
