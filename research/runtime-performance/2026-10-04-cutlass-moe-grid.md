# CUTLASS MoEの有限cluster/stage探索

更新日: 2026-10-04 JST。ユーザーの実装・探索・E2E確認指示に基づく継続実験。大域最適性の保証ではない。

固定SGLang fc275674a / CUTLASS 57e3cfbでは、M128 N128 K128 Cooperativeのcluster 1x1/1x2/2x1/2x2とAuto/2/3/4段を最大16構成で比較する。FP8入力・scale128・FP32蓄積・BF16出力・top8を保持する。固定headerはStageCountAutoCarveoutだけに対応するためreserveで段数を制限し、resolved段数と実SharedStorage容量をcompile-time検査・manifestへ保存する。同一resolved構成はprobeで重複除外する。過去に遅かったPingpongと未成立N256 layoutは今回の範囲外。

16構成を4 TUへ分割、MAX_JOBS=1で一buildにまとめる。各trialの再compile・全E2Eは行わない。up/downは相方stockのwholeMoE graph時間で別探索し、8192 balanced/skewで全5 paired rounds改善・精度通過したrole上位2つだけjoint最大4組をheldoutで検証する。合成routingであり実expert分布は未取得。小さい離散空間なのでまずgridを使い、Optuna/runtime依存は追加しない。局所winnerを自動採用せず有望幅を確認してから同newimageのshuffleon/tileoff対照と固定candidate E2Eへ進む。

9e207a7 / Actions37193147613はclusterタグ無しでvalidationのみ成功し、native compileは未実施。新しい[build-only]は[require-build][cluster:aspire2ap]を伴いimageだけ作り、通常PDを抑止する。[cutlass-grid-probe][nodes:1][cluster:aspire2ap]はそのimageを再利用する専用経路で、両特殊タグの併用は拒否する。通常clusterタグのPD経路を維持する。

microprobeはnormal routeのshared 1GPU、14 CPU/235GB、place=free、wall15分・実処理timeout720秒。モデルstagingや公式score生成は行わず、schedulerが割り当てたGPU UUIDだけをEnrootへ渡しtorch device_count=1を確認する。固定026/030の既存反復on PBS216140は健全なQを維持、取消・配置変更なし。1GPU結果は最終固定pair16GPUの公式E2E比較を代替しない。sourceimage hash、module/source hash、production ID、runtime/probe hashを分離して保存する。

## ビルド成立と1GPU測定開始

[89195b0 / build37194029820](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/actions/runs/37194029820)はnative compile/link/importを含め成功。16構成を4 geometry objectとbindingの5 TUで一度のimagebuildへ収録、MAX_JOBS1。image SHA256 `66925d036b288267adaeaafba65397b1a08313ab03b3e7efb5f05fec9bbb2dab`、サイズ12,232,060,928 bytes、module `f680e7e424536544b02b47d9d1e1e4bd1d615db28e5eaf2cd9a2922696fd1737`、source `01b81e1d034e93aab8f02f203e808322a069d2caf8b5b74b1f27cf794ba89d00`。Auto段数6、固定2/3/4が全clusterで成立し16unique、stage重複なし。SharedStorageは218624/84992/118272/152064 bytes、GEMM register168/helper30。Carveoutは実SMEM増量ではなくstage制約であり、register削減は確認されない。GPU can_implement/精度・速度はこの時点では未確認。

[9de0957 / finite1GPU37195511116](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/actions/runs/37195511116)を同image再利用で投入。最大16config、各train最大34variant×2shape×5round、heldout最大6variant×3shape×5round＝430 graph測定・43000 timed replays。各測定3 eager warmup＋3 graph warmup、720秒timeout/15分wall。probe source/hash・native/productionID照合、1visibleGPU・精度gateを含む19CPUtests/Ruff/独立review通過。合成wholeMoE測定であり、公式採用評価・候補自動採用は行わない。結果待ち、旧repeat-on216140 Qを維持する。

## 完走監査・探索結果

run37195511116はActions全体success/PBS216217 Exit0、DGX038のH100 1GPU、wall3分55秒。正本`qwen-artifacts/37195511116/1`の全12hash、image/probe/source/module/productionID、実allocatedUUIDだけのcommand、score非生成、全回収・cleanupを照合した。live JSONとterminal正本はbyte一致。全74測定entryでeager/graph bitwise同一・L2=0、全候補can_implement/graph成功。下記はwholeMoE時間であり個別GEMM TF/sではない。


37195511116 terminal正本JSONを解析。live snapshotとbyte一致を独立確認、auditの全12hash/PBS216217 Exit0/3:55/cleanup監査を参照。実nodeはDGX038、1GPU。全16config・up/down各役割×balanced/skewの品質・graphを確認。全測定でeager/graph L2=0、bitwise同一、finite、quality_eligible。未対応候補無し。

|候補|balanced 全MoE対stock時間差 / 改善round数|skew 時間差 / 改善round数|
|---|---:|---:|
|up:coop128_c1x2_sauto|-0.119% / 2/5|-0.165% / 3/5|
|down:coop128_c1x2_sauto|-0.065% / 3/5|+0.092% / 2/5|
|up:coop128_c1x2_s2|+41.765% / 0/5|+35.667% / 0/5|
|down:coop128_c1x2_s2|+10.131% / 0/5|+9.206% / 0/5|
|up:coop128_c1x2_s3|+11.653% / 0/5|+8.566% / 0/5|
|down:coop128_c1x2_s3|+0.363% / 1/5|+1.177% / 0/5|
|up:coop128_c1x2_s4|+3.514% / 0/5|+1.237% / 0/5|
|down:coop128_c1x2_s4|-0.952% / 5/5|-0.205% / 4/5|
|up:coop128_c1x1_sauto|-0.517% / 4/5|+1.887% / 0/5|
|down:coop128_c1x1_sauto|-0.466% / 4/5|+0.187% / 1/5|
|up:coop128_c1x1_s2|+42.899% / 0/5|+34.815% / 0/5|
|down:coop128_c1x1_s2|+12.176% / 0/5|+9.356% / 0/5|
|up:coop128_c1x1_s3|+10.790% / 0/5|+8.165% / 0/5|
|down:coop128_c1x1_s3|+0.815% / 0/5|+0.970% / 0/5|
|up:coop128_c1x1_s4|+2.260% / 0/5|+1.940% / 0/5|
|down:coop128_c1x1_s4|-1.061% / 5/5|-0.169% / 4/5|
|up:coop128_c2x1_sauto|-0.511% / 5/5|+3.381% / 0/5|
|down:coop128_c2x1_sauto|+0.513% / 0/5|+4.186% / 0/5|
|up:coop128_c2x1_s2|+43.632% / 0/5|+52.941% / 0/5|
|down:coop128_c2x1_s2|+12.462% / 0/5|+16.145% / 0/5|
|up:coop128_c2x1_s3|+11.830% / 0/5|+17.308% / 0/5|
|down:coop128_c2x1_s3|+0.656% / 0/5|+5.417% / 0/5|
|up:coop128_c2x1_s4|+3.401% / 0/5|+7.255% / 0/5|
|down:coop128_c2x1_s4|-0.663% / 5/5|+3.580% / 0/5|
|up:coop128_c2x2_sauto|+0.569% / 0/5|+4.213% / 0/5|
|down:coop128_c2x2_sauto|+2.764% / 0/5|+7.147% / 0/5|
|up:coop128_c2x2_s2|+51.396% / 0/5|+63.004% / 0/5|
|down:coop128_c2x2_s2|+14.565% / 0/5|+20.784% / 0/5|
|up:coop128_c2x2_s3|+15.939% / 0/5|+21.489% / 0/5|
|down:coop128_c2x2_s3|+3.148% / 0/5|+9.274% / 0/5|
|up:coop128_c2x2_s4|+5.650% / 0/5|+8.801% / 0/5|
|down:coop128_c2x2_s4|+1.211% / 0/5|+6.597% / 0/5|

clone/wheel差はbalanced+0.054%、skew+0.123%、heldout4096-0.058%、hotonly8192-0.072%。T32+4.03%はprobe強制large geometryとwheel smallMの差でproduction性能候補ではない。stockは全出力bitwise同一。

down stage4はbalanced約1%改善・skew約0.2%改善の点推定だが、skewで1round微悪化し事前「両patterns全5round改善」基準を通らない。wheel比ではskewの改善はさらに小さい。up cluster2x1 autoはbalanced改善対skew退行。stage2/3等に頑健な勝者なし。shortlistは両role空、joint heldoutにはwheel/stockしか測定されておらずcandidateのheldout成立を主張できない。

**判断:** 現事前基準ではE2Eへ進む候補無し。事後に基準を緩めない。これはstage4downの効果が理論的にないという結論ではなく、実router分布未知・小差・heldout未検証のため追加E2E根拠不足という判断。個別GEMM時間へ全MoE改善率を逆算しない。shape key/quality/5pairedround詳細はanalysis.json参照。

device: H10080GBHBM3、132SM、visible1、Torch2.11cu130、CUDA13.0。clock時系列はJSONに無いので外部gpu-environment.csvのsnapshotと区別。image/source/modulehashはaudit独立照合対象。

terminal rawのgpu-environment.csvはprobe開始前snapshotでSM clock345MHz/32°C、driver595.71.05。これはidle時のclockで、計測中345MHzだったとは解釈しない。今回連続clock samplingは無いのでactive clockは未確認。元profileDGX026の1980MHz実測と同clock/同node条件を主張せず、今回同GPU内の交互順比較だけを使う。

有限16構成の探索で事前gateを満たすwinner無し、追加候補E2Eを投入しない。Optunaはこの小さい離散空間の全grid結果を置き換える理由がなく未使用。これは未知空間の最適性証明ではなく、未探索mainloop/layout/実routing分布は残る。既shuffle融合の優位性・最後の反復Qは別キャンペーンとして継続。mainruntime/source採用なし。

## 独立stage4確認実験の事前計画

初回gridの基準未達・候補無しという結論を維持する。一方、balanced約1%/skew約.2%のdown stage4仮説について、ユーザーの最適探索指示とroot判断に従い同imageの短い独立1GPU確認を一回だけ追加する。candidateは`coop128_c1x2_s4`と`coop128_c1x1_s4`の2つ、upはwheel固定。wheel/stockclone対照、新seed、20paired roundsと順序反転、各測定前に0.2秒以上の連続GPU warmupを行う。allocatedUUIDだけの1秒clocks/power/utilization samplingを計測区間外でも記録し、他GPUへアクセスしない。

事前確認gateはcandidate/wheel medianが8192 balancedで0.995以下・skewで0.999以下、各pattern15/20round以上改善、全finite/relativeL2<=1e-3。heldout4096 skew/8192 moderateは順位選定に使わずmedian退行を除外する頑健性gate。bootstrap95%区間は診断であり20roundを20独立runとは数えない。初回gridの全5sign条件を黙って置換するのではなく、別実験・別条件commitの確認基準として記録する。tinygainの場合は公式E2E費用対効果をrootへ提示し、自動採用しない。sourcevariant/stage5/newbuildなし、既healthy PBS216140 Q維持。

## 独立確認の完走・採否

[bce182d / run37196596176](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/actions/runs/37196596176)はsameimage66925d03再利用、新build無し、Actions全体success/PBS216221 Exit0・5分20秒、DGX038のallocated H100 1GPUで完走。正本`qwen-artifacts/37196596176/1`全14hash、module/image/probe/productionID、20round全warmup>=200ms・400連続replay、実commandUUID限定、sampler正常終了・stderr無し、公式score非生成、home回収・runtime/PBScleanupを確認。旧healthy216140 Qは変更なし。


事前gateを独立再計算し、両候補False。追加E2Eを勧めない。初回16候補gridの「頑健な勝者なし」は維持する。

|候補|8192 balanced 中央値時間差 / paired勝数|8192 skew 中央値時間差 / paired勝数|heldout4096 skew|heldout8192 moderate|
|---|---|---|---|---|
|down c1x2 stage4 / up wheel|-0.6488% / 20/20|-0.1502% / 11/20|-0.7008% / 18/20|-0.3875% / 20/20|
|down c1x1 stage4 / up wheel|-0.6879% / 20/20|+0.0023% / 10/20|-0.2685% / 15/20|-0.2754% / 16/20|

表の時間差はcandidate median / wheel median - 1。paired ratio中央値とは異なる。c1x2 skewはpaired中央値-0.0147%、診断bootstrap95%区間[0.998212,1.001142]で1を含む。c1x1 skewはpaired中央値+0.0381%、区間[0.998200,1.002029]。balancedは両候補改善するが、skewの最低15/20勝を満たさない（c1x1は中央値閾値も未達）。観測後に閾値は変えない。heldoutは退行なしだが、選定に使わず、training未達を救済しない。

全4shape×4variant×20roundのeager/graph出力はfinite、relativeL2=0、bitwise一致。stock clone/wheel中央値差はbalanced-0.0159%、skew+0.0042%、heldout4096-0.0839%、moderate+0.0806%。候補の約0.65% balanced差を単にbuild差とは説明しにくいが、実route skewで頑健な改善の証拠にはならない。

wholeMoE CUDA graph時間であり、down GEMM単独改善率へ逆算しない。実router分布未収集の合成入力、同一run内20roundは独立runではないためresampling区間は診断のみ。新seedとheldoutで過適合を点検したが、公式E2E改善は未確認。

H10080GB HBM3 / SM132 / visible1、UUID f28ea8ef-b4e7-5547-f51e-0bd45cfb6168。1秒samplerのutil>0全232sampleでSM1845–1980MHz。CSV timestampにtimezoneが無く、probeUTCと8h差の時間整合からUTC+8を仮定するとbalanced1890–1965、skew1905–1980、heldout4096 1845–1965、moderate1890–1965MHz。これはshape全windowの範囲で、variant別の完全matched clock証明ではない。prelaunch345MHzを実行clockとして扱わない。

正本JSON SHA256: 52f5db2415bd514a903b69a1df6254a77a1973fac79cf5ebdaf2d66be50aad6c。独立解析script/JSONは同dirに保持。raw改変、追加GPU投入、自動採用なし。

両候補の事前gate未達により候補E2Eは投入しない。pass時の同newimage shuffleon/tileoff vs down-IDだけ変更のmatched E2EはCPU計画のみ（通常12分×16GPU、off/on pair約24分・6.4GPUh、queue別）。その費用をこの微小・分布依存結果だけで追加消費しない。initialgrid verdictやprecision/runtime既定は維持し、改善未確認を最適性証明に置き換えない。branch `experiment/prefill-cutlass-confirm-20261004` は再現用保持、mainruntime変更なし。
