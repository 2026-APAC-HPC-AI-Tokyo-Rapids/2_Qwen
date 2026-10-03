# Hopper MoE ESの局所比較と通常E2E検証

2026-10-04。ユーザーの「理論性能比を考慮して改善案を立て、実装・実行し性能向上確認」に対する実験履歴。実測改善はまだ未確定。

## 分析に基づく条件

[演算効率分析](2026-10-04-moe-gemm-efficiency.md)で現Pのupはdense FP8 peak比約46%、down約28%。downはK384と大きいBF16出力により帯域roofも低く、低いpeak比をそのまま高速化余地としない。既存Hopper expert-specialization (ES)をup/down別に比較し、全面GEMM新作より既存実装を先に試す。NVLSと交絡させない。

成功nonprofile c48 `ecd89520e2ec900741b42cb80f336280629e5620` を親とするbranch `research/aspire-moe-es-20261004`。条件commit `b557fcd200f60b9f9ce7133f3da835b3afc95f6b` / [Actions37140273788](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37140273788) を `[require-build] [cluster:aspire2ap] [nodes:2]` で投入した。profileではなく通常benchmark。P2/D2×TP4 EP1 DP1 PP1、c48、dgx026/030、FP8 target/FP8 KV、BF16 EAGLE3 draft、FA3、既存SiLU融合/draft-head/Dgraph13/15を固定。

## 最小改修と事前確認

本repoの `patches/sglang/prefill-cutlass-es.patch` を固定SGLang `fc275674` の私有build cloneだけに適用し、mirrorは読取専用。`Fp8MoEMethod` のCUTLASS経路に `SGLANG_PREFILL_CUTLASS_ES=off/up/down/both` を接続する。PrefillかつSM90だけで有効、default off、MXFP8の既存ES必須経路を保持する。BF16 draftはこのFP8 MoE methodを通らず、Dとcheckpoint/top8/FP8境界/FP32 accumulator/BF16 output/target verificationを変更しない。初回conditionはPのみ **off固定**。

既存SM90 ES native APIはpre-compute後にmain streamからpoolの2streamへeventでforkし、最後に両streamをmainへjoinする。main-stream CUDA event計測は完了待ちを含む。各callは2backup workspaceも割り当てる。固定CUTLASS pin `57e3cfb47a2d9e0d46eb6335c3dc411498efa198` のdescriptor式から協調kernel workspaceの上限は `(mainloop2 + epilogue最大2) × TMA128 × SM132 = 67584 bytes`、identity fusion/group schedulerは0。現90000bytesを維持し、無根拠な1GiB増量はしない。nativeにdynamic workspace-size guardがない制約は残るため、実global graph capture/replayと出力検証で確認する。既存image20c3a383のkernel wheelはsglang-kernel0.4.5。

187 CPU tests、Ruff、bash、diff whitespaceを通過。3patchは固定sourceへapply確認し、変更PythonをAST parse。独立Sol reviewでAPI/shape/stride/transpose、routingのtop8重複なし、closure、stream/join/workspace式、誤差評価の範囲にblockerなし。これはGPU実成立や改善の証明ではない。

## 実機の順序と精度評価

初回allocationのPnode GPU0で採点前にoff/up/down/bothを同new image・同入力で有限比較する。全probe timeout180秒。T8192（現profileの実shape）/4096/32、E128/H4096/I384/top8、balanced/skew/zero-expertを使う。route countsは合成で、checkpointの実router分布ではない。up N768,K4096、down N4096,K384。nativeをcapture前にwarmupし、eagerおよびglobal CUDA graphのCUDA event平均とhost wallを分けて保存する。

全shapeでnonfinite・大きい系統的誤りを確認。小T32では同FP8 weightをdequantizeした独立BF16/FP32 dense参照に対し、baselineとcandidateのrelative L2を比較する。candidate envelopeはbaseline×1.25＋0.005、large shape stock-relative L2<0.1は粗大誤り検出に限る。stockとの差とnormalized aggregate diffは診断併記で、elementwise一致を要求せず合法な丸め順序差を許す。これらはfixture品質検査であり、下流品質の証明にしない。

局所結果からserving modeを自動切替しない。probe後は **同new imageのoff固定** で公式c48 E2Eを実行する。有望variantだけ次の追跡commitでmodeを固定しimageを再利用、同条件非profile E2Eと必要な反復で改善を確認する。局所速度だけで完了や採用としない。旧image20c3a383との直接比較で新imageの依存差を交絡させない。未知の失敗への複雑retry/fallbackは追加しない。

## 正本と完了条件

Lunaがbuild/配備/PBS/probe/E2Eを監視し、最初fatal・probe完成・terminalでSolへ復帰通知する。raw正本は `qwen-artifacts/{run}/1`、解析・preflight・式は `research-issue-17/20261004-moe-es/`。初回runと候補のimage/source/patch来歴、local probe、公式parser/raw request、8greedy probe、checksum、残留PBS/runtimeと作業treeを監査し、改善なしもそのまま記録する。runtime sourceは優位性確認前にmainへmergeしない。

## 初回結果と次の条件

37140273788はbuild・ASPIRE・公式採点・回収・cleanup成功。optional Firmus archiveはSSH接続切断が初発で失敗したが、ASPIRE imageを無効にせずworkflow全体はsuccess。新image SHA256は `4618103a36b6cae6fd86054b5e26a58358106d9c97ad4e5fb5c10a6bd6a3c0aa`。52manifest対象全hash一致、parser再計算一致、1665 requests/error0、good fraction98.9189%、score2679.351 tokens/s、8件greedy各64tokens/logprobs、PBS216067 Exit0/11m53、runtime削除と残留PBSなしを確認した。これはES-off新image対照であり改善実績ではない。

全4shape×4modeで精度gateとglobal graph capture成功。T8192 balancedのgraph時間はoff1.3841ms、up1.3980、down1.4014、both1.4018。対off時間差はup+1.01%/down+1.25%/both+1.28%。T8192 skewは+0.15/+0.69/+0.31%、T4096 skewは+2.32/+1.94/+3.45%、T32は+23.51/+24.75/+30.81%。eagerも全candidate点推定で遅い。単一ordered20repeatの僅差を有意悪化とは断定しないが、有望条件がないためES candidate E2Eは投入せず不採用。これは合成routingのfusedMoE全体時間であり、個別GEMMのTFLOP/sへ逆算しない。

次は別要因の[Prefill NVLS](2026-10-04-prefill-nvls-campaign.md)を同new imageのruntime条件で比較する。MoE GEMMの改善とは区別する。

## 量化scale shuffle融合のCPU候補（保留）

別branch `experiment/prefill-shuffle-fusion-20261004`（b557fcd起点）で量化値とscaleのrow shuffle融合をCPU準備した。default off、P/Hopper/EP1/K4096G32の限定経路、既存quantize-onceを保持。205tests/Ruff/bash/diff、固定sourceの4patch適用とAST検査を通過したが、commit/build/push/GPUは未実施、実機の速度・精度・graph成立は未確認。

既存profile P0TP0ではpaired copy211.35µs/call（FP8169.49＋scale41.86）。scale費用を全除去してもactive5秒窓の65.67ms（1.31%）、半減なら32.84ms（0.66%）が局所削減上限で、FP8 copy12.35%退行で相殺する。bulk bytesは変わらず、graph node一つ削減をCPU launch一回削減と同一視しない。これはE2E向上率ではなく局所費用上限。独立新image buildと同image off/enable反復費用に対し現時点の根拠は弱く、単独build/GPUは保留。次に別理由でsource buildを行う機会へ有限probeを同梱する候補として保持する。NVLSを優先する。
