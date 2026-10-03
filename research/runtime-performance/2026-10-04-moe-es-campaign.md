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
