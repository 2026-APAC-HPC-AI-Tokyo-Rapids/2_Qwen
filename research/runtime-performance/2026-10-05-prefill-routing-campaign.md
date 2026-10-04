# Prefill routing helperの部分置換

更新日2026-10-05 JST。ユーザーの実測高速化要求に対し、[独立設計review](2026-10-05-next-moe-routing-review.md)を実装へ進めた。既CUTLASS gridとstage4確認は事前gate未達のまま、GEMM候補を採用しない。既shuffle最終反復37177667936/PBS216140も変更・取消せずLunaが監視する。

## 実装と条件

実験branch `experiment/prefill-routing-20261005`、source/build commit `9c5a5a371524304bbab960cd8d3d86d67de56adb`。expertごとの全route走査をB512のtile histogram、expert/tile整数exclusive prefix、tile-local shared cursor scatterへ置換する。独立CUDA 1 TU・3整数kernelのみ、MAX_JOBS=1。GEMMテンプレートを再compileせず、既wheel GEMM、FP8 scale128/quantize once、BF16中間丸め、FP32 top8集約、target verificationを維持する。P/Hopper/EP1、BF16入力、E128/top8/H4096/I384/T≤8192に限定し、default off。T0はwhole MoEを早期returnし、helper単独ではmetadataをresetする。

新imageでは私有cloneへの適用順をSiLU→draft selected-head→FP8/scale shuffle→routingの4patchに固定。不採用ES/tile patchと大きいnative tile extensionはbuild対象から撤去した。source cleanupを性能gainと混同しないため、以後のoff/on比較は同じ新imageを使う。D/既c48/FA3/FP8 KV/EAGLE3 BF16 draftは維持する。

旧profileのhelper計154.587ms/5.004391秒（3.09%）は全消去感度上限。論理IDs読取の削減はL2/HBM実trafficの削減量を証明せず、実改善は未測定。

## 検証と進行

274 CPU tests、routing/probe Ruff、bash syntax、actionlint、actual依存patch chainと固定source AST fixture、Docker呼出script実行bit、diff checks通過。独立review GO。CPU検査はnativecompile/GPU正しさの証明ではない。

[build-only Actions 37218171768](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/actions/runs/37218171768)を投入。`[require-build] [cluster:aspire2ap] [build-only]`でGPU計算は起動しない。first compiler fatal、module import/source/.so manifest、terminal archiveをLunaが監視する。native compileとimage来歴はまだ未確認。

build成功後、別runtime commitで同imageのstock/tiledをscheduler経由1GPU・15分walltime・720秒timeoutの有限microprobeで測る。allocated GPU UUIDのみmount/監視、active clock/powerを1秒CSVに記録する。モデルstaging・公式score生成は行わない。

事前gateは8192 balanced/skewのwholeMoEそれぞれ中央値ratio≤0.995かつ20paired round中15勝以上。heldout4096skew/8192hot-only/65tailは中央値退行無し。全eager/graph finite/relativeL2≤1e-3、bitwise差診断、T32独立BF16/FP32 oracle envelope、map逆写像/offset/count、同graphでIDsを変えたreset、T0を必須にする。候補2・training2shape・heldout3shape・oracle1shape、最大404測定/40400timed replay、交互順序と連続≥200ms warmupで有限終了する。synthetic routeは実expert分布ではない。micro有望時だけ同新image・shuffleon・routing off/onの公式E2E比較条件と費用をrootへ報告する。自動採用・実行中探索は導入しない。

## helper build成功

37218171768はterminal success。source `d958b6ce7ea382c9ac42a8ebf8f4f6c4b68a781900129e15aa9fa650f2cb38ce`、build Python `ba9dd52211441c69529ce76d712a520c4204f71a3a76fa287a480a23e4094c8e` はcheckoutと一致。module `620e625331d74985538fbef04f077daf299b5af7eb9d97b2e25e2bd608666e1d` がlink/import済み。ptxasはscatter32/prefix28/histogram18 registers、各512B shared、1TU/3kernel/MAX_JOBS1を確認。EC2 elapsed1568秒、image12231192576 bytes、S3/全cluster archive成功、GPU/PBS無し。

条件のみの空commit `4b34df8` はpaths-ignore付きpush workflowを生成しなかった。GPU消費無し。PBS job/runtime名をrouting専用にした実runtime commit `f041fe1` で[finite 1GPU run37220486463](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/actions/runs/37220486463)を起動した。source/image/gateは変更しない。公式E2E性能改善はまだ未確認。
