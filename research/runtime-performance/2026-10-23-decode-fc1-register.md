# Decode FC1 register保持・部分amax交換（実装候補、GPU未検証）

`patches/deep-gemm-bench/decode-fc1-register/sm90-decode-fc1-register.patch` は image 39899235870 のheaderと `configs/firmus-framework.json` の採用overlay 5本を適用した対照に重ねる。対照header SHA256は `77ee7caea6e5736803b6034f83575f9a846d30c54ce4970ced7c65450a9be497`。本変更は候補であり、GPU正しさ・速度・公式serving結果は未確認。ユーザーの追加指示に従いActions、scheduler、GPUジョブ、pushは実施していない。

2026-10-18のFC1量子化改善は全量FP32 shared stagingのbank conflictを除いた。本候補は残る16 KiBのSwiGLU store→loadを削る。gate accumulatorの死んだslotsに各laneのSwiGLU値を保持し、8 warp×64 tokenの部分amax（2 KiB）だけを書き出す。両N-split WGの64 channelを4 threads/tokenで集約し、SFは1 thread/tokenが従来helperで計算する。inverse SFの64 float（256 B）を別scratchへ公開して、元のlaneが保持値を量子化する。shared allocation自体と最終FP8 tileの位置は変えない。

部分amaxの添字は `warp * 64 + (token ^ (warp * 4))`。各warpの4 writer laneと、4 threads/tokenによる8 warp分の読み出しの両方でbank conflictを避ける。隣接channelが4 lane離れるSwapAB layoutをxor 4/8のshuffleで4-byte FP8へpackし、既存の非swizzle `[token][64]` tileへ書く。この最終storeには理論上4 distinct words/bankの競合が残る。全量staging削除の利得をshuffle、store競合、追加barrierが上回る可能性があり、改善を主張しない。

SwiGLUのclamp、`expf`/`__expf`、逆数、積の順序、SF helper、`__nv_fp8x2_e4m3`丸め、FC2後のTop-K重みは変更しない。amaxは0から `cute::max` で始めてNaNを取り込まない既存動作を保持。invalid tokenは最大値に0を与え、SF/FP8をstoreしない。

## 同期と公開

元の2 CTA barrierにinverse SF公開用を1つ加える。第1 barrierが部分amaxを公開し、前tileのFP8 TMAを発行したwarpの `wait<0>` より後に新tileのFP8書込みが来るよう保護する。追加barrierがinverse SFを公開し、第3 barrierが全warpのFP8書込みとscratch読出しを完了させる。既存のTMA fence、TMA store、issuing warpでのwait、L2 arrivalのreleaseは無変更。前tileと次tileのscratch再利用も第3 barrierで区切られる。

初版は各warpでSFを計算したが、SASSのSF helper CALLが1→30へ増えたので、GPU投入前に上記の1 thread/tokenへ修正した。SFの重複除去にはbarrier1回を追加する費用がある。

## CPU検査

`qwen-experiment-metrics/tests/test_decode_fc1_register_layout.py` の2 testは全 `valid_m=1..64` についてlane所有の全単射、cross-WG amax（NaN/Inf/signed zero/別WG最大値を含む）、FP8の任意byte列の4-byte packと出力番地の一意性、scratchの全単射・writer/gather bank配置を列挙して検査する。pytest、ruffを通過。候補はprovisionerの要求する `deep-gemm-bench/<candidate>/` 配下に置き、flat patch拒否を含む既存kernel-bench fixtureも再検証した。GPU演算やmemory orderingをCPUテストで実証したとは扱わない。

## CUDA式を抽出したhost数値検査

`scripts/check-decode-fc1-arithmetic.py` は canonical image header のコピーに対照overlay 5本と候補patchを順に適用し、両方の実C++からSwiGLU本体、SF helper、amaxのscalar fold、FP8変換の乗算・constructor・pack式を抽出する。g++ 13.3の `-O2 -fno-fast-math -ffp-contract=off` でhost shared libraryを作り、CUDA 13.0.88の公式 `cuda_fp8.h` にある `__nv_fp8x2_e4m3(float2)` のhost fallbackを実行する。CUTLASSの `cute/numeric/math.hpp` も直接includeしてmax/min/absのNaN規則を使用する。独自FP8 converterやcute演算shimで置換していない。生成C++と結果JSONは `--output-dir` に残す。

2577組のgate/up入力をfast math 2条件×clamp無効/8で検査した。入力は±0、subnormal、最大有限値、NaN payload、±Inf、任意のFP32 bit列を含む。全有限E4M3中点とその上下1 FP32 ULP、SF下限 `1e-10f` 前後、別WG最大値、全NaN行などを含め、`valid_m=1..64` の有効行で144196値を量子化した。非NaNのSwiGLU・量子化直前積（±0を含む）、amax、SF、inverse SF、packed FP8が対照と候補でbit一致した。独立Python scalarモデルは各演算の後にFP32へ丸め、SwiGLUには同じhost `expf` primitiveを使用する。FP8参照は全127非負有限値を列挙したnearest-neighbor/ties-to-even判定で、公式headerのhost変換と一致した。

NaNのpayload/符号は同一演算でもhost最適化された対照と候補でclamp無効の各mode10件、対照とPython scalarモデルでclamp無効の各mode78件異なった（clamp=8はいずれも0件）。両方NaNであることを必須とし、差を結果JSONへ記録する。公式host FP8変換によるcanonical化後のpacked bytesは一致した。これをNaN中間値までbit一致したとは扱わない。

この検査はCUDAの `__expf` と `math::fast_rcp` を共通host演算で代替する。device近似命令、CUDA compilerによる融合/FTZ/register保持の影響、WGMMA結果、実shuffle・barrier・TMA・無効行のmemory accessは証明しない。amaxのshuffle topologyはhostで組み直しているため、実際のlane/invalid-row mappingは前掲の独立layout testとソースレビューで補う。CPU数値検査をGPU replayや公式servingの代替としない。

再現例（header、CUDA、CUTLASS includeの所在は環境に合わせる）:

```bash
python3 scripts/check-decode-fc1-arithmetic.py --base-header /path/to/image/sm90_fp8_mega_moe.cuh --cuda-include /path/to/cuda/include --cutlass-include /path/to/cutlass/include --output-dir /tmp/fc1-host-arithmetic
```

## CPU CUDA compile

Firmus loginからimage内CUDA 13.0.88を用い、実JIT相当の `--register-usage-level=10` で対照・候補をcompile。共有toolchain/headersは読取り専用、出力は専用directory。Decode wave32/wave16とPrefill 2WGを検査し、register/spill、WGMMA warning、Prefill SASS不変を確認した。結果とhashは後掲。

raw source、compile script/log、cubin、SASSは `/tmp/qwen-astra-fc1-register-evidence-20261023/compile-evidence.tar.gz` へ回収済み。Firmus側の専用compile directoryと共有抽出toolchainは全候補のCPU検査後に削除した。この候補からschedulerジョブは作成していない。

| geometry | 対照/候補 registers | spill store/load | stack bytes | normalized SASS行数 対照→候補 |
| --- | --- | --- | --- | --- |
| decode32 | 168 / 168 | 0 / 0 | 8 | 13648 → 15776 |
| decode16 | 168 / 168 | 0 / 0 | 8 | 13664 → 15792 |
| prefill2wg | 168 / 168 | 0 / 0 | 0 | 14016 → 14016 |

SASS行数は命令行とencoding行を含む静的なcode量であり、実行命令数ではない。Prefill 2WGのnormalized SASS SHA256は対照・候補とも `f4dfa3b8bba194252571e838604df44885ea652d8bb06b15e8d4d074edcc3f86`。候補DecodeにはWGMMA直列化warningなし。8-byte stackは対照と同じでspillではない。

最終候補header SHA256: `fa4a86e57417b92dd7dfd25b30b4c7c1f38ac04ac4264656c8876779a58378b2`。
