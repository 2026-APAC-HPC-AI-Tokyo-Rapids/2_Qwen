# Round 35：MegaMoE FC2 初回 K128 の再調査

**段階1は不通過。FC2の初回恒等rescaleとaccumulatorゼロ初期化は現mainで既に消えており、追加の削減は0命令だった。段階2は実施せず、実行コード・config・image pinを変更しない。**

親mainは `4b06ffd`、作業branchは `perf/pp4-r35-fc2-firstk-20261104`。AGENTS.md、r31設計調査、r32〜34のnote・コミット本文、tracker/modelとr24 profile JSON、保存Actions由来のr33 fixture・予算、scratchに保存されたr31/r32のCodex会話記録を確認した。SSH、Actions操作、Slurm投入、GPU実行、image buildはしていない。mirror、`~/.codex`、`/tmp/qwen-*`、他worktreeは変更していない。

## 対象とsourceの確認

対象は同じPrefill EP1のT7988/E128/top8/H4096/I1536、M128/N256/K128、4 stage、lean、split accumulator。BF16入力・出力、FP8 E4M3積、FP32 scale/accumulator、group128 quant、FC2 BF16 partial、FP32 weighted combineを維持する。DeepGEMM `1c8138a59b3df5dac50b76b730834ed57982fdfb` と、現configの14本のheader overlayを使う。14本すべての順序・SHA256がr32 proofと一致し、fixtureのcontrol headerに現FC1 overlayを適用したheader SHA256は `75085f03b05c8ab0cb2dd072994b27b1584225456928d49e66e67820ed1dee56`。今回のcontrolは**FC1 first-K有効**であり、r32以前のFC1をcontrolに戻していない。imageは採用済み `3a47b757`、今回のCPU compileはそのCUDA 13.0.88を検証するものではない。

`deep_gemm/impls/sm90_fp8_mega_moe.cuh` の `kL1SplitUnroll=8` に対し、`kL2SplitUnroll=12`。FC1は32 blockを4 chunkに分けるので入口と定常部の分離が必要だった。一方、FC2は `kNumL2BlockKs=1536/128=12` とchunk長が等しく、外側K chunk loopは1回だけで、内側12 blockは既存のpragmaで展開される。source上は `r0_start/r1_start=1`、`final_accum={}`、`split_k_loop` のrescaleとaccumulate=trueが残るが、compilerは初回であることを確定できる。

FC1と同じ `first_k_split_loop` をFC2の呼出しにも使う一時probeをworktree内のscratchだけに作った。probe macroは `DG_SM90_PREFILL_FC2_FIRST_K`、FC1 macroはすべてのcompileで1に固定。最初のK128の各half先頭K32だけscale-d=0にし、他のK積、scale、順序、12 blockのdrain境界は維持した。これはSASS差を調べるためのCPU候補であり、配備用overlay・host macro・gateは実装していない。初回宣言のゼロを手動削除する変更もしていない。

## CPU SASS gate

r32 proof archive（SHA256 `e2faa2a495c306729bf41f711bd28e5cdceb1a206454ef4df04f7ceaf21f8015`）のsourceと依存headerをworktree内へ展開し、mirrorに触れず再現した。CUDA `/usr/local/cuda-13.4`、nvcc/ptxas **13.4.92**、sm_90a、C++20/O3。r32と同じ24代表instantiationについて、control・probe flag 0・probe flag 1の計72本を新規compileし、`nvdisasm -hex` の命令wordと制御wordをlittle-endian byte列へ復元して比較した。ELF metadataやdisassemblyの文字列比較ではない。

| 対象構成の項目 | control（FC1採用済み） | FC2 probe有効 | 差 |
|---|---:|---:|---:|
| FC2初回K128のaccumulator FMUL | 0 | 0 | 0 |
| FC2有効入口のaccumulatorゼロ初期化命令 | 0 | 0 | 0 |
| FC2残り11 blockのaccumulator FMUL | 1408 | 1408 | 0 |
| FC2全12 blockのQGMMA | 96 | 96 | 0 |
| FC2 setup〜最終QGMMAの静的範囲 | 2250命令 | 2250命令 | 0 |
| FC1定常8×K128の静的範囲 | 1611命令 | 1611命令 | 0 |
| kernel全体の静的命令数 | 11320 | 11320 | 0 |
| SASS encoding byte | 181120 | 181120 | 0（0%） |
| ptxas launch register | 168 | 168 | 0 |
| 最大参照register | R206 | R206 | 0 |
| math setmaxnreg枠 | 224 | 224 | 0 |
| stack / spill store / spill load | 0 / 0 / 0 | 0 / 0 / 0 | 0 |

flag offは**24/24構成で全byte一致**。flag onも**24/24構成で全byte一致**し、対象のencoding SHA256はともに `f157a6dddb926f5944f751e46448afb69091eb9cfbcccd17e21fe852205e8c41`。対象のLOCAL load/storeは0、C75xx警告なし。legacy geometryの既存spillまで0になったという意味ではない。dynamic sharedは228560 byteで、launch/tilingは不変。

FC2入口は `.L_x_104` / `0x12000`。初回8本のQGMMAのうち、half 0の `0x122b0` とhalf 1の `0x12440` が既に `RZ, !UPT`、残り6本は通常accumulate。初回最後は `0x124d0`。入口からここまでのaccumulator R24〜R151へのFMUL・ゼロ初期化は0本。次のK128以降のFMULは `0x12640` から始まり、最終QGMMA `0x1ac90` まで **11×128=1408本**。各blockでscaleが変わるため、これは恒等処理ではない。scale計算等のFMULとaccumulator FMULを混同しない。

FC2のK loopは全展開済みで、定常K loopのback-edge自体がない。表の2250命令はsetup `0x12000`〜最後のQGMMA `0x1ac90` の包括的な静的範囲であり、実行時間や待機反復数ではない。初回後の範囲 `0x12500..0x1ac90` は2170命令、QGMMA88本。FC1の定常範囲は `0x7a90..0xdf30`。候補全byte一致により、定常命令・register・spill・code sizeの非増加は確認できたが、**正の命令削減という必要条件を満たさない**。

## 同kernel内のその他の累積

有効FC1の入口はr34採用済みの2本のQGMMA初期化で処理され、同じ初回FMUL・ゼロ初期化は残っていない。padding-only WGのゼロは有効FC1/FC2の入口と分岐が違い、空行を処理する経路の意味を保つために必要である。r32で棄却したpaddingゼロの手動削除を繰り返さない。

FC1 epilogueのSwiGLU・amax/group128 quantは出力変換・最大値reductionであり、K blockをcarried scaleで累積する構造ではない。dispatchのcount/offsetやarrivalは整数処理・同期で、同種のFP累積ではない。

combineには `float2 reduced[...]={}` と `sum=fmaf(value, weight, sum)` がある。現SASSの `0x1cd60..0x1ce50` に16本のCS2R、`0x1cd60..0x1e1f0` に64本のFFMAがあるが、これは動的なvalid top-k maskから順に取り出すweighted reductionであり、**初回identity rescaleは存在しない**。ゼロはmaskが空の出力も定義する。初回FMAを積に置き換えても積自体は必要で、K scaleの恒等処理を消す今回の案とは別の設計になる。既存のcombine融合・FP32 precombine等の棄却理由も踏まえ、別案へ範囲を広げない。

## 理論見積もり

FC1で削除したのはactive math warpの入口当たりFMUL128本＋ゼロ初期化64本、計192命令。過去の完全MegaMoEの短縮16〜25 µs/callを単純に割ると、**0.0833〜0.1302 µs／数えた入口命令**となる。これは全tileで反復する仕事を代表warpの静的本数で正規化した経験的な係数であり、GPUの1命令レイテンシではない。FC1の定常命令・code layout変更も含む実測なので、因果を初回192本だけへ分離できない。

今回追加で削れるFC2命令は0本で、同じ命令量比例模型の**期待短縮は0 µs/call**。実GPUの時間差は未測定で、0 µsを実測値とはしない。

仮にFC2にも192本の未最適化入口が残っていたなら、Kに対する割合はFC1の1/32に対し1/12で約2.67倍、全output tile数は4096/3072倍なので、FC1の経験係数による全callの仮想的な換算は **21.3〜33.3 µs/call**になる。しかし、その入口は既にcompilerが消している。Kが短いことを根拠に、この仮想利得を現mainへ加算しない。

有効仕事はFC1 1.608196816896 TFLOP/層、FC2 0.804098408448 TFLOP/層、合計2.412295225344 TFLOP/層、historical実行行68288で不変。r31の主要論理I/O模型はFC1 47.572 GB/24層、FC2 34.325 GB/24層のまま。実HBM byteと新たなGPU時間は未測定。EP1 expert間NVLink転送はなく、PP/KV転送の削減でもない。

## 段階2と予算

段階1の正味削減が0なので、段階2は行わない。overlay、host selector、12 GPU gate、prewarm、`results/prefill-r35-dispatch-{0,1,2}.txt` 回収は追加しない。image pinは変更しない。CPU一時probeは配備せず、fresh image buildも不要である。

将来、正の削減を得た別候補を同じ経路へ入れる場合でも、FC1とFC2に105秒の独立gateを二つ直列に置く設計は予算外である。r34開始249秒なら `249+105+105+630=1089>1020`、r33開始257秒なら1097秒となる。一つのgate内でcontrolをFC1採用済みmain、candidateをFC1+FC2へ置き換えるなら、既存105秒予算で984/992秒、条件 `SECONDS+105+630<=1020` を満たす。ただし新候補のcompile時間が同予算に収まることは未確認であり、今回はその実装も新gateの予算予約もしない。44 case×4成分、1200 call、pooled中央値、公開結果・UUID照合、ready前prewarmの条件を緩める理由はない。

## 検証と証拠

`export PATH=$HOME/.local/bin:$PATH` の後、指定の以下2本を実行した。全pytestは終了コード0、**2043 passed / 90 skipped**（約14分）。Ruffも終了コード0。skipをGPU成功とは扱わない。

```sh
uv run --project qwen-experiment-metrics --with 'numpy>=1.26,<3' --with pytest python -m pytest -q -p no:cacheprovider qwen-experiment-metrics/tests
uv run --project qwen-experiment-metrics ruff check qwen-experiment-metrics scripts containers
```

実行コードとtestを変更しておらず、git履歴に依存するtestも追加していない。CPU proofは新規scratch `codex-scratch/r35/r35-cpu-proof-20261104.tar.gz`（21223301 byte、SHA256 `ec11afd63899612354a68d7a87aaab9aa32f251dbdc3468d2bb94b8d18ebffc0`）へ保管した。gzip検査とarchive内72本すべてのencoding hash照合を通過し、worktree内の展開source・cubin・一時probe等は削除した。本作業から投入したjobはない。
