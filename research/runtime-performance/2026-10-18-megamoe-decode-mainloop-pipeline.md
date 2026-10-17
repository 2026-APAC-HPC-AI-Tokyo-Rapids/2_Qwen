# 2026-10-18 SM90 MegaMoE Decode mainloop の software pipeline 案と N_SWAP hoist

[kernel 単体 bench と timeline](2026-10-18-megamoe-kernel-timeline.md) の Decode-128 で、GEMM 区間の重み読み出しが実効 2.7 TB/s にとどまり、math warpgroup はほとんど stage を待たず、TMA loader は空き stage を 150〜190 µs 待っていた。律速は math 側の k-block ごとの直列処理（full wait → SwapAB WGMMA → `warpgroup_wait<0>` → per-token scale を掛けて FP32 に promote → empty arrive）と考えられる。そこで、ある k-block の WGMMA と前の k-block の promote・barrier 処理を重ねる 2 段の software pipeline を、精度を変えない（出力 bit 一致）bench overlay として実装し、[kernel-bench] 経路で測った。

結論：**pipeline は効かなかった。効いたのは、その前提として入れた「N_SWAP の選択を block ごとに 1 回にし、K loop 全体をその specialization の中で回す」変更（hoist）だけだった。** hoist だけの最終案は全 12 case で出力 hash が image と一致し、Decode を 2.4〜11.2% 短縮した（2 run。Decode-128 は image 3 run 平均 0.2295 ms に対し 0.2048 / 0.2065 ms）。pipeline を重ねると、どの chunk 長でも hoist だけより遅い。最終案はブランチ `perf/firmus-decode-mainloop-pipeline-20261018` の `2d91b50`（`patches/deep-gemm-bench/sm90-decode-swap-ab-hoist.patch`）に置き、main には入れていない。

## 1. 実装

対象は image `e29dfde…` の `sm90_fp8_mega_moe.cuh`（`patches/deep-gemm/*.patch` 適用済み）の非 reuse 経路のうち SwapAB 経路（Decode：BLOCK_M 64 / BLOCK_N 128、math 256 thread = N 方向に分けた 2 warpgroup、8 stage）。Prefill の reuse 経路には触れていない。

### 1.1 pipeline 案（不採用）

- `if constexpr (kSwapABActive)` の分岐を新設し、N_SWAP（valid token 数を 8 単位に切り上げた WGMMA の N）を block ごとに 1 回選んで、その specialization の中で K loop を回す（hoist）。こうしないと 2 つの accumulator を register に保ったまま k-block をまたげない。
- L1：k-block k を `accum[k % 2]` に発行し、`warpgroup_wait<1>` で k−1 の完了を待って k−1 を promote し、その stage の empty barrier を arrive する。weight scale は 1 k-block 前に読み、activation scale は arrive 前の smem_sfa から読む。
- L2：per-64 の K 半分を 1 単位とし、k の low 半分の WGMMA と k−1 の high 半分の promote を重ねる。stage の解放は両半分の promote 後。
- どの単位も 0 から積算を始め、promote の順序（k 昇順、L2 は low→high）を変えないので、出力は bit 一致する（実測でも全 run で一致、3 節）。

コンパイル（Firmus login で image の CUDA 13.0 nvcc を `unsquashfs` で取り出し、`containers/build-sm90-megamoe.py` と同じ Prefill/Decode の instantiation と `--ptxas-options=-v,-warn-spills` で確認）で分かった制約：

1. **WGMMA が loop の back-edge をまたいで実行中で、その accumulator を次の反復で読むと、ptxas は kernel 中の全 WGMMA を直列化する**（`C7514 ... wgmma.mma_async instructions are serialized due to non wgmma instructions reading accumulator registers`）。k loop を素直に 2 段 pipeline にすると必ずこうなる（各反復末尾で `warpgroup_wait<0>` を入れると消えることで確認）。このため pipeline は「k-block を chunk 長だけ unroll した中で重ね、chunk 末尾で drain する」形でしか書けない。
2. 本体が大きくなると NVVM が `process_math_block` と Decode の `for_each_block` callback を inline しなくなり、`C7510`（関数境界をまたぐ WGMMA の直列化）と 6.5 KB の spill が出る。両 lambda に `CUTLASS_LAMBDA_FUNC_INLINE` を付けると解消する（Prefill はもともと inline 済みで SASS は不変）。
3. 上の 2 KB の spill は、未初期化の accumulator が loop を回っていたため。0 で初期化すると spill は消えたが C7514 は残った（chunk 化後は accumulator が loop をまたがないので関係しない）。
4. **math region の register 割り当ては `setmaxnreg.inc` の値ではなく launch bound（384 thread → 168）で上限が決まる。** `kNumEpilogueRegisters` を 224 にしても SASS の最大 register は R165 のままで spill も変わらなかった。予算を増やす案は効かないので、すべての案は 168 のまま spill 0 で収めた。

### 1.2 hoist だけの最終案（`2d91b50`）

pipeline の仕組みをすべて外し、hoist だけを直接書いたもの。block ごとに weight scale の行 pointer を 1 回計算し、`__ldg` を full barrier の wait の前に置く。k-block ごとの処理順は image と同じ（L1：発行 → `wait<0>` → promote → arrive、L2：low・high の 2 group を発行 → `wait<1>` → low を promote → `wait<0>` → high を promote → arrive）。本体が小さいので `CUTLASS_LAMBDA_FUNC_INLINE` なしでも inline され、付けても SASS は同一なので付けていない。

### 1.3 register / spill / コード量

| 版 | Decode register | Decode spill (st/ld byte) | WGMMA 直列化 | Decode cubin | Prefill |
|---|---:|---:|---|---:|---|
| image | 168 | 0 / 0 | なし | 120 KB | 96 reg、spill 40/40 |
| pipeline、loop をまたぐ素直な形（inline なし） | 168 | 6504 / 7176 | C7510 | — | SASS 同一 |
| 同、inline 強制後 | 168 | 2020 / 1976 | C7514 | — | SASS 同一 |
| pipeline chunk 32/12、8/6、4/2、2/2、2/1、1/2、1/1 | 168 | 0 / 0 | なし | 1002 / 399 / 219 / 181 / 153 / 163 / 134 KB | SASS 同一 |
| 最終案（hoist のみ） | 168 | 0 / 0 | なし | 135 KB | SASS 同一 |

chunk の表記は L1 / L2 の chunk 長（k-block 数）。1 は重ねない（L2 は image と同じ low/high の重なりだけ）。Prefill の「SASS 同一」は `cuobjdump -sass` の出力が image の header から作ったものと byte 一致したという意味（cubin は metadata だけ異なる）。chunk 長は Prefill では実体化されない `if constexpr (kSwapABActive)` の中にしかなく、8/6、4/2、4/4、2/2、2/1、1/2、1/1 と最終案で確認した。

timeline 計装は候補の header に合わせて作り直し、per-CTA counter `ML1Loop` / `ML2Loop`（math warpgroup 0 の block 開始 → K loop 終了、epilogue を含まない）を足した。これで FC1/FC2 の時間を K loop と epilogue に分けられる。

## 2. 実行記録

すべて `[cluster:firmus] [nodes:1] [kernel-bench]`、image `e29dfde…`、Firmus hpcai 1 ノード 4×H200。

| run | commit | ブランチ | 内容 |
|---|---|---|---|
| 39053820377 | b90cc52 | main | image（基準、前回） |
| 39054922367 | 9606b24 | 削除済み | image（中身のない候補、前回） |
| 39058111866 | 9b6e785 | experiment/megamoe-timeline-loop-split-20261018（削除） | image + loop/epilogue 分割 counter 付き timeline（対照） |
| 39058491946 | 91152db | perf/firmus-decode-mainloop-pipeline-20261018 | pipeline chunk 8/6 |
| 39058112448 | 846f28b | experiment/megamoe-mainloop-chunk32x12-20261018（削除） | chunk 32/12（K loop を全 unroll） |
| 39059225948 | 1fd9254 | experiment/megamoe-mainloop-chunk4x2-20261018（削除） | chunk 4/2 |
| 39059209277 | c76cd93 | experiment/megamoe-mainloop-chunk2x2-20261018（削除） | chunk 2/2（L1+L2 pipeline） |
| 39060897888 | c3019ef | experiment/megamoe-mainloop-chunk2x1-20261018（削除） | chunk 2/1（L1 だけ pipeline） |
| 39060915255 | 1de221c | experiment/megamoe-mainloop-chunk1x2-20261018（削除） | chunk 1/2（L2 だけ pipeline） |
| 39060930640 | b21acb3 | experiment/megamoe-mainloop-chunk1x1-20261018（削除） | chunk 1/1（hoist のみ、pipeline の仕組みは残る） |
| 39062255256 | 2d91b50 | perf/firmus-decode-mainloop-pipeline-20261018 | 最終案（hoist のみを直接記述） |
| 39062255256 attempt 2 | 2d91b50 | 同上 | 最終案の再測定 |

最初の push で 3 ブランチを同時に送ったとき、91152db の run が作られなかった（他の 2 本は起動）。ブランチを消して送り直しても起動しなかったので、同じ内容で commit 時刻だけ変えた 91152db を作って push した。以後は 1 ブランチずつ push している。

## 3. 結果

### 3.1 出力の一致

全 run の全 12 case（Decode 7、Prefill 5）で `output_sha256` が 39053820377 と一致した。数値 gate（`check-firmus-megamoe.py`）も全 run で PASS、計装版の出力も各 run の計装なし版と bit 一致した。

### 3.2 計時（median ms、iteration ごとの rank 間最大値の 30 回 median）

| case | image（3 run） | hoist 最終案 | 再測定 | hoist（1/1） | 2/1 | 1/2 | 2/2 | 4/2 | 8/6 | 32/12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode-balanced-4 | 0.1380 / 0.1379 / 0.1398 | 0.1314 | 0.1314 | 0.1310 | 0.1335 | 0.1335 | 0.1327 | 0.1356 | 0.1345 | 0.1545 |
| decode-balanced-16 | 0.1880 / 0.1880 / 0.1880 | 0.1825 | 0.1818 | 0.1814 | 0.1832 | 0.1820 | 0.1821 | 0.1859 | 0.1849 | 0.2137 |
| decode-balanced-32 | 0.1944 / 0.1948 / 0.1952 | 0.1900 | 0.1889 | 0.1889 | 0.1915 | 0.1899 | 0.1894 | 0.1915 | 0.1934 | 0.2517 |
| decode-balanced-64 | 0.2059 / 0.2076 / 0.2073 | 0.1974 | 0.1954 | 0.1956 | 0.1980 | 0.1969 | 0.1970 | 0.1986 | 0.2056 | 0.2719 |
| decode-balanced-96 | 0.2186 / 0.2169 / 0.2168 | 0.2006 | 0.2007 | 0.1987 | 0.2013 | 0.2015 | 0.2017 | 0.2052 | 0.2293 | 0.3084 |
| decode-balanced-128 | 0.2292 / 0.2305 / 0.2289 | 0.2048 | 0.2065 | 0.2032 | 0.2072 | 0.2107 | 0.2092 | 0.2175 | 0.2497 | 0.3718 |
| decode-local-128 | 0.2256 / 0.2265 / 0.2278 | 0.2027 | 0.2013 | 0.2011 | 0.2046 | 0.2082 | 0.2053 | 0.2085 | 0.2426 | 0.3479 |
| prefill-balanced-1024 | 0.5288 / 0.5186 / 0.5175 | 0.5171 | 0.5306 | 0.5306 | 0.5180 | 0.5287 | 0.5282 | 0.5179 | 0.5185 | 0.5300 |
| prefill-balanced-2048 | 0.9246 / 0.9069 / 0.9117 | 0.9098 | 0.9241 | 0.9210 | 0.9108 | 0.9308 | 0.9301 | 0.9036 | 0.9089 | 0.9235 |
| prefill-balanced-4096 | 1.6680 / 1.7427 / 1.6554 | 1.6955 | 1.6923 | 1.6877 | 1.6853 | 1.6743 | 1.6648 | 1.6978 | 1.7025 | 1.6813 |
| prefill-local-4096 | 1.6683 / 1.5964 / 1.5925 | 1.5767 | 1.5942 | 1.5832 | 1.5986 | 1.6310 | 1.6956 | 1.5658 | 1.6136 | 1.6431 |
| prefill-skew-4096 | 1.8661 / 1.8421 / 1.8287 | 1.8254 | 1.8515 | 1.8736 | 1.8222 | 1.8566 | 1.8605 | 1.8292 | 1.8844 | 1.8638 |

- image の Decode は 3 run で ±1% に収まる。image 3 run の平均に対し、最終案は 4/16/32/64/96/128/local-128 token で −5.2 / −2.9 / −2.4 / −4.6 / −7.7 / −10.8 / −10.6%、再測定は −5.2 / −3.3 / −3.0 / −5.6 / −7.7 / −10.0 / −11.2%。hoist（1/1、−3.0〜−11.5%）とも同じ水準で、差は run 間の揺らぎの範囲。
- hoist（1/1）と最終案は Decode の全 size で image より速い。pipeline を重ねた 2/1・1/2・2/2 は Decode の全 size で 1/1 より 0.3〜4% 遅く、chunk を長くするほど遅くなり、32/12 は image より 62% 遅い。
- Prefill は全版で image の run 間の揺らぎ（±4.5%、前回の記録どおり二峰）の範囲。Prefill の SASS は全版で image と同一なので、差は clock と run 間の揺らぎである。

### 3.3 timeline（Decode-128、計装版、代表 call、4 rank の中央値。µs）

| 版（run） | kernel | FC1 span（CTA median / max） | FC2 span | GEMM 区間 | 実効帯域 FC1 / FC2 / GEMM（TB/s） | TMA A / B の empty 待ち | K loop L1 / L2（per CTA） | full 待ち L1 / L2 | epilogue L1 / L2（per CTA） |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| image + 分割 counter（39058111866） | 257.8 | 137.5 / 141.2 | 78.5 / 82.5 | 223.5 | 2.85 / 2.44 / 2.70 | 152 / 194 | 123.2 / 65.9 | 21.0 / 6.6 | 19.9 / 9.9 |
| chunk 8/6（39058491946） | 254.7 | 112.8 / 132.7 | 72.2 / 93.1 | 211.8 | 3.03 / 2.16 / 2.85 | 117 / 163 | 95.8 / 53.1 | 18.6 / 4.4 | 22.3 / 16.0 |
| chunk 2/2（39059209277） | 213.8 | 108.1 / 115.6 | 59.3 / 70.9 | 178.5 | 3.48 / 2.84 / 3.38 | 93 / 139 | 91.2 / 42.3 | 25.8 / 4.2 | 22.4 / 14.2 |
| chunk 2/1（39060897888） | 209.4 | 108.2 / 114.9 | 58.8 / 68.4 | 175.3 | 3.50 / 2.94 / 3.45 | 92 / 141 | 89.9 / 45.8 | 24.9 / 4.6 | 23.0 / 10.3 |
| hoist 1/1（39060930640） | 214.3 | 112.5 / 117.6 | 59.4 / 65.2 | 180.4 | 3.42 / 3.09 / 3.35 | 97 / 147 | 96.9 / 47.5 | 21.0 / 5.4 | 21.3 / 9.1 |
| 最終案（39062255256） | 215.8 | 112.7 / 118.5 | 60.4 / 66.0 | 181.7 | 3.40 / 3.05 / 3.32 | 98 / 150 | 97.1 / 47.5 | 20.7 / 5.4 | 21.0 / 10.1 |
| 最終案の再測定（39062255256 attempt 2） | 216.1 | 112.7 / 118.4 | 60.5 / 65.6 | 180.9 | 3.40 / 3.07 / 3.34 | 98 / 148 | 97.1 / 47.8 | 20.8 / 5.4 | 20.9 / 9.9 |

実効帯域は FC1 402.7 MB、FC2 201.3 MB、合計 604 MB/rank の重みを、FC1/FC2 は最も遅い CTA の span、GEMM 区間は最初の FC1 stage 到着から最後の FC2 終了までで割った値。K loop・full 待ち・epilogue は per-CTA counter の CTA median（epilogue = block 時間の合計 − K loop の合計）。計装の割り増しは image で +16%、最終案で +9%（計装は k-block ごとに乗るので、k-block あたりの時間が短い版ほど割り増しが小さく見える）。

読み方：

- **hoist だけで K loop が 2 割短くなる**（L1 123 → 97 µs、L2 66 → 48 µs、FC1 span 137 → 113 µs、FC2 79 → 60 µs、GEMM 区間 224 → 181 µs）。image は k-block ごとに N_SWAP の 8 分岐を通り直しており、hoist はその分岐と合流を K loop から外す（weight scale の `__ldg` を full wait の前に移した効果とは分けて測っていない）。重みを読む HBM 側は変えていない。
- pipeline を重ねても K loop は数 % しか縮まない（2/1 で L1 90 µs）。promote と barrier 処理は k-block あたりの費用のうち小さい部分で、重ねても得るものが少ない。計装版では 2/1 が最短だが、計装なしでは 1/1 より遅い。
- pipeline はコード量の代償が大きい。8 個の N_SWAP specialization のそれぞれに chunk 分の unroll 本体が入る。L2 を pipeline にした版（1/2、2/2、4/2、8/6）は FC2 epilogue が CTA あたり 9〜10 → 14〜16 µs に増え、chunk を長くすると CTA 間のばらつき（FC1 span の max − median、image 4 µs → 8/6 で 20 µs）も増える。命令 cache から epilogue と loop のコードが追い出されているとみられる。token 数が増えるほど（使う N_SWAP の種類が増えるほど）遅くなること、1 種類しか使わない Decode-4 では 8/6 でも速いことも、これと合う。

## 4. 残っている時間（最終案、Decode-128）

- 計装なしの kernel は約 200 µs（bench 0.2048 ms は pre-dispatch を含む）。GEMM 区間は計装版で 181 µs、計装の割り増し（約 9%）を除くと約 166 µs で、604 MB に対して実効 3.6 TB/s 前後。HBM 下限（4.8 TB/s で 126 µs）との差は約 40 µs。
- TMA loader はまだ空き stage を 98 / 150 µs 待っている。HBM ではなく math 側が律速という状態は変わらない。
- **FC1 epilogue が CTA あたり約 21 µs（block あたり約 3.5 µs）**。SwapAB の L1 epilogue は、FP32 の [token][64 列] tile を token ごとに 1 thread が 64 列ずつ読んで amax と FP8 量子化を行う。行の stride が 64 float なので warp 内の 32 thread が同じ bank を読む（32-way の bank conflict）うえ、active な thread は最大 64/256 しかない。列方向に warp を割り当て、shuffle で amax をまとめれば、出力は bit 一致のまま数分の一にできる見込み。FC2 epilogue は CTA あたり約 10 µs。
- dispatch（起動から最初の FC1 stage まで）約 20 µs と、最後の FC2 から終了まで 12〜17 µs（rank 内の遅い CTA 待ち median 9〜10 µs、max 17〜26 µs）は前回から変わらない。

## 5. 判断と次の手順

- 採用候補は hoist だけの最終案（`perf/firmus-decode-mainloop-pipeline-20261018` の `2d91b50`）。kernel 単体で Decode を 2〜11% 短縮し（2 run）、出力は bit 一致、Prefill の SASS は不変。serving での効果は、通常の `[require-build]` の image build と serving run で確かめる必要がある（未実施）。
- pipeline 案（91152db と chunk 違いの各版）は不採用。ブランチ履歴（91152db）と本稿に記録を残した。
- 次の kernel 候補は FC1 SwapAB epilogue の bank conflict 解消（4 節）。同じ bench で数分で判定できる。
