# 2026-10-18 SM90 MegaMoE Decode の末尾・起動区間の短縮

[kernel 単体 bench と timeline](2026-10-18-megamoe-kernel-timeline.md) の Decode-128 では、最後の FC2 から kernel 終了まで 12〜17 µs、起動から最初の FC1 まで約 19 µs かかっていた。[計画ノート](2026-10-18-firmus-maximum-performance-plan.md) の 2-5 はこの二つで合計 −20〜25 µs/call を見込んでいた。本稿は、8 patch 適用済みの image（header `135a43be…`、image `7c6286e5…`）の上に重ねる Decode 向けの改善案を作り、[kernel-bench] の複数案 job で測った結果をまとめる。

結論：

- **採用候補は 3 patch（計測済み）。** komoriyuta の `sm90-combine-parallel`（origin/perf/firmus-evidence-20261018。旧 header `0e34a083` 向けだったものを 8 patch の上に載せ直した）を、dispatch の不要な読み出しの削除と並列 combine の 2 ファイルに分け、その上に動的な block 割り当てを重ねた。全 Decode size で image より速く、Decode-128 は 203.0 → 196.1 / 197.1 µs（−3.4 / −2.9%、同一 job の 2 回）。Decode-4/16/32 は −4〜−5%。12 case の出力 hash はすべて image と一致し、数値 gate も通った。
- **目標の −20 µs には届かなかった。** Decode-128 で −6〜−7 µs。rank 内の遅い CTA 待ちは median 7〜8 → 2〜6 µs、combine は 9 → 5〜6 µs に縮んだ。残りは rank 間の負荷差（routing で決まる）と、rank 間の barrier・cleanup barrier の往復。
- 一つ目の job の flat barrier と pre-pull barrier の省略は、**rank ごとに geometry が変わる EP collective で通信手順が食い違い、gate で落ちた**（2.3 節）。修正版は Prefill を含む全 geometry に同じ手順を入れる patch として作り、GPU 未計測（GPU-pending）で置いた。
- FC1 epilogue の TMA store 完了待ちを store を出した warp だけにする patch も、GPU 未計測で置いた。

## 1. 改善案

すべて `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh` だけを変える。Decode だけの変更は SwapAB 形状の template 条件（`kDecodeTail = kSwapABActive`）の内側に置き、Prefill の SASS は変えていない（1 と 5 を除く。4 節）。

### 1.1 dispatch の L1 top-k 重みの読み出し削除（`sm90-dispatch-drop-l1-weight.patch`、komoriyuta の案）

dispatch は token ごとに remote rank の top-k 重みを NVLink で読み、L1 重み pool に書いていた。late FP32 top-k patch 以降、combine は重みを自 rank の入力 buffer から読み、L1 重み pool を読む場所はない。Decode 分岐と Prefill 分岐の両方からこの読み書きを外した。pool 自体は host と共有する buffer layout に残る。Prefill の SASS は変わる（spill 40 → 20 byte）。これは coordinator が認めた例外で、条件は「純粋な不要 load の削除」「Prefill の出力も bit 一致」「単独の patch ファイル」の三つ。Prefill 側の作業はこの patch の上に rebase する。

### 1.2 並列 combine（`sm90-combine-parallel.patch`、komoriyuta の案）

従来の combine は 1 warp が 1 token の 8 KiB 行を受け持つので、128 token なら 132 CTA のうち 16 CTA の 128 warp しか動かない。komoriyuta の案は 1 token を 1 KiB の chunk 8 個に分け、warp ごとに (token, chunk) を受け持たせる。各要素は同じ slot 順に同じ FP32 FMA で足すので、出力は変わらない。旧 header 向けの patch は combine の hunk がそのまま当たったので、dispatch の hunk（1.1）と分けた。

同じ考えで、自作版（live な top-k 行を 1 つの mbarrier にまとめて一度に読む）も作った。1 回目の job で komoriyuta 版と差がなかった（Decode-128 で 200.2 / 199.4 µs）ので、単純な komoriyuta 版を土台にした。

### 1.3 動的な block 割り当て（`sm90-decode-dynamic-schedule.patch`）

静的 scheduler は CTA i に block i, i+132, … を渡す。Decode-128（32 expert、各 1 m-block）では FC1 768 block（32 k-block）と FC2 1024 block（12 k-block）を割り振り、CTA ごとの block 数は「FC1 6 + FC2 8」が 76 CTA、「6 + 7」が 32 CTA、「5 + 8」が 24 CTA になる。image の timeline（run 39065587762）では、各組の最後の FC2 終了の median が 190 / 183 / 179 µs、最も遅い CTA が 195 µs で、全 CTA が combine 前にその CTA を待っていた。

変更点：

- CTA i は最初に block i を処理する。次からは A loader が、今の block の TMA load を出し終えた時点で rank 内の counter に `atomicAdd` して次の block を取る。取った位置との差を shared memory の 4 slot ring（expert-count 領域の空き 768 byte 目から、full / empty mbarrier 付き）で B loader と math warp に渡す。
- block 番号の空間と解釈は既存 scheduler のもの（wave の FC1 block、次にその wave の FC2 block）をそのまま使い、`get_next_block` が足した kNumSMs を差し引いて取った位置へ進める。各 CTA の番号は増える一方で、block は自分より小さい番号の block（同じ expert の FC1 と dispatch）しか待たないので、待ちは循環しない。block の計算はどの CTA で走っても同じなので出力も変わらない。
- counter は L2 arrival-mask の、BLOCK_M=64 の pool block（2208 個）より後ろの slot に置いた。workspace はこの領域を最小の候補 BLOCK_M（8）向けに確保しているので、確保済みで、buffer の zero 初期化で 0 になり、他から触られない。SM 0 が cleanup で 0 に戻す。その時点で、pre-combine barrier の前段の grid sync を全 CTA の math が通過しているので、この call の取得はすべて終わっている。
- Decode の combine buffer を expert-count 領域の後ろへずらした。ring の最後の mbarrier arrive（B loader）は combine より前に来る保証がないため。
- register：ring の slot・phase を 1 つの反復番号にまとめ、ring には絶対位置ではなく差を載せ、counter の pointer を使う場所で計算し、empty の arrive を lane 0 だけでなく全 thread で行う。こうして Decode 168 register・spill 0 に収めた。途中の版では、FC1/FC2 の k-block 数を `?:` で選ぶと 2 要素の local 配列になる、述語を詰めた word が spill する、といった問題が出た。

### 1.4 FC1 store 完了待ちの局所化（`sm90-decode-l1-store-wait.patch`、GPU-pending）

SwapAB の FC1 epilogue は、TMA store を出した後、全 256 thread が CTA barrier でその完了を待ち、その後で 1 thread が arrival を出していた。本案では store を出した thread がそのまま `tma_store_wait<0>` と `red.release.gpu` を行い、2 つ目の CTA barrier をなくす。他の warp は次の block へ進む。FP8 staging tile に次に書くのは次の FC1 epilogue の CTA barrier の後で、store を出した warp は待ちを終えてからでないとそこに着かない。L2 SF の store は 1 つ目の barrier の前にあり、release の累積性で覆われる。計算は変えていないので出力は bit 一致する。

### 1.5 全 geometry 共通の flat barrier（`sm90-flat-nvlink-barriers.patch`、GPU-pending、任意）

1 回目の job で落ちた案（2.3 節）を、全 geometry に同じ手順として入れ直したもの。

- pre-combine barrier と cleanup barrier：各 CTA が CTA 内で同期した後、全 rank の 64 bit counter に release.sys で 1 を足す（thread r が rank (rank_idx + r) へ）。pre-combine は全 CTA が kNumSMs × kNumRanks 回の到着を待つ。cleanup barrier は従来どおり SM 0 だけが待つ。grid sync は使わない。ある call の barrier は、どの rank も次の call の同じ barrier に着く前に全 rank で終わる。このため待つ目標は自 CTA の local add が返した値から決まり、counter は増える一方で reset しない。counter は L2 arrival-mask の最後の 2 slot（`num_max_pool_blocks − 1, − 2`）に置いた。番地が geometry に依存しないので、remote rank から同じ番地に届く。
- pre-pull barrier の省略：SM 0 は per-expert 合計を release.sys で加算する。pull loop は既存の spin（全 rank 分そろうまで）の後に `fence.acq_rel.sys` を置いて、per-rank 数と送信元 index を acquire する。remote の token を読むのは、その rank が publish した後、つまりその rank の pre-dispatch が終わった後になる。
- Prefill の SASS は変わる（spill 20 → 28 byte）。Prefill 側の patch と同じ箇所（pre-combine barrier、cleanup）に触れるので、統合時は衝突に注意が要る。Decode は 168 register・spill 0。
- 期待値は 1 回目の timeline からの見積もりで、Decode あたり 4〜7 µs（pre-combine の grid sync 2 回分、cleanup の grid sync、pre-pull barrier の epilogue grid sync）。Prefill にも同じ量が効く。

### 1.6 精度規則の緩和の後で見直したこと

coordinator から、bit 一致の代わりに「数値 gate が通ること、精度を下げないこと、仕事を省かないこと」を条件とすると連絡があった。SwapAB mainloop の promote は、すでに `final += (scale × sf) × acc` の FMA 形になっている。weight scale（128×128 block）と activation scale（token × 128 K）は k-block ごとに変わるので、k-block をまたいで promote をまとめることもできない。1 つ前の作業の timeline でも、K loop は計装分を除くと HBM 下限の約 135/126 で、残りは epilogue（CTA あたり FC1 12.5、FC2 10.5 µs）と CTA 間・rank 間の待ちだった。規則の緩和で効く mainloop の案は見つからず、どの候補も bit 一致のままにした。

## 2. 実行記録

すべて `[cluster:firmus] [nodes:1] [kernel-bench]`、image `7c6286e5…`、Firmus hpcai 1 ノード 4×H200、複数案 job。

| run | commit | ブランチ | 内容 |
|---|---|---|---|
| 39086770417 | f1b393b | experiment/firmus-decode-tail-20261018（archive tag のみ） | komoriyuta、decode-combine（自作 combine）、decode-sched（+ flat barrier + 動的割り当て）、decode-all（+ pre-pull barrier 省略、timeline）。Slurm 24441 |
| 39093822925 | f5daaef | experiment/firmus-decode-sched-20261018（archive tag のみ） | komoriyuta、kom-sched（+ 動的割り当て、timeline）、kom-sched-b（同じ patch で再測定）、sched（動的割り当てのみ）。Slurm 24463 |

### 2.1 1 回目（39086770417）

komoriyuta と decode-combine は gate を通り、12 case の hash が image と一致した。decode-sched と decode-all は gate で落ちた（2.3 節）。

### 2.2 2 回目（39093822925）

全 4 案が gate を通り、全 12 case の hash が image と一致した（各 step 58〜68 秒）。

### 2.3 1 回目の gate 失敗の原因

両案とも rank 3 の `torch.cuda.synchronize()`（`check-firmus-megamoe.py:89`）が `unspecified launch failure` を報告した。gate の起動から約 5 分 20 秒後で、gate の最初の MegaMoE case（`cross-rank-empty-tail`、rank 別 token 数 [7, 0, 65, 129]）の最初の forward だった。

host（`csrc/jit_kernels/heuristics/sm90_mega_moe.hpp` の `should_use_swap_ab_for_mega_moe_sm90`）は、**各 rank の自分の token 数**で geometry を選ぶ。SwapAB は num_tokens ≤ 128 のときだけ使う。このため、129 token の rank 3 だけが非 SwapAB の kernel を、ほかの rank は SwapAB の kernel を同時に実行する。flat barrier と pre-pull barrier の省略は SwapAB の kernel の rank 間手順だけを変えていた。rank 3 は従来の NVLink barrier で来ない signal を待ち続け、300 秒の timeout trap に入った。serving でも geometry は混ざる。A1 profile の prefill-0 には SwapAB の kernel が 94 回現れる。

rank 間の手順（barrier、publish）を変える案は、全 geometry に同じ変更を入れる必要がある。2 回目の案は rank 内で閉じる変更だけにした。全 geometry 版は 1.5 節。

## 3. 結果

### 3.1 計時（median µs、iteration ごとの rank 間最大値の 30 回 median。括弧は同じ job の image 比）

| case | image（2 回目） | komoriyuta | kom-sched | kom-sched-b | sched | 1 回目 image | 1 回目 komoriyuta | 1 回目 decode-combine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode-balanced-4 | 132.3 | 128.8（−2.6%） | 126.3（−4.5%） | 126.9（−4.1%） | 129.2（−2.4%） | 132.1 | 130.0（−1.5%） | 128.4（−2.8%） |
| decode-balanced-16 | 181.1 | 177.8（−1.8%） | 174.0（−3.9%） | 173.7（−4.1%） | 177.6（−1.9%） | 181.8 | 177.9（−2.2%） | 177.7（−2.3%） |
| decode-balanced-32 | 190.0 | 185.6（−2.3%） | 180.2（−5.2%） | 180.4（−5.1%） | 183.6（−3.4%） | 187.8 | 184.5（−1.8%） | 185.3（−1.3%） |
| decode-balanced-64 | 193.9 | 190.8（−1.6%） | 188.8（−2.6%） | 188.7（−2.7%） | 192.2（−0.9%） | 195.1 | 191.4（−1.9%） | 190.7（−2.3%） |
| decode-balanced-96 | 197.3 | 194.9（−1.2%） | 194.2（−1.6%） | 194.1（−1.6%） | 197.6（+0.1%） | 198.6 | 193.2（−2.7%） | 195.7（−1.4%） |
| decode-balanced-128 | 203.0 | 198.7（−2.1%） | 196.1（−3.4%） | 197.1（−2.9%） | 200.5（−1.2%） | 205.1 | 199.4（−2.8%） | 200.2（−2.4%） |
| decode-local-128 | 200.0 | 194.5（−2.8%） | 194.8（−2.6%） | 195.1（−2.5%） | 197.9（−1.1%） | 200.8 | 195.0（−2.9%） | 196.6（−2.1%） |
| prefill-balanced-1024 | 517.6 | 515.3 | 515.5 | 516.6 | 515.6 | 521.4 | 514.4 | 514.4 |
| prefill-balanced-2048 | 908.3 | 908.2 | 904.4 | 906.0 | 906.8 | 911.2 | 903.0 | 908.4 |
| prefill-balanced-4096 | 1662.9 | 1666.5 | 1712.8 | 1693.4 | 1687.7 | 1702.4 | 1656.8 | 1662.9 |
| prefill-local-4096 | 1585.2 | 1592.1 | 1568.3 | 1615.3 | 1576.7 | 1571.8 | 1593.9 | 1598.1 |
| prefill-skew-4096 | 1825.7 | 1824.0 | 1813.6 | 1813.0 | 1864.9 | 1829.9 | 1852.3 | 1823.9 |

- kom-sched と kom-sched-b（同じ patch）の差は Decode で 1 µs 以内。両方とも全 Decode size で komoriyuta より速い。動的割り当ての上乗せは、4〜32 token で −2〜−3%（16 token 以下は wave あたり 16 expert の instantiation）、64〜128 token で −0.4〜−1.3%。
- sched（並列 combine なし）は image 比 −1〜−3% だった。並列 combine と動的割り当ての効果はほぼ足し算になる。
- 自作の combine と komoriyuta 版は、同じ job で差が 1% 以内（ほとんどの size で揺らぎの範囲）だった。
- Prefill の SASS は komoriyuta・kom-sched・sched で同一（1.1 の削除による 1 種類のみ）。Prefill の差は、run 間の揺らぎ（±4.5%、二峰）と同じ程度。

### 3.2 timeline（Decode-128、計装版、代表 call、µs。CTA の median / max）

| 段階 | image（39065587762） rank 0 / 1 / 2 / 3 | kom-sched（39093822925） rank 0 / 1 / 2 / 3 |
|---|---|---|
| 最初の FC1 stage | 19.4 / 17.4 / 20.7 / 20.1 | 17.1 / 19.1 / 18.4 / 18.9 |
| 最後の FC2 終了（CTA median / max） | 188.4/195.1, 183.7/189.6, 188.0/194.1, 188.9/195.1 | 190.5/191.9, 182.1/187.5, 189.0/190.6, 192.7/194.0 |
| rank 内の遅い CTA 待ち（median / max） | 8.3/18.2, 6.9/19.9, 7.5/22.0, 8.1/23.5 | 2.5/9.8, 6.4/9.3, 3.2/10.0, 2.3/9.7 |
| pre-combine NVLink barrier（rank 間待ち） | 2.8, 8.3, 5.0, 2.1 | 3.9, 9.2, 5.1, 2.0 |
| combine（max） | 9.2, 9.3, 9.3, 9.8 | 6.3, 5.3, 5.5, 6.8 |
| dispatch 終了（cleanup barrier 後） | 204.5, 203.8, 205.5, 204.5 | 203.6, 204.1, 203.7, 204.2 |
| kernel 全体 | 208.3, 208.0, 209.3, 208.6 | 203.6, 204.1, 203.7, 204.2 |

- 動的割り当てで、rank 内で最後の FC2 が終わる時刻のばらつきは median−max 差 6〜7 µs → 1.4〜5.4 µs になった。代わりに FC2 の span の max が伸びる（66 → 78 µs）。最後の block を早く終えた CTA が次の FC2 block を取るためで、kernel 時間には効かない。
- 残る待ちは rank 間の差。rank 1 は受信 pair が最も少なく（995 対 1072）、他 rank より 5〜10 µs 早く終わって pre-combine barrier で待つ。routing の偏りで決まり、kernel の中では消せない。
- combine は 9 µs → 5〜6 µs。それでも kernel 終了は、combine と、dispatch 側の cleanup + 最終 barrier（pre-combine barrier から約 6 µs）がほぼ同時に決めている。combine だけをさらに縮めても終了は早まらない。自作の一括読み combine の利点が出なかった理由でもある。1.5 の flat barrier は cleanup 側も縮める案。

## 4. register / spill / SASS（Firmus login、image の CUDA 13.0 nvcc、`containers/build-sm90-megamoe.py` と同じ instantiation に wave 16 の Decode を加えた）

| chain（8 patch の上） | header sha256 | Decode / Decode-16 | Decode spill | Prefill |
|---|---|---|---|---|
| image（8 patch） | `135a43be…` | 168 / 168 | 0 | spill 40/40、SASS `c1f5f67f…` |
| + drop-l1-weight | `c28e4ea7…` | 168 / 168 | 0 | spill 20/20、SASS `fc516eee…` |
| + combine-parallel | `50a75c51…` | 168 / 168 | 0 | `fc516eee…` |
| + decode-dynamic-schedule | `7bd1deac…` | 168 / 168 | 0 | `fc516eee…` |
| + decode-l1-store-wait | `af518d97…` | 168 / 168 | 0 | `fc516eee…` |
| + flat-nvlink-barriers | `4cd72987…` | 168 / 168 | 0 | spill 28/28、SASS `4fe79157…` |

Decode の「8 byte stack frame」は image から変わらない（ptxas は spill に数えない）。timeline 計装版（kom-sched）は Decode spill 28/136 byte で、従来の計装版と同じ程度。

## 5. 統合の方法と残り

- 配線（integrator が行う）：`patches/deep-gemm/` の chain の末尾（`sm90-dispatch-pull-overlap.patch` の後）に、`sm90-dispatch-drop-l1-weight.patch` → `sm90-combine-parallel.patch` → `sm90-decode-dynamic-schedule.patch` の順で足す。header sha は `7bd1deacb2066e88fdd0088f8e32cc4ec7d7b660434137ec8836105e69cec05f`。GPU-pending の 2 つも入れる場合は、続けて `sm90-decode-l1-store-wait.patch`（`af518d976abf0626a9f398f8f547bd0fb557924da387d0257df84f24d3b4134d`）、`sm90-flat-nvlink-barriers.patch`（`4cd729874af3e46bcfd88d359f064374b9e7407cd4d2ef0952404f097dd5ea27`）。`containers/build-sm90-megamoe.py`、Dockerfile、`scripts/check-firmus-image.sh` の header pin と patch sha の記録を、この順に合わせて更新する。
- Prefill 側の patch は、drop-l1-weight の上に rebase する前提で作る。flat-nvlink-barriers は Prefill の pre-combine barrier と cleanup barrier も書き換えるので、Prefill tail の patch と同じ箇所を触る。
- GPU-pending の 2 つは、統合用の一括 kernel bench で gate と計時を取る。flat barrier の確認点は gate の `cross-rank-empty-tail`（geometry 混在）、graph replay、idle rank。
- serving の Decode は bench（203 µs）より遅い 212.6 µs/call で、routing の偏りが大きい。動的割り当ては expert ごとの token 数の偏りも吸収するので、serving での効果は bench より大きい可能性がある（未確認）。
