# 2026-10-18 Prefill MegaMoE の geometry 変更（Tier 3-1）と Prefill 限定の冗長 expert（Tier 3-5）の検討

[Firmus 上限計画](2026-10-18-firmus-maximum-performance-plan.md) の Tier 3 のうち、3-1（DeepGEMM host library `_C.so` を再ビルドして Prefill MegaMoE の geometry を選び直す）と 3-5（Prefill だけ hot expert を複製する）を、ソースと既存の計測から机上で評価した。対象は image `e29dfde…` の sgl-deep-gemm 0.1.5（host は tag v0.1.5 の `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp` と `csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp`、kernel は `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh` に main の `patches/deep-gemm/*.patch` を当てたもの）と、Qwen EP4 Prefill 形状（4096 token/rank、local expert 32、hidden 4096、intermediate 1536、top-8）。

## 1. 結論

- **3-1：採用できる geometry はない。GPU job は投入していない（予算 3 回のうち 0 回）。** host heuristic が選べる値のうち、stage 数は 128×256 では SMEM で 3 が上限、wave あたり expert 数は既に最大の 32、epilogue register は予算上限の 112 で、cluster/TMA multicast と PDL は kernel 側に実装がない。出力を bit 一致に保てる別 geometry は 128×128（2 warpgroup、5 stage）と 192×128（3 warpgroup、4 stage）だけで、どちらも律速要因（k-block ごとの promote と tile ごとの同期 epilogue）を悪化させ、stage を増やす利点は timeline 上ほぼない（§2.4）。計画の「MegaMoE −10〜15%」は host 側の geometry 変更では裏付けられない。
- 残っている損失は kernel 構造（header）側にある：(1) reuse-accum 方式の promote（k-block ごと、accumulator 1 要素に FMUL 2 回）、(2) 4 warpgroup が同じ tile を分担するため epilogue 中は tensor core が止まる、(3) FC2 は per-64 SF のため k-block ごとに WGMMA 2 group・`wait<0>` 2 回・scale 4 回。(1) を通常の promote（`final += s·acc`、FFMA 1 回）に替える案は丸めが変わり bit 一致でなくなるので、精度方針の判断なしには進めない（§2.5）。
- **3-5：静的な冗長 expert は効かない。** rank 負荷 max/mean（94 layer 平均）は R=0（現行の静的配置）1.0341 → R=4 1.0313 → R=8 1.0284。Prefill MegaMoE は比例上限でも −0.27% / −0.55%、Prefill 全体では −0.16% / −0.32% で、実装の閾値 1.5% に遠く届かない。残っている 1.034 は pass ごとの揺らぎで、全 pass 合計では既に 1.000003 なので、固定の分割では追えない。pass ごとに最適に分ける動的分割なら R=4 で 1.0031 まで下がるが、Prefill 全体では比例上限で −1.7%、配置 run での実現率（36%）を掛けると −0.6% で、rank 間で count を交換する分の起動コストも引かれる。**実装しない。**

## 2. 3-1：host heuristic と kernel template が選べるもの

### 2.1 host の選択（v0.1.5）

| 項目 | Prefill（4096 token） | 決め方 |
|---|---|---|
| BLOCK_M / math thread | 128 / 512（4 warpgroup） | expert あたり期待 token ≥ 64 なら `{128, 512}` 固定、それ以外は `{64, 256}` |
| BLOCK_N | 256 | 128×512 のとき 256 固定 |
| BLOCK_K | 128 | 固定（per-128 SF） |
| cluster | 1 | 定数 `cluster_size = 1` |
| wave あたり expert | 32 | expert あたり期待 token が 1〜4 の外なら local expert 全部 |
| dispatch / TMA thread | 64 / 64 | math 512 のとき 64 |
| stage / SMEM | 3 / 225,600 B | `(232448 − 固定部) / stage あたり` |
| epilogue register | 112 | math 512 のとき 112（64×32 + 64×24 + 512×112 = 60,928 ≤ 64,512） |
| reuse accum as final | true | `block_m == 128` に固定で結びつく |
| L2 arrival counter | true | 128×256×512 のときだけ |
| PDL | 実行時の大域フラグ（`deep_gemm.set_pdl`） | MegaMoE kernel には `griddepcontrol` がない |

上の式で計算した SMEM 225,600 B は nsys の起動 metadata と一致する。環境変数で geometry を変える口は無く（`DG_SM90_FP8_SWAP_AB` は Decode の SwapAB だけ）、変えるには `_C.so` の再ビルドが要る。

### 2.2 kernel template の制約（header）

- warpgroup ごとの tile は必ず 64 行。BLOCK_M 128 で 4 warpgroup のときだけ 2×2 分割（各 64×128）、BLOCK_M 128 で 2 warpgroup なら M 方向 2 分割（各 64×BLOCK_N）、BLOCK_M 64 なら N 方向分割。
- weight SF の添字（`gate_n = sf_n_block_idx / 2`、L2 は warpgroup あたり `l2_sf` 1 個）と L1 出力の per-64 SF（warpgroup あたり amax 1 個）は **warpgroup の N 幅 128 を前提にしている**。64×256 の warpgroup tile はこのままでは黙って誤った結果になる。さらに launch bound（384 thread → 168 register、[Decode mainloop note](2026-10-18-megamoe-decode-mainloop-pipeline.md) §1.1-4）では accumulator 128 個と SwiGLU の一時値 64 個が収まらない。
- BLOCK_M の候補は layout の `kCandidateBlockM = {8, 16, 32, 64, 96, 128, 192}` で、256 は pool の大きさが足りない。
- 「Pipeline (cluster=1, no TMA multicast)」と明記され、`LOAD_BLOCK_M = BLOCK_M` で multicast の分割も同期も無い。host の cluster を 2 にしても CTA が 2 個組で起動されるだけで、データは共有されない。
- `kReuseAccumAsFinal`（BLOCK_M 128）は accumulator を最終値として使い回し、k-block ごとに「1/scale を掛ける → WGMMA で積算 → scale を掛ける」を行う。BLOCK_M 64 系は別の accumulator に積んで `final += s·acc` で promote する。**両者は丸めが異なるので、BLOCK_M 64 系の geometry は bit 一致しない。**

### 2.3 候補の SMEM と stage（host の式で計算）

| geometry | warpgroup tile | stage | SMEM B | WGMMA の SMEM 読み / TMA 書き（128×256×128 相当あたり） | bit 一致 |
|---|---|---:|---:|---|---|
| 現行 128×256、4 WG | 64×128 | 3 | 225,600 | 96 KB / 48 KB | — |
| 128×128、2 WG | 64×128 | 5 | 219,376 | 96 KB / 64 KB | する見込み（reuse 経路のまま） |
| 192×128、3 WG（register 予算の指定が要る） | 64×128 | 4 | 228,624 | 96 KB / 53 KB | する見込み（reuse を明示した場合） |
| 128×256、2 WG（64×256） | 64×256 | 2（dispatch 64 なら 3） | 183,488 | 80 KB / 48 KB | header 書き換えが要り、register が足りない |
| 64×256、2 WG（N 分割） | 64×128 | 4 | 208,080 | 96 KB / 80 KB | しない（非 reuse） |
| 128×256 で 4 stage | 64×128 | — | 275,792 必要 | — | 不可能。固定部を 31.7 KB 以下にする必要があるが、L2 の BF16 出力 staging だけで 64 KB |

### 2.4 なぜどの候補も勝てないか

[kernel timeline](2026-10-18-megamoe-kernel-timeline.md) §4.3 の Prefill-4096 を tile 単位に直すと（rank0、1.54 GHz。tensor core の理想値は 8192 FLOP/clk/SM として計算）:

| 段階 | 実測 | tensor core の理想値 | 効率 |
|---|---:|---:|---:|
| FC1 1 tile（128×256、K 4096 = 32 k-block） | 926 µs / 25 = 37.0 µs | 21.3 µs | 58% |
| FC2 1 tile（128×256、K 1536 = 12 k-block） | 604 µs / 33 = 18.3 µs | 8.0 µs | 44% |

- **stage を増やす利点がない。** TMA warp は空き stage を 1030〜1280 µs（GEMM 約 1.5 ms のうち）待っており、loader は常に先行している。math が stage を待つのは FC1 103〜116 µs（うち最初の stage まで 26〜40 µs）と FC2 46〜51 µs、合わせて約 9% で、その一部は dispatch 到着（L1 到着待ち 32〜37 µs）と L2 の arrival 待ち。pipeline の深さで取り戻せるのは多めに見ても数 % にとどまる。時間で見た先行量も、128×128 の 5 stage（5 × 約 0.5 µs）は現行 3 stage（3 × 約 1 µs）と大差ない。
- **128×128 / 2 WG は損失要因を増やす。** k-block あたりの promote 演算は accumulator 1 要素あたり FMUL 2 回で geometry によらず、これを隠す warp が SMSP あたり 4 本から 2 本に減る。tile 数は 2 倍になり、tile ごとの epilogue・barrier の回数も 2 倍になる。TMA の SMEM 書き込みは FLOP あたり 1.33 倍。さらに host の結びつきで L2 arrival counter が外れ、L1 epilogue に CTA 全体の同期が戻る。
- **192×128 / 3 WG** は stage が 4 になる代わりに expert ごとの M 端数の padding が増え（平均 1024 token/expert で 128 行なら約 6%、192 行なら約 9%）、warp も 3 本に減る。
- **cluster 2 / multicast** は L2 → SMEM の転送を減らす手段だが、TMA は既に先行しており律速ではない。kernel 側も未実装。
- **PDL** は直前の kernel の末尾と起動を重ねるだけで、上限は数 µs/call（< 0.3%）。MegaMoE kernel は `griddepcontrol.wait` を持たないので、有効にすると pre-dispatch の出力が見える前に読む危険がある。225.6 KB の SMEM を使う CTA は前の kernel の CTA と同じ SM に同居もできない。
- **SMEM 帯域**（推定）：SS mode の m64n128k32 は 1 命令 64 clk で A 2 KB と B 4 KB を読み、96 B/clk。TMA の書き込み 47 B/clk を足すと 143 B/clk で、SM あたり 128 B/clk を超える。tensor core の上限は約 90% に頭打ちになるが、現状の 44〜58% から見ると主因ではない。64×256 の warpgroup tile なら読みは 80 KB に減るが、§2.2 のとおり header の書き換えと register の両方で成り立たない。

FC1 と FC2 の差から k-block 単価と tile ごとの固定費を分けることはできない（FC2 の k-block は scale 処理が 2 倍で、仮定次第で tile 固定費は 0.7〜7 µs のどちらにもなる）。どちらの場合も損失は k-block ごとの promote と同期 epilogue にあり、tile の形の選択では減らない。

### 2.5 kernel 構造側の案（3-1 の範囲外。記録のみ）

| 案 | 仕組み | bit 一致 | 備考 |
|---|---|---|---|
| 通常の promote（`final += s·acc`）を 128×128 / 2 WG（168 register）で使う | k-block あたり ALU 半減（FMUL 2 → FFMA 1）、DeepGEMM 通常 GEMM と同じ構成 | **しない**（丸め順が変わる） | host の `reuse_accum_as_final` を外す再ビルドと、§2.4 の 2 WG の不利の両方を伴う。精度方針の判断が先 |
| warpgroup を 2 組に分けて別 tile を処理（ping-pong） | 片方の epilogue 中にもう片方が tensor core を使う | する見込み | register と SMEM（CD 64 KB × 2）が足りず、tile 形状の見直しを伴う大改修 |
| FC2 の per-64 SF 分割の扱い | k-block あたり `wait<0>` 2 回と scale 4 回 | 実装次第 | Prefill tail を扱う別作業（`perf/firmus-max-prefill-moe-20261018`）と同じ区間なので、ここでは触れない |

### 2.6 `_C.so` の再ビルド経路（今回は不要）

採用案が無いので image build への追加は行わない。将来 host を変える場合の要点だけ残す：sgl-deep-gemm v0.1.5 の `setup.py` は `CUDAExtension`（`csrc/python_api.cpp`、include は `deep_gemm/include`・`third-party/cutlass/include`・`third-party/fmt/include`、link は `cudart` と `nvrtc`）なので、image の `/opt/sglang/venv` の torch と CUDA 13.0 で `_C.so` だけを作れる。kernel-bench の候補 overlay は `deep_gemm` package を丸ごと `PYTHONPATH` の先頭に置くので、再ビルドした `_C.so` を候補ディレクトリに入れれば image を作り直さずに測れる。

### 2.7 計算上の補足

timeline note は FP8 dense の理論値を 1979 × clock / 1.98 GHz で換算している（1.64 GHz で 1640 TFLOPS、速い rank の FC1 950 TFLOPS が 58%）。仕様値 1979 TFLOPS は 8192 FLOP/clk/SM × 132 SM × 1.83 GHz に相当するので、clock 比で換算すると 1.64 GHz の理論値は約 1770 TFLOPS、950 TFLOPS は 54% になる。§2.4 の表は tile 単位（padding 行の計算を含む）なので 58% / 44% と出る。どちらで数えても結論は変わらない。

## 3. 3-5：Prefill 限定の冗長 expert

### 3.1 方法

`2026-10-18-megamoe-geometry-and-redundancy.py`（結果は同名の `.json`）で、[配置 note](2026-10-18-megamoe-expert-placement.md) と同じ run 39051152456 の stat dump（3 worker、1640 pass、94 layer）を使った。

1. 複製：load / 複製数が最大の expert に複製を 1 つ足す（SGLang EPLB と同じ規則）、を R 回繰り返す。1 expert の複製は最大 4（rank ごとに 1 つ）。
2. 配置：128+R 個の物理 expert を rank あたり (128+R)/4 個、同じ expert の複製は別 rank に、複製あたり load の大きい順に greedy で詰める。R=0 は現行の `configs/firmus-prefill-expert-location.json` と同じ規則で、指標も 1.0341 で一致する。
3. 評価指標は配置 note と同じ「pass ごとの最重 rank / 平均を token 数で加重平均したもの」の 94 layer 平均。複製された expert の token の分け方は次の 2 通り。
   - 静的：複製に均等に分ける。token index の偶奇や送信元 rank ごとの固定割り当てなど、どの固定分割もこれより良くはならない。
   - pass ごとに最適：その pass の count を見て分け、最重 rank を最小にする。最適値は「rank の部分集合 U について（U 上の非複製 expert の load ＋ 複製が全部 U に入っている expert の load）/ |U|」の最大値（Hall の条件）で、動的分割の上限を与える。

### 3.2 結果

| R | rank あたり expert | 複製された expert/layer | 静的 max/mean（p90 / 最悪 layer） | 静的、128 行 tile 換算 | 静的、out-of-sample | pass ごと最適 | 追加 VRAM/rank |
|---:|---:|---:|---|---:|---|---:|---:|
| 0 | 32 | 0 | 1.0341（1.044 / 1.054） | 1.0327 | 1.0340〜1.0349 | 1.0341 | 0 |
| 4 | 33 | 3.7 | 1.0313（1.041 / 1.053） | 1.0300 | 1.0308〜1.0320 | 1.0031 | 1.78 GB |
| 8 | 34 | 6.2 | 1.0284（1.036 / 1.053） | 1.0273 | 1.0281〜1.0291 | 1.0001 | 3.55 GB |
| 12 | 35 | 8.1 | 1.0259（1.033 / 1.044） | 1.0249 | 1.0260〜1.0268 | 1.0000 | 5.32 GB |

VRAM は 1 expert 18.88 MB（w13 3072×4096 と w2 4096×1536 の FP8 に block128 scale）× 94 layer × R/4。

### 3.3 Prefill 時間への換算

kernel 時間が最重 rank の load に比例するとした上限（kernel bench の skew 1.13 → +12% はほぼ比例）と、静的配置の serving run で実現した割合（予測 −22.7% に対し実測 −8.1%、36%）の 2 通りで換算した。MegaMoE は Prefill step の 58%（[計画 §2.3](2026-10-18-firmus-maximum-performance-plan.md)）。

| 方式 | MegaMoE（比例上限） | Prefill step（比例上限） | Prefill step（実現率 36%） |
|---|---:|---:|---:|
| 静的 R=4 | −0.27% | −0.16% | −0.06% |
| 静的 R=8 | −0.55% | −0.32% | −0.11% |
| 静的 R=12 | −0.80% | −0.46% | −0.17% |
| pass ごと最適 R=4 | −3.0% | −1.74% | −0.63% |
| pass ごと最適 R=8 | −3.3% | −1.91% | −0.69% |

### 3.4 判断

- 静的な冗長 expert は R=12（5.3 GB/rank）でも Prefill −0.5% 未満で、閾値 1.5% を大きく下回る。静的配置で全 pass 合計の偏りは 1.000003 まで消えており、残りの 1.034 は pass ごとの揺らぎである。最も熱い expert でも割当の約 11%（rank の取り分 25% の半分以下）なので、expert の粒度は LPT の制約になっていない。複製は粒度の問題を解く手段で、揺らぎには効かない。
- 動的分割は比例上限なら閾値に届くが、実現率を掛けると −0.6〜0.7% にとどまる。しかも複製された expert の全 rank の count を分割前に知る必要があり、kernel 内で count を交換する NVLink barrier を 1 回足す（現状の pre-pull barrier で 5〜10 µs、MegaMoE 1.8 ms の 0.3〜0.5%）か、kernel の外で all-reduce する（94 layer で数 ms）ことになり、得られる分の大半を食う。
- 実装するなら必要なもの（記録のみ）：SGLang の MegaMoE adapter の guard（`args.ep_num_redundant_experts == 0`）を Prefill に限って外す、`--ep-num-redundant-experts R` と 128+R 列の `physical_to_logical_map`、MegaMoE weight の local expert 数 33/34。local expert 数は `_C.so` の定数ではなく、weight の形から host が JIT template 引数 `kNumExperts` として渡すので host の再ビルドは要らない。ただし 32 の倍数でない local expert 数（scheduler の `kNumExpertsPerLane = 2`）が正しく動くかは未検証。SGLang の static dispatch は送信元 rank ごとに 1 つの複製を選ぶ固定割り当てで、§3.1 の静的分割の一種である。出力は複製の weight が同一で、行ごとの計算と combine の順序が計算場所に依存しないので bit 一致する。
- **実装しない。** 3-5 は計画から外すか、「Prefill 全体 −0.1〜0.3%、VRAM 1.8〜3.6 GB/rank」と書き換えるのが妥当。

## 4. 作業記録

- GPU job：なし。ブランチ `perf/firmus-max-geometry-20261018` には、この note と解析 script・結果だけを置いた。実験ブランチは作っていない。
- 再現：`uv run --no-project --python 3.12 --with numpy --with torch --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match python -I research/runtime-performance/2026-10-18-megamoe-geometry-and-redundancy.py /srv/qwen-artifacts/39051152456/1/results/results/expert-distribution/*/*.pt --report-out report.json`（dump は worker ごとのディレクトリ名で区別するので、元の配置のまま渡す）。
