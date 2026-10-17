# 2026-10-18 SM90 MegaMoE Decode の FC1 epilogue 量子化と dispatch pull の改善案

[N_SWAP hoist](2026-10-18-megamoe-decode-mainloop-pipeline.md)（`2d91b50`）を入れた後の Decode-128 には、FC1 epilogue（CTA あたり約 21 µs）と、起動から最初の FC1 stage まで約 20 µs かかる dispatch が残っていた。本稿では、hoist の上に重ねる bench overlay として次の候補を作り、[kernel-bench] 経路で 1 つずつ測った。どれも出力を bit 一致に保つ。

1. FC1（SwapAB L1）epilogue の量子化器：shared memory の bank conflict をなくし、256 thread 全部で per-token amax と FP8 量子化を行う。
2. dispatch pull：remote の SF と top-k 重みを続けて読む（2a）。さらに TMA store の完了待ちを次の token の処理と重ねる（2b）。
3. 対照：hoist の weight scale 先読みを外した版。

結論：**採用は 1 と 2a の組み合わせ。** 組み合わせ版は 2 run（g4、g1）とも全 12 case の出力 hash が image と一致した。hoist だけの版に対して Decode を 0.5〜2.4% 短縮し、image に対しては 3.6〜12.5% 短縮した（Decode-128 は 0.2018 / 0.1998 ms。hoist は 0.2048 / 0.2065、image 3 run 平均は 0.2295）。効果の大半は 1 によるもので、FC1 epilogue は CTA あたり 21 → 12.5 µs になった。2a は単独では揺らぎの範囲の改善しかない。2b は最初の FC1 を遅らせたので不採用。3 は hoist と差がなく、hoist の効果は N_SWAP の選択を block ごとに 1 回にしたこと自体による。採用した patch 一式はブランチ `perf/firmus-decode-kernel-20261018`（`901288a`）に置き、main には入れていない。image build と serving run はしていない。

## 1. 実装

基準は image `e29dfde…` の header（`patches/deep-gemm/*.patch` 適用済み）に hoist patch を当てたもの。各候補は `patches/deep-gemm-bench/` の別ファイルにしてあり、名前順に重ねて当てられる（hoist → l1-quant → dispatch-pull-overlap → dispatch-pull-pipeline）。2b だけは 2a の上に作った。dispatch と量子化器は別の場所なので、1 と 2a は単独でも組み合わせても当たる。

### 1.1 FC1 量子化器（`sm90-decode-swap-ab-l1-quant.patch`、採用）

従来の SwapAB L1 epilogue は、SwiGLU 後の FP32 tile を smem に [token][64] で置き（64 列は 1 つの per-64 SF group で、CTA の L1 出力 tile 全体）、barrier の後に token ごとに 1 thread が 64 値を読んでいた（コンパイル後は LDS.128 が 16 本）。amax を取った後にもう一度 64 値を読み、FP8 の 2 byte 対を 32 回書く。行の stride が 256 byte なので、LDS.128 の各 phase（8 thread）が同じ 4 bank に集中する。動く thread も最大で 64/256 しかない。staging の書き込みも、1 命令内の 4 つの token 対が同じ 8 bank に載る 4-way conflict だった。

変更点：

- staging の列を `8 * ((token / 2) % 4)` で XOR swizzle する。warp の staging store が 32 bank に散り、4 列単位の連続性と 16 byte 境界は保たれる。
- 1 token を 4 thread で受け持つ（256 thread = 64 token = BLOCK_M で 1 回）。各 thread は token 行の float4 を 4 本読む。slot の順番を `(token + 1) / 2` だけ回すので、読み（1 phase に 2 token）も FP8 の 4 byte store（1 warp に 8 token）も conflict しない。値は register に置いたまま amax と量子化の両方に使う。
- amax は、thread ごとに 16 値の逐次 max を取った後、quad 内で XOR shuffle を 2 回する。`cute::max(a, b) = a < b ? b : a` は NaN を取り込まず、入力はすべて非負なので、64 値の逐次 max と同じ float になる。
- scale・clamp・逆数は従来どおり `sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv` で求め、同じ `sf_inv` を掛けて同じ `__nv_fp8x2_e4m3` で変換する。2 対を同じ byte 順で 32 bit にまとめて書く。SF 行（SF_BLOCK_M stride）と FP8 byte（valid_m 未満の token だけ、TMA store 用の非 swizzle [token][64]）は従来と同じものを書く。
- barrier の数、TMA store、arrival は変えていない。CTA barrier は増やさず、register 予算も変えていない。komoriyuta の warp-redux-r176（1 warp で 1 token、5 段の shuffle、8 token を逐次処理、176 register）は serving で遅かった（[計画ノート](2026-10-17-firmus-acceleration-plan.md) の run B）。本案は 168 register のまま、64 token を 1 回の処理で終える。

### 1.2 dispatch の remote load 重ね（`sm90-dispatch-pull-overlap.patch`、採用）

従来の 1 token の pull は、remote token の TMA load を発行し、remote SF 32 個（1 lane 1 個）を読んで書き、`__syncwarp` を挟んでから remote top-k 重みを読む（2 回目の NVLink 往復）。その後 TMA load を待ち、行を local に TMA store し、完了を待って arrival `red_add` を出す。本案では SF と重みを続けて register に読み、その後で従来と同じ lane が同じ local 番地に書く（重みは lane 0。従来の elected lane と同じ）。token の配置、SF layout、metadata、TMA store、完了待ち、arrival は変えていない。

dispatch のコードは Prefill と共有している。何も条件を付けずに変えると Prefill の SASS と spill（40 → 44 byte）が変わったので、SwapAB 形状の `if constexpr` で Decode だけに限った。

1 warp で 2 token を同時に扱う案は作っていない。dispatch の smem send buffer は host が dispatch warp あたりちょうど 1 行（4 KiB × 2 warp = 8192 byte、1024 境界で余りなし）として確保している。ほかに大きな smem 領域は GEMM stage と CD staging しかなく、どちらも dispatch がまだ pull している間に loader と math warp が使いうるので、空いていると保証できる領域がない。

### 1.3 arrival の後回し（`sm90-dispatch-pull-pipeline.patch`、不採用、`fab9771`）

1.2 の上に作った。TMA store を発行したら完了を待たずに次の token に進む。次の token の expert/rank 選択と source-index load の後に、まず pull buffer の読み出し完了を待ち（`wait_group.read`）、次の token の TMA load を発行する。続けて前の token の store の完了を待ち（`wait_group 0`）、arrival を出す。release が次の token の remote load を待たないように、remote load より前に置いた。loop の後に残り 1 個の arrival を出す。各 token の arrival は store 完了後にちょうど 1 回出るので、数と配置は変わらない。Decode に限定し、Prefill の SASS は変わらない。

### 1.4 対照：weight scale を full barrier の後で読む（`sm90-decode-swap-ab-sf-after-wait.patch`、`64da796`）

hoist patch は、(a) N_SWAP の選択を block ごとに 1 回にすること、(b) k-block ごとの weight scale の `__ldg` を full barrier の wait より前に移すこと、の 2 つを同時に入れていた。(b) の分を測るため、hoist の loop の中で 3 つの `__ldg`（L1 gate/up、L2）だけを wait の後に戻した。

### 1.5 register / spill / コード量

Firmus login で image の CUDA 13.0 nvcc（`usr/local/cuda-13.0`）を `unsquashfs` で取り出した。`containers/build-sm90-megamoe.py` と同じ Prefill/Decode の instantiation に、wave あたり 16 expert の Decode（4/16 token で使われる）を加え、`--ptxas-options=-v,-warn-spills` で確認した。

| 版 | Decode / Decode-16 register | spill (st/ld byte) | Decode cubin | Prefill |
|---|---:|---:|---:|---|
| image | 168 / 168 | 0 / 0 | 119640 | 96 reg、spill 40/40、SASS sha256 `c1f5f67f…` |
| hoist | 168 / 168 | 0 / 0 | 136024 | SASS 同一 |
| hoist + l1-quant | 168 / 168 | 0 / 0 | 131928 | SASS 同一 |
| hoist + dispatch overlap | 168 / 168 | 0 / 0 | 136024 | SASS 同一（条件なし版は変化） |
| hoist + sf-after-wait | 168 / 168 | 0 / 0 | 133976 | SASS 同一 |
| hoist + overlap + pipeline | 168 / 168 | 0 / 0 | 136024 | SASS 同一 |
| **採用：hoist + l1-quant + overlap** | 168 / 168 | 0 / 0 | 131928 | SASS 同一 |

量子化器は cubin を 4 KB 小さくした。従来の 64 回 unroll の 2 loop が、thread あたり LDS.128 4 本、F2FP 8 本、SHFL 2 本になったためである。timeline 計装版は Decode で spill 8/112 byte（dispatch counter 追加後は 28/136 byte）が出るが、計時する候補版には入らない。

## 2. 実行記録

すべて `[cluster:firmus] [nodes:1] [kernel-bench]`、image `e29dfde…`、Firmus hpcai 1 ノード 4×H200。

| run | commit | ブランチ | 内容 | node |
|---|---|---|---|---|
| 39053820377 | b90cc52 | main | image（基準、hash の正本） | g1 |
| 39062255256 attempt 1 / 2 | 2d91b50 | perf/firmus-decode-mainloop-pipeline-20261018 | hoist（前回） | g4 / g1 |
| 39064611698 | 0768be6 | experiment/firmus-decode-l1-quant-20261018（削除） | hoist + l1-quant | g4 |
| 39064761765 | db2c8ad | experiment/firmus-decode-dispatch-overlap-20261018（削除） | hoist + dispatch overlap | g1 |
| 39064910947 | 64da796 | experiment/firmus-decode-sf-after-wait-20261018（削除） | hoist + sf-after-wait（対照） | g4 |
| 39065587762 attempt 1 / 2 | cff95f1 | experiment/firmus-decode-quant-dispatch-20261018（削除、内容は perf ブランチに残る） | hoist + l1-quant + overlap、timeline に dispatch counter 3 個を追加 | g4 / g1 |
| 39066974393 | fab9771 | experiment/firmus-decode-dispatch-pipeline-20261018（削除） | hoist + overlap + pipeline | g1 |

ブランチ `perf/firmus-decode-kernel-20261018` は `cff95f1` に、結果を記録しただけの空の `[skip ci]` commit（`901288a`）を足したもの。push で bench は起動していない。dispatch の timeline patch は候補ごとに context を合わせて作り直した。

## 3. 結果

### 3.1 出力の一致

全 run の全 12 case（Decode 7、Prefill 5）で `output_sha256` が 39053820377 と一致した。数値 gate も全 run で PASS した。計装版の出力も、各 run の計装なし版と bit 単位で一致した。

### 3.2 計時（median ms、iteration ごとの rank 間最大値の 30 回 median）

| case | image（3 run 平均） | hoist（2 run） | l1-quant | overlap | sf-after-wait | 採用の組み合わせ（2 run） | 組み合わせ vs hoist | 組み合わせ vs image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode-balanced-4 | 0.1386 | 0.1314 / 0.1314 | 0.1331 | 0.1318 | 0.1334 | 0.1306 / 0.1295 | −1.0% | −6.1% |
| decode-balanced-16 | 0.1880 | 0.1825 / 0.1818 | 0.1828 | 0.1805 | 0.1828 | 0.1823 / 0.1801 | −0.5% | −3.6% |
| decode-balanced-32 | 0.1948 | 0.1900 / 0.1889 | 0.1896 | 0.1880 | 0.1892 | 0.1876 / 0.1875 | −1.0% | −3.7% |
| decode-balanced-64 | 0.2069 | 0.1974 / 0.1954 | 0.1971 | 0.1947 | 0.1974 | 0.1939 / 0.1946 | −1.1% | −6.1% |
| decode-balanced-96 | 0.2174 | 0.2006 / 0.2007 | 0.1989 | 0.1995 | 0.2007 | 0.1969 / 0.1963 | −2.0% | −9.6% |
| decode-balanced-128 | 0.2295 | 0.2048 / 0.2065 | 0.2017 | 0.2050 | 0.2052 | 0.2018 / 0.1998 | −2.4% | −12.5% |
| decode-local-128 | 0.2266 | 0.2027 / 0.2013 | 0.1990 | 0.2035 | 0.2022 | 0.2006 / 0.1974 | −1.5% | −12.2% |

- 単独で hoist の 2 run 平均と比べると、l1-quant は 96/128/local-128 で −0.9 / −1.9 / −1.5%、4〜64 token では +1.3〜0.0%（Decode-4 の +1.3% は sf-after-wait 対照の +1.5% と同程度で、run 間の揺らぎ。組み合わせ版の 2 run では −1.0%）。overlap は 16/32/64/96 で −0.9 / −0.8 / −0.9 / −0.6%、128 と local-128 では −0.3 / +0.7% で、どれも揺らぎ（同じ kernel の run 間で ±1%）と同程度。
- 組み合わせ版は 2 run とも全 Decode size で hoist より速く、別の node（g4、g1）で再現した。足し合わせると、96 token 以上は l1-quant、64 token 以下は overlap が主に効いている。
- 4/16 token の 16 expert wave instantiation は、hoist（−5.2 / −3.1%）に組み合わせ版で −1.0 / −0.5% が加わる程度。この区間はどの候補も揺らぎの範囲で、改善も悪化も判定できない。
- sf-after-wait は全 Decode size で hoist との差が −0.1〜+1.5% に収まり、揺らぎと区別できない。hoist の効果（Decode-128 で −10%）は weight scale の先読みではなく、N_SWAP の選択を外に出したことによる。
- pipeline（1.3）は overlap 版（39064761765）と比べて 64/96/128 で +1.1 / +1.0 / +0.6% 遅い（0.1968 / 0.2015 / 0.2063）。
- Prefill は全版で image の run 間の揺らぎ（±4.5%）の範囲。SASS は全版で image と同一。

### 3.3 timeline（Decode-128、計装版、代表 call、4 rank の中央値。µs）

| 版（run） | kernel | 最初の FC1 stage | 最初の token pull 完了 | FC1 span | FC2 span | L1 K loop | L1 epilogue | L2 K loop | L2 epilogue | rank 内の遅い CTA 待ち | 最後の FC2 → 終了 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hoist（39062255256-1） | 215.8 | 20.2 | 18.2 | 112.7 | 60.3 | 97.1 | 21.1 | 47.5 | 10.1 | 9.3 | 13.5 |
| hoist（39062255256-2） | 216.1 | 19.8 | 17.8 | 112.7 | 60.5 | 97.1 | 21.0 | 47.8 | 9.9 | 9.5 | 16.3 |
| l1-quant（39064611698） | 208.6 | 20.8 | 18.4 | 105.0 | 60.7 | 98.5 | **12.5** | 47.6 | 10.4 | 8.6 | 15.0 |
| overlap（39064761765） | 215.8 | 18.9 | 17.5 | 112.7 | 60.0 | 96.2 | 20.9 | 47.4 | 9.9 | 9.7 | 16.1 |
| sf-after-wait（39064910947） | 214.5 | 19.6 | 17.6 | 112.0 | 60.5 | 95.5 | 21.3 | 45.2 | 12.5 | 8.0 | 14.2 |
| pipeline（39066974393） | 219.9 | 22.9 | 20.6 | 113.0 | 60.5 | 100.5 | 21.0 | 48.0 | 9.8 | 8.1 | 16.6 |
| 組み合わせ（39065587762-1） | 209.2 | 19.8 | 18.2 | 105.8 | 61.0 | 98.2 | 12.5 | 47.7 | 10.5 | 7.8 | 14.6 |
| 組み合わせ（39065587762-2） | 208.1 | 18.7 | 17.4 | 106.0 | 61.3 | 98.2 | 12.8 | 47.9 | 10.6 | 6.5 | 15.5 |

K loop と epilogue は per-CTA counter の CTA median（epilogue = block 時間の合計 − K loop の合計、L1 は 6 block、L2 は 8 block）。組み合わせ版と pipeline 版の timeline には dispatch counter の計装が増えている（spill 28/136 byte）ので、dispatch の列は他の版より少し長めに出る。

- **FC1 epilogue は CTA あたり 21 → 12.5 µs（block あたり 3.5 → 2.1 µs）。** FC1 span は 112.7 → 105 µs、計装版の kernel は 216 → 208〜209 µs。K loop と FC2 は変わらない。
- overlap は、最初の token の pull 完了と最初の FC1 stage をおよそ 0.5〜1 µs 早めた（19〜20 µs → 18.9 µs）。kernel 全体の差は揺らぎに埋もれる。
- pipeline は最初の token の pull 完了を 2.5〜3 µs 遅らせた（17.5 → 20.6 µs）。最初の FC1 も 22.9 µs に遅れた。

### 3.4 dispatch の token あたり内訳（組み合わせ版、dispatch warp 0、約 4 token/warp、per-CTA の合計 µs、4 rank の中央値の範囲）

| 区間 | attempt 1 | attempt 2 | pipeline 版 |
|---|---:|---:|---:|
| loop 先頭 → TMA 発行（expert/rank 選択、source-index load） | 6.3〜7.0 | 6.4〜6.9 | 5.5〜5.9 |
| TMA 発行 → remote SF・重みのコピー完了 | 5.5〜6.8 | 5.5〜7.0 | 6.6〜7.8 |
| TMA load の待ち | 1.0〜1.1 | 0.8〜1.4 | 0.2 |
| TMA load 完了 → local TMA store 完了 + arrival | 9.9〜11.0 | 9.7〜10.7 | 10.5〜11.9（後回しにした待ちを含む） |

1 token は約 6 µs で、最も大きいのは local への 4 KiB TMA store の完了待ち（約 2.6 µs/token）。remote の 2 つの load をまとめた後の remote 区間は約 1.6 µs/token で、NVLink の往復 1 回分程度。pipeline 版は store の完了待ちを次の token の選択処理（約 1.4 µs）と重ねるつもりだった。しかし完了待ちの合計はほとんど減らず（store の完了は発行から 2.6 µs 以上かかる）、最初の token の arrival が 2 番目の token の選択処理と `wait_group.read` の後まで遅れた。最初の FC1 block は各 warp の最初の token で決まる（各 CTA の最初の block は先頭の expert のもので、その token は全 warp の 1 番目の pull に当たる）ので、この遅れはそのまま kernel 時間に出た。2 番目以降の token は律速していない（TMA A warp の L1 到着待ちは最初の block の約 18 µs がほぼ全部）。

## 4. 残っている時間（採用版、Decode-128）

- 計装なしの kernel は約 199〜200 µs（bench 0.2008 ms は pre-dispatch を含む）。
- GEMM 区間（最初の FC1 stage → 最後の FC2 終了）は計装版で 170〜177 µs。計装の割り増し（約 9%）を除くと約 160 µs で、重み 604 MB を読む HBM 下限（126 µs）との差は約 35 µs。このうち K loop は L1 約 98 µs、L2 約 48 µs/CTA（計装込み）で、割り増しを除くとそれだけで下限の約 135/126 になる。残りは CTA あたり約 23 µs の epilogue（FC1 12.5、FC2 10.5）と CTA 間のばらつき。loader は今も空き stage を待っている（A 91〜95 µs、B 141〜145 µs）ので、律速は引き続き math 側。k-block 処理の pipeline 化は前回効かなかった。
- epilogue：FC1 12.5 µs/CTA（2.1 µs/block）、FC2 10.5 µs/CTA。FC1 の残りは、SwiGLU と staging、3 回の CTA barrier、4 KiB の TMA store の完了待ちと arrival。
- dispatch：起動から最初の FC1 stage まで約 19 µs。内訳は計数・atomic・index 書き込み 3 µs、pre-pull NVLink barrier 7〜9 µs（rank 間の起動ずれを含む）、最初の token 約 6 µs（選択処理 1.6、remote 1.6、store 完了 2.6）。store の完了待ちは release の前提なので、最初の token では消せない。
- tail：最後の FC2 から終了まで 14〜16 µs（rank 内の遅い CTA 待ちが median 7〜9、max 20〜26 µs、combine、最終 barrier）。

## 5. 判断と次の手順

- 採用候補は `perf/firmus-decode-kernel-20261018`（`901288a`）の 3 patch（hoist、l1-quant、dispatch-pull-overlap）。kernel 単体で image に対して Decode を 3.6〜12.5% 短縮し（2 run）、出力は bit 一致、Prefill の SASS は不変。overlap の寄与は 1% 前後で揺らぎと区別しにくいが、悪化した size はなく、Prefill に影響しないので残した。
- serving での効果は、通常の `[require-build]` の image build と serving run で確かめる必要がある（未実施）。そのときは `patches/deep-gemm-bench/` の 3 ファイルを `patches/deep-gemm/` の chain に移し、`build-sm90-megamoe.py` と `check-firmus-image.sh` の header pin を更新する。
- pipeline（`fab9771`）と sf-after-wait 対照（`64da796`）は不採用・記録のみ。
- 次の候補：(1) 最初の token の選択処理を短くする（recv count と rank 別 count を 1 回の load 波で読み、依存する load を 3 段から 2 段にする）、(2) tail の rank 内 CTA ばらつき、(3) FC1 epilogue の残りの barrier 構成。どれも見込める改善は 1% 前後。
