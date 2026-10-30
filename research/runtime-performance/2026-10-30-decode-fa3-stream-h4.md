# Decode FA3 verify: 4 KV head/CTA の stream split（512 B 行読み、2026-10-30 round 6）

読み取りのみの解析と CPU 検証。GPU job・Actions・ssh は使っていない（round 6 は玄界も使わない）。基準は main cd262ed（9832.74、run 41005872470、image 40991744289）。Decode は TP4/DP4/EP4、EAGLE3 3/1/4 verify、cap 48/rank、FP8 KV（page 64、[token][4 head][128]）。FA3 verify の pin は `p128split2-s3-fp8`。

## 1. h2 行（256 B）が勝てなかった理由

40991744289 と 41005872470 の `results/results/decode-selection/fa3.json`（各 4 GPU × 2 block、1 block = 20 対の交互 replay、µs/層、combine 込み）。40964714831 の FA3 選択は `decode_kv_read_probe` の import 失敗で全 rank が timing 前に終わっており、h2 の値はない。

| batch | pin（同じ対の中） | split2-h2 | split3-h2 | split2-h2 tile | split3-h2 tile | pin tile |
|---|---|---|---|---|---|---|
| 44 | 72.7（split2 対）／75.1（split3 対） | 74.7（−2.0） | **68.1（+7.1、範囲 +5.9〜+8.5）** | 176 = 132 + 44 | **264 = 2 × 132** | 352 = 264 + 88 |
| 46 | 74.9／74.1 | 75.3（−0.4） | 83.6（−9.6） | 184 = 132 + 52 | 276 = 264 + 12 | 368 = 264 + 104 |
| 48 | 76.9／76.2 | 76.3（+0.6） | 85.1（−9.0） | 192 = 132 + 60 | 288 = 264 + 24 | 384 = 264 + 120 |

括弧内は pin − 候補（正が短縮）の 16 block の中央値。読み粒度 probe（同じ byte と program 数）は 128／256／512 B で 2.90／3.03／3.27 TB/s（B44）、2.97／3.10／3.31（B46）、3.03／3.17／3.36（B48）。

- **256 B 行は帯域には効いていた。** 132 SM がちょうど 2 round で埋まる B44 の split3-h2 は 68.1 µs。combine を pin と同じ約 5 µs とすると本体は約 63 µs で、183.7 MB を 2.92 TB/s で読んでいる。pin の本体（72.7 − 4.6 ≈ 68 µs）は 2.70 TB/s。256 B probe（3.03）の 96%、pin は 128 B probe（2.90）の 93%。
- **h2 は tile 数を半分にし、tile あたり byte を 2 倍にする。** そのため最後の round の tail が大きくなる。split3-h2 の tile 数は 6B で、132 の倍数は B44 だけ。B46/48 では 12／24 tile の 3 round 目ができ、byte の増加分（+4.6%／+9.2%）を除いても 83.6 − 68.1 × 192.1/183.7 = 12.4 µs、85.1 − 74.4 = 10.7 µs かかる（700 KB の tile 1 個を 1 SM 約 60 GB/s、約 31 B/clk で読む）。split2-h2 は 4B tile（1.05 MB）が 132 + 44〜60 の 2 round になり、2 round 目の tail が帯域の利得を打ち消して pin と同じになる。
- pin（128 B、8B tile）も 3 round 目は 88〜120 tile と部分的だが、tile が小さい（525 KB）ので tail が短い。
- **結論。** 粒度の利得を時間に変えるには、tile の数と大きさを batch に合わせて全 SM で同時に終わるように分ける必要がある。h2 の split 数を変えるだけでは、B46/48 で整数の分け方がない（4B・6B・8B のどれも 132 の倍数にならない）。

## 2. 512 B 行を読む設計の評価（CPU の数字）

| 方向 | 数字 | 判断 |
|---|---|---|
| 1 CTA に 4 KV head、consumer 4 WG（各 WG は pin と同じ 1 head・M64 の計算）、kBlockN 64 | 640 thread で launch 96 register。consumer 120／producer 32 で ptxas は 604 B spill（consumer の S/O accumulator が local へ出る） | 不可 |
| 同上、kBlockN 32（S が 32→16 register） | consumer 112／producer 64 で consumer の main loop に local memory なし（spill 64 B は segment ごとの前後処理だけ）、producer の load loop も spill なし。consumer 120／producer 32 では producer の load loop に毎 iteration 約 15 個の LDL/STL | **採用** |
| 2 WG × 2 head（WG が head を順に処理） | O が 2 head 分で 128 register、S 32、P 8 で約 200（224 以内）だが、mma loop を 2 head 用に書き直す必要がある | 今回は見送り（次の手段） |
| 1 head あたりの Q 行を減らす | GQA16 × verify 4 = 64 行が WGMMA の最小 M（64）そのもの | 不可 |
| split-K（WG ごとに全 head の KV 範囲） | WG が 4 head 分の O（256 register）を持つ | 不可 |
| TMA（`cp.async.bulk`）で 512 B 行を smem へ | smem は [head][stage][token] で、1 回の TMA box（[head][token]）と並びが違う。V は FP8 WGMMA のため転置が必要で producer の register は減らない。page 32 KB 単位の bulk copy は kBlockN 64 が前提で、その consumer は上の 604 B spill | 今回は見送り |
| 別 kernel で KV を gather して連続配置 | 読み 1 回 + 書き 1 回 + 読み 1 回で byte が 3 倍 | 不可 |

## 3. 実装：`p32stream-h4-fp8`（`kernels/fa3-verify/decode-stream-h4.patch`）

`decode-wide-h2.patch` の head 群の仕組みを 4 head に広げ、1 段の stream split と組み合わせた。

- **CTA の中。** 1 CTA が 1 request の KV 区間について 4 KV head 全部を扱う。producer WG は cp.async で token 行 512 B を 1 warp 命令で読む（`PagedKVManager` の `FullRow`：32 thread/行、`kBlockKGmem` = 512）。各 head の 128 B は h2 と同じ `heads_in_row` で head 別の swizzle 済み tile に置く。consumer は 4 WG で、WG h は pin と同じ 1 WG・M64 の計算（RS PV、IntraWGOverlap、FP8 転置 V）を head h に行う。kBlockN 32、3 stage。register は 128 × (64 + 4 × 112) = 65536。
- **stream split（`VerifyStreamScheduler`）。** 全 request の 32 token block を request 順に並べ、G 個（G ≤ 132）の等しい区間に切る。CTA c は block [cN/G, (c+1)N/G) を受け持つ。区間が request の境界をまたぐ所で segment に分かれ、各 segment は slot = c − (その request の先頭 CTA) の FP32 partial を書く。G は区間長が ⌈最大 block 数 / 5⌉ 以上になるように選ぶので、1 request の slot は 6 以下になる。各 warp が `seqused_k` から（lane あたり 2 request の scan で）分割を計算し直すので、main loop の間に生きている状態は現在の segment だけ。CTA 0 の最初の producer warp が request ごとの slot 数を dynamic-split metadata に書き、combine はそれを読む（`S == 6`）。segment の範囲は split 座標に slot | lo << 3 | hi << 17 として入れ、`BlockMN`（StreamSplit）と epilogue（slot = 下位 3 bit、常に partial）が取り出す。
- **companion。** `candidates.json` に 1 行（block_n 32、stages 3、splits 6、fp8、heads 4）。`build.py` は patch を最後（prefill-pipe の後）に当て、表に heads 列を足す。`wrapper.cpp` は heads 4 で prepare を省き（main kernel が slot 数を書く）、6 slot・fp8・merge なしを確認する。`combine.cu` は S = 6 の instantiation を足すだけで、既存 6 個の SASS は変わらない。SGLang 側の変更はない（partial buffer は splits = 6 から確保され、metadata の大きさは同じ）。
- **精度。** head ごと・block ごとの計算は pin と同じ（FP8 QK、FP32 online softmax、FP8 P × V、FP32 partial）。変わるのは KV の分け方（block 32、区間の境界）と partial の数で、FP 和の順序だけ。精度方針の範囲。split が pin と違うので byte 一致は求めず、既存の数値 gate を使う。

### 分割の CPU 検証（`stream_slot_counts` と同じ整数式の模型）

| 入力 | block 数 | CTA | segment（partial） | 最大 slot | CTA あたり block（最大／平均） | CTA あたり segment |
|---|---|---|---|---|---|---|
| 選択の長さ B44（3900〜4200） | 5583 | 132 | 173 | 5 | 43／42.3 | ≤ 2 |
| 同 B46 | 5838 | 132 | 175 | 4 | 45／44.2 | ≤ 2 |
| 同 B48 | 6097 | 132 | 178 | 4 | 47／46.2 | ≤ 2 |
| serving 相当 B44／46／48（4004〜4204） | 5669／5922／6178 | 132 | 170／175／177 | 5／4／4 | 43／45／47 | ≤ 2 |
| 境界長（4〜129）B32〜48、乱数 20,000 例（1〜64 request、1〜131072 token） | | | | ≤ 6 | | |

全例で、全 block がちょうど 1 回ずつ覆われ、各 request の slot が 0..count−1 を 1 回ずつ使う。CTA 間の差は 1 block（32 token）以内。

## 4. SASS（CUDA 13.4、sm_90a）

| | pin `verify_n128_s3_split1_merge0` | `verify_n32_s3_split1_merge0_h4` |
|---|---|---|
| launch register／spill | 255／0 | 96（setmaxnreg producer 64・consumer 112）／64 B（consumer の segment 前後と producer の初期化、両方の main loop は 0） |
| shared memory（`SharedStorageSize`、host で計算） | 158,720 B | 198,656 B（227 KiB 以内、build が assert） |
| consumer main loop（mask なし） | 435 命令／128 token・1 head（QGMMA 8、MUFU 66、FMUL 66） | 222 命令／32 token・1 head（QGMMA 5、MUFU 18、FMUL 66） |
| producer load loop | 260 命令／32 KB（LDGSTS 16、LDSM 8、STSM 8） | 256 命令／32 KB（LDGSTS 16、LDSM 8、STSM 8） |

- **bytes/cycle。** 32 KB（32 token × 4 head × K+V）あたり、SM の発行は consumer 16 warp × 222 + producer 4 warp × 256 = 4576 warp 命令、4 scheduler で 1144 cycle。512 B probe の帯域（B44〜48 で 3.27〜3.36 TB/s、1.9 GHz で 13.0〜13.4 B/clk/SM）なら 32 KB に 2450〜2520 cycle かかるので、発行は 45%。tensor は 4.2 MFLOP / 8192 = 512 cycle、MUFU は 16 × 18 × 32 / 16 = 576 cycle。kBlockN 32 にしたことで O の rescale（FMUL 66）が 4 倍の頻度になり、consumer の命令は byte あたり pin の 2.0 倍になるが、SM の上限は約 28.6 B/clk で、必要な 13 B/clk の 2.2 倍ある。memory 律速のまま。
- pin は 1 WG の遅延律速（単独 tile で約 14.5 B/clk/SM）で、全 SM 稼働時の 2.7 TB/s（10.8 B/clk/SM）は 128 B 読みの上限。
- **触らない kernel。** patch 前後の全 27 TU（Decode verify 10、Prefill 15、combine、decode_prepare、upstream の prepare）を同じ flag で compile し、mangled name ごとに SASS（address・命令・encoding）を比べた。共通の 32 関数（合計約 13 万命令）はすべて一致し、追加は h4 の main kernel（新 TU）と `combine_fixed<6, true>` だけ。kernel の mangled name も変わらない（新しい template 引数は内部の `PagedKVManager` と `BlockMN` にだけ足した）。pin（`verify_n128_s3_split1_merge0`、4768 命令）は byte 一致。

## 5. 見込み

- **時間の式。** 本体 = KV byte ÷ 実効帯域 + partial の追加書き込み、これに combine と前後処理を足す。実効帯域は probe に対する比で置く。pin は 128 B probe の 93%、split3-h2 は 256 B probe の 96% だった。h4 は 512 B probe の 93〜97% とする（B44 3.04〜3.17、B46 3.08〜3.21、B48 3.12〜3.26 TB/s）。partial は約 175 segment × 128 KB = 22.4 MB（split3-h2 の B44 は 16.9 MB、pin は 11.3〜12.6 MB）。split3-h2 との差 5.5 MB を DRAM 書き込みとして 0〜+1.8 µs、combine と起動間隙は 5〜6 µs、前後処理は split3-h2 と同じ（CTA あたり 2 segment）。
- **B44。** 183.7 MB → 本体 57.9〜60.4 µs、合計 62.9〜68.2 µs。pin（72.7〜75.1）に対して −4.5〜−12 µs。
- **B46。** 192.1 MB → 本体 59.8〜62.4 µs、合計 64.8〜70.2 µs。pin（74.1〜74.9）に対して −4〜−10 µs。
- **B48。** 200.6 MB → 本体 61.5〜64.3 µs、合計 66.5〜72.1 µs。pin（76.2〜76.9）に対して −4〜−10 µs。
- **悲観側。** 512 B 行が probe の利得を出さず、split3-h2 の 2.92 TB/s に留まる場合は B44 −2〜−6、B46 −0.5〜−3、B48 0〜−2.3 µs。B46/48 で 1 µs の条件を割る GPU があれば採用されない。
- **Decode step あたり。** 中央は −6 µs/層（上の範囲の下側寄り。kBlockN 32 の 4 WG と 2 倍の partial が想定どおりに隠れない分を見込む）× 94 層 = **−0.56 ms/step**（34 ms の 1.6%）。範囲は −0.1〜−1.0 ms。ITL 平均 15.40 ms は約 15.15 ms になる。係数は 0.8〜1.0。同じ byte を速く読む memory 律速 kernel の改善で、Decode GPU は上限近くの clock で動いている。選択は Decode GPU 上の実 clock で対にして測る。

## 6. 検証と gate

- **CPU。** `qwen-experiment-metrics/tests/test_fa3_verify_stream.py` が行・build 順・patch の符号化と復号・wrapper と combine の 6 slot・分割の被覆と slot 上限（選択長、境界長、乱数）・endurance の規則を確かめる。companion の patch chain は pristine の sgl-attn に `--fuzz=0` で全部当たり、結果は作業 tree と一致する。`wrapper.cpp` は torch header で g++ の構文検査を通る（前後とも）。
- **選択（`select-firmus-decode-fa3.py --pinned-env`、PP4 の Decode GPU 連鎖の先頭）。** 対象は h4 の 1 行だけ（h2 行は表に残すが screen しない。`decode_kv_read_probe.py` は役目を終えたので削除）。採用条件は 4 つ。(1) 既存の数値 gate（FP32 参照、stock 比較、境界長、NaN を入れた partial、入力を変えた graph replay）を通る。(2) main kernel が書いた slot 数が CPU の分割（`stream_slot_counts`）と一致する。(3) 44/46/48 の全 GPU・両 block で 1 µs/層以上速い。(4) **endurance gate**：各 GPU・各 batch で、timing の後に 94 層 graph を 300 回続けて replay する（KV 長を 0〜63 token 縮めた 8 通りで回し、block 境界と区間を毎回変える）。最後に元の長さでもう 1 回 replay し、最初の出力と byte 一致すること。合計 GPU あたり 903 replay、h4 kernel の起動は 84,882 回。どれかが欠けたら pin のまま。hang は既存の `timeout --kill-after=5 175` で終わり、env は書かれないので pin が使われる。
- **GPU でしか決まらないこと。** 512 B 行の実効帯域、kBlockN 32 の 4 WG が memory 律速に届くか、数値（FP 和の順序の範囲）、slot 数、hang しないこと。すべて上の選択と gate で確かめ、落ちれば pin。

## 7. job 内時間

FA3 選択は 2 行 × 3 batch + probe（40991744289 で 23.2 s）から、1 行 × 3 batch + endurance になる。timing は 1 行分（約 −0.6 s × 3）、probe は削除（Triton の compile と 3 粒度 × 40 起動、約 −3〜−5 s）、endurance は batch あたり 301 replay × 約 7 ms = 2.1 s、3 batch で約 +6.5 s。合計は約 22〜27 s で、Decode GPU の連鎖はほぼ変わらない（増えても約 +4 s）。image の再 build が必要（companion の AOT build）。

## 8. 残りの手段（大きい順）

| 手段 | 見込み | 前提 |
|---|---|---|
| 2 WG × 2 head（WG が O を 2 head 分持つ、kBlockN 64） | consumer の命令を byte あたり pin 並みに戻し（rescale が 1/2）、本体の発行余裕を 2.2 倍から約 3 倍にする。帯域律速が続くなら時間は同じ | mma loop の 2 head 版。h4 が帯域に届かなかった場合の次の手 |
| page 単位（32 KB 連続）の TMA bulk 読み | 128 MiB 読み reference の 3.7 TB/s に近づけば本体 −6〜−8 µs/層 | smem を [stage][head] に並べ替え、kBlockN 64 の consumer の register（2 WG × 2 head と組み合わせ） |
| combine を次の norm/quant と融合 | 約 5 µs/層の一部 | partial 22 MB の読み直しと起動間隙 |
| h2 行と c2 行を表から外す | build の TU が 2 個減る | 今回は既存 test の範囲を変えないため残した |
