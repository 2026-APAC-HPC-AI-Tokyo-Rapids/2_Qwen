# Decode step round 3: MegaMoE の経路確認、床との差、MoE 入力の融合（2026-10-30）

読み取りのみの解析と CPU 検証。GPU job・Actions・ssh は使っていない。前提は [2026-10-29-decode-step-breakdown.md](2026-10-29-decode-step-breakdown.md) と [2026-10-30-decode-step-two-cta-fa3.md](2026-10-30-decode-step-two-cta-fa3.md)。基準は main 9f4f891（9590.42、Decode cap 48/rank、graph bs ≤ 48）。

## 1. cap 48 で Decode MegaMoE は速い経路に残る

- host（image 40911121619 の `patches/deep-gemm-host/series` 先頭の `sm90-swap-ab-192.patch`）の `should_use_swap_ab_for_mega_moe_sm90` は `num_tokens <= 192` で SwapAB を選ぶ。10-18 の cap 136 の失敗（`<= 128` の境界で 90% の call が非 SwapAB 229 µs）を受けて 144、さらに 192 に広げたもの。
- `get_block_config_for_mega_moe_sm90` は 1 expert あたり期待 token 数 `num_tokens × 4 × 8 / 128` が 64 以上で Prefill 形状（BLOCK_M 128/N 256）へ移る。192 行では 48 なので BLOCK_M 64・split-N 2 WG のまま。1 wave の expert 数は 32（期待 48 > 4）。
- serving は `DG_SM90_DECODE_INDEPENDENT_WG=1`（decode-0.log の `independent_wg=1`）なので、host は 6 stage を渡し、header は `kIndependentWG`（WG ごとに独立した 3 stage の task queue）になる。
- `run_sm90_mega_routed` は y の行数をそのまま `num_tokens` に渡し、DP attention の verify graph は rank ごとに bs × 4 ≤ 192 行。**cap 48 は SwapAB + IWG のまま**。49 要求/rank（196 行）以上では非 SwapAB に落ちるので、cap を上げるなら host の 192 も同時に上げる必要がある。

## 2. 主要項目と床の差（なぜ残りが小さいか）

| 項目 | n≈42 の時間 | 実効帯域・状態 | 判断 |
|---|---|---|---|
| MegaMoE | profile 223.5 µs/層（serving 換算 ≈204） | kernel bench（39945097647）balanced-16 で 593 MB を約 157 µs、balanced-128 で 604 MB を約 160 µs（起動・後段の約 30 µs を除く）。どちらも 3.77 TB/s で、128 MiB 読出し reference の 3.70〜3.73 TB/s と同じ | GEMM 区間は実用 HBM 帯域に達している。残りは dispatch（約 19 µs）と後段（約 12〜17 µs）の待ち |
| FA3 verify | 71.6 µs/層（n=44） | split2 の 352 tile × 525 KB = 185 MB。1 CTA/SM でも 2 CTA/SM（2026-10-30 の候補）でも約 70〜75 µs、すなわち全体で約 2.6〜2.8 TB/s が上限 | 2 CTA が効かなかった理由はこの全体上限で説明できる（並列度を倍にしても 185 MB / 2.7 TB/s ≈ 68 µs）。KV が [token][4 head][128] の FP8 なので 1 tile は 512 B ごとに 128 B だけ読む。上限の原因候補はこの粒度 |
| dense QKV/O | 23.3 + 17.4 µs（M=176） | QKV は M=128 で 15.2 µs、M=160〜176 で 23.3〜23.7 µs。BM128 の 2 本目の M tile（48 行）が 1 本分の時間を使う | B の multicast（cluster 2）は M=128（BM64 で M tile 2 本）で 1.0〜1.3 µs 遅かった（39901974117、39950338791）ので 176〜192 行でも候補にしない |

MegaMoE の serving と bench の差は約 4%（224/1.095 ≈ 204 対 balanced 約 196 µs）。routing の偏りによる M tile の増加は上限でもこの程度。

## 3. 実装：Decode の MoE 入力融合（`decode-moe-input.patch`、job 内選択）

TARGET_VERIFY の層では現在、post-attention の flashinfer `fused_add_rmsnorm`（3.5 µs）→ gate GEMM（6.9 µs）→ Triton router（3.5 µs）→ `mega_moe_pre_dispatch_sm90`（2.2 µs、y の BF16 を読み直して FP8 化し top-k を buffer へ写す）→ MegaMoE と流れる。Prefill では同じ並びを `prefill-moe-input.patch` の "fused" が置き換え、40849459077・40883389891 の両 run で 4 GPU とも採用され、PP4 では固定されている（7988 行で 123.7 → 104.1 µs/層）。

- **変更。** `SGLANG_DECODE_MOE_INPUT=fused`（既定 control）で、TARGET_VERIFY の 1〜192 行に Prefill の "fused" を使う。(1) `post_norm_moe_input` が residual・y（BF16、in place）・FP8 x・FP32 scale を書く。Decode では kernel は自分の行しか読まないので、pad slot（4096 行の buffer の残り）は書かず grid を行数にする（`pad_slots=False`）。(2) gate GEMM は同じ。(3) `route_into_buffer` は `select_experts` と同じ router launch（bias なし、`FUSE_DECODE_PADDING`、`num_token_non_padded`）で int64 id と重みを buffer に直接書く。pre-dispatch kernel はなくなる。Prefill の呼び出しは既定引数のままで、EXTEND 側の条件も変わらない。
- **精度。** 精度方針の範囲（FP32 和の順序）。FP8 x・scale・top-k は、同じ y から pre-dispatch と router が書く値と bit 一致が条件。y は flashinfer と BF16 1 ulp 以内（Prefill と同じ gate）。
- **CPU 検証。** Triton interpreter（torch 2.9 CPU、triton 3.5）で 176/172、184/184、192/188、4/4 行を確認した。router の id・重みは `moe_fused_gate(..., num_token_non_padded=)` と bit 一致、padding 行の id は −1、buffer の行数より後ろは書かれない、scale は bit 一致。residual と FP8 値の比較は interpreter が BF16/FP8 変換を切り捨てで行う（GPU は最近接偶数）ため一致しない。kernel 本体は Prefill で GPU の bit 一致 gate を通ったものから変えていない。
- **選択**（`select-firmus-decode-moe-input.py`、`select-firmus-decode-kernels.sh moe-input`）。Decode GPU 4 枚で、batch 44/46/48（176/184/192 行、最後の要求を padding）、192 行 padding なし、1 要求の 5 ケースに上記の数値 gate を掛ける。時間は 94 層の CUDA graph を経路ごと・batch ごとに作り、2 block × 10 対を交互に replay する。全 GPU・全 batch・両 block で max(1 µs, 2%)/層 以上速い場合だけ `SGLANG_DECODE_MOE_INPUT=fused` を decode-kernels.env に追記する。失敗・欠落・遅延では何も書かず、pre-dispatch のまま。
- **見込み**〔条件〕。pre-dispatch 2.2 µs と起動間隙 0.5〜1 µs が消え、norm は 3.5 µs から FP8 書き込みを含めて 3〜3.5 µs になる。**−2〜−3.2 µs/層 = −0.19〜−0.30 ms/step（serving の 34.4 ms に対して −0.55〜−0.9%）**。kernel 1 個と y の 1.4 MB の読み直しを消す仕事の削減なので、power-cap 係数は約 1.0。
- **job 内の追加時間。** Triton 2 kernel と pre-dispatch の JIT を含め 30〜45 s（timeout 150 s）。Decode GPU の連鎖（Prefill attention 選択 → FA3 → copy engine → これ）の最後に置く。連鎖は Prefill GPU の MegaMoE 選択（1.5〜2 分）と並行なので、job の延びは 0〜45 s。
- **image の再 build は不要**（SGLang overlay と script だけ）。

## 4. 5% に届かない理由と次の手段

この round で CPU の証拠と job 内 gate で出せる短縮は −0.2〜−0.3 ms（約 0.6〜0.9%）にとどまる。2 節のとおり、大きい 3 項目は GEMM が帯域、FA3 が全体の読出し上限、dense が tile 量子化で止まっており、5% には設計の変更が要る。大きい順：

| 手段 | 見込み〔係数〕 | 前提・作業 |
|---|---:|---|
| FA3 verify を 1 CTA で 4 KV head（512 B の token 行を連続で読む） | 上限が読み粒度によるなら 185 MB を 3.5 TB/s で 53 µs、−18 µs/層 = −1.7 ms〔約 1.0、帯域効率〕 | 新しい mainloop（4 consumer WG または kBlockN 32）と image の再 build。先に、同じ KV pool を 128 B と 512 B の粒度で読むだけの probe を次の GPU run に入れて、2.7 TB/s の原因を確かめる |
| Decode の静的 expert 配置 | serving と bench の差 4% が上限、−0.4〜−0.8 ms〔約 1.0〕 | Decode routing の採取（`decode-routing-capture.patch`、無効のまま）が先。Prefill の routing を proxy にすると、168 行・trivial 配置で最も重い rank の M tile が 32 から 40〜44 に増える |
| dense QKV の M=176〜192 で 2 本目の M tile（48〜64 行）を短くする | M=128 の 15.2 µs と M=176 の 23.3 µs の差の一部、−4〜−6 µs/層 = −0.4〜−0.55 ms〔約 1.0〕 | 1d2d に BLOCK_M 192 か 128+64 の混在 tile が要る。host と header の変更、image の再 build |
| MegaMoE dispatch の pull を 1 warp 2 token に（空き smem 約 30 KB を使う） | −3〜−6 µs/層 raw、−0.1〜−0.35 ms〔0.36〜0.6、待ちの削減〕 | header と host の env 切替、image の再 build。sparse dispatch（16,896 atomic の削減）は 0 だった |
| FA3 backend の draft extend を CUDA graph に | −0.1〜−0.25 ms〔0.36〜0.6〕 | FA3 backend には draft_extend_v2 の capture/replay があるが、`eagle_worker_v2` の graph 対応 backend 一覧に入っていない。model load 後にしか確かめられないので、worker 内の数値比較と eager への自動 fallback が要る |
| min-tokens penalty を疎に（停止 token の位置だけ −inf を加える） | −0.1〜−0.15 ms〔約 1.0〕 | bit 一致にできる。penalizer orchestrator の密な (bs, vocab) 加算 API を変える必要がある |

## 5. GPU でしか決まらないこと

- 融合 norm の Decode 行数での時間（1 行 1 program、8 warp）。job 内選択の 3 batch × 2 block で決まり、遅ければ採用されない。
- GPU の丸めでの bit 一致（residual、FP8 x、scale、top-k）。選択の数値 gate で確認する。
- serving で step と ITL に反映されるか。通常の serving 1 回で確認する。Decode worker log の `MoE input fused: active` と source/decode-kernels.env で経路を確かめる。

## 6. 追加：image 再 build に載せる 2 候補（同じ branch、job 内選択）

coordinator の指示で 4 節の 1 位（FA3 の読み粒度）と 3 位（dense QKV の 176〜192 行 tile）を、次の bundle build に載せる候補として実装した。どちらも選択で負ければ pin のまま動く。

### 6.1 FA3 verify：1 CTA で 2 KV head（`decode-wide-h2.patch`、行 `p128split2-h2-fp8`・`p128split3-h2-fp8`）

- 1 CTA が同じ（要求、split）の KV head 2j と 2j+1 を受け持つ。producer は token 行の連続 256 B を cp.async で読み、16 B ごとに head 別の swizzle 済み tile へ置く。consumer は 2 warpgroup で、各 WG が pin と同じ 1 WG・M64 の計算を自分の head に行う。scheduler には head 対を 2 head として見せるので、metadata、partial の配置、combine は pin と同じ。
- 4 head（512 B）にしなかったのは、consumer 4 WG では register が足りないため。2 head で 384 thread、producer 56・consumer 224 register（launch 168、spill 0）、shared memory 216,064 B なので stage は 2。
- CUDA 13.4・sm_90a：QGMMA 24、MUFU 213、FFMA 208、FMUL 206、FADD 196、FMNMX 202、F2FP 128 は pin（s3）と同数。LDGSTS は 36 → 72、LDSM/STSM は 16/24 → 32/40。split が同じ `p128split2-h2-fp8` は pin と byte 一致するはずで、選択はそれを要求する（`exact_reference`）。
- 既存の s2/s3 split、s3 merge、s2 非 split、two-CTA、Prefill fixed の 6 kernel は、patch 前後で SASS が関数名を除いて一致した。companion の patch chain は pristine の sgl-attn に `--fuzz=0` で全部当たる。
- 選択：`select-firmus-decode-fa3.py --pinned-env` の対象を two-CTA 行（40925558339 で負け）から h2 行に替えた。採用条件は従来どおり、数値 gate、split が同じ行の byte 一致、44/46/48 の全 GPU・両 block で 1 µs/層以上の短縮。

### 6.2 読み粒度の probe（`decode_kv_read_probe.py`、診断のみ）

FA3 選択の各 batch で、同じ KV（64 token page、[token][4 head][128] FP8）を同じ page table で、1 program あたり 1・2・4 head（token 行の 128・256・512 B）ずつ読む Triton kernel を測る。program 数（要求 × 8）と読む byte 数はどの粒度でも同じ。結果は `fa3.json` の各 rank の `read_probe` に TB/s で残る。h2 行が負けても、2.7 TB/s が読み粒度によるものかどうかがこの run で分かる。kernel は Triton 3.5 で sm_90 向けに offline compile できる（interpreter はデータ依存の loop 上限を扱えないので、そこでは動かしていない）。

### 6.3 dense QKV：BLOCK_M 192（host `sm90-1d2d-decode-bm192.patch`、header `deep-gemm-bench/decode-dense-bm192/sm90-1d2d-decode-bm192.patch`）

- `DG_SM90_1D2D_DECODE_CONFIG` に `QKV>128:`／`O>128:` を足した。129〜192 行だけに効き、そこでは先頭の項目より優先される。BLOCK_M 192 はこの項目でしか使えない。BLOCK_M 192 では math warpgroup が 3 個（384 thread、register 152）、1d2d の wave は 192 行になる。それ以外の geometry は header の式が変わらない。
- 候補は `QKV>128:192,72,5,1,0`（9216/72 = 128 tile、132 SM で 1 wave）。M=176〜192 の行が 1 本の M tile に入り、BLOCK_M 128 の 2 本目（48〜64 行で 1 本分の時間）がなくなる。tensor の仕事は 256 行 → 192 行（−25%）。CTA あたりの A＋B の読み込みは 192×4096 + 72×4096 = 1.08 MB で、今の 128×144 と同じ。
- 1 行の計算は WGMMA の N と K の分け方が同じなので、出力は pin（`QKV:128,144,5,1,0`）と bit 一致するはず。選択はそれを条件にする。
- CUDA 13.4：192×72 と 192×80 は 128 register（launch 512 thread の上限）、spill 0。既存の 128×144、128×80、64×144、64×96 と Prefill schedule の 256×128 は、patch 前後で SASS が一致した。host は v0.1.5 に series 全体を当てた tree で `tvm_ffi_api.cpp` の g++ syntax check を通した。
- O（K=8192）は BLOCK_M 192 にすると CTA あたりの A が 1.5 MB になり、今より読み込みが増えるので候補にしない。
- 選択：`select-firmus-decode-dense.py --pinned-env` を、pin を control に 176/184/192 行で BLOCK_M 192 だけを測る形にした（SwapAB 行は 3 run で負けたので外す）。条件は 3 行すべてで pin と bit 一致、かつ全 GPU を合わせた median で 0.5 µs 以上速く、両 block でも速いこと。採用時は変わった `DG_SM90_1D2D_DECODE_CONFIG` の 1 行だけを decode-kernels.env に追記する。

### 6.4 見込みと job 内時間

| 候補 | 見込み〔係数〕 | job 内時間 |
|---|---|---|
| FA3 h2 | 読み粒度が原因なら 185 MB を 3.2〜3.5 TB/s、−10〜−18 µs/層 = −0.9〜−1.7 ms〔約 1.0〕。原因でなければ 0、producer が 2 倍の copy と transpose を担うので負けることもある | FA3 選択 約 35 s（two-CTA 3 行が h2 2 行と probe に替わり、従来の約 30 s とほぼ同じ） |
| dense BLOCK_M 192 | QKV 23.3 µs の 2 本目の tile 分の一部、−3〜−6 µs/層 = −0.3〜−0.55 ms〔約 1.0〕 | 約 45〜55 s（DeepGEMM JIT 2 個） |
| MoE 入力融合（3 節） | −0.19〜−0.30 ms〔約 1.0〕 | 30〜45 s |

Decode GPU の連鎖（Prefill attention 選択 → FA3 → copy engine → MoE 入力 → dense）は約 220 s で、40925558339 の連鎖（約 145 s、MegaMoE 選択とほぼ同時に終了）より約 75 s 長い。目標の「MegaMoE 選択 + 60 s」を約 15 s 超える。Prefill の FP32 Q/K norm は 40911121619 と 40925558339 の両方で全 GPU で勝っている。これを `prepare-firmus-pp4-settings.py` で固定して Prefill attention 選択（約 60〜70 s）を外せば予算に収まる。ただし他の worker がこの選択に候補を足していないかを親が確かめる必要がある。
