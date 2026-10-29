# Decode step の再分解と、FA3 verify の 2 CTA/SM 候補（2026-10-30）

読み取りのみの解析と CPU 検証。GPU job・Actions・ssh は使っていない。前回の内訳と床の表は [2026-10-29-decode-step-breakdown.md](2026-10-29-decode-step-breakdown.md)。

## 1. 出典と方法

- run 40825417205 の `profile/nsys-decode-0.nsys-rep`（4 GPU）。その後の Decode kernel の変更は wide norm/quant（in-job 6.06 → 2.51 µs/層）と verify penalty の broadcast（bit 一致、−0.07 ms）だけなので、以下の表は profile の値のまま、この 2 点を本文で補正する。
- [`analyze-decode-step-profile.py`](analyze-decode-step-profile.py) を細分化した。verify graph（94 層、層 0 の非融合 norm を含む）、verify tail（target LM head、logits の FP32 化）、eager の sampling/verification、eager の draft extend forward（最初の nvjet から draft の argmax まで）、batch bookkeeping、draft graph（2 step）に分け、各区分の a + b·n を全 n と n ≥ 36（FA3 split2 が 3 round の領域）で回帰する。結果は [`decode-step-profile-40825417205.json`](decode-step-profile-40825417205.json)。
- profile pass の周期は serving より約 10% 長い（n≈40 で 37.4 ms、serving は ITL 15.30 ms × 受理長 2.217 ≈ 33.9 ms）。以下は比率で扱う。

## 2. 1 step の内訳〔kernel実測、ms/step〕

| 区分 | n=40 | n=42 | n=44 | a（n≥36） | b µs/要求 |
|---|---:|---:|---:|---:|---:|
| MegaMoE（EP4 dispatch・FC1・FC2・combine を 1 kernel） | 20.88 | 21.01 | 21.01 | 14.92 | 144.9 |
| MoE 前段（gate GEMM＋reduce、top-k、pre-dispatch） | 1.20 | 1.18 | 1.19 | 0.97 | 5.3 |
| target attention（FA3 verify＋combine＋prepare、QK-norm RoPE） | 7.48 | 7.52 | 7.58 | 5.97 | 37.3 |
| dense QKV＋O | 4.03 | 4.06 | 4.07 | 3.39 | 15.8 |
| norm（input norm＋quant、post-attn） | 1.16 | 1.16 | 1.17 | 0.98 | 4.3 |
| 層内の空き＋その他 | 0.39 | 0.40 | 0.40 | 0.31 | 2.1 |
| verify tail（target LM head 0.35、logits FP32 化 0.08） | 0.43 | 0.44 | 0.45 | 0.25 | 4.4 |
| sampling/verification（penalty 0.18、argmax 0.05、tree） | 0.23 | 0.24 | 0.24 | 0.04 | 4.7 |
| draft extend（eager forward、168 行） | 0.53 | 0.53 | 0.53 | 0.50 | 0.8 |
| draft graph（2 step、42〜44 行）＋空き | 0.85 | 0.85 | 0.85 | 0.82 | 0.6 |
| eager bookkeeping＋空き | 0.39 | 0.39 | 0.39 | 0.39 | 0.1 |
| **周期** | **37.44** | **37.67** | **37.78** | **28.43** | **220** |

区分の和が周期を 0.1 ms 上回るのは、bookkeeping の一部が別 stream（s136）で重なるため。

- **a + b·n。** 全 n で a = 27.78 ms、b = 0.236 ms/要求、n ≥ 36 で a = 28.43、b = 0.220。n = 40→44 の局所傾きは 0.085 ms/要求で、b の大半は n = 36→40 の MegaMoE（20.0 → 20.9 ms）による。
- **draft（3 forward）は計 1.38 ms/step（3.7%）。** attention 0.30（eager extend 101 µs、graph 2 × 97 µs。同じ形の target verify は 71 µs）、LM head 0.41（3 × 約 135 µs、262 MB を約 1.95 TB/s）、GEMM 0.40（eager extend の 168 行で 210 µs。graph の 42 行では 2 × 94 µs。cutlass FP8 の QKV が 168 行で 56 µs、42 行で 12.5 µs）、小 kernel 0.20。
- **CPU・graph の空き**は層内 0.40、eager 0.24、bookkeeping 0.15、draft graph 0.05 で計約 0.85 ms（2.3%）。eager の小 kernel は 1 個あたり約 1.2 µs の起動間隙で並ぶ。
- **MegaMoE**（55.8%）の内部（dispatch・FC1・FC2・combine）は nsys では分けられない。旧 timeline（10-18、decode-128）では起動→FC1 約 19〜37 µs、FC1 106〜137、FC2 60〜78、後段 15〜20 µs。

## 3. n = 45〜50 で何が律速するか

- **上限 176（44/rank）と graph bs ≤ 44。** 現在 n を 44 より上げる手段がない。serving の p99/平均 ITL = 1.176 から、p99 < 20 ms は step ≤ 1.111 × step(39.5) に相当する。p99/平均の比が保たれると仮定すると、どちらの回帰でも n ≈ 57〜58 まで ITL は SLA 内に収まる。**+10〜15%（n = 44〜46）を受けるのに必要なのは、まず `decode_max_running_requests` 192 と graph bs 46・48 である**（n ≤ 44 の step は変わらない）。kernel 側の guard（FA3 B ≤ 48、MegaMoE SwapAB ≤ 192 行、1d2d ≤ 192 行、norm 128〜192 行、top-k ≤ 192）はすべて n = 48 まで対応済み。KV pool は (48 + 20) × 4224 = 287,232 ≤ 393,216 token/rank。
- **n ≥ 50 の段差。** FA3 split2 は tile = 8n で、1 CTA/SM の 132 slot に対し n ≥ 50 で 4 round 目に入る（+約 19 µs/層 = +1.8 ms）。SwapAB と FA3 の B ≤ 48 guard もここで切れる。

## 4. FA3 verify は CTA あたりの latency 律速

| n | split2 tile | round | FA3 µs/層 |
|---:|---:|---:|---:|
| 32 | 256 | 2 | 49.2 |
| 33（n=34 bucket の p10） | 264 | 2 | 52.0 |
| 34 | 272 | 3（3 round 目 8 tile） | 67.9（中央値） |
| 36〜44 | 288〜352 | 3（24〜88 tile） | 68.8〜71.6 |

3 round 目は 8 tile でも 88 tile でも 19〜22 µs かかる。1 tile（KV 16 block）を GPU がほぼ空いた状態で処理しても約 19 µs で、132 SM がすべて動く round の 24.6 µs の 77% にあたる。全 SM が動く round の読出しは 134 MB / 49.2 µs = 2.7 TB/s で、128 MiB 読出し reference の 3.7 TB/s に届いていない。job 内選択（4 run）でも、B40 の split2・split3・uneven（528 unit）はすべて 70.2〜70.8 µs だった。CTA あたり 3 tile でも 4 tile でも同じ時間なので、時間を決めているのは KV の byte 数ではなく tile あたりの latency である。消費側は 1 MMA warpgroup（SMSP あたり 1 warp）だけで、1 KV block あたり約 2200 cycle かかる。tensor 約 512 cycle、MUFU 約 512 cycle に対して大きく、latency が支配している。

## 5. 候補：2 CTA/SM（`decode-two-cta.patch`、job 内選択）

- **変更。** `FlashAttnFwdSm90` に `MinBlocks_`（既定 1）を加える。verify の 2 CTA 版は launch bounds を (256, 2) にし、launch 時 128 register、setmaxnreg を producer 56 / consumer 200 とする（2 CTA × 128 thread × 256 = 65,536）。stage は 2（109,568 B、2 × (109,568 + 1,024) ≤ 228 KiB は build.py で assert）で、persistent grid は 2 × 132 CTA。mainloop・epilogue・scheduler・prepare・combine は変えない。FLOP と byte も同じで、1 SM で独立な 2 tile を並べて latency を隠すだけである。
- **候補。** `p128split2-c2-fp8`、`p128split3-c2-fp8`、`p128-uneven-c2-fp8`。uneven は B34〜44 で 528 unit = 2 × 264 slot になる。
- **SASS（CUDA 13.4、sm_90a）。** 2 CTA 版は launch 128 register、spill 0、最大 register R186（consumer 200 の範囲内）、`USETMAXREG.TRY_ALLOC 0xc8 / DEALLOC 0x38`、4832 命令。同じ split の 1 CTA 版（s2）は 255 register、R251、4728 命令。QGMMA 24、MUFU 213、FFMA 208、FMUL 206、FADD 196、FMNMX 202、F2FP 128、LDGSTS 36 は完全に一致する。差は整数の address 計算が uniform datapath（UIMAD 0→55、UIADD3 46→84、ULDC 95→132）へ移った分だけである。PTX の FP・tensor の opcode 集合も一致する（1376 個）。したがって split2-c2 の出力は、pin されている split2-s3 と byte 一致するはずである。
- **触らない kernel の SASS。** patch 前後で、verify main 4 種（s2/s3 split、s3 merge、s2 非 split）と Prefill fixed の命令列（address・encoding）は、関数名を除いて完全に一致した。mangle 名だけは既定の template 引数が 1 個増えたため変わる。
- **見込み**〔条件〕。κ ≈ 1（Decode は memory/latency 律速で clock は上限近く）。HBM 上限を 3.7 TB/s とおく。
  - split2-c2（n=44）：264 tile の round は 139 MB / 3.7 TB/s = 37.5 µs、残り 88 tile は約 19 µs で、計 57〜60 µs（現行 71.6）。
  - split3-c2・uneven-c2（n=40〜44）：480〜528 unit を帯域律速の 2 round（各 23〜25 µs）で処理し、46〜50 µs ＋ combine 約 5.5 µs（現行 約 76）。
  - FA3＋combine で −12〜−22 µs/層、**−1.1〜−2.1 ms/step（−3〜−5.5%）**、中央 −1.6 ms（−4.3%）。paged cp.async の読み方が 2.7 TB/s で頭打ちなら split3-c2 は 2 × 33 µs ＋ 5.5 で、−0.5 ms が下限になる。
  - split2 は 264 slot に対して n ≤ 66 まで 2 round に収まるので、§3 の n ≥ 50 の段差も消える。
- **安全策。** PP4 の job では Decode GPU 上の選択の連鎖（MoE input → dense → **FA3** → copy engine）に `select-firmus-decode-kernels.sh fa3 --pinned-env` を加える。pin されている候補（現 `p128split2-s3-fp8`）を基準に、3 行だけを B = cap/4 で測る。採用条件は 3 つ。(1) 既存の数値 gate（FP32 参照、stock 比較、境界長、NaN を入れた partial、changed-input graph replay）を通る。(2) split 数が pin と同じ行は、pin の出力と byte 一致する。(3) 4 GPU × 2 block すべてで 1 µs/層以上速い。勝者は `SGLANG_DECODE_FA3_VERIFY=<勝者>` として decode-kernels.env に追記し、source 時に pin を上書きする。失敗・遅延・不一致のときは何も書かず、pin がそのまま使われる。
- **job 内の追加時間。** 選択は 3 行 × 3 batch。過去の 16 行版が 36〜47 s だったので、20〜40 s と見込む。Decode GPU の連鎖は Prefill GPU の MegaMoE・FA3 選択と並行して走る。40883389891 では連鎖（dense は 22:04:30 頃に終了）が Prefill worker の起動（22:05:2x）とほぼ同時に終わっていたので、job は 0〜40 s 延びる。
- **image の再 build が要る**（AOT companion `kernels/fa3-verify`）。

## 6. kernel 内 merge（perf/decode-verify-merge-20261028）は直さない

玄界の診断 run 40626477479（B40、split 3、1 GPU）では、最初の不一致が `merge.final_counters` = 480 だった（期待値 0）。480 = 40 要求 × 4 head × 3 split で、すべての到着が数えられたまま、どの (要求, head) でも reset されていない。reset は「最後の到着」の分岐の中にだけあるので、`old == count − 1` が一度も成立しなかったことになる。tile の split 座標から復号した split 数（`split >> 16`）が実際の到着数と一致していない疑いが強い。その結果 merge が走らず、出力には NaN を入れた workspace が残る（`loaded_O` が NaN）。

直したとしても、得られるのは combine kernel（4.6 µs/層、0.43 ms/step）から、最後に到着した CTA が critical path に足す merge の分を引いた残り、約 0.2〜0.3 ms である。2 CTA 化（−1.1〜−2.1 ms）より 1 桁小さく、2 CTA 化では combine を別 kernel のまま残せる。数値の失敗は 3 run と診断 1 回で続いている。今は直さない。再開するなら、probe の record[9..10]（old/last）と復号した count を出力して、まず count の不一致を確かめる。

## 7. 実装しなかった手段（大きい順）

| 手段 | 見込み | 理由・前提 |
|---|---:|---|
| Decode 上限 192、graph bs 46・48 | n ≤ 44 は 0。+10〜15% の要求を受ける前提 | 設定の変更だけ。FA3・dense の選択 batch が 48 に動くので、親で一括して変える |
| MegaMoE decode の前段（16,896 atomic → prepare）と FC2→combine の chunk 化 | 5〜20 µs/層（0.5〜1.9 ms） | 21 ms・55.8% と最大。prepare は gate と束で +34〜39 µs だった（40689226339）。単独の寄与を先に測る |
| Decode の MoE 入力融合（Prefill の `post_norm_moe_input` を TARGET_VERIFY に） | −2〜3 µs/層（0.2〜0.3 ms） | pre-dispatch 2.2 µs と起動間隙を消す。Decode 形での選択が要る |
| sampling：疎な min-tokens penalty と BF16 logits の argmax 融合 | 0.15〜0.2 ms | penalty の生成 48 µs、加算 73 µs、FP32 化 79 µs、argmax 48 µs。argmax の NaN・同値の規則を再現する必要がある |
| draft LM head（3 × 135 µs、約 1.95 TB/s） | 約 0.15 ms | BF16 のまま帯域を出す GEMM の設定が要る |
| draft extend の FP8 GEMM（168 行で 210 µs、cutlass の設定） | 約 0.1 ms | sgl-kernel の dispatch 変更は image 側の作業 |
| dense QKV/O | — | SwapAB は 25.6 対 23.1 µs で負けた。split-K/stream-K の再設計が必要 |

## 8. GPU でしか決まらないこと

- 2 CTA が実際に 1 SM に同居すること。tile の latency がどこまで重なるかと、HBM の実効帯域。これらは job 内選択の時間と、serving の profile pass で確認する。
- split2-c2 が pin と byte 一致すること（選択内で検査）。split3/uneven-c2 の数値（既存 gate）。
- serving の ITL と step への反映。通常の serving 1 回で確認する。
