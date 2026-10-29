# Decode step の内訳（FP8 draft、EP4）と −5% 候補（2026-10-29）

読み取りのみの解析と CPU 検証。GPU job・Actions・ssh は使っていない。

## 1. 出典と方法

- profile：run 40825417205 の `profile/nsys-decode-0.nsys-rep`（Decode は 9235.28 の EP4 最良構成と同じ引数：TP4/DP4-attention/EP4 MegaMoE、EAGLE3 3/1/4、FP8 draft、cap 176。選択は `p128split2-s3-fp8`、QKV `128,144,5`、O stock）。profile pass は別 client だが、rank あたりの running は 32〜44（verify 行 128〜176）で、n = 40/42/44 に 185/160/45 step ある。
- `nsys export --type sqlite` → [`analyze-decode-step-profile.py`](analyze-decode-step-profile.py)。94 個の MegaMoE を含む TARGET_VERIFY graph の起動から次の起動までを 1 step とし、4 GPU 全部を集計。結果は [`decode-step-profile-40825417205.json`](decode-step-profile-40825417205.json)。
- 時間は kernel 時間（同じ stream 上の直列）と、直前 kernel 終了からの空き。clock は profile pass のもの（Decode は上限 690 W 付近でも 1.86〜1.95 GHz、memory/latency 律速なので κ ≈ 1 として扱う）。

## 2. 1 step の内訳〔kernel実測〕

| 区分 | n=40 ms | n=42 ms | n=44 ms | n=42 µs/層 |
|---|---:|---:|---:|---:|
| MegaMoE | 20.88 | 21.01 | 21.01 | 223.5 |
| FA3 verify（p128s2）＋ combine | 7.07 | 7.11 | 7.17 | 71.0 ＋ 4.6 |
| dense QKV ＋ O（1d2d） | 4.03 | 4.06 | 4.07 | 43.2（QKV 24.3、O 18.7） |
| input norm＋group quant | 0.83 | 0.83 | 0.83 | **8.8** |
| router gate（nvjet split-K ＋ reduce） | 0.66 | 0.65 | 0.65 | 6.9 |
| QK-norm RoPE / post-attn norm / topk / pre-dispatch | 1.27 | 1.27 | 1.28 | 4.4 / 3.5 / 3.5 / 2.2 |
| 層内の空き | 0.39 | 0.40 | 0.40 | 4.2（qknorm 前 1.1、post-attn norm 前 0.9 など） |
| verify tail（target LM head 0.35 ＋ logits FP32 化 0.08） | 0.42 | 0.43 | 0.44 | |
| eager（sampling、penalty、draft extend） | 0.91 ＋ 空き 0.25 | 0.92 ＋ 0.24 | 0.93 ＋ 0.24 | |
| draft graph（2 step） | 0.79 ＋ 0.05 | 0.79 ＋ 0.05 | 0.80 ＋ 0.05 | |
| **周期** | **37.45** | **37.67** | **37.78** | |

- この profile では n=40〜44 の周期がほぼ平坦（約 85 µs/要求）。n=32〜44 全体の回帰では 236 µs/要求で、傾きの大半は n≤38 側の MegaMoE と FA3 の段差に由来する。10-26 の a ＋ b·n（b = 0.61 ms）は n=40〜44 の局所傾きを過大に見ている可能性がある（理論分析 §5 Q6 の答え）。
- eager の中身（n=42、1 step）：penalty 関連 約 190 µs（`acc_additive_penalties` の生成 48 µs、`repeat_interleave` の複製 69 µs、加算 73 µs）、argmax 46 µs、draft extend の forward（eager、LM head 134 µs を含む）約 530 µs、小 kernel の起動間隙。penalty は AIPerf の `min_tokens: 200` で MinNewTokens penalizer が常に有効なため。
- draft の LM head（vocab 32000、BF16、M=42〜44）は 1 回 131〜134 µs で 1 step に 3 回（0.40 ms）。262 MB の読み出しは 3.7 TB/s で 71 µs なので約 2 TB/s しか出ていない。

## 3. 床との比較（理論分析 §2.2 の更新、n=42 = 168 行）

| kernel | 実測 µs/層 | HBM 床 4.8 TB/s（3.7〜4.0） | 比 | §2.2（旧 run 40310899466） |
|---|---:|---:|---:|---|
| MegaMoE | 223.5 | 125.8（151〜163） | 1.78（1.4） | 229.0 |
| FA3 verify ＋ combine | 75.6 | KV 176 MB：36.7（44〜48） | 2.06（1.6） | 71.7 ＋ 4.5 |
| QKV ＋ O | 43.2 | 71.3 MB：14.9（18〜19） | 2.9（2.3） | 24.6 ＋ 18.8 |
| input norm＋quant | 8.8 | 4.7 MB：1.0 | 9 | 9.2 |
| gate ＋ reduce | 6.9 | 2.4 MB：0.5 | 14 | 約 5 ＋ 2 |
| target LM head（/step） | 345 | 1.245 GB BF16：259（311〜336） | 1.03〜1.1 | — |
| draft LM head ×3（/step） | 397 | 3×262 MB：164（212〜213） | 1.9 | — |

LM head は床にある（FP8 化は精度方針で不可）。旧 run との差は小さく、FP8 draft 化は draft graph（0.79 ms）にだけ効いている。

## 4. 候補と根拠

必要な短縮は s(43) ≈ 37.7 → 35.5 ms（−2.2 ms、−5.8%）。Decode は κ ≈ 1。

### 4.1 dense QKV/O の SwapAB（`decode-dense-swapab.patch`、job 内選択）

- 現行 1d2d は token 168 行を M=128 tile 2 本（256 行）に載せ、weight tile を 2 回読み、QKV は tile FLOP が 1.455 倍。QKV の L2→SM 転送は約 142 MB、O（64×96、tile 3 本）は約 165 MB。
- 新 kernel（SGLang JIT、`jit/csrc/gemm/decode_swapab_fp8.cuh`）は 1 CTA が weight 128 行 × token 176 列（WGMMA m64n176k32、2 math WG）。weight は 1 回だけ読む。DeepGEMM の TMA・barrier・WGMMA 部品と 1d2d と同じ K block ごとの FP32 promotion（`final += (sw·sx)·acc`）を使い、出力は stmatrix.trans で token 行に並べ替えて 128B swizzle の TMA store。O は K を 4 分割し、CTA cluster の分散 shared memory で rank 順に FP32 和をとってから 1 回だけ BF16 に丸める（FP 和の順序だけが変わる）。
- 候補：QKV `swapab-176-s5`（72 CTA）、`swapab-176-s5-mc2`（2 CTA cluster で token tile を multicast、L2→SM 約 65 MB）、O `swapab-176-s5-k4`（128 CTA）。DG1（cp.async 全 thread copy、K block ごとの CTA 同期、別 kernel の fixup、107/119 µs）とは producer/consumer の構造が違う。
- 見込み〔条件〕：QKV は tensor 時間 32 K block × 704 cycle ≈ 11.9 µs（1.9 GHz）＋前後 2〜3 µs で 14〜17 µs、O は 16 K block で 6 µs ＋ cluster 和・前後で 9〜12 µs。合計 −13〜−20 µs/層、**−1.2〜−1.9 ms/step**（κ≈1）。1d2d と同じく 2 WG が同じ K block を同時に処理すると promotion（1 thread あたり FFMA 88・FMUL 44・LDS 22）の間 Tensor Core が空き、1 K block 約 1000 cycle になる（QKV 17〜19 µs、下側の見込み）。そこで最初の K block だけ WG1 の発行を WG0 の後ろに並べる named barrier を置き、stage が満ちている間は一方の promotion と他方の WGMMA が重なる順序を保つ。
- 安全策：選択は 176 行で計測、serving は 129〜176 行だけで使用（それ以下は stock）。数値は既存の dense gate（FP32 参照 相対 L2 ≤ 0.005、stock 比較、changed-input graph、境界行 4/64/65/128/129/160/176、決定性）。

### 4.2 TARGET_VERIFY の input norm＋quant を 512 lane に（`decode-norm-quant-wide.patch`、job 内選択）

- 現行 kernel は 1 行を 64 lane で処理し、1 thread 約 1800 命令（CUDA 13.4 SASS、89 register）を SM あたり 2 warp で直列に流す。10-23 の記録で融合前の 2 kernel（flashinfer norm 3.3 ＋ quant 約 1.9 µs）より遅く、現在も 8.8 µs。
- 新 kernel は 1 行 512 lane（1 lane 16 byte）、344 命令・38 register。要素ごとの演算は同じで、二乗和の順序だけが変わる（CPU 模擬で gate の境界内、FP8 code の差は 1% 未満）。weight は前 kernel の出力ではないので PDL wait の前に読む。
- 見込み：flashinfer の同規模 norm（3.3 µs）並みの 3〜4 µs で **−5 µs/層 = −0.47 ms/step**。

### 4.3 verify penalty の broadcast（`verify-penalty-broadcast.patch`、bit 一致）

`repeat_interleave` で (bs·4, vocab) の FP32 penalty を作ってから加える代わりに、logits を (bs, 4, vocab) と見て行ごとの penalty を broadcast 加算する。同じ要素どうしの同じ FP32 加算なので出力は bit 一致。**−0.07 ms/step**（69 µs の複製 kernel ＋起動間隙）。

### 4.4 合計

中央値 −1.5（dense）−0.47（norm）−0.07（penalty）＋ SwapAB kernel の早期 `launch_dependents` で次の kernel（QK-norm RoPE、flashinfer norm はどちらも PDL wait あり）の起動間隙 0.1〜0.2 ms ≈ **−2.1 ms/step（−5.6%）**。範囲 −1.4〜−2.6 ms。dense が下側に着地すると −5% に届かない。

## 5. 見送ったもの

| 候補 | 理由 |
|---|---|
| router gate の 1 kernel 化 | Triton full-K は +3.4 µs/層（39881882325）、cluster 版は prepare と束で +34〜39 µs/層（40689226339）。どちらが遅いか分かっていない。差の上限は約 4 µs/層 |
| MegaMoE 前段（prepare）・後段 | 上と同じ束の測定で、prepare 単独の寄与は未分離。SASS と timeline なしに設計を変えない |
| FA3 verify の kernel 内 merge | 相対 L2 0.7315 の不具合が未解決（perf/decode-verify-merge-20261028）。得られても combine 4.6 µs/層 |
| draft LM head（3×131 µs） | 床 71 µs で −0.18 ms/step の余地。BF16 SwapAB を別に作る必要があり、今回は dense を優先 |
| penalty 加算と argmax の融合 | −0.07 ms/step 程度。argmax の NaN・同値の規則を新 kernel で再現する必要がある |

## 6. GPU でしか決まらないこと

- SwapAB 3 候補と wide norm の時間（job 内選択の 2 block・4 GPU、0.5 µs 以上かつ両 block で速い場合だけ採用）。数値は選択内の gate、失敗した候補は除外して stock のまま。
- serving での step 短縮と ITL への反映（通常の serving 1 回）。
- 選択の追加時間：rank 0 の JIT（SwapAB 3 module 各 10〜15 s、wide norm 約 5 s）。dense 選択は FA3 選択（36 s）＋ Prefill 選択（56 s）と並行なので、全体の遅れは 0〜20 s 程度。選ばれた場合は Decode worker の起動時にも JIT が 10〜20 s かかる。
