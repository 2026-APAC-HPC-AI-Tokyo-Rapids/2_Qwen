# Prefill MegaMoE timeline（2 WG、2026-10-24時点の kernel）

> 2026-10-28訂正：以下は旧runの記録。現行の内部時間として流用しない。`megamoe-phases-40057117293.json`にraw再解析を保存した。dense FP8 peakは`8192×132×clock`で、同call/rank別clockとrecv_pairsを用いたuseful/peak÷CTA中央値mainloopはFC1約69–70%、FC2約66–69%。旧73–75%と1.98GHz基準は不正確。real43 fixtureのremote率は53.40%、全rank平均FC2 remote payload約143.34MB/rankであり、下記190MBは過大。転送完了時間は独立に未測定。独立WGの−9〜10%予想は後述の失敗により棄却済み。

kernel bench [40057117293](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40057117293)（24de06f、image 39932105942、control = serving の overlay chain ＋ `DG_SM90_DECODE_INDEPENDENT_WG=1`）。timeline 版の出力は bench 版と bit 一致、計装の遅れは +0.9〜3.1%。全表は run の summary（`scripts/summarize-megamoe-kernel-bench.py`）。

## prefill-real43-replica4-4096（serving に近い実 routing ＋ hot-4 複製、1 層 1424 µs、SM 1.64 GHz）

1 CTA の流れ（rank の中央値、µs。WG0 と WG1 はほぼ同じ）：

| 区間 | 時間 | 割合 | 性質 |
| --- | ---: | ---: | --- |
| dispatch／最初の stage 待ち | 23〜29 | 2% | |
| FC1 mainloop（25 block） | 654〜690 | 47% | 824 GFLOP/rank で約 1.2 PFLOPS。1.64 GHz の FP8 理論性能（約 1.64 PF）の約 75%。うち stage 待ち 61〜69 |
| FC1 epilogue（SwiGLU＋FP8＋store） | 46〜50 | 3% | Tensor Core は空く |
| FC2 mainloop（32〜33 block） | 339〜355 | 24% | 412 GFLOP で約 1.2 PFLOPS（約 73%）。うち stage 待ち 36〜46 |
| **FC2 epilogue（BF16＋NVLink scatter）** | **182〜209** | **14%** | Tensor Core は空く。balanced では 233〜259 |
| combine 前の待ち（grid＋NVLink） | rank 0：75、1：50、2：12、3：16 | 1〜5% | rank 2/3 の mainloop が約 5% 遅い（block 数は同じ）→ rank 間の速度差 |
| combine | 78〜81 | 6% | 受信した BF16 partial（約 268 MB/rank）の HBM 読み出しでほぼ律速 |

結論：
- GEMM の mainloop 自体は、clock を下げた後の理論性能の 7 割台で、大きな余地はない。「理論性能の 4 割」の大半は、**Tensor Core が止まる区間**（FC1/FC2 epilogue 約 240 µs、combine 80 µs、rank 間の待ち 40〜65 µs、dispatch 25 µs、合計約 400 µs ＝ 28%）と、電力上限による clock 低下（1.98 → 1.64 GHz）から来ている。
- FC2 の区間（mainloop＋epilogue 約 560 µs）では、約 190 MB（出力の remote 分、推定）を NVLink で送る。片方向 450 GB/s なら下限は約 420〜470 µs。
- 2 つの math WG は、同じ tile を一緒に処理し、epilogue にも同時に入る。

## 次の候補（見込みは推定）

1. **Prefill の 2 WG を独立化（ping-pong）**：片方の WG の epilogue（FC1 の SwiGLU/量子化、FC2 の BF16/NVLink scatter）を、もう片方の WG の mainloop と重ねる。Decode の independent-WG（−2.3%、serving +1.2%）と同じ考え方。FC2 は NVLink の下限で止まるので、FC2 で約 −90 µs、FC1 epilogue で約 −47 µs、合わせて MegaMoE **−9〜10%**（Prefill −5〜6%）。
2. **rank 間の速度差を見込んだ配置**：rank 2/3 の約 5% 遅い分、仕事を少し rank 0/1 に寄せ、combine 前の待ち 40〜65 µs を削る（MegaMoE −2〜3%）。serving の 2 ノード・各 GPU でも同じ傾向かを先に確かめる必要がある。
3. combine（80 µs）と dispatch（25 µs）は HBM と通信でほぼ律速で、以前の試み（combine 前の合算、stream combine）は遅くなった。後回し。

## Prefill 2 WG 独立化 A/B と FC1 early release の Firmus kernel bench（[40069880580](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40069880580)、3e2e0c1、image 40064516380）

control は komoriyuta の 8793.13 の serving overlay から FC1 early release を外したもの。3 候補とも数値 gate と出力 hash の一致（22/22）を通過。

| Prefill case | control | FC1 early release | 独立化 A（64×256、2＋2 stage） | 独立化 B（2＋1 stage） |
| --- | ---: | ---: | ---: | ---: |
| real43-replica4-4096 | 1.424 ms | +2.0% | +25.3% | +29.1% |
| real62-replica4-4096 | 1.436 | +2.0% | +26.8% | +30.0% |
| real7-replica4-4096 | 1.452 | +1.1% | +22.3% | +35.8% |
| balanced-4096 | 1.509 | +1.7% | +22.2% | +32.2% |

- FC1 early release は Firmus では遅い（玄界の 1 GPU probe では −2.5〜4%）。
- 独立化は、stage の減少（3 → 2、B は片方 1）と B operand の TMA の重複（+67%）が、epilogue を重ねる効果を大きく上回った。shared memory の制約の下で、WG を分けて epilogue を隠す方式は不採用。Decode の MegaMoE はどの候補でも差なし。

## serving（[40071892077](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40071892077)、bd3345a、image 40064516380）：**8854.80**

komoriyuta の 8793.13（40059609107）から、FC1 early release を外し、Decode の QK-norm/RoPE の FP8 変換を hardware 変換＋厳密な修正（`patches/sglang/decode-qknorm-fast-fp8.patch`）にした。job 内の選択は FA3 p128split3、QKV BM128/BN144/5。

| | 8793.13（40059609107） | **8854.80（40071892077）** |
| --- | ---: | ---: |
| 要求数 / good | 5613 / 5478 | 5665 / 5516 |
| good fraction | 97.59% | 97.37% |
| TTFT 平均 / p99 | 1927 / 4350 ms | 1928 / 4768 ms |
| ITL 平均 / p99 | 16.58 / 19.37 ms | 16.35 / 19.17 ms |
| Prefill MegaMoE（1 層） | 1561.4 µs | **1513.6 µs** |
| Prefill 1 batch（kernel 合計） | 254.1 ms | **249.7 ms（−1.7%）** |
| Decode QK-norm/RoPE（1 層） | 9.4 µs（前回の調査値） | **4.3 µs** |

- 要求数は +0.9%、score は +0.7%。
- good でない要求（149 件、2.6%）の大半は、計測開始の一斉到着による TTFT 超過と見られる（komoriyuta の記録では 8793 の run で 121 件）。
