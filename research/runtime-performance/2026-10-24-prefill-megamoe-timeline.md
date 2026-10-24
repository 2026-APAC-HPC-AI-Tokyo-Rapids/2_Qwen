# Prefill MegaMoE timeline（2 WG、今の serving と同じ kernel）（2026-10-24）

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
