# 2026-10-24 系全体の上限と構造的レバーの再検討（8854.80 = run 40071892077 を起点）

局所的な kernel 改善の利得が +0.5〜1% まで細ったため、理論上限から逆算し直した。Fable 5.1（系の待ち行列モデルと第一原理）と Codex GPT-6 Astra（nsys 4 本を SQLite 化した全 94 層の kernel 台帳）に独立に分析させ、本稿で突き合わせた。GPU job は使っていない。数値は run 40071892077 の artifact（profile pass の nsys、profile_export.jsonl、worker log）から再計算したもの。

## 1. 両者が一致した事実

| 項目 | 値 | 意味 |
|---|---|---|
| serving 中の Prefill clock | 1.38〜1.40 GHz（1.30〜1.53） | kernel bench（1.47〜1.67 GHz）より低い。bench の順位は、NVLink 律速部と compute 律速部の比が serving と違う |
| Prefill Tensor pipe 稼働 | 54〜56% | GEMM の clock あたり効率は天井近く、Tensor Core が止まっている時間が問題 |
| Prefill 1 step（4032 token/rank） | 255 ms、kernel 稼働 97%、空き 6.6 ms | CPU やスケジューラの空きは 2.6% しかない |
| Prefill MegaMoE | 1494 µs/層、useful 808 TFLOPS（観測 clock の FP8 peak 比 54%） | 床との差 64 ms/step、最大の予算 |
| Prefill FA3 | 450 µs/層、581 TFLOPS（39%） | 床との差 26 ms/step |
| Prefill dense QKV＋O | 81〜83% | 余地は小さい |
| Prefill 小 kernel 合計 | 約 208 µs/層（qknorm/RoPE 66 µs は HBM の 36%） | 融合の余地あり |
| Decode 1 step（40 要求/rank × verify 4） | 36.7〜37.0 ms、kernel 稼働 98%、空き 0.7 ms | CPU 改修での回収は最大 1.8% |
| Decode MegaMoE | 218 µs/層（重み HBM 床 126 µs、帯域 58%） | 床との差 8.7 ms/step |
| Decode FA3 verify | 69.5 µs/層（KV 床 35 µs、50%）＋ combine 5.3 µs | 差 3.2 ms/step |
| P/D の釣り合い | Prefill 供給 47.0 req/s、Decode の SLA 内容量 49.6 req/s | Prefill だけ速くしても +2% で Decode の ITL に当たる。Decode だけなら 0% |
| 損失 | 初期一斉到着の bad 120 件（= N − 12 rank × 10）、立ち上がり 2.2%、drain 0.9% | bad を全部救っても +2.7%。N 240 → 200〜210 で +0.5% 程度 |

「10000 は物理上限のすぐ内側」（[10-18 plan](2026-10-18-firmus-maximum-performance-plan.md)）は両者とも退けた。必須演算だけの上界は観測 clock で約 23,600、電力上限を考えた現実的な目標は 11,000〜12,000（Prefill 205 ms / Decode 30 ms 相当）。

## 2. 見解が分かれた点

- **上限の見積もり**：Fable は「電力上限で duty を上げると clock が下がる」線形モデルで 13,100（duty 0.85）、現実的 11,200。Codex は clock を固定し 75〜80% roofline で 12,700〜13,600。どちらも 11,000 前後は理論上矛盾しないとする。
- **EAGLE steps 2 / draft 3**：Fable は「要求数を 48/rank に増やせば verify 行数が 160 → 144 に減り、SLA 内容量が 49.6 → 57 req/s」と推す。Codex は 10-20 の実測（受理長 2.01、step −7%）から、同じ要求数では step < 32.9 ms が要ると指摘。両者は矛盾せず、K=2 は「要求数を増やして Decode の余力を作る」手段で、Prefill が 49 req/s を超えて初めて効く。p99/mean 比 1.172 なので平均 ITL 16.9 ms は p99 で 19.8 ms と縁にある。
- **FA3 prefill の余地**：Fable +1.4%、Codex +6〜9%（Decode 改善と組んだ場合）。4k causal FP8 の FA3 として 581 → 700 TFLOPS 程度は妥当（step −7 ms、−2.9%）。
- **EAGLE 受理長 2.21**：Fable は nebius の model card（K=7 で 4.25〜4.32）から 25〜30% 低いとして、PD 経路・aux hidden state 層・FP8 target の不一致を疑う。玄界の非 PD で τ(K) を測れば安く切り分けられる。

## 3. 構造的レバー（両者の統合順位）

| 順位 | レバー | 側 | 見込み（片側のみの効果ではなく対で） | 費用 |
|---|---|---|---|---|
| 1 | MegaMoE Prefill の段階分離（dispatch/epilogue/scatter の所有者を math WG から分け、ready expert から進める） | P | 1.49 → 1.13〜1.30 ms/層、P 221〜237 ms | 大、build 1〜2 |
| 2 | Decode MegaMoE を重み供給と expert readiness 主体に（work queue と buffer epoch の再設計） | D | 218 → 160〜180 µs/層、D −3.6〜5.5 ms | 大、build 1〜2 |
| 3 | Decode verify attention の KV 共有・split 集約の再設計 | D | 75 → 50〜55 µs/層、D −1.9〜2.3 ms | 中〜大 |
| 4 | Prefill FA3 の 4k causal 専用 pipeline | P | 450 → 300〜370 µs/層、P −7〜20 ms | 中〜大 |
| 5 | Prefill 小 kernel の融合（add＋RMSNorm＋quant＋pre-dispatch、qknorm/RoPE） | P | 208 → 120〜150 µs/層、P −5〜8 ms | 中 |
| 6 | EAGLE 受理長の診断（玄界 非 PD で τ(K)） | D | 欠陥なら Decode +13〜27% | 玄界 1 job |
| 7 | Decode K=2 ＋ cap 192、N 200〜210、hot expert 複製 8 | D/系 | Decode 余力 +15%、+0.5〜1% | config のみ（Prefill 改善と同乗） |
| 8 | P:D 13:3（4+4+5 / EP3 Decode） | 系 | +8%、Decode 行費用 −20% が前提 | 大、条件付き |

採らない：chunk 32768/rank、EP8 Prefill、TP attention、tree 投機、2 GPU Decode、ping-pong 系（既に負け）。

## 4. 進め方

Prefill と Decode を必ず対で速くする。第 1 段の合格線は Prefill 225 ms ＋ Decode 33 ms（予測 約 10,000）、第 2 段は Prefill 205 ms ＋ Decode 30 ms（約 11,000）。kernel bench の判定は serving 時の clock（1.38 GHz）との差を前提にし、NVLink 律速部は clock で縮まない。

## 5. 正本

分析の作業ファイル（SQLite、全 kernel CSV、全 94 層台帳、計算スクリプト）は作業端末の scratchpad にあり、リポジトリには入れていない。本稿の数値はそこから転記した。
