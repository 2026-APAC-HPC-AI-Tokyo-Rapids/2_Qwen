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

## 6. 長期課題 1 回目：FC2 非同期 bulk scatter（D1）と padding WG skip（D5a）、kernel bench [40095847738](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40095847738)（4f64903）

Opus が source と過去の失敗を要約し、Fable が「実ルーティング（replica4、remote 比率 中央値 48.6%、p90 56.7%）では FC2 の送信を計算の裏に隠せるはず」と判断して D1 を最優先にした。過去の bulk scatter 否定（39352299232）は均等ルーティングだけの測定だったため、実ルーティングで再評価した。host と SMEM 確保量（225,472 B）は変えず、CD は 128 行 × 512 B の行優先、quad 内 shuffle と奇数行の chunk 回転で STS.128 は全 phase 1-way。全候補で出力は control と bit 一致（19/19 ケース）、末尾の control 再実行は ±0.3%。

| case | control ms | D1 | D5a | D1＋D5a |
|---|---:|---:|---:|---:|
| real43-replica4 | 1.4113 | +1.3% | −0.4% | +0.9% |
| real62-replica4 | 1.4262 | +1.3% | −1.8% | +0.2% |
| real7-replica4 | 1.4294 | +1.4% | −1.0% | +0.1% |
| local | 1.3043 | +4.9% | −2.0% | +0.8% |
| balanced | 1.4958 | +3.2% | −1.5% | +2.6% |

timeline（per CTA、µs、WG0 rank0）：

| | control real43-replica4 | D1 real43-replica4 | control local | D1 local |
|---|---:|---:|---:|---:|
| FC2 stage wait | 38.5 | 57.1 | 32.3 | 32.7 |
| FC2 epilogue | 189.3 | 198.8 | 94.1 | 114.9 |
| ML2Drain（最終の完了待ち） | — | 0.0 | — | 0.0 |

**D1 は不採用。** 機構は 2 つ。(1) bulk store が同じ SM の TMA load と TMA 機構を取り合い、FC2 の stage wait が +19 µs。(2) epilogue は短くならない。NVLink を使わない local でも FC2 epilogue は 94 µs かかり、これは rank あたり 268 MB の BF16 combine payload を書く時間（≈ 2.85 TB/s）に相当する。つまり FC2 epilogue は発行コストではなく combine payload の書き込み量そのもので決まっており、remote 分（実ルーティングで約 +95 µs）が NVLink に乗る。最終 drain が 0 µs なのは、次 tile の STS 前の `.read` 待ちで転送待ちがすでに吸収されているためで、CD が 1 面しかないため重ねられる区間は次 tile の mainloop 1 回分しかない。

**D5a は採用候補。** 有効行 64 以下の tile で WG1 の WGMMA・rescale・SF load を省くだけ（padding 行のみ、出力 bit 一致）で、実ルーティング −0.4〜−1.8%、local −2.0%。D1 と組むと D1 の損で相殺される。serving overlay への組み込みは次の build に同乗させる。

**次の論点。** FC2 epilogue（約 190 µs）と combine（約 80 µs）は、どちらも combine payload（1 token あたり top-8 × 4096 × BF16 = 64 KiB）の書き込みと読み出しで決まる。重ねるより、payload を減らす方が筋が良い。同じ expert rank 上の同一 token の出力を FP32 で合算し、1 度だけ BF16 に丸めて送れば、丸め回数は現行（expert ごとに BF16）以下になる。演算順序の違いに当たるので規則上は許容と読めるが、確認は要る。過去の pre-reduce（39355994503、serving −5.8%）は、HBM staging と行ごとの atomic で負けており、方式の問題だった。

## 7. attention・MegaMoE 候補の job 内選択つき serving [40175897453](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40175897453)（87a85dd、新 image）

**9022.10 tok/s**（公式判定 有効）。8854.80（40071892077）比 +1.9%、komoriyuta の 8922.77（40116925551）比 +1.1%。要求 5744 / good 5611（97.68%）、TTFT 平均 / p99 1819 / 4248 ms、ITL 平均 / p99 16.52 / 19.30 ms、受理長 2.21。profile なし（selector 込みで 17 分枠に収めるため）。

| 選択 | 勝者 | 根拠（selector の計測） |
|---|---|---|
| Prefill FA3（3 worker × 4 rank、L=3986/4027/4096） | **n192** | control 比 中央値 −9.3 µs/層（最小 −7.1）。regs −1.5、lazy −0.4、regs-lazy −1.4 は閾値 5 µs 未満で不採用。c2・regs-c2・regs-lazy-c2 は V byte image の dump 照合に失敗（gate-or-timing-failed） |
| Decode FA3 verify（B40） | **p128split3-s3-fp8** | stock 比 −21.7 µs/層（従来の p128split3 は −18.2）。FP8 出力の融合で約 −3 µs、3 段で約 −0.5 µs。上位は split2-s3-fp8 −22.0、uneven-s3-fp8 −21.8 と僅差。kernel 内 merge の 7 候補はすべて candidate-failed |
| Decode dense | QKV 128,144,5（従来どおり） | |
| MegaMoE | padding-WG skip（D5a）を overlay で常時適用 | |

Prefill selector は 94 層 graph（QKV producer 込み）で計り、1 GPU probe（regs232 −5.3%、N192 −4.0%）と違い、serving 形では regs の効果が消えた。固定経路の control は probe の control と register 配分が違う（spill 72/120 B 対 80/140 B）ことが一因とみられる。Decode selector の所要は 46.7 秒、Prefill selector と合わせて MaxTime 内に収まった。

残課題：c2 の byte image の照合失敗と、Decode の kernel 内 merge の失敗はどちらも実装の不具合で、次の build までに直す。e4（木構造の行最大値）は Firmus probe 40179002643 で +1.2〜+6% と遅く不採用。e1（PV 待ちの位置）は 22 通りの source 変更で ptxas の配置を変えられず未達。

## 8. 2 回目の serving：PV 待ち対策・c2 修正・DG1・merge 修正・norm 融合・上限 176 [40201493911](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40201493911)（3287a43、image 40197369083）

**9036.12 tok/s**（公式判定 有効）。9022.10 比 +0.16% で、単回差としては区別できない。要求 5749、good 97.67%、TTFT 平均 / p99 1818 / 4250 ms、ITL 平均 / p99 16.5 / 19.2 ms、受理長 2.21。image build の run [40197369083](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40197369083)（cbca2b3）は、Decode norm 融合の gate が flashinfer との bit 一致を要求していたため、server 起動前に停止した。既存の 2 行融合も同じ比較で 127 行あたり約 500 code 異なる（scale は一致）ので、Prefill と同じ丸め水準の判定に揃え（3287a43）、同じ image で再実行した。

| 項目 | 結果 |
|---|---|
| Prefill FA3 の PV 待ち対策（pv-live-regs / -n192 / -regs-n192） | control 比 中央値 **+10〜+17 µs/層の悪化**。checksum と恒等 PRMT の追加命令が PV の重なりの利得を上回った。不採用 |
| c2（V byte image 修正） | 数値 gate は通過。中央値 −1.9 µs/層、最小 +3.1 µs で閾値未満。不採用 |
| Prefill 勝者 | n192（中央値 −9.3 µs/層、前回と同じ） |
| Decode dense DG1（SwapAB + stream-K、176 行） | QKV 107.4 µs・O 118.1 µs、stock は 24.2・17.4 µs。**不採用**。QKV は dg-128-144-5（23.4 µs）、O は stock |
| Decode FA3 の merge 可視性修正 | 全 merge 候補が修正前と同じ相対 L2 0.7315 で失敗。原因は L1 可視性ではない。勝者は p128split2-s3-fp8（71.6 µs、control96 は 93.7 µs） |
| Decode norm 1 行/CTA | 丸め水準の gate を通過（127/128/144 行で 483/533/644 code 差、scale 一致）。単独の時間は未計測 |
| Decode 上限 176 | Decode の running は中央値 40 / p90 43 / 最大 44。**Decode の待ち行列は平均 3.81 → 0.13** |

**律速が Prefill に戻った。** Decode の待ちはほぼ消えた一方、Prefill 3 worker の待ち行列は平均 4.24（前回 3.3）。Decode 側の余力は Prefill の供給を超えたので、Decode 側の改善（MegaMoE M1 など）は今の構成では score に効かない。

**Prefill は電力上限で動いている。** 1 Hz の telemetry（今回から記録）で、負荷中の Prefill 12 GPU は電力中央値 676〜693 W、SM クロック中央値 1500〜1560 MHz（最低 1380 MHz）。throttle 理由は SW power cap（0x4）だった。Decode の 4 GPU は同じ約 690 W で 1860〜1935 MHz。Prefill の速さは命令数よりも 1 演算あたりのエネルギーで決まる。命令を足して重なりを増やす変更（今回の PV 待ち対策、e4）が serving 形で負けるのと整合する。次に効く方向は、Prefill の総仕事量またはデータ移動量を減らす変更（FC2 combine payload の削減、GEMM の無駄な tile の除去など）であり、重なりの改善ではない。
