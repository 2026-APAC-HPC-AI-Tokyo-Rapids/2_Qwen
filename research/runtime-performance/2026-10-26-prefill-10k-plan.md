# 2026-10-26 10,000 tok/s に向けた Prefill の方策（profile・NCU・source に基づく再評価）

起点は main の 9120 構成（image 40197369083）。数値は 40310899466（Nsight）、40313004834（`ReqTimeStats`、telemetry）、FA3 probe 40172396762（NCU）と 40172404753（M1 計時）から取った。GPU job は使っていない。

**2026-10-28 更新:** 以下の改善幅・合計予測は過去の検討として保存する。今後の優先順位と理論比較は[カーネルの理論値と改善台帳](kernel-theory-tracker.md)を正本とする。同runのraw profileを再集計し、Prefill実row3994/3995、target/draft分離、PDLの重なりを補正した。公称1979 TFLOPSを1.98GHzで割る補正、33MBだけによるnorm類の床、融合済み処理の再計上を撤回。§2の一律50〜60%電力係数と§9の到達予測は採用判断に使わない。hot4複製とinput norm/quant・QKnorm/RoPE/FP8 Q融合は導入済み。§8のDecode workerでPrefillする案はBasicの候補にしない。routerの通常測定結果は[2026-10-27-router-first-response.json](2026-10-27-router-first-response.json)に記録済みで、TTFT短縮だけではgoodput改善にならなかった。

**2026-10-27 検証による訂正:** §4.5 F1 の「`ex2.approx.f16x2` で MUFU 命令が半減する」という前提は、CUDA 13.0.88 / SM90a の本番形状の native compile で否定された。2要素は2本の `MUFU.EX2.F16` に展開され、変換とspillも増えたためGPU測定は行っていない。したがって F1 の −3.5〜−7 ms と、それを含む §9 の 9,500〜9,900 見積りは現時点の到達予測として使用しない。QKVの追加2案も採用条件に届かなかった。実測と次の限定候補は [10-27 の台帳](2026-10-27-prefill-memory.md) を参照。

## 1. 現状の収支

| 項目 | 値 | 出所 |
|---|---:|---|
| score | 9118.01 / 9129.28（good 0.9771 / 0.9778、5802 要求） | 両 run の score.json |
| raw 要求率 | 46.7 req/s | aiperf `request_throughput` |
| Prefill 1 反復 | **251.4 ms**（kernel 稼働 249.4 ms、空き 2.0 ms） | nsys prefill-0 GPU0、17 反復 |
| Prefill 供給 | 4032 / 0.2514 × 12 / 3986 = **48.3 req/s** | |
| 供給 → raw の損失 | 0.967（立ち上がり約 0.5 s ＋ drain 約 3.5 s / 124 s） | |
| ITL 平均 / p90 / p99 | 15.9 / 17.4 / 18.8 ms | aiperf |
| TTFT 平均 / p99 | 1.89 / 4.43 s | aiperf |
| 受理長 | 2.21（4 DP state とも） | eagle-acceptance.json |
| 電力 | **16 GPU すべて中央値 689〜693 W、throttle 理由 0x4（SW power cap）が 95% 超** | telemetry g1/g4 |
| SM clock 中央値 | Prefill 1485〜1545 MHz、Decode 1860〜1950 MHz | 同上 |

Decode の step は running に対して約 0.57 ms/要求で伸びる（decode-0.log の `Decode batch` 行、step = running × accept / throughput）。

| running/rank | 37 | 39 | 40 | 41 | 42 | 43 | 44 |
|---|---:|---:|---:|---:|---:|---:|---:|
| step ms（中央値） | 35.1 | 36.2 | 37.1 | 37.3 | 37.7 | 38.6 | 39.4 |

cap 176（44/rank）に張り付くと step 39.4 ms → ITL 平均 17.8 ms、要求ごとの p99（p99/平均 1.18）は 21 ms で SLA を割る。Decode の SLA 内容量は running 42 前後、**約 50 req/s** であり、Prefill 供給 48.3 とほぼ釣り合っている。

## 2. 10,000 の必要条件

`score = raw × good × 200`。good 0.978 なら raw ≥ 51.1 req/s。

- Prefill 供給 ≥ 51.1 / 0.967 = 52.9 req/s → **1 反復 ≤ 229 ms（−22 ms、−8.9%）**。
- Decode は 51.1 req/s を ITL p99 < 20 ms で受ける必要がある。running 42〜43/rank で step ≤ 36.5 ms → **step −3〜−5%**。
- good を 0.99 に上げられれば raw 50.5、Prefill ≤ 232 ms に緩む。

**電力上限の係数。** Prefill 12 GPU は 700 W の cap で動いており、kernel の短縮は平均電力を上げて clock を下げる。実測は async admission（空き −5.3 ms → 反復 −1.9 ms、36%）だけだが、これは空き→稼働の変換で最悪側である。命令を減らす変更は動的エネルギーも減るので、**見込みは raw の 50〜60%** とする。したがって kernel 時間で **raw −35〜−45 ms** を見つける必要がある。

## 3. Prefill 1 反復の内訳（nsys 40310899466、prefill-0 GPU0、251.4 ms）

| kernel | ms/反復 | 割合 | 回/反復 | µs/回 |
|---|---:|---:|---:|---:|
| MegaMoE `sm90_fp8_mega_moe_impl` | 142.5 | 56.7% | 94 | 1516 |
| dense QKV＋O `sm90_fp8_gemm_1d2d_impl` | 44.3 | 17.6% | 188 | 236 |
| FA3 prefill | 41.6 | 16.5% | 95 | 438 |
| qknorm_rope_prefill_fast | 6.24 | 2.5% | 94 | 66 |
| fused_add_rmsnorm（flashinfer） | 3.19 | 1.3% | 100 | 32 |
| norm_quant | 2.86 | 1.1% | 90 | 32 |
| pre_dispatch | 1.54 | 0.6% | 94 | 16 |
| router gate GEMM（BF16 nvjet 64x80） | 1.41 | 0.6% | 94 | 15 |
| EAGLE draft extend（BF16 nvjet 4 本＋act） | 3.9 | 1.6% | 6 | 140〜1183 |
| router triton、mask、copy ほか | 1.4 | 0.6% | | |
| 空き | 2.0 | 0.8% | | |

dense GEMM は QKV 304 GFLOP / O 270 GFLOP を 236 µs で、**約 1220 TFLOPS（観測 clock の peak 比 81〜87%）**。余地は −1〜−2 ms。

## 4. FA3 prefill：NCU の結果（probe 40172396762、control N224、shape4096、1 launch 370.8 µs、clock 固定なし）

kernel は `FlashAttnFwdSm90`、PackGQA（tile M128 = 8 token × 16 Q head、N224、K128、FP8、2 consumer WG ＋ 1 producer WG、`IntraWGOverlap`）。tile 訪問は 19,760、warp-tile（WG-tile × 4 warp）は 158,080。

### 4.1 pipe の占有

| pipe | warp 命令数 | 占有（elapsed 比） |
|---|---:|---:|
| tensor（QGMMA 64x224x32 ＋ 64x128x32） | 1.74 M | **42.3%** |
| XU（MUFU.EX2 18.02 M、RCP 0.10 M、LG2 0.03 M） | 18.17 M | **約 43%**（16 lane/SM、8 cycle/warp 命令で換算） |
| ALU（FMNMX 20.7 M、F2FP 9.9 M、PRMT 4.8 M、ISETP、LOP3、FSEL …） | 50.0 M | 29.9% |
| FMA（FFMA 18.4 M、FADD 17.8 M、FMUL 11.7 M、IMAD 5.1 M） | 53.8 M | 16.2% |
| issue 全体 | 143.7 M | 約 43% |

**F2FP（FP32 → E4M3、`F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` 9.38 M）は XU ではなく ALU で実行されている**（XU の命令数が MUFU の数と一致する）。10-25 の probe 準備で未判定だった点はこれで確定した。

どの pipe も 45% を超えない。tensor と XU がちょうど同じ占有なのは FA3 論文の FP8・hdim128 の見積り（exp 需要 = MMA 需要）どおりで、**kernel は throughput ではなく依存と同期で律速されている**。

### 4.2 stall の内訳（PC sampling 40,484 sample）

| 理由 | 割合 | 主な場所 |
|---|---:|---|
| barrier | 20.0%（未 issue 12.7%） | `WARPGROUP.DEPBAR.LE` ×2（PV QGMMA 直後、9.9%）、named barrier 後の `UMOV`（ping-pong の受け渡し、4.6%） |
| wait（固定 latency 依存） | 19.3% | `MUFU`（13.4% のうち 44%）、`FMNMX`（10.5% のうち 37%） |
| long_sb | 14.7% | producer warp の TMA barrier 待ち loop（`SYNCS.PHASECHK.TRYWAIT` → `@!P1 BRA`、8.6%）。consumer が律速で producer は満杯待ち |
| selected（issue 中） | 14.5% | |
| short_sb / mio | 7.9% / 5.7% | `PRMT`（FP8 P の register 並べ替え）、LDSM/STSM（V の転置） |

### 4.3 softmax 区間の命令数（warp-tile あたり、112 要素/thread）

| 命令 | 数 | pipe | cycle |
|---|---:|---|---:|
| MUFU.EX2 | 114 | XU（8 cycle/命令） | **912** |
| FMNMX | 131 | ALU（2 cycle/命令） | 262 |
| F2FP E4M3 | 63 | ALU | 126 |
| PRMT、ISETP、LOP3、FSEL、SEL、整数 | 約 122 | ALU | 244 |
| FFMA（x·scale − max） | 116 | FMA（1 cycle） | 116 |
| FADD（行和） | 112 | FMA | 112 |
| FMUL（O の rescale 64 ＋ α） | 74 | FMA | 74 |
| その他（IMAD、SHFL、LDL/STL、barrier、分岐） | 約 177 | 混合 | |
| 合計 | 909 | | |

tile あたりの周期は 4,200〜4,600 cycle（elapsed ÷ 149.7 tile/SM、clock 1.71〜1.86 GHz の幅）で、M1 の固定費 20% を除くと KV tile あたり 3,400〜3,700 cycle。ping-pong では周期 = 自 WG の GEMM 896 ＋ softmax 区間なので、**softmax 区間は 1 WG あたり約 2,500〜2,800 cycle**。XU 912 ＋ ALU 632 ＋ FMA 340 の直列和 1,884 に、latency と barrier 待ち 700〜900 を足した値である。過去の結果とも整合する：

- lazy rescale（FMUL −64）は −0.4 µs/層：**FMA pipe の命令は隠れており、減らしても効かない**。
- regs232（spill 除去、LDL/STL −5）は −1.5 µs：同上。
- e4（木構造の行最大値、命令追加）は +1.2〜+6%：**命令を足すと必ず負ける**。
- PV 待ち対策（checksum/PRMT 追加）は +10〜+17 µs：同上。

**結論：FA3 で効くのは XU と ALU の命令を減らす変更だけ**で、重なりを増やす変更は効かない。

### 4.4 M1（40172404753、1860 MHz）：KV 反復あたり 15.4 ns、M-tile あたり 37.8 ns

shape4096 では I=19,760、T=2,048 で、**M-tile の固定費が 20%（77 µs/層）**、残り 80% が KV block ごとの費用。固定費は 1 M-tile あたり約 5 µs（SM あたり 15.5 M-tile）で、KV 反復 2.4 回分に相当する。中身は ping-pong の fill/drain、対角 block の mask と半分の無駄、epilogue（FP8 group-128 出力）、tile scheduler の推定で、PC sample を epilogue 区間に限定した再集計で確定できる。

### 4.5 FA3 の方策

**F1. exp を f16x2 で計算する（最大）。** `softmax.h` の `scale_apply_exp2` は `exp2f`（MUFU.EX2 FP32）を要素ごとに呼ぶ。P はこの直後に `convert_type_out` で E4M3 に丸められる（`mainloop_fwd_sm90_tma_gmma_ws.hpp` 1361、1412、1531 行）。`ex2.approx.f16x2` に置き換えると MUFU は 114 → 57（XU 912 → 456 cycle）で、E4M3 への変換は `cvt.rn.satfinite.e4m3x2.f16x2` で命令数が変わらない。FP32 → f16x2 の変換（F2FP 56、ALU +112 cycle）が増える。引数 x·scale − max を f16 に丸めるため、P の相対誤差は |x| ≤ 8 で 2^-8、|x| ≤ 16 で 2^-7.5 程度になり、E4M3 の丸め（2^-4）より 6〜11 倍小さい。行和は f16 の P から FP32 で積算する。見込みは softmax 区間 −344 cycle（−13%）、MUFU の wait stall（MUFU sample の 44%）が比例して減れば −20%。KV tile の周期では −10〜−15%、固定費 20% を除いて **FA3 −8〜−16%（−35〜−70 µs/層、−3.5〜−7 ms/反復 raw）**。bit 一致ではなく「新しい量子化点を増やさず、既存の E4M3 丸めより小さい精度差」なので、精度方針の判断が要る。

**F2. M-tile 固定費（20%）の分解。** PC sample を epilogue と対角 tile に限定して集計し、fill/drain・mask・epilogue のどれが大きいかを確定する。epilogue の FP8 group-128 出力（SHFL 1.7 M を含む）が主なら、出力を BF16 のままにして次の quant に任せる案（fp8-output.patch の逆）を candidate にする。見込みは **−1〜−3 ms raw**。

F1 と F2 で **FA3 −5〜−10 ms raw、電力係数込みで −2.5〜−5.5 ms**。

## 5. MegaMoE（142.5 ms、56.7%）：残る手は配置だけ

10-24 の timeline（bench 40057117293、1 層 1424 µs）での非 MMA 区間は FC2 epilogue 182〜209 µs、combine 78〜81、FC1 epilogue 46〜50、dispatch 23〜29、rank 間待ち 12〜75 で、合計約 350〜440 µs（23〜29%）。FC2 epilogue は local のみでも 94 µs（268 MB の BF16 combine payload を 2.85 TB/s で書く時間）で、残りは remote 分が NVLink に乗る分である。

重ねる方向はすべて実測で棄却済み：WG 独立化（+22〜+36%）、ping-pong（SMEM 帯域 125%）、FC2 interleave（+8%）、bulk scatter（+1.3〜+4.9%）、FP32 事前合算（serving −5.8%、bytes 収支で 2〜4 倍の負け）、L2 hint prefetch、mma-order rotation。FC2 epilogue は書き込み量で決まるので、**payload（token あたり top-8 × 4096 × BF16 = 64 KiB）を減らす変更しか残っていない**。精度方針の下で残る手段は remote 行を減らす expert 配置（P1'）で、JSON の差し替えだけ、出力は bit 一致。

必要なのは Prefill の rank 別 routing 統計である。`decode-routing-capture.patch` は Decode 用で source EP rank 別に JSONL を書く。Prefill 側は同じ形で `megamoe` の pre-dispatch 境界に差し込み、**次の検証 run の profile pass でだけ**有効にする（normal pass には入れない）。上界は CPU で計算し、−2 ms に満たなければ捨てる。見込み **−2〜−5 ms raw**。

EP2 Prefill（2 GPU worker × 6）は remote 比率を 3/4 → 1/2 にするが、rank あたりの expert 重みが 62 → 124 GB になり H200 141 GB に収まらない。採らない。

## 6. 小 kernel（約 15 ms、6%）、draft、dense

- **S1. 融合**（P3）：qknorm_rope 66 µs、fused_add_rmsnorm 32、norm_quant 32、pre_dispatch 16、router gate 15 ＋ triton 6、mask/copy 5。4032 × 4096 × BF16 = 33 MB の 1 pass は HBM 4 TB/s で 8 µs なので、qknorm_rope は床の 8 倍、norm 系は 4 倍。①add＋RMSNorm＋group quant＋pre-dispatch を 1 pass、②qknorm＋RoPE＋FP8 Q を 1 pass（既存 `prefill-qkv-fast-fp8` の延長）。wide qknorm が負けた原因（CTA 数 1/8、97 register）を踏まえ、CTA 数は維持する。見込み **−4〜−7 ms raw**、bit 一致を要求できる。
- **S2. draft extend を FP8 で**：BF16 GEMM 3.9 ms/反復。greedy（temperature 0）では検証が target の argmax を取るため、**draft の精度は出力に一切影響しない**（受理長だけが変わる）。ASPIRE の試験（759227b、36538766257）は score 中立で受理長の記録はない。見込み **−1.5〜−2 ms raw**。`--speculative-draft-model-quantization fp8` の flag だけ。受理長を eagle-acceptance.json で確認する。
- **S3. dense**：−1〜−2 ms raw。他の build に同乗。

## 7. Decode step −3〜−5%（10,000 で Decode が蓋にならないため）

running 1 件あたり 0.57 ms の傾きは、verify attention の KV 読み（44 × 4100 token × 96 KB = 17 GB/step を 6.7 ms、HBM の 54%）と dense の行数で決まる。
- **D2. verify attention**：p128split2-s3-fp8 が 71.6 µs/層。kernel 内 merge の候補は全部失敗（相対 L2 0.7315、実装不具合）。split の partial を global に書かず 1 CTA で FP32 積算する形を直す。見込み −1.4〜−1.9 ms/step。
- **D1. MegaMoE 供給**（dispatch 集計 19 → 8 µs、FC2 chunk 順の combine）：−1〜−2 ms/step。
- S2 の draft FP8 は Decode の draft 3 step にも効く（未計測）。

## 8. good 率（+1.0〜+1.3%）

[§6](2026-10-26-firmus-9120-profile-and-request-stats.md) の router 修正（約 12 件、全要求 TTFT −0.2 s）と Prefill の立ち上がり整列（約 11 件）で +0.4%。残り約 108 件は 12 rank の容量である。開始 2 s は Decode 4 GPU が空いているので、**Decode worker が開始直後の要求を自前で prefill する**（通常 scheduler の extend 経路を Decode の event loop に許可し、router が最初の N 件を Decode へ直送する）と 4 rank × 11 件 = 約 44 件を救える（+0.7%）。Decode の running は数件なので ITL への影響は小さいが、SGLang の Decode scheduler に extend batch を組む改修で工数は大きい。

## 9. 積み上げと判定

| 項 | raw −ms | 電力係数後 −ms | 種別 |
|---|---:|---:|---|
| F1 FA3 exp f16x2 | 3.5〜7 | 2〜4 | build（sgl-attn 候補） |
| F2 FA3 固定費 | 1〜3 | 0.5〜2 | build |
| S1 小 kernel 融合 | 4〜7 | 2〜4 | overlay |
| P1' 局所性配置 | 2〜5 | 1〜3 | JSON（統計 run 要） |
| S2 draft FP8 | 1.5〜2 | 1 | flag |
| S3 dense | 1〜2 | 0.5〜1 | overlay |
| 合計 | **13〜26** | **7〜15** | |

Prefill 251 → **236〜244 ms**（供給 +3〜+6.5%）。good +1.0〜+1.3% を足して **9,500〜9,900**。10,000 は **F1 と S1 が上側に着地し、good が 0.99 近くまで上がり、Decode が −5% を確保した場合**に届く。中央値では届かない。MegaMoE に大きな 1 手が無いこと、Prefill が 700 W の cap で動いていることが理由である。

## 10. 進め方（投入は build 1 回・serving 1 回）

1. 玄界 1 GPU（`kernels/fa3-verify` の prefill candidate と `Genkai kernel probe`）で F1 の f16x2 変種を control と対で計る。数値 gate は E4M3 の P と出力の差を記録し、採否は µs/層。同じ job で S1 の融合 kernel を bit 一致で計る。
2. F2 は既存の M2 ncu-rep（`fa3probe/m2`）の PC sample を区間別に再集計するだけで、GPU は不要。
3. 採用候補を 1 回の build にまとめ（F1/F2 は image 内の FA3 binary、S1/S3 は overlay、S2 は flag）、Firmus serving 1 回で in-job selection と profile pass を取る。profile pass で Prefill の rank 別 routing を記録し、P1' は次の run に回す。
4. 台帳：Prefill 反復 ms、FA3/MegaMoE/小 kernel の µs/層、clock、受理長、Decode step、good 率。

## 11. 決定が必要な事項

1. **F1 の精度方針**：exp の引数と P を f16 で扱う（E4M3 丸めより 6〜11 倍小さい差、新しい量子化点なし）を許容するか。
2. **S2 draft FP8**：greedy では出力不変。受理長の低下がなければ採用してよいか。
3. **§8 の Decode 自前 prefill**：+0.7% に対して scheduler 改修を行うか。
