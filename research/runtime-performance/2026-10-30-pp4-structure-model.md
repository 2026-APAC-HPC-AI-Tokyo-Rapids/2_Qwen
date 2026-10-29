# 2026-10-30 PP4 の kernel より上の律速：校正した系の模型、段と GPU の配置

読み取りのみの解析と CPU 検証（GPU job・Actions・ssh なし）。出典は run
40883389891（9392.85、PP4 c225）、40825417205（8952.69、PP4、nsys）、40630160781
（9235.28、EP4）と、telemetry のある Firmus run 16 本の archive。

## 1. 要約

- **系の模型**（[`pp4-serving-model.py`](pp4-serving-model.py)、結果
  [`pp4-serving-model-20261030.json`](pp4-serving-model-20261030.json)）は
  40883389891 を 9404（+0.12%）、構成の違う 40825417205 を 8992（+0.44%）で再現する。
  同じ系統の EP4 模型（`scratchpad/sys/des.py`）は 40630160781 を 9230（−0.06%）で再現する。
- **周期は律速 GPU の電力上限での計算時間でほぼ決まる。** 3 pipeline の周期 120.8 /
  120.2 / 123.9 ms は「24 層の段のうち最も遅い GPU の SM clock」に反比例する
  （周期 × clock = 174.4〜175.4 M cycle）。CPU・同期・転送の残りは 1 周期 0.5〜1 ms 以下。
- **段の配分 24/24/24/22 は層の粒度（1 層 ≈ 4.2%）の中で既に均衡している。** 最終段は
  他より 1〜2% 軽い。この軽さを、各 pipeline で最も遅い GPU に載せる（stage→GPU の
  並べ替え）のが唯一の残りで、**Prefill 容量 +0.35〜0.6%、score +0.3〜0.6%**。実装した。
- 3 要求 microbatch、release（router の保持）の短縮、router の方針、Decode の上限 48 は、
  現状では得にならない（§5〜§9）。concurrency は 225 のまま（215 と差 0.3% 以内）。
- **Prefill だけ速くしても Decode がすぐ律速になる。** Decode の上限 44/rank での天井は
  約 50.7 req/s（score 約 9,650 相当）。他の worker の目標（MegaMoE −5%、dense −3%、
  Decode −5%）が全部入って模型 9,920、10,000 には Prefill −6.5% と Decode −5% が要る。

## 2. 模型と校正

client（225 slot の一斉投入と再送）、router の保持（Prefill の通知まで最初の token を
止める）、Decode（4 DP lockstep、要求ごとの受理）は EP4 模型の系統。Prefill は 3 本の
pipeline で、周期 P_w ごとに 2 要求の microbatch が入り、4 段後に終わる。形成から開始
（1.015 P）、開始から最終段の終了（3.907 P）、終了から通知（4.041 P）、終了から Decode の
受け入れ（230 ms）は前 worker の loop 模型（depth 1、stage 0 の admission、最終段の
result 先行、copy engine）の値。Decode の step は 40883389891 の Decode log から
`running × accept len / gen throughput` で求め、n = 41.9/rank で 37.4 ms、局所の傾き
0.72 ms/要求（n=37〜44 の中央値 32.9→38.8 ms）。

| | 40883389891 実測 | 模型 | 40825417205 実測 | 模型 |
|---|---:|---:|---:|---:|
| score | 9392.85 | 9404 | 8952.69 | 8992 |
| 一斉投入の合格 / TTFT 不合格 | 98 / 127 | 100 / 126 | 96 / 139 | 84 / 158 |
| ITL 平均 / p99（ms） | 15.30 / 18.00 | 15.16 / 18.06 | 16.14 / 18.99 | 15.70 / 18.47 |
| 定常 TTFT 中央値（s） | 1.48 | 1.47 | 1.72 | 1.78 |
| 定常の完了率（req/s） | 49.28 | 49.29 | 47.28 | 47.54 |

限界：模型の Prefill は厳密な FIFO で、開始後 2.5〜3.5 s に送られた第 2 波の TTFT を
約 0.2 s 長く見る（実測は中央値 2.57 s、最大 **2.99 s**。3 s の崖の手前にいる）。
差分として読み、絶対値は ±0.5%。

## 3. 段の均衡（問 1）

nsys（40825417205 の prefill-0、45 microbatch）の層ごとの開始間隔は stage 0/1/2 で
4.95 / 4.98 / 5.03 ms、MegaMoE は 2.79 / 2.80 / 2.83 ms。stage 3 は 22 層 4.91 ms ＋
lm_head・draft の後段 10.6 ms で計 115.3 ms（他は 118.3〜120.2）。

- 電力上限では段の時間は「仕事量 ÷ その GPU の上限での clock」。40883389891 の
  telemetry で、最終段の GPU は同じ GPU の EP4（対称負荷）での clock より 1〜2% 高く、
  24 層の段は 2〜5% 低い。最終段は 24 層の段より約 1〜2% 軽い。
- 1 層の移動は 4.2% なので、24/24/23/23（最終段が最重になる）も 25/24/23/22 も悪化する。
  SGLang の `SGLANG_PP_LAYER_PARTITION`（既に使用中）を pipeline ごとに変えても同じ。
- 残る手は **最終段（軽い段）を最も遅い GPU に置く** こと（§7、実装）。

## 4. 周期の残り（問 2）

copy engine 前（40825417205、profile）：stage 2 の計算 120.2 ms、NCCL のみ 2.6 ms、
空き 0.4 ms。copy engine 後（40883389891）の周期の平均は 121.6 ms で、前 worker の
予測 121.8 ms と一致した。

- 周期の減り（log の周期 123.4/123.3/125.0 → 120.8/120.2/123.9、平均 −2.3 ms）は、
  kernel の短縮（FA3 varlen/direct-O と MoE 入力の融合、約 1.9 ms、κ≈1）と、
  NCCL のみの時間 2.6〜4.1 ms の除去（待ちの除去は κ 0.36〜0.6 で 1〜2.4 ms）の和
  2.9〜4.3 ms よりやや小さい。同じ GPU の SM clock は 1〜1.5% 下がった
  （g1:0 1467→1444 MHz）。待ちが消えて電力上限での計算に置き換わった。
- 律速段は周期のほぼ全部を計算している（周期 × clock がどの pipeline でも
  174〜175 M cycle）。CPU の起動（25 ms/microbatch）、result・consensus は depth 1 の
  loop の余裕（約 90 ms）の内側にあり、周期に出ていない。残りは段の境の空き
  0.5 ms 前後で、消しても κ 0.36〜0.6 で 0.2〜0.3 ms（0.2%）。

## 5. microbatch の大きさ（問 3）

実 routing（`kernels/probes/megamoe-local-ep1-routing.npz`、3 層 × 8 要求）で、
expert ごとの行を 64 行単位（tail の WG1 skip を含む現行）に切り上げた有効率は
1 要求 87.8%、2 要求 93.65%、3 要求 95.75%、4 要求 96.8%。3 要求で MegaMoE の
GEMM 量は 1 token あたり −2.2%。MegaMoE（層の 57%）と microbatch ごとの固定分
（0.5〜1 ms）を合わせて **1 token あたり −1〜3%**（中央 −1.9%）。dense GEMM の wave
端数は −0.8%×（層の 19%）、FA3 は要求ごとなので変わらない。KV は inflight 10→15 要求
（131072 token の pool に 60k）で入る。

代わりに pipeline の遅延が 1.5 倍（形成〜通知 1.08 s → 1.6 s）になり、一斉投入の
合格が 100→81 に減る。Little の法則で Prefill 内の要求が約 20 増えるので c は 240 が要る。

| 1 token の短縮 | c225 | c235 | c240 | 現行 c225 比 |
|---|---:|---:|---:|---:|
| 2.0% | 9325 | 9372 | 9389 | −0.2% |
| 3.0% | 9351 | 9404 | 9414 | +0.1% |
| 4.5% | 9380 | 9443 | 9454 | +0.5% |

見込み（1.9%）では負け。前 worker の判断（3 要求は不利）と同じ結論で、根拠を kernel の
padding から付け直した。chunk の大きさも同じ理由で 2 要求のまま。

## 6. concurrency と一斉投入（問 4）

一斉投入の 225 件のうち合格は「Prefill の供給 × (3 s − 最初の token までの遅れ)」で
決まり（実測 98、0.5 s ごとに 22〜27 件が 0.75〜5.7 s に一定の速さで出る）、
不合格はすべて容量で決まる（127 件、全て開始 0.08 s 以内の送信）。

| c | 205 | 215 | 225 | 235 | 240 |
|---|---:|---:|---:|---:|---:|
| 模型（現行の配置） | 9348 | 9408 | 9404 | 9381 | 9348 |
| 模型（§7 の配置） | 9365 | 9431 | **9458** | 9429 | 9405 |

- c225 を維持。c を上げると、第 2 波（実測で最大 2.99 s）が崖を越える。
- router の保持の短縮（PP4 の release consensus の lag 2→1 に相当する −0.12 s、
  −0.24 s）は、一斉投入の合格 +6〜12 と引き換えに全要求の ITL が +0.6〜1.1 ms 増え、
  模型で ±0（9457 / 9440）。EP4 での実測（−0.29%）と同じ構図で、行わない。
- 期限を見た順序、意図的に落とす、benchmark の時刻を見るものは
  `system-level-analysis.md` §3.4 の通り効果が無いか規約上不可。新しい合法の機構は無い。
- 既存の start alignment は DP lockstep 用で、PP4（pipeline ごとに 1 scheduler）には
  作用しない。

## 7. GPU の配置（問 5、実装）

telemetry（1 Hz、busy window の平均 SM clock）。EP4 の Prefill は worker 内の 4 GPU が
同じ仕事を lockstep でするので、clock の差は GPU の性質。10 run の平均（sd）：

| GPU | g1:0 | g1:1 | g1:2 | g1:3 | g1:4 | g1:5 | g1:6 | g1:7 | g4:0 | g4:1 | g4:2 | g4:3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EP4 MHz | 1504 (15) | 1508 (10) | 1527 (10) | **1481** (6) | 1520 (8) | 1504 (8) | **1494** (6) | 1506 (8) | 1515 (11) | 1487 (7) | **1476** (7) | 1495 (8) |
| PP4 40883389891 段 | 0 | 1 | 2 | 3 | 0 | 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| PP4 MHz | 1444 | 1464 | 1475 | 1456 | 1466 | 1466 | 1459 | 1498 | 1433 | 1435 | 1416 | 1445 |

- 順位は run によらない。g1:6 は g1:7 より 10 run 全部で 3〜23 MHz 低く、g4:2 は g4:3 より
  全部で 14〜28 MHz 低い。prefill-2 の 3% の遅れは g4:2（24 層の段 2）が律速であるため。
- 各 pipeline の律速は「24 層の段の最も遅い GPU」で、prefill-1 は g1:6、prefill-2 は g4:2
  （どちらも段 2）。prefill-0 は最も遅い g1:3 が既に最終段。
- **最終段を最も遅い GPU に移す**（prefill-1・2 の CUDA_VISIBLE_DEVICES を 0,1,3,2）と、
  律速は次に遅い GPU に移る：prefill-1 +0.5〜0.7%、prefill-2 +0.5〜1.1%
  （EP4 の clock 差と PP4 の clock 差の範囲）。Prefill 容量 +0.35〜0.6%、模型 c225 で
  9404 → 9458（+0.57%）。最終段は 1〜2% 軽いので、遅い GPU（差 0.7〜1.1%）に載せても
  律速にならない。
- Decode の 4 GPU（g4:4〜7）も 692 W の上限にいる（1832〜1901 MHz）。Prefill の負荷を
  かけた記録が無いので、Decode と prefill-2 の node 内の入れ替えは賭けになる。Decode も
  上限 44/rank の天井（§9）に近く、遅い GPU を受け持たせる余裕が無い。行わない。

## 8. router（問 6）

3 pipeline の待ち行列は定常で各 4.5 要求と均等（min-load）。全 pipeline が飽和している
間は、配分は総供給を変えず、遅い pipeline の 3% は待ち時間に 4.5 × 62 ms × 3% ≈ 8 ms
出るだけ。一斉投入でも各 pipeline の 75 件が合格可能数（約 33）を超えるので、合格数は
供給の和で決まる。変更の余地は無い。

## 9. Decode の容量（問 7）

- 1 要求は 199 / 2.224 ≈ 89.5 verify step。running/rank は
  n = λ × 89.5 × step(n) / 4。log の step(n) は n=40 で 35.9、44 で 38.8 ms、40883389891 は
  n=41.9（中央値 42）で、上限 44 に 21% の標本（58/277）で当たっている。
- **上限 44/rank の天井は λ = 176 / (89.5 × 38.8 ms) ≈ 50.7 req/s**（score 約 9,650 相当）。
  現在の Prefill 供給 49.3 req/s の 3% 上にある。
- 10.5k（定常 ≈ 55 req/s）には n ≈ 48/rank が要る。傾き 0.3 ms/要求なら step 39.4 ms、
  ITL p99 ≈ 18.2 ms で成り立つ（Decode −5%、上限 48 = graph bs 46/48 と
  `decode_max_running_requests` 192、c235〜240）。傾きが log の 0.72 なら n が発散し、
  上限 48 でも 10.2k 止まり。profile（10-29 の decode-step 稿）の n=40〜44 の傾きは
  0.08〜0.24 ms で、log と食い違う。**傾きは次の serving の Decode log（Decode が
  速くなった後の n と step）で決める。** 現状の速さでは上限 48 は模型で ±0 なので入れない。

## 10. 組み合わせの予測（模型、§7 の配置込み、最良の c）

| 条件（周期の倍率） | c | score | 備考 |
|---|---:|---:|---|
| 現行（実測 9392.85） | 225 | 9404 | |
| §7 の配置 | 225 | 9458 | +0.6% |
| ＋ MegaMoE −2%（×0.989、four-stage 採用相当） | 225 | 9501 | ITL 不合格 9、Decode 律速が始まる |
| ＋ MegaMoE −5%、dense −3%（×0.958） | 225 | 9541 | ITL 不合格 15。Decode が天井 |
| MegaMoE −2%、Decode −5% | 215 | 9617 | |
| MegaMoE −2%、dense −3%、Decode −5% | 215 | 9743 | |
| MegaMoE −5%、Decode −5% | 215 | 9788 | |
| MegaMoE −5%、dense −3%、Decode −5% | 215 | 9918 | |
| Prefill ×0.935、Decode −5% | 225 | 10065〜10129 | 傾き 0.72〜0.3 |
| Prefill ×0.90、Decode −5%、上限 48 | 240 | 10166〜10525 | 傾き 0.72〜0.3 |

MegaMoE は PP4 の層の 57%、dense/attention は 43% として周期に換算（κ≈1）。Decode が
速くなると Little の法則で Decode 中の要求が減るので、最適な c は 215 に下がる。

## 11. 実装：最終段を最も遅い GPU へ

- `configs/firmus-gpu-sm-clock.json`：Prefill 12 GPU の EP4 での平均 SM clock（UUID で
  引く。`scripts/rank-firmus-gpu-clocks.py` で archive から再生成でき、同じ表になる）。
- `scripts/order-firmus-pp4-gpus.py`：worker の 4 GPU の UUID（nvidia-smi）を表で引き、
  最も遅い GPU が今の最終段の GPU より 5 MHz 以上遅ければ末尾に移す。表に無い GPU が
  あれば並びはそのまま。
- `sglang.sbatch` の worker.sh（PP4 Prefill の分岐）で CUDA_VISIBLE_DEVICES に設定し、
  既存の `gpu=` の log 行に並びが残る。provision は `.prefill.gpu_sm_clock` があるときだけ
  表と script を置く（`configs/firmus-framework.json` は対象外）。
- job 内の追加時間：worker の起動時に nvidia-smi 1 回と python 1 回（1 s 未満、3 worker
  並行）。image の再 build は不要。
- 判定：prefill-0 は並びが変わらない対照。次の run の log の周期比
  prefill-1/prefill-0 と prefill-2/prefill-0 が、40883389891 の 0.995 / 1.026 から
  下がること（期待 0.990 / 1.014）、telemetry で g4:2 が最終段の clock に上がること。

## 12. 実装しなかった手（順位順）

1. Decode の上限 48/rank と c235〜240：Prefill が 6% 以上速くなり、Decode log の傾きが
   0.3 ms/要求程度と確かめられた後。
2. Decode と prefill-2 の node 内入れ替え：g4:4〜7 の Prefill 負荷での clock が分かれば
   判断できる（prefill-2 は最大 +1〜2%、Decode は同じだけ遅くなりうる）。
3. 3 要求 microbatch：MegaMoE の 1 token あたりの短縮が 4% を超えると分かった場合のみ
   （c240 と組で +0.5%）。
