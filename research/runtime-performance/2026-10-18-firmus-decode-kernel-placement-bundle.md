# 2026-10-18: Firmus の Decode MegaMoE kernel 3 patch と Prefill 静的 expert 配置の束ね評価

## 1. 目的と結論

P3/D1（c192）では、throughput は Prefill の供給量で決まり、Decode は ITL 上限に張り付いている。そのため、Decode kernel だけを速くしても主に ITL>20 ms の不合格が減るだけである。Prefill の静的 expert 配置だけでは Prefill MegaMoE が約 8% 速くなったが、ITL 不合格が増えた（6418.50、[expert placement note](2026-10-18-megamoe-expert-placement.md)）。そこで、kernel bench で効果を確認済みの Decode 側 3 patch と Prefill 配置を 1 つの image・構成に束ね、参照構成の再実行と交互に 2 回走らせた。

結論:

- 束ね構成は **7251.58 / 7185.18**（平均 7218.38）good output tok/s だった。参照構成は 6521.98（38945855851）/ 6485.02（同日の再実行 R2）で、平均 6503.50 に対して **+11.0%** になる。束ね構成の低い方の run でも、参照の高い方の run より 10.2% 高い。同一構成の run 間ばらつき（約 ±3%、6521.98 対 6699.30）を大きく超えている。recorder を入れた統計 run の 6699.30 と比べても +7.7%。
- 主因は ITL>20 ms 不合格の解消で、82 / 106 件から 7 / 9 件に減った。ITL の平均は 17.3–17.5 → 16.1 ms、p99 は 20.3–20.6 → 18.9 ms。raw の output tok/s は 6837–6875 → 7402–7430（+8%）、req/s は 34.2–34.4 → 37.0–37.2 に上がった。TTFT>3 s 不合格は 114–140 → 104–126 件で、ほぼ横ばいである。
- profile pass の MegaMoE kernel 平均は次のとおり。Decode は 0.2415 / 0.2491 → 0.2126 / 0.2124 ms（−13%）、Prefill の主形状は 2.1635 / 2.1519 → 1.8205 / 1.8169 ms（−16%）。
- 数値 gate、official validation、静的配置の読み込み、sanity chat の内容、EAGLE3 の accept length は全 run で期待どおりだった（§4）。
- branch `perf/firmus-decode-kernel-placement-20261018` に置き、main には未 merge（merge 判断はユーザーが行う）。

## 2. 束ねた内容と commit

branch `perf/firmus-decode-kernel-placement-20261018` は origin/main 0064dd7 から作った。

| commit | 内容 |
|---|---|
| c3e94eb | `perf/firmus-expert-placement-20261018`（a61d08d）の merge。Qwen3 MegaMoE 経路が `ExpertLocationDispatchInfo` を topk に渡す変更、`configs/firmus-prefill-expert-location.json`、`prefill.expert_location` を Prefill だけの `--init-expert-location` へつなぐ配線、server-info 検査、sanity chat |
| d7bd3af | `perf/firmus-decode-kernel-20261018`（901288a）の merge。bench overlay の 3 patch、それらの上に rebase した timeline overlay、summarize script |
| 0374c55 | A1。3 patch を `patches/deep-gemm-bench/` から `patches/deep-gemm/` へ無変更で移した（`git mv`）。`containers/build-sm90-megamoe.py` は既存 5 patch の後に bench と同じ名前順（hoist → l1-quant → dispatch-pull-overlap）で適用する。build context と Dockerfile へのコピー、build record の `swap_ab_hoist_patch_sha256` / `l1_quant_patch_sha256` / `dispatch_overlap_patch_sha256` を追加し、`scripts/check-firmus-image.sh` と image test の header pin を `135a43be…` に更新した。`[require-build]` |
| c272f7e | A2 の 1 回目。`configs/firmus-framework.json` の `images.sglang` を A1 の image に pin した（再利用時、provision は選ばれた image が config の pin と一致することを要求する） |
| bc207d6 | A2 の再実行。router の待機を `/health` から `/readiness` に変更した（§5） |

patch の sha256 は、hoist `88351653…`、l1-quant `012220e3…`、dispatch-pull-overlap `450d2749…` で、901288a の overlay と byte 単位で同じである。header は、元の 0.1.5 header `a53d83ca…` に 8 patch を当てると `135a43be…` になる。これは元 header と e29dfde image 内の header の両方からローカルで再現し、A1 の build record とも一致した。A1 の build log では、Decode は 168 registers・spill 0、Prefill は 96 registers・spill 40/40 byte で、image e29dfde と同じだった。

push 前の検査として、shell 構文、行継続の検査、kernel 実験 tag の検査、ruff、pytest 全件を実行した。さらに、15 patch の SGLang chain を pin した source fc275674 に適用し、`check-firmus-staging` / `megamoe-source`（placement で写像した expert id を含む）/ `latency-source` を通した。DeepGEMM の 8 patch chain が元 header に適用できること、timeline overlay がその上に適用できることも確認した。

kernel 側の根拠: [mainloop pipeline note](2026-10-18-megamoe-decode-mainloop-pipeline.md) と [epilogue/dispatch note](2026-10-18-megamoe-decode-epilogue-dispatch.md)。Decode-128 は 229.5 → 200.8 µs で、全 12 case の出力 hash が一致し、Prefill SASS は byte 単位で同じ。

## 3. Run 一覧と比較

条件はすべて、c192、P3/D1、各 worker TP4/DP4/EP4 MegaMoE、decode cap 128、EAGLE3 steps 3 / draft 4、公式 60/120/30 client、`[nodes:2] [profile]`（通常計測と profile 専用 client は別の pass）である。error はどの run でも 0 件、official validation はすべて valid だった。

| run | commit | Actions | image | score | eligible 割合 | TTFT>3 s | ITL>20 ms | TTFT avg / p99 (ms) | ITL avg / p99 (ms) | raw out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R（参照） | a6acf5f | [38945855851](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38945855851) | e29dfde4… | 6521.98 | 0.9486 | 140 | 82 | 2037.8 / 5119.9 | 17.3 / 20.3 | 6875.4 | 34.38 |
| R2（参照の再実行） | 0b18a7c（main 0064dd7 + 空 commit） | [39069954937](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39069954937) | e29dfde4…（再利用） | 6485.02 | 0.9484 | 114 | 106 | 2033.8 / 5003.8 | 17.5 / 20.6 | 6837.5 | 34.19 |
| A1（束ね） | 0374c55 | [39070273796](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39070273796) | 7c6286e5…（新規 build） | **7251.58** | 0.9759 | 104 | 7 | 1871.1 / 4647.5 | 16.1 / 18.9 | 7430.2 | 37.15 |
| A2（束ねの再実行） | bc207d6 | [39078761629](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39078761629) | 7c6286e5…（再利用） | **7185.18** | 0.9706 | 126 | 9 | 1891.3 / 4638.9 | 16.1 / 18.9 | 7402.4 | 37.01 |

実行順は R2 → A1 → A2 で、Firmus の同じ 2 node（g1/g4）で直列に走った。R2 は空 commit を push しても `paths-ignore` のため workflow が起動しなかったので、同じ commit に対して `workflow_dispatch`（run_firmus、node_count 2、profile、require_build false）で起動した。条件の切り替えには使っていない。

serving 内部の状態:

| run | Decode 平均 running/rank | accept len | Prefill queue 平均 | queue が空でない割合 |
|---|---|---|---|---|
| R | 29.2 | 2.208 | 0.87–0.93 | 40.1–40.3% |
| R2 | 29.2 | 2.207 | 0.75–0.81 | 32.9–34.2% |
| A1 | 29.5 | 2.210 | 0.75–0.79 | 32.3–33.0% |
| A2 | 29.4 | 2.209 | 0.81–0.85 | 36.5–37.8% |

Decode の同時実行数（cap 32/rank 付近）と accept length は変わっていない。同じ slot 数で 1 step が短くなったため、ITL が下がり、req/s が上がった。Prefill の queue 統計は R/R2 間のばらつきの範囲内である。

## 4. kernel 時間と検証

profile pass の nsys kernel summary から求めた MegaMoE の平均。Prefill は 3 worker の call 数加重平均である。

| run | Prefill 主形状（block 128×256） | Decode |
|---|---|---|
| R | 2.1635 ms | 0.2415 ms |
| R2 | 2.1519 ms | 0.2491 ms |
| 配置のみ（39053527142、参考） | 1.9871 ms | 0.2469 ms |
| A1 | 1.8205 ms | 0.2126 ms |
| A2 | 1.8169 ms | 0.2124 ms |

- Decode は −13%（R/R2 平均 0.2453 → 0.2125 ms）。kernel bench の Decode-128 で得た −12.5% とほぼ同じ。
- Prefill は −16% で、配置のみの run の −8% より大きい。Prefill の kernel SASS は image e29dfde と byte 単位で同じなので、kernel patch の効果ではない。原因は確かめていない。候補は profile 窓の batch 構成と、Prefill GPU の電力制限下での clock 差。A1 と A2 は 0.2% の差で揃っている。
- 配置ありの run（配置のみ、A1）では、Prefill worker に SwapAB 形状（block 64×128、8 stage）の MegaMoE instance がわずかに現れた。A1 の prefill-0 では 94 call・平均 3.29 ms、配置のみの run では各 worker 564–752 call・約 3.45 ms。R/R2 には無く、A2 でも profile 窓には現れなかった。上の表の Prefill 値はこれを除いた主形状である。含めた場合、A1 は 1.8276 ms になる。

全 run で確認した項目:

- MegaMoE 数値 gate: 全 run で PASS（`SM90 MegaMoE EP4 numerical gate; original FP8 scales, late FP32 combine, changing masked Decode graph inputs`）。
- image: R/R2 は e29dfde4…（header `0e34a083…`）、A1/A2 は 7c6286e5…（header `135a43be…`、build record に追加した 3 patch の sha も入っている）。A2 が A1 の image を使ったことを `sglang-image.sha256` で確認した。
- 配置: A1/A2 の Prefill 3 worker は `init_expert_location=…/prefill-expert-location.json` で、4 DP state すべてが `ep_dispatch_algorithm=static`。各 Prefill log に `init_expert_location from init_by_mapping` が 4 行ある。Decode は `trivial` / `None` で、そのログは 0 行。R/R2 は全 worker が trivial。
- sanity chat（greedy、Prefill 経由）: A1 と A2 の応答は配置のみの run と同じ文字列だった（"The capital of France is Paris. It is historically important because …"）。
- EAGLE3 の accept length: 2.207–2.210 で、全 run で変化なし。

## 5. A2 の 1 回目の失敗と修正

A2 の 1 回目（[39074570813](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39074570813)、c272f7e、Slurm 24418）は、AIPerf の前に失敗した。4 worker が healthy になり、router の `/health` も 23:17:42 に 200 を返した。しかし 1 秒後の sanity chat が 503 になった。router log には pd_router の `No prefill workers available` が出ていた。sglang-router 0.3.2 の `/health` は、単に生存を返すだけの handler である。worker の登録は非同期の workflow（`local_worker_registration`）で進むので、`/health` が 200 でも登録が済んでいるとは限らない。placement の変更で入った sanity chat が READY 直後の最初の request になり、`set -e` 下の `curl --fail` で job が止まった。A1 と配置のみの run は、登録が先に終わったので偶然通っていた。

修正（bc207d6）: `sglang.sbatch` の router 待機を `/readiness` に変更した。pin した router source（`sglang_router-0.3.2.tar.gz`、sha256 `bdbea1d5…`、`src/server.rs`）では、PD mode の `/readiness` は、healthy な Prefill と Decode が 1 つ以上登録されるまで 503 を返す。`wait_http` は 503 を再試行する。READY が登録の分だけ遅れる以外、計測と serving の設定は変わらない（READY after は R2 361 s、A1 380 s、A2 364 s）。main の sbatch には sanity chat が無いので、この競合は main では表に出ていない。束ね branch を merge すると、この修正も一緒に入る。

## 6. 注意点と次の手

- 比較は束ね 2 run と参照 2 run（うち 1 run は同日の交互実行）である。差（+10–12%）は既知のばらつきの 3 倍以上あり、ITL 不合格の減少という機構とも合っている。ただし、Decode kernel 単独と配置単独の寄与は分けていない。配置単独では ITL 不合格が増えたので、2 つの変更は相補的に働いたと解釈している。
- Decode の ITL 余裕（p99 18.9 ms）ができたので、次は Decode の slot 数（cap、c）か Prefill の供給を増やす方向を、この image を基点に評価できる。
- branch: `perf/firmus-decode-kernel-placement-20261018` は残す。R2 branch は `archive/experiment/firmus-reference-rerun-20261018` tag にして削除した。元の 2 branch と mainloop pipeline branch の扱いは commit 本文と最終報告に記録する。
