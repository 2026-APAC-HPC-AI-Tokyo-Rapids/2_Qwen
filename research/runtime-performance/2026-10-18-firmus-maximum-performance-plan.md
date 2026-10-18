# 2026-10-18 Firmus の到達上限とその引き上げ計画（main 6c10548 = 7251.58 / 7185.18 を起点）

目的は Firmus H200×16・ISL 4000 / OSL 200・SLA（TTFT ≤ 3 s、要求平均 ITL ≤ 20 ms）の下で、精度を変えずに score（goodput req/s × 200）を可能な限り上げること。本稿は現行最良 A1/A2（[bundle note](2026-10-18-firmus-decode-kernel-placement-bundle.md)）の AIPerf 記録、worker ログ、profile pass の kernel-summary、kernel bench の timeline（[kernel timeline note](2026-10-18-megamoe-kernel-timeline.md)）から、(1) 何が throughput を決めているか、(2) 物理的な上限はどこか、(3) どの順に何を当てるかを決める。実装はまだ行っていない。

## 1. 結論

| 段階 | 内容 | 期待 score | 必要なもの |
|---|---|---:|---|
| 現在 | A1/A2 平均 | 7218 | — |
| Tier 1 | runtime のみ（Decode 上限、handoff 遅延、DP 同期）、既存 image | 7600〜7900 | serving 2〜3 run |
| Tier 2 | Prefill/Decode の kernel・patch を束ねた新 image | 8500〜8900 | kernel bench 数回 ＋ build 1 ＋ serving 2 run |
| Tier 3 | MegaMoE の geometry（host library 再ビルド）、FA3 prefill、EAGLE 段数 | 9300〜9700 | ビルド経路の整備 ＋ bench ＋ serving |

10000 には Prefill 1 要求あたりの GPU 時間を 293 ms → 230 ms 以下にする必要があり、Tier 3 まで全部当たって届くかどうかという位置にある（§3）。各 Tier は独立に score を押し上げるので、上から順に着手し、判定は同条件 2 run（run 間ばらつき ±3%）で行う。

## 2. 現状の収支（A1/A2）

### 2.1 閉ループの構造

c192 の閉ループでは throughput = c ÷ 1 要求の所要時間。A1/A2 は 192 ÷ 5.17 s = 37.1 req/s で、所要時間は TTFT 1.88 s ＋ 199 token × ITL 16.3 ms = 3.25 s。

要求の滞在場所（AIPerf の start/ack/end と Decode ログの平均。Little の法則で時間に換算、37.1 req/s）:

| 場所 | 滞在数 | 時間 | 根拠 |
|---|---:|---:|---|
| Decode running | 125（31.5/rank、上限 32） | 3.25 s | AIPerf ack→end、Decode ログ `#running-req` |
| Decode slot 待ち（KV 到着済み） | 19（4.7/rank） | 0.51 s | Decode ログ `#queue-req` |
| Decode KV 転送中 | 34（8.5/rank） | 0.92 s | `#transfer-req`（prealloc 9.5 を含む） |
| Prefill 計算中（batch 内） | 約 22 | 0.6 s | 3986 token の要求が 4096 token chunk を 2 iteration またぐ（[plan §3.1](2026-10-17-firmus-acceleration-plan.md)） |
| Prefill queue ＋ bootstrap 待ち | 約 10〜20 | 0.3〜0.5 s | 同上。A1 の Prefill ログには batch 行が無く再計測が要る |

Prefill 側滞在 67 要求のうち GPU を使っているのは 12〜22 要求分で、残り約 45 要求（1.2 s）は待ちである。ただし閉ループなので待ちを削っても、Prefill 供給か Decode slot のどちらかが空いていなければ throughput は増えない。

### 2.2 どちらが縛っているか

- **Decode は slot 上限に張り付いている。** running 31.5/rank（上限 32）、slot 待ち 19 要求。ITL は平均 16.3 / p99 19.0 ms で、上限 20 ms まで 1 ms 強の余裕がある。
- **Prefill は 90〜96% 稼働。** profile pass の kernel 稼働率 96.2%（prefill-0 device 0）。1 step（4027 token/rank）の GPU 時間は約 296 ms（§2.3）なので、12 rank の供給上限は 12 × 0.96 ÷ 0.293 s ≒ 39 req/s。通常 pass の 37.1 req/s はこれに近い。
- したがって **runtime だけで得られる raw は +5〜8%（≒ 40 req/s）が上限**で、それ以上は Prefill の kernel 時間を削るしかない。Decode 側は slot を増やす余地があるが、ITL の余裕 1 ms を食う。

### 2.3 GPU 時間の内訳（A1 profile pass、device 0、1 step あたり）

Prefill（4027 token/rank、1 step ≒ 296 ms、稼働 96%）:

| kernel | 1 層 | 1 step | 割合 | 備考 |
|---|---:|---:|---:|---|
| MegaMoE（BLOCK 128×256） | 1.828 ms | 171.8 ms | 58% | 1.24 TFLOP/層 → 680 TFLOPS。GEMM 区間は約 1.5 ms（throttled peak の 58%）、非重複部（起動 40 µs ＋ 末尾 95〜240 µs）が 9〜16% |
| FA3 prefill | 0.489 ms | 46.0 ms | 15.5% | 4k causal、FP8 Q/KV。約 270 GFLOP → 545 TFLOPS |
| dense GEMM（QKV＋O） | 0.451 ms | 42.4 ms | 14.3% | 約 1270 TFLOPS、改善余地なし |
| 融合 QK-norm＋RoPE＋FP8 Q | 0.141 ms | 13.3 ms | 4.5% | 4027 × 8192 BF16 の読み書き 130 MB に対し 141 µs は帯域の 1/4 |
| RMSNorm ×2、量子化 ×2、elementwise、router、pre-dispatch | 約 0.14 ms | 13.5 ms | 4.5% | |
| EAGLE3 draft（nvjet GEMM） | — | 約 4.8 ms | 1.6% | |
| SwapAB 形状の MegaMoE（迷い込み） | 3.29 ms × 94 call/窓 | 約 4 ms 相当 | 1.4% | 本来 0.14 ms の小 batch 呼び出しが 3.3 ms。DP rank 間の非同期（他 rank が 4096 token の call 中）で待っている可能性が高い |

Decode（31.5 要求/rank、verify 4 token、1 step ≒ 39.7 ms（profile 時）、稼働 83%）:

| kernel | 1 step | 割合 | 備考 |
|---|---:|---:|---|
| MegaMoE（SwapAB 64×128） | 20.0 ms（212.6 µs/層） | 50% | HBM 床 126 µs。起動 19 µs、末尾 14〜16 µs、mainloop が床より約 40 µs 長い |
| FA3 decode | 4.7 ms（49.6 µs/層） | 12% | KV 129 MB/層 → 2.6 TB/s |
| dense GEMM ×2 | 2.9 ms | 7% | 重み読み出し律速 |
| 小 kernel（RMSNorm、RoPE、量子化、elementwise、splitK、router、pre-dispatch、store_kv、prepare_varlen） | 約 3.1 ms | 8% | 1 層あたり約 33 µs、12 回の起動 |
| lm_head・draft GEMM | 約 0.8 ms | 2% | |
| kernel 間の空き | 約 6.6 ms | 16.6% | overlap scheduler は有効（spec v2 worker）。DP attention の同期 all_gather は Gloo（CPU group）で、`get_next_batch_to_run` 約 20 ms のうち GPU と重ならない分 |

## 3. 上限の計算

### 3.1 Prefill 供給

| Prefill 1 要求の GPU 時間 | 供給上限（稼働 96%） | 良好率 97.5% の score |
|---:|---:|---:|
| 293 ms（現在） | 39.3 req/s | 7670 |
| 255 ms（Tier 2） | 45.2 req/s | 8810 |
| 230 ms（Tier 3） | 50.1 req/s | 9770 |
| 215 ms（MegaMoE が DeepGEMM 単体の最良 1400 TFLOPS 相当） | 53.6 req/s | 10450 |

電力上限は 700 W（仕様上限、nvidia-smi 記録）で、Prefill 中は 1.47〜1.67 GHz まで落ちる。管理者設定で取り戻せる分はない。最後の行は電力制限下では出ない値で、10000 は物理上限のすぐ内側にある。

### 3.2 Decode slot と ITL

必要な running 数 = req/s × 199 × ITL。ITL は要求平均なので、step 時間 ÷ accept 長（2.21）。

| 目標 req/s | ITL 16.3 ms | ITL 18 ms |
|---:|---:|---:|
| 40 | 130（33/rank） | 143（36/rank） |
| 45 | 146（37/rank） | 161（40/rank） |
| 50 | 162（41/rank） | 179（45/rank） |

slot を 32 → 40/rank にすると verify 行が 126 → 160 になり、現行 kernel では step +3.5 ms（MegaMoE 0.4 µs/行/層 ×34 行 ×94 ＝ 1.3、FA3 +1.3、CPU +1）、ITL +1.6 ms。p99 が 20 ms を超えて不合格が増えるので、slot 増と Decode の行コスト削減は対で進める。KV は 393216 token/rank に対し 40 × 4200 = 168k で問題ない（token usage 0.46 → 0.60）。

### 3.3 開始一斉投入の TTFT 不合格

計測開始時に c 要求が同時に届き、Prefill 供給 S で捌くので、不合格 ≒ c − 3 s × S。A1/A2 の 104〜126 件（2.3〜2.7%）はこれで説明できる。c は「S × 所要時間」より大きくできないので、TTFT と ITL を縮めるほど同じ S を小さい c で出せ、一斉投入分も減る。定常 TTFT p99 は 2.4〜2.6 s で上限 3 s に近く、**handoff 遅延を削る前に c を上げてはいけない**。

## 4. 施策

### Tier 1: runtime のみ（image 不変）

| # | 施策 | 根拠 | 期待 | 判定 |
|---|---|---|---|---|
| 1-1 | Decode 上限 128 → 144（36/rank）、`decode_cuda_graph_bs` に 34・36 を追加 | slot 待ち 19 要求、ITL 余裕 1 ms | raw +4〜6%、ITL +0.8 ms | Decode ログ `#queue-req` ≒ 0、ITL p99 ≤ 20 |
| 1-2 | `SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1`（Decode） | DP 同期が Gloo CPU group。`dp_attn.py` に NCCL device group へ切り替える env がある | step −1〜3 ms（空き 6.6 ms の一部） | profile の kernel 間空き、ITL |
| 1-3 | `--optimistic-prefill-attempts 2`（Prefill） | run E で bootstrap 待ち 1.0 → 0、失敗 0、TTFT p99 2.68 → 2.16 s | TTFT −0.3 s | Prefill ログ `#bootstrap-req` ≒ 0 |
| 1-4 | Prefill の chunk を要求境界に揃える（`PrefillAdder` の patch: 残り budget が要求長未満なら次の要求を入れない） | 3986 token が 4096 chunk を 2 iteration またぎ、batch 内滞在が 2 倍。DP rank 間の非同期（迷い込み SwapAB 呼び出し）も減る | TTFT −0.3 s、Prefill −1% | batch あたり token 3986、`#new-seq` 1、SwapAB 呼び出し消滅 |
| 1-5 | c の再設定 | 1-1〜1-4 後の所要時間 × 供給 ＋ 余裕 | 一斉投入不合格 −0.5〜1% | TTFT 不合格の内訳 |

Tier 1 の到達は Prefill 供給 39〜40 req/s に頭打ちで、score 7600〜7900。1-1＋1-2 を run T1a、1-3＋1-4＋1-5 を run T1b として 2 run、良ければ main へ。

### Tier 2: kernel と patch（新 image）

Prefill（1 要求 293 → 約 255 ms）:

| # | 施策 | 根拠 | 期待 |
|---|---|---|---|
| 2-1 | MegaMoE prefill の末尾: combine を最後の FC2 と重ねる、rank 内の遅い CTA 待ち（median 27〜40、max 75 µs）を減らす block 割り当て、起動 40 µs の短縮 | timeline で非重複部 150〜280 µs/call | MegaMoE −8〜12% → −15〜20 ms |
| 2-2 | 融合 QK-norm＋RoPE＋FP8 Q の帯域化（141 → 約 45 µs） | 130 MB の読み書きに 141 µs は 0.9 TB/s | −9 ms |
| 2-3 | RMSNorm＋量子化、残差加算の融合、pre-dispatch の統合 | 1 層 0.14 ms の小 kernel 群 | −4〜6 ms |
| 2-4 | 1-4 の恒久化（迷い込み SwapAB の解消） | 1.4% | −4 ms |

Decode（step 36 → 約 30 ms at 31.5/rank、40/rank で約 33 ms）:

| # | 施策 | 根拠 | 期待 |
|---|---|---|---|
| 2-5 | MegaMoE decode: dispatch の直列 NVLink 往復（19 µs）を mbarrier 1 回にまとめる、末尾の grid sync＋combine（14〜16 µs）短縮 | timeline | −20〜25 µs/層 → −2〜2.4 ms |
| 2-6 | FA3 decode の split 数・tile 調整（49.6 µs → 約 40 µs、2.6 → 3.3 TB/s） | KV 読み出し 129 MB/層 | −0.9 ms |
| 2-7 | 小 kernel 融合（RMSNorm＋量子化、RoPE＋store_kv） | 1 層 12 起動 33 µs | −1 ms |
| 2-8 | 空き 6.6 ms の分解（M1）後、残る CPU 分の削減 | 1-2 の結果次第 | −1〜3 ms |

Tier 2 後は Decode 40〜44/rank を ITL 平均 15〜16 ms で回せるので、Prefill 供給 45 req/s を Decode が受けられる。c は 45 × 4.0 s ≒ 180〜200 で足り、一斉投入不合格は 1.5% 程度。score 8500〜8900。

### Tier 3: 構造に近い変更

| # | 施策 | 根拠 | 期待 | 前提 |
|---|---|---|---|---|
| 3-1 | DeepGEMM host library（`_C.so`）をソースから再ビルドし、MegaMoE prefill の geometry（stage 数、BLOCK_N、cluster/TMA multicast）を選び直す | GEMM 区間が throttled peak の 58%、3 stage・cluster 1・SMEM 225.6 KB/227 KB。header では変えられない | MegaMoE −10〜15% → −17〜25 ms | sgl-deep-gemm 0.1.5 のビルド経路を image build に追加 |
| 3-2 | FA3 prefill の tile/pipeline（545 → 700 TFLOPS） | FA3 FP8 の公表値との差 | −10 ms | 単体 bench で差を確認してから |
| 3-3 | EAGLE3 steps 4 / draft 5 | accept 2.21 は低く、Decode 行コストが下がれば verify 行の追加が割に合う可能性 | ITL ±5% | Tier 2 の後に 1 run |
| 3-4 | Decode 上限 44〜48/rank、c 再設定 | 3-1 で Prefill 供給 50 req/s に届く場合 | score 9300〜9700 | Tier 2 の Decode 改善 |

### 先に測るもの

- M1. A1 の nsys（`profile/torch` の CPU span）で Decode の空き 6.6 ms を、Gloo all_gather・spec worker・転送 poll・prealloc に分解する。1-2 と 2-8 の設計に必要。
- M2. A1 の Prefill ログに `Prefill batch` 行が無い理由（R にはあった）を確認し、batch 構成・queue・bootstrap の計数を復旧する。1-3/1-4 の判定に必要。
- M3. 迷い込み SwapAB 呼び出し（3.3 ms）が DP 非同期の待ちであることを nsys timeline で確認する（他 rank の同時刻の kernel）。
- M4. kernel bench の Prefill timeline を 2-1 の各案で採る（既存経路、1 回 10 分）。

### 採らないもの

- PD 統合（1 つの worker で prefill と decode）: Prefill の GPU 時間が支配的で、Decode GPU に prefill を混ぜると ITL が上限を超える。
- P/D 比の変更: D を 2 GPU にすると重み 118 GB で KV が 86 要求分しか取れず slot が足りない。P を減らす理由はない。
- 精度を変えるもの（INT4/FP4 重み、accept threshold の緩和、低精度 KV の追加）。
- 動的 EPLB・冗長 expert（MegaMoE patch のガードで禁止、32 expert/rank 固定）。
- 2 micro-batch overlap（TBO）の新規実装: MegaMoE は通信を kernel 内で既に重ねており、重ならない部分は 10% 程度。2-1 で直接削る方が安い。
- prefix cache: 公式 client は `cache-bust first_turn_prefix` で prompt ごとに一意。

## 5. 実行計画

| run | 条件 | 期待 | 合格 |
|---|---|---|---|
| T1a | main image、Decode 上限 144・graph bs 34/36、NCCL 同期 env | 7450〜7650 | ITL p99 ≤ 20、`#queue-req` ≒ 0、score > 7250 |
| T1b | T1a ＋ optimistic 2 ＋ chunk 境界 patch（runtime patch は SGLang の Python なので image 不変で当てられるか確認。不可なら Tier 2 へ） | 7600〜7900 | TTFT p99 ≤ 2.3 s、一斉投入不合格 ≤ 2% |
| bench | 2-1、2-5 を kernel bench で各案 1 回、hash 一致と timeline で採否 | | Prefill −8%、Decode −20 µs |
| T2 | 新 image（2-1〜2-7）、上限 160、c 200 | 8500〜8900 | 2 run の平均で T1b 比 +8% 以上 |
| T3 | host library 再ビルド image、上限 176、c 216 | 9300〜9700 | 同上 |

ブランチ運用は従来どおり。Tier 1 は runtime 設定のみなので `perf/firmus-runtime-20261019`、Tier 2 以降は kernel ブランチで bench → `[require-build]` → serving 2 run → main へ merge。各 run は `[cluster:firmus] [nodes:2] [profile]` で profile pass も採り、kernel-summary で内訳を更新する。
