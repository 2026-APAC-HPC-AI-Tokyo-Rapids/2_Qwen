# 2026-10-18 Firmus Tier 1（runtime と小 patch）の計測と準備（main 6c10548 起点）

[最大性能計画](2026-10-18-firmus-maximum-performance-plan.md)の Tier 1（1-1〜1-5）と、先に測るもの M1〜M3 を、A1（[39070273796](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39070273796)）と A2（[39078761629](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39078761629)）の raw 出力（worker ログ、AIPerf 記録、profile pass の nsys と Torch CPU trace）、pin した SGLang source（fc275674 ＋ 15 patch）から詰めた。Firmus の GPU job はこの作業では投入していない。成果物は branch `perf/firmus-max-runtime-20261018` に置き、統合担当が他の kernel 作業と 1 つの image・1〜2 run にまとめる前提で、GPU を使わない検査はすべて済ませた。

## 1. 結論

| 項目 | 結果 | branch での扱い |
|---|---|---|
| M1 Decode の空き | profile 時の空き 6.87 ms/step（17.3%）の主因は gloo ではなく、**新規要求を受け入れる DP rank の blocking H2D/D2H が WAR barrier で fence された `schedule_stream` 上で走り、その rank の CPU が自 GPU の verify 終了まで止まる**こと。MLP sync の all_gather が 4 rank を待ち合わせるので、1 rank の停止が全 rank の次 step 起動を遅らせる。gloo 自体の遅延は 0.35 ms/step | §2 |
| M2 Prefill batch 行 | A1 の `prefill-0.log` だけ tqdm 出力に途中で切れた UTF-8 があり、`grep`（ugrep、`-I`）が binary として黙って読み飛ばした。行は R と同じ頻度で全部ある。runtime flag は不要 | §3、`grep -a` で読む |
| M3 迷い込み SwapAB | DP3 の **IDLE forward**（自 rank に処理可能な要求が無い step）で、MegaMoE に 1 行の `y` を渡すため Decode 形状が選ばれ、他 rank の 4058 token を Decode tile で処理して 3.29 ms/層かかっていた。chunk 境界とは無関係で、request-aligned chunk では消えない | §4、別 patch 候補として記録 |
| 1-2 NCCL DP 同期 | `SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1` は**入れない**。fence 済み `schedule_stream` 上の H2D/D2H になるので、現在 overlap が効いている 18% の step も verify 終了待ちになり、空きが 6.9 → 7.3 ms に増える見込み。`SGLANG_SCHEDULER_SKIP_ALL_GATHER`、`--speculative-skip-dp-mlp-sync` も不採用 | §5。代わりに Decode の `--disaggregation-decode-polling-interval 2` |
| 1-4 request-aligned chunk | `patches/sglang/prefill-request-aligned-chunks.patch`（新規、16 番目）。disaggregation Prefill で、batch に既に要求がある rank は、その batch で終わらない要求を新たに始めない。env `SGLANG_PREFILL_REQUEST_ALIGNED_CHUNKS=1` で有効（既定は upstream 動作） | §6。image への配線は統合担当 |
| 1-1 Decode 上限 | 128 → **136**（34/rank）、`decode_cuda_graph_bs` に 34 を追加 | §7 |
| 1-3 optimistic prefill | `--optimistic-prefill-attempts 2`（Prefill） | §7 |
| 1-5 concurrency | 0% 改善前提の既定は **192 のまま**。他 worker の Decode 短縮量に応じた表は §7.3 | §7 |
| 任意 router TCP_NODELAY | sglang-router 0.3.2 の listener は `TCP_NODELAY` を設定していない。axum-server 0.8.0 の `NoDelayAcceptor` を使う 1 行の patch を用意した | §8、`patches/sglang-router/no-delay-acceptor.patch`。配線は統合担当 |

## 2. M1: Decode の kernel 間の空き（A1 profile pass、decode-0、4 rank）

nsys の SQLite export と Torch CPU trace は同じプロセスの時刻で、約 30 µs の精度で揃う（各 target-verify の `cudaGraphLaunch` が Torch の `step[TARGET_VERIFY]` に入り、gloo all_gather の開始・終了が main thread の `pthread_cond_wait` と一致する）。窓内は 131 step。

### 2.1 GPU 側（rank 0。rank 1〜3 も ±0.05 ms）

1 step の並びは draft graph（70 node、0.80 ms）→ 約 2.3 ms の空き（TV graph の launch 中）→ TV graph（1791 node）と eager の draft_extend（約 118 kernel）が連続して約 32 ms → 末尾の空き → 小さな eager の準備 kernel → 次の draft graph。

| 指標 | 平均 | 中央値 | p10 | p90 |
|---|---:|---:|---:|---:|
| step 周期 | 39.75 ms | 40.23 | 33.56 | 42.70 |
| GPU busy（全 stream の和集合） | 32.88 ms | | | |
| 空き | **6.87 ms（17.3%）** | | | |

空きのうち graph 内の kernel 間の細かい隙間が 0.69 ms（約 1417 個/step）、launch 間が 6.18 ms。step は 2 群に分かれる。

| 群 | step 数 | 空き | 周期 |
|---|---:|---:|---:|
| CPU が遅れる step | 108 / 131（82%） | 8.14 ms | 41.0 ms |
| CPU が先行する step | 23 / 131（18%） | 0.88 ms | 33.9 ms |

### 2.2 遅れる step の critical path（GPU step の終了から）

| 区間 | 時間 | 中身 |
|---|---:|---|
| 最も遅い rank が all_gather に入るまで | 2.76 ms | うち GPU 終了の rank 差は 0.27 ms。残り 2.49 ms はその rank の sync 解放後の Python（CUDA API 約 0.4 ms を含む） |
| gloo all_gather の遅延（最後の到着から退出まで） | 0.35 ms | p10 0.29 / p90 0.41 |
| all_gather 退出から最初の TV kernel まで | 4.97 ms | run_batch まで 0.31、draft 準備 1.28、draft launch 0.19、TV 準備 0.94、TV launch 2.25。うち約 0.8 ms は draft graph の実行と重なる |

止めている sync: `torch.tensor(..., device=cuda)`（pageable）は `cudaMemcpyAsync` の後に `cudaStreamSynchronize(schedule_stream)` を呼ぶ。`schedule_stream` は `_apply_war_barrier`（`managers/scheduler.py:1562-1577`）により、run_batch のたびに forward が draft_extend の直前に記録する read-done event を待つ。rank 0 では 25〜31 ms 止まる sync が 51 回（計 1.35 s）あり、すべてこの stream 上の 8 / 15976 / 31952 byte の H2D だった。発生箇所は 2 つ。

- `alloc_for_decode_prealloc`（`disaggregation/decode.py:1714-1716` の `last_loc`、`:1752-1757` の prefix/seq lens）。`pop_preallocated` から。さらに直後の `kv_to_page_indices`（`mem_cache/common.py:35`）は GPU 上で割り当てた page 番号を Prefill へ送るため `.cpu()` するので、H2D を non-blocking にしても D2H でまた fence を待つ。
- `prepare_for_prebuilt`（`disaggregation/decode_schedule_batch_mixin.py:80`、int32 `input_ids` など、要求あたり約 16 KB）。`get_next_batch_to_run` から。

108 / 131 step で 4 rank のどれかがこれに当たり、最後に到着する rank は 27 / 27 / 32 / 22 回とほぼ均等だった。到着率から見ても、37 req/s の prealloc と転送完了が 4 rank に毎 step 平均 2.7 件発生するので、どの rank も何もしない step は e^−2.7 ≒ 7% しかない。

`process_batch_result` の `copy_done.synchronize()` は 2.8 ms/step あるが、CPU が先行している iteration でしか起きないので空きへの寄与は 0.02 ms/step。

### 2.3 gloo collective（1 step あたり）

| collective | 回数 | 発行元 | 費用 | critical path |
|---|---:|---|---|---|
| all_gather（MLP sync） | 1 | `scheduler_components/dp_attn.py`、overlap 有効なので `tp_group.cpu_group` | span 平均 12.9 ms（中央値 1.84、p90 29.5）はほぼ他 rank 待ち。実遅延 0.35 ms | 載る |
| broadcast | 1 | `request_receiver.py:190` の `broadcast_pyobj(control_reqs)` | 0.49 ms | 載らない（GPU 実行中） |
| all_reduce | 1.86 | `poll_and_all_reduce`（`decode.py:776` prealloc、`:1960` transfer）。attention-TP group（size 1） | 0.012 ms | 載らない |

### 2.4 profile 窓と通常 pass の差

nsys の graph node trace は TV の launch（2.3〜2.5 ms）を水増ししている。通常 pass の step は ITL 16.3 ms × accept 2.21 ≒ 36.0 ms で、profile 時（39.75 ms）より 3.7 ms 短い。通常 pass の実際の空きは 3〜4 ms/step と見るのが妥当で、先行 step の周期を約 33 ms とすると、遅れる step は約 36.7 ms になる。

### 2.5 削れる量

- 遅れる step を先行 step にできれば Decode step は通常 pass で約 −8%（36.0 → 33 ms）。根本対策は admission の H2D を pinned ＋ `non_blocking` にし、prealloc の page 割り当てを CPU 側で決める（または fence の外で行う正当性を示す）こと。後者は allocator の構造変更なので Tier 2 の SGLang patch として扱い、今回は作らない。
- runtime だけでできるのは、admission を k step に 1 回へまとめる `--disaggregation-decode-polling-interval k`（§5.4）。

## 3. M2: A1 の Prefill ログに `Prefill batch` 行が「無い」理由

- A1 の `prefill-0.log` は byte 37265 で UTF-8 が途中で切れている（4 rank の tqdm「DeepGEMM warmup」の進捗棒が `\xe2\x96` の途中で他 rank のログ行に割り込まれた）。この環境の `grep` は ugrep の `-I`（binary を無視）付きで、このファイルを binary と判定して何も出さない。`file` も「Non-ISO extended-ASCII」と出す。R/R2/A2/E と A1 の `prefill-1/2.log`、`decode-0.log` は正しい UTF-8 で、普通に読める。
- 行はある。`grep -a`（または `LC_ALL=C grep`、Python なら bytes で読んで `decode("utf-8", "replace")`）で数えると、A1 の prefill-0 は 2680 行で R の 2503 行、A2 の 2691 行と同じ頻度。SGLang は Prefill batch を毎 batch 1 行出している。runtime flag の追加は不要。
- 通常 pass（profile 開始前、64 token の health 要求を除く）の batch 構成:

| run | batch 数（DP0/1/2/3、prefill-0） | 2 要求入りの batch | 平均 `#new-token`（page 丸め） | `#queue-req` | `#bootstrap-req` | queue 空で bootstrap 待ちありの起動 |
|---|---|---:|---:|---:|---:|---:|
| R2 | 626 / 616 / 595 / 551 | 80% | 4042 | 0.80 | 1.03 | 53% |
| A1 | 678 / 665 / 643 / 596 | 79% | 4046 | 0.80 | 0.94 | 53% |
| A2 | 677 / 668 / 643 / 604 | 80% | 4048 | 0.85 | 0.97 | 50% |
| E（c176、optimistic 2） | 627 / 619 / 608 / 544 | 86% | 4057 | 0.85 | 0.00 | 0% |

- 「2 要求入り 80%」は 1 要求まるごと（3994 token、page 丸め 4032）＋次の要求の先頭 64 token という構成で、ほぼすべての要求が 2 batch にまたがっていることを示す（§6）。
- DP rank ごとの batch 数の差（A1 で DP0 678、DP3 596）は、ログに出ない IDLE forward の数に当たる（§4）。

## 4. M3: Prefill の迷い込み SwapAB 呼び出し

- 94 call はすべて **device 3（DP3）** の 1 forward だった。device 0〜2 には無い。
- 形状: Prefill 形状は `sm90_fp8_mega_moe_impl<…,128,256,3 stage,…,640 thread>`、225600 B SMEM。SwapAB は `<…,64,128,8 stage,swap,…,384 thread>`、230672 B SMEM。どちらも grid 132。
- token 数は `sm90_mega_moe_pre_dispatch` の grid（`n + ceil((4224 − n)·8/512)`）から求め、store_kvcache と router の grid でも確認した。窓内の 18 forward（約 289 ms/forward）:

| forward | DP0〜2 | DP3 | DP3 の MoE 形状 |
|---|---|---|---|
| 0〜14 | 各 4058 token | 4058 | Prefill |
| 15 | 4058 | 1946（chunk の残り、queue 空） | Prefill |
| 16 | 4058 | 3994（新規 1 要求） | Prefill |
| **17** | **4058、Prefill 形状 2.13 ms/層** | **0 token（IDLE）** | **SwapAB 3.29 ms/層** |

- Torch trace でも DP3 は `step[IDLE bs=0]`、DP0〜2 は `step[EXTEND bs=2 toks=4058]`。DP3 の最後の batch 行は `#new-token 4032, #queue-req 0, #bootstrap-req 2` で、新しい要求は bootstrap 待ちだった。
- 機構: IDLE rank も MegaMoE の集団 kernel に入る必要がある。`forward_megamoe` は `num_tokens=0` で `run_sm90_mega_routed` を呼び、`y` を `max(0, 1) = 1` 行で確保する。DeepGEMM は `y.size(0) = 1` から SwapAB（≤128 token）を選ぶ。各 rank は自分の行数で形状を選ぶので、この forward では 3 rank が Prefill、1 rank が SwapAB だった。dispatch は pull 型で、rank 間で共有する量（4224 行/rank、pool 141312、SF pool 2260992、32 expert/rank）は両形状で同じなので正しさの問題は無い。性能だけの問題で、DP3 の 32 expert は他 rank から約 24k の token-expert 対（expert あたり約 760）を受け取り、それを Decode tile で処理する。3.29 ms は他 rank の attention 待ち約 1.24 ms と実処理約 2.05 ms。
- EAGLE3 draft（`step[IDLE]` 0.5 ms、MoE なし）、health 要求、chunk の残り（1946 token は Prefill 形状）は原因ではない。
- 費用: IDLE forward 1 回で DP3 の GPU が 94 × 3.29 = 310 ms 埋まる。他 rank の MegaMoE は最後の rank が層に入ってから終わるまで 2.12 ms/層で、通常（1.74〜1.83 ms、4 rank 満載）より長い。forward は 309.8 ms（通常約 289 ms）で、処理 token は 12174（通常 16232）。
- 頻度: batch 行数の差から、A1 の各 worker で DP3 約 82、DP2 約 35、DP1 約 13、DP0 0〜1 回の IDLE forward（約 680 forward 中）。rank-forward の 4.8% が IDLE。run E（optimistic 2）でも DP3 は 83 回少なく、bootstrap 待ちが消えても偏りは残る。`DPBudget.dispatch` の `total_requests.index(min(...))` が同数のとき小さい index を選ぶので、供給が Prefill 能力を下回る時間に DP3 から先に干上がると考えられる（ソースからの推定で、データでの確認はしていない）。
- **request-aligned chunk では消えない。** IDLE forward は処理可能な要求が無い rank で起きる。対策候補（いずれも未実装・未検証）: (1) IDLE の Prefill rank では `y` を 129 行以上で確保して Prefill 形状を選ばせる（pre-dispatch が余りの行を埋めるので、余り行の topk の扱いを先に確認する）、(2) 形状を DP 全体の最大 token 数で選ぶ、(3) DP dispatch の同数時の選び方を回す。(1)/(2) は MegaMoE の SGLang patch の担当者向けで、Prefill の時間の約 1%（IDLE forward 1 回あたり約 +21 ms × 約 130 回/worker ÷ 210 s）に効く。

## 5. 1-2: DP 同期 knob の安全性と推奨（fc275674 ＋ 15 patch）

### 5.1 `SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1`

`prepare_mlp_sync_batch_raw`（`dp_attn.py`）は、offload が無く、overlap 無効または env が立っているとき `tp_group.device_group` / `tp_group.device` を使う。`MLPSyncBatchInfo.all_gather` は `torch.tensor(..., device=cuda)` で局所 tensor を作り（blocking H2D）、`all_gather_into_tensor` を NCCL stream に積み（NCCL stream は current stream = `schedule_stream` を待つ）、`tp0_info[:, :2].cpu()` と `.item()` で CPU へ戻す。scheduler thread の current stream は `schedule_stream` で、`_apply_war_barrier` により実行中 step の read-done event（spec v2 では draft_extend の直前、つまり target verify の終了）を待っている。

- **順序と deadlock。** NCCL の all_gather kernel は verify 終了後にしか走らず、次 step は all_gather の結果を CPU が受け取るまで launch されないので、MegaMoE（132 SM を占有し rank 間で待ち合わせる persistent kernel）と同時に走ることはない。IDLE rank の forward で read-done event が無い場合は `wait_stream(forward_stream)` になり、さらに保守的。CUDA graph の capture/replay にも影響しない（all_gather は graph の外）。deadlock は起きない。ただしこれは fence の位置に依存した結果で、fence を外す変更と組み合わせると、NCCL kernel が SM を握ったまま他 rank を待ち、他 rank の MegaMoE が全 CTA の常駐を待つ循環待ちがあり得る。
- **性能。** env を立てると、毎 step の CPU 処理が verify 終了を待つ。遅れる step（82%）は既に prealloc/prebuilt の sync で verify 終了を待っているので変わらず約 8.1 ms。先行していた step（18%）は all_gather 退出後の 4.97 ms（draft_extend と draft graph で約 2 ms 隠れる）が新たに空きになり、0.9 → 約 3.5 ms。期待値は 0.82 × 8.1 ＋ 0.18 × 3.5 ≒ 7.3 ms で、現状 6.87 ms より悪い。gloo をやめて得られるのは実遅延 0.35 ms のうち最大 0.3 ms で、待ちの本体（遅い rank の CPU）は NCCL でも残る。
- Prefill でも同じ順序（`eagle_worker_v2.py` は extend でも target 終了に fence を置く）になり、batch の組み立てが 290 ms forward の終了待ちになる。gloo の遅延はもともと forward に隠れているので得るものが無い。
- **推奨: Decode・Prefill とも設定しない。**

### 5.2 `SGLANG_SCHEDULER_SKIP_ALL_GATHER`

DWDP（`server_args.py:6292`、`moe_a2a_backend=none`、CUDA graph 無効）のために入った knob。Decode は `require_mlp_tp_gather` が False（`moe_dense_tp_size 1 == tp/dp`、`enable_dp_lm_head`、a2a megamoe）なので `global_num_tokens` は自 rank の値だけで足り、MegaMoE も rank ごとの行数の違いを kernel 内で扱う。それでも次の理由で使わない。

- `need_idle_batch = dp_size > 1` になり、全 rank が空でも毎 iteration IDLE forward（MegaMoE の集団呼び出し）を回し続ける。
- `is_extend_in_batch` / `global_forward_mode` / TBO 準備 / `recv_skipper_forward_mode` が rank 間で揃わなくなる。overlap 無効化の判定（`is_disable_overlap_for_batch`）はこの同期値を前提にしている。
- 得られるものが無い。scheduler での待ち合わせが消えても、速い rank は次 step の最初の MegaMoE 層で遅い rank を kernel 内で待つ（SM を回したまま）。step 時間は遅い rank の停止＋Python＋launch で決まる。
- MegaMoE ＋ decode CUDA graph ＋ spec v2 overlap の組み合わせで upstream が試した形跡が無い。

### 5.3 `--speculative-skip-dp-mlp-sync`

`arg_groups/speculative_hook.py:132` が `speculative_algorithm == "EAGLE"` を assert するので、EAGLE3 では起動時に失敗する。効くのも非 disaggregation の `get_next_batch_to_run`（`scheduler.py:2867`）の追加 sync だけで、disaggregation Decode の loop（`get_next_disagg_decode_batch_to_run`、1 step 1 回の sync）には関係しない。**不採用。**

### 5.4 代わりに入れるもの: `--disaggregation-decode-polling-interval 2`（Decode）

`process_decode_queue`（`decode.py:2320-2352`）は `polling_count % polling_interval == 0` の iteration でだけ `pop_preallocated` と `pop_transferred` を行う。1 iteration に 1 回の all_gather で全 rank は同じ iteration 数を進むので、k > 1 にすると admission（§2.2 の停止の原因）が全 rank で同じ iteration に集まる。

| k | 停止する step の割合 | 通常 pass の step（推定） | 追加の TTFT | slot の空転 |
|---:|---:|---:|---:|---:|
| 1（現状） | 82% | 36.0 ms | — | — |
| 2 | 約 50% | 約 35.2 ms（−2.3%） | 約 +37 ms | 約 0.5% |
| 3 | 約 33% | 約 34.7 ms（−3.7%） | 約 +75 ms | 約 1.1% |

推定は「先行 step 33.0 ms、停止 step 36.7 ms ＋ まとめて受け入れる件数に応じて 1 件あたり約 0.7 ms」の 2 群モデルで、まとめた分の Python がどれだけ伸びるかは未測定。k = 2 は Decode 律速の間でも差し引き +1.5〜2%、Prefill 律速になれば TTFT +37 ms だけが残る。上限 3 s に対して定常 TTFT p99 は 2.43 s で、影響は小さい。config の `decode_polling_interval` を 2 にした。効果の判定は §9。

## 6. 1-4: request-aligned prefill chunks（`patches/sglang/prefill-request-aligned-chunks.patch`）

### 6.1 現象

Prefill は DP4 で `chunked_prefill_size 16384` → 4096 token/rank。要求は 3994 token（page 丸めで 4032）。1 要求目を入れると残り 64 token で、`PrefillAdder.add_one_req` は 2 要求目を 64 token に切って入れる。続く batch は 2 要求目の残り 3930 token ＋ 3 要求目の 128 token … となり、ほぼ全要求が 2 batch にまたがる（§3 の「2 要求入り 80%」）。

backlog がある間は、どちらの方式でも 1 batch で約 1 要求が終わるので完了時刻は変わらない（token 単位の FIFO で、batch が 1.6% 短くなるだけ）。差が出るのは queue が薄いとき（A1 は queue 平均 0.8、起動の 53% で queue が空）。各 rank に 1 batch 周期あたり約 1 要求が届く状態だと、現行方式では新しい要求が前の要求の残りと同じ batch に先頭だけ入って次の batch で終わり、batch 内滞在が 2 周期になる。Little の法則で見た batch 内滞在 22 要求（0.6 s ≒ 2 × 290 ms）はこれと合う。要求境界で揃えれば 1 周期で終わり、TTFT は最大で約 −0.29 s。実際の削減は backlog の時間割合に応じて −0.15〜−0.29 s と見る。開始一斉投入は backlog 状態なので、その不合格数はこの patch では変わらない。

### 6.2 変更

- `schedule_policy.py`: `PrefillAdder.__init__` に `request_aligned_chunks` 引数を追加し、モジュール定数 `_PREFILL_REQUEST_ALIGNED_CHUNKS`（`SGLANG_PREFILL_REQUEST_ALIGNED_CHUNKS == "1"`）と AND する。`add_one_req` の chunk 分岐（要求がこの batch に収まらず切るしかない分岐）の先頭で、`can_run_list` が空でなければ `AddReqResult.OTHER` を返す。`add_one_req_ignore_eos`（radix cache 無効時の経路）の chunk 分岐にも同じ判定を入れた。
- `scheduler.py`: `_get_new_batch_prefill_raw` が `request_aligned_chunks=self.disaggregation_mode == DisaggregationMode.PREFILL` を渡す。Decode や非 disaggregation の scheduler には効かない。
- 返す位置は、既存の `trunc_len <= 0` で `OTHER` を返す箇所と同じ分岐内（lock・prefill delayer の交渉・host load-back の後）で、既存の早期 return と同じ後始末になる。`OTHER` は `batch_is_full` を立てずに waiting queue の走査を止めるので、要求は queue の先頭に残り、次の batch で最初に入る。
- 単独で予算を超える要求（`can_run_list` が空）は従来どおり chunk する。chunk 継続中の要求（`add_chunked_req`）はそのまま続き、その batch に新しい要求は入らない。64 token の health 要求のように残り予算に収まる要求はそのまま入る。
- env を既定 off にしたのは、同じ image で T1a（off）と T1b（on）を比べられるようにするため（既存の `SGLANG_PREFILL_FUSED_FP8_SHUFFLE` などと同じ扱い）。branch の `worker.sh` は Prefill のときだけ `SGLANG_PREFILL_REQUEST_ALIGNED_CHUNKS=1` を export し、起動行に `aligned_chunks=1` を出す。patch の無い image では無視される。

### 6.3 他の機構との関係

- **optimistic prefill（`--optimistic-prefill-attempts 2`）。** handshake 未完了の要求も waiting queue に入って forward される。chunk 境界で handshake 未完了だと、handshake 済みの要求に譲って KV を捨て、やり直す（`process_prefill_chunk`、`optimistic_release_and_requeue`）。要求が 1 batch で終わるようになるとこの譲り（KV の再計算）が起きなくなる。1 batch で終わった要求は handshake まで `disagg_prefill_inflight_queue` で KV を持って待ち、送信だけを止める（`req.pending_bootstrap`）。N=2 はそのままでよい（N=1 でも足りるが、patch の env を外したときに N=2 が要る）。
- **chunk 継続経路。** `self.chunked_req` の扱い（`init_next_round_input`、`inflight_middle_chunks`、overlap 時の `tmp_end_idx`）は変えていない。継続中の要求の後ろに新しい要求が入らなくなるだけ。
- **DP 振り分け（`total_requests`）。** rank ごとの queue/running の数え方は変わらない。queue に 2 件ある rank が 1 batch で 4058 token ではなく 3994 token を処理する（−1.6%）が、forward 時間もほぼ比例して縮む。step の固定費の分（数%）だけ Prefill の供給が −0.1% 程度になる見込み。
- **prefill-ready-handoff。** 結果が準備できた batch の処理を次の forward の前に行う fast path（attention TP1/CP1 のとき）で、要求が 1 batch で終わると最終 chunk の送信が 1 周期早く始まるだけ。chunk 準備の順序（`tmp_end_idx` と `inflight_middle_chunks`）には触れない。
- **firmus-staged-eagle-kv。** sender は scheduler の chunk とは独立に `chunked_prefill_size // page_size` ページ単位で chunk_id を振り直すので、scheduler の chunk が 1 つになっても chunk_id の連番は保たれる。Firmus は `SGLANG_DISAGG_STAGING_BUFFER=0`（直接 page 転送）。
- **MegaMoE の行数上限** `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096` は超えない（batch は 4096 token 以下のまま）。
- 出力: 重み・kernel・KV は同じで、batch 構成だけが変わる。round_robin や到着の揺らぎと同種で、意味的に不変（bit 一致は保証しない）。判定は公式 parser で行う。

### 6.4 検査（GPU なし）

- 15 patch を `scripts/build-sglang-container.sh` と同じ順に fc275674 へ `git apply --check` → `git apply` し、その後にこの patch を当てて通ることを確認した。既存 15 patch は `schedule_policy.py` / `scheduler.py` に触れない。変更された Python ファイル 21 本はすべて `py_compile` を通る。sha256 は `2d355e79…`。
- 既存の source 検査（`check-firmus-staging.py`、`check-firmus-megamoe-source.py`、`check-firmus-latency-source.py`）は 16 patch の tree でも PASS。
- 新しい CPU 検査 `scripts/check-firmus-prefill-chunks.py <source>`: patch 済み source の `PrefillAdder` と `AddReqResult` を AST で取り出して実行する（KV pool と要求は mock、radix 有効・無効の両経路）。(1) 従来動作では 2 要求目が 64 token に切られる、(2) aligned では 2 要求目が `OTHER` で queue に残り chunk が作られない、(3) 残り予算に収まる 64 token 要求は入る、(4) 単独の 5000 token 要求は 4096 token に chunk される、(5) chunk 継続の後ろに新規要求は入らない、(6) env が無ければ scheduler の引数だけでは何も変わらない、(7) scheduler が `self.disaggregation_mode == DisaggregationMode.PREFILL` を渡す、を確かめる。patch を当てない tree では「request-aligned chunk switch is missing」で失敗する。

## 7. 1-1 / 1-3 / 1-5: runtime 設定

### 7.1 変更（`configs/firmus-framework.json`、`platforms/firmus/framework-comparison/sglang.sbatch`）

| 項目 | 変更前 | 変更後 |
|---|---|---|
| `decode_max_running_requests` | 128 | 136（34/rank） |
| `decode_cuda_graph_bs` | `[…, 30, 32]` | `[…, 30, 32, 34]` |
| `decode_polling_interval`（新規） → Decode の `--disaggregation-decode-polling-interval` | （1） | 2 |
| `prefill.optimistic_prefill_attempts`（新規） → Prefill の `--optimistic-prefill-attempts` | （0） | 2 |
| Prefill の env `SGLANG_PREFILL_REQUEST_ALIGNED_CHUNKS` | なし | 1（worker.sh） |
| `concurrency` | 192 | 192 |

- graph の bucket 列は上限と一致させる（`bisect_left` で次の bucket へ padding するので、上限 34 に 32 の次の 36 しか無いと 36 の graph を回す）。上限を 144 にする場合は 36 を追加する。
- req slot は 34 ＋ extra 16 = 50/rank × 4 = 200 ≥ c192。KV は 393216 token/rank に対し 34 × 4200 ≒ 143k で、token usage は 0.46 → 約 0.49。
- 起動後の検査（`scripts/check-firmus-server-info.py`、Actions の「GPU終了後に回収した実設定を検証」）に、Prefill 全 state の `optimistic_prefill_attempts` と Decode 全 state の `disaggregation_decode_polling_interval` を加えた。キーの無い古い config は SGLang の既定（0、1）として扱う。A1 の実 server-info と A1 の config で PASS を確認した。

### 7.2 cap と c のモデル（A1/A2 の要求ごとの記録から）

- ITL と Decode 同時実行数: 要求の存続中の Decode 滞在数（ack〜end）の時間平均で要求をまとめると、中央値は N=60 で 11.2〜11.5、100 で 14.8〜15.0、116 で 15.4〜15.6、124 で 16.4 ms（A1/A2）。N≥100 の傾きは約 0.06 ms/要求（全 rank 合計）。kernel からの見積もり（MegaMoE は hoist 後の bench で 96→128 行が 0.13 µs/行/層、FA3 は KV 量に比例して 0.149 ms/要求/rank、ほかに小 kernel と CPU）では 0.045〜0.08。0.06 を中心に置き、0.045 と 0.08 で感度を見た。定常の run 内回帰（N の幅が狭い）は傾きが run ごとに 0.03〜0.17 とばらつくので使わない。
- ITL の分布は平均に比例して伸縮させる（A1/A2 の p99 − 平均 = 2.6〜2.7 ms は平均の 16%）。ITL 不合格率は、A1/A2 の 9214 要求の ITL を倍率 r 倍したとき 20 ms を超える割合。
- Decode 容量 = N ÷（199 × ITL ＋ 0.11 s）。Prefill 供給 S = 39.3 req/s（計画 §3.1）。Prefill 律速なら Decode は S × T_dec 件しか持たない。
- c = N ＋ X × T_handoff ＋ 8（slot 待ちの余裕）。T_handoff（slot 待ちを除く TTFT）は A1 で 1.3 s。開始一斉投入の不合格 ≒ c − B、B（3 s 以内に返せる数）は A1 で 88。
- score ≒ 200 × X ×（1 − ITL 不合格 − 一斉投入不合格 − 0.2%）。モデルは A1 の実測（7251/7185）に対して上限 128 で 7353 と、約 2% 高めに出る（c を 182 と置くため）。比較にだけ使う。

### 7.3 推奨（他 worker の Decode step 短縮量 x ごと）

傾き 0.06 ms、S = 39.3 req/s、T_handoff 1.3 s の場合。

| x | 上限/rank（上限） | Decode 保持数 | ITL 平均 / p99 | ITL 不合格 | X（req/s） | 必要な c | モデル score |
|---|---|---:|---|---:|---:|---:|---:|
| 0% | 32（128、現状） | 125 | 16.13 / 18.92 | 0.17% | 37.7 | 182 | 7353 |
| 0% | **34（136）** | 133 | 16.60 / 19.47 | 0.38% | 38.9 | 191 | 7574 |
| 0% | 35（140） | 135（Prefill 律速） | 16.75 / 19.64 | 0.44% | 39.3 | 194 | 7638 |
| −8% | 34〜36（136〜144） | 117（Prefill 律速） | 14.39 / 16.88 | 0.00% | 39.3 | 176 | 7702 |
| −15% | 34〜36（136〜144） | 103（Prefill 律速） | 12.57 / 14.74 | 0.00% | 39.3 | 162 | 7725 |

- **x = 0%:** 上限 136。傾きを 0.08 と悲観しても p99 19.66 ms、ITL 不合格 0.46% に収まる。140 以上は傾き 0.08 のとき p99 が 20 ms を超え、不合格が 1%〜1.7% に跳ねる。c は 192 のまま（必要量 191）。
- **x = −8%:** Decode は Prefill の供給（39.3 req/s）より速くなり、上限は効かなくなる（保持 117 件）。上限は 136〜144 のどちらでもよく、揺らぎを吸うため 144（graph に 36 を追加）でもよい。c は 176 前後。
- **x = −15%:** 同様に Prefill 律速。c は 160〜165。
- 1-3/1-4 で T_handoff が 1.3 → 1.0 s、B が 88 → 100 になれば、必要な c はさらに X × 0.3 ≒ 12 下がる（x = 0% で約 180）。S が 41 req/s まで上がれば（IDLE forward が減るなど）、x = 0% でも上限 140〜144 が得になる（S = 41、T_handoff 1.0、B 100 で 144 が 7778）。ただし c を下げるのは、TTFT p99 と Decode の slot 待ち queue が下がったことを run で確かめてからにする（計画 §3.3）。c が足りずに Decode が空くと 1% 単位で失い、c が 4 多すぎても一斉投入不合格が 0.09% 増えるだけで、損失は非対称。

## 8. 任意: router の TCP_NODELAY（`patches/sglang-router/no-delay-acceptor.patch`）

- pin した sglang-router 0.3.2（`sglang_router-0.3.2.tar.gz`、sha256 `bdbea1d5…` を再確認）の `src/server.rs:1041-1060` は `axum_server::bind(addr).handle(handle).serve(...)` で、Cargo.lock の axum-server 0.8.0 の既定 `DefaultAcceptor`（`accept.rs:29-43`）は何もしない。`TCP_NODELAY` を設定するのは opt-in の `NoDelayAcceptor`（`accept.rs:46-64`、`stream.set_nodelay(true)`）だけ。hyper 1.12.0 / hyper-util 0.1.21 の server 側にも nodelay は無い。
- worker への reqwest client は `.tcp_nodelay(true)` を明示している（`src/app_context.rs:338-344`、reqwest 0.12.28 の既定も true）。router → Prefill/Decode は影響を受けない。
- PD の streaming 応答（`pd_router.rs` の `create_streaming_response`）は worker の SSE chunk を 1 つずつ書く。ヘッダの書き込みの直後に最初の SSE chunk を書くと、Nagle がヘッダの ACK を待ち、client の delayed ACK（Linux で約 40 ms）だけ遅れる。router の終了時刻と client TTFT の差が一定 41 ms（±1 ms）であることと合う。
- patch は TLS でない分岐に `.acceptor(axum_server::accept::NoDelayAcceptor::new())` を 1 行足すだけで、依存も Cargo.lock も変わらない。`prefill-load-lifetime.patch` の後に当たることを確認し、ローカルで Rust 1.95.0 の `cargo check --locked --bin smg`（protoc 29.3）を通した。sha256 は `de7f1a42…`。
- 効果は TTFT −41 ms（全要求）で、TTFT 不合格の裾と、同じ供給を保つのに必要な c（約 −1.5）に効く。確認は §9。

## 9. serving run で確かめること

| 施策 | 効いた印 | 失敗・副作用の印 |
|---|---|---|
| 1-1 上限 136 | Decode ログの `#running-req` が 33〜34、`#queue-req`（slot 待ち）が 4.7/rank から減る。req/s +3% 前後 | ITL p99 > 19.7 ms、ITL 不合格 > 0.5%。`check-firmus-server-info` の `effective_max_running_requests_per_dp=34` |
| polling 2 | profile pass の Decode step 周期（TV launch 間隔）が短くなり、空き 6.9 ms から減る。通常 pass の ITL × accept が 36.0 ms より短い | TTFT が 40 ms 以上伸びる、`#transfer-req` が増え続ける |
| 1-3 optimistic 2 | Prefill の batch 行で `#bootstrap-req` ≒ 0、`#optimistic-req > 0`、「queue 空で bootstrap 待ちあり」の起動が 53% → 0%（run E と同じ） | `Prefill bootstrap failed`、`finalize` の AssertionError、NIXL の `has no transfer metadata`、`#inflight-req` と token usage の単調増加 |
| 1-4 aligned chunk | 起動行の `aligned_chunks=1`。batch 行で `#new-seq` が 1、`#new-token` が 4032（page 丸め）、「2 要求入り」が 80% → ほぼ 0%。定常 TTFT p50 −0.15〜−0.29 s | Prefill の req/s が −1% を超えて下がる。DP rank 間の batch 数差（IDLE forward）が増える |
| router nodelay | client TTFT と router の終了時刻の差が 41 ms → 1 ms 程度 | なし（router が起動しなければ build の問題） |
| NCCL env | 設定しない。ログの server_args で確認 | — |

注意: Prefill のログは `grep -a` で読む（§3）。IDLE forward（§4）は batch 行に出ないので、DP rank ごとの batch 数の差で数える。

## 10. 統合担当への配線

branch `perf/firmus-max-runtime-20261018` は origin/main 332e30c から作り、ff3c85f（aligned chunk patch と CPU 検査）、843c90a（runtime 設定と起動後検査）、0879bb8（router patch）、この文書の commit を置いた。image に関わる配線（build script、Dockerfile、image test）は他の worker の patch とまとめて行うため、この branch では行っていない。

SGLang patch（`patches/sglang/prefill-request-aligned-chunks.patch`、sha256 `2d355e79…`）:

1. `scripts/build-sglang-container.sh` に `aligned_chunk_patch="patches/sglang/prefill-request-aligned-chunks.patch"` を足し、2 つの patch 列（apply の for と `patches_json` の for）の末尾（`${qknorm_rope_fp8_patch}` の後）に加える。`targets` の case に `python/sglang/srt/managers/schedule_policy.py python/sglang/srt/managers/scheduler.py` を足す。他の worker の patch が同じファイルに触れる場合は順序を決めて `git apply --check` を通し直す。
2. 同じ script の source 検査（`check-firmus-latency-source.py` の次）に `python3 "${GITHUB_WORKSPACE}/scripts/check-firmus-prefill-chunks.py" "${source_dir}"` を足す。
3. Dockerfile と image test（`scripts/check-firmus-image.sh`、`qwen-experiment-metrics/tests/test_firmus_image.py` など）が patch の一覧や sha を固定していれば、そこにも加える。source の変更なので `[require-build]`。
4. 無効にするときは `sglang.sbatch` の worker.sh の `export SGLANG_PREFILL_REQUEST_ALIGNED_CHUNKS=1` の行を消す（image は変えない）。

router patch（`patches/sglang-router/no-delay-acceptor.patch`、sha256 `de7f1a42…`）:

1. `containers/Dockerfile` 22 行目の `COPY prefill-load-lifetime.patch sglang-router.Cargo.lock /opt/apac/` に `no-delay-acceptor.patch` を加え、`scripts/build-sglang-container.sh` の build context への `cp`（`prefill-load-lifetime.patch` の隣）に同じファイルを加える。
2. `containers/build-sglang-router.sh` の `patch -p1 < /opt/apac/prefill-load-lifetime.patch` の次に `patch -p1 < /opt/apac/no-delay-acceptor.patch` を足し、build record の `names` に `"nodelay_patch_sha256":"no-delay-acceptor.patch"` を加える。`patches_json` に router patch を登録している行（`router_patch=…`）にも同様に加える。
3. router binary が変わるので `[require-build]`。

runtime 設定は image を変えないので、この branch の config/sbatch をそのまま使える。§7.3 の表に沿って上限と c を選ぶ場合は、`decode_max_running_requests`・`decode_cuda_graph_bs`・`concurrency` を 1 commit で変える（`qwen-experiment-metrics/tests/test_firmus_reference.py` が上限 136 と c 192 を固定しているので一緒に直す）。
