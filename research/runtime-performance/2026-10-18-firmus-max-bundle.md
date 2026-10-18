# 2026-10-18 Firmus 最大性能 bundle（branch `perf/firmus-max-20261018`）の統合記録

W1（runtime tier 1/2）、W2（Prefill tail）、W3（Decode tail）、W4（SGLang 小 kernel）、W5（Prefill mainloop prefetch）の各 branch を merge commit で束ね、一つの image に配線した。GPU・build job は投入していない。投入は統合担当が一度だけ行う。

## 1. SGLang patch の順序（fc275674 に `git apply --check` → `git apply` で全通過）

既存 15 本 → `prefill-request-aligned-chunks` → `qknorm-rope-fp8-kv` → `qwen-topk-padding` → `qknorm-rope-fp8-token-warps` → `add-rmsnorm-group-quant` → `decode-async-admission` → `dp-total-requests-rotate-ties` → `megamoe-idle-prefill-geometry`。順序の問題は無かった。`check-firmus-staging.py`、`check-firmus-megamoe-source.py`、`check-firmus-latency-source.py`、`check-firmus-prefill-chunks.py`、`check-firmus-tier2-source.py admission dp-ties idle-geometry` はすべて patch 済み tree で PASS。`check-firmus-latency-source.py --installed` は image 内でしか動かせない（torch と build 済み sglang が要る）ので、ローカルでは未実行。

router は `prefill-load-lifetime.patch` → `no-delay-acceptor.patch` の順で sglang_router-0.3.2 に当たる。

## 2. DeepGEMM patch の順序と header pin

`patches/deep-gemm/series` が適用順、`patches/deep-gemm/header.sha256` が全 patch 適用後の `sm90_fp8_mega_moe.cuh` の SHA-256。build は `patches/deep-gemm` ディレクトリごと image に入れ、`build-sm90-megamoe.py` が series を適用して header を照合し、`sm90-megamoe-build.json` の `patches`（名前 → sha256）に記録する。patch を足すときは series に 1 行と header.sha256 の更新だけで済む。

| 順 | patch | 由来 |
|---|---|---|
| 1–8 | late-topk-fp32、fc2-pipeline、prefill-rescale、decode-exact-tiles、prefill-phase、decode-swap-ab-hoist、decode-swap-ab-l1-quant、dispatch-pull-overlap | main（header `135a43be…`） |
| 9 | dispatch-drop-l1-weight | W3 |
| 10 | prefill-tail | W2 |
| 11 | prefill-mainloop-prefetch | W5 |
| 12 | combine-parallel | W3（下記のとおり rebase） |
| 13 | decode-dynamic-schedule | W3（下記のとおり rebase） |
| 14 | decode-l1-store-wait | W3 |

最終 header sha256 は `ff7a58cca00aa42a90386d12b0437b5d8caf35eb0ad082a8a1a0b4b74001a2a1`（upstream v0.1.5 の header `a53d83ca…` から全 14 本を当てて再計算）。途中の値は、8 本 `135a43be…`、+drop `c28e4ea7…`、+tail `45e93fd4…`、+mainloop `23de592b…`。W2 単独 `6b321026…`、W5 単独 `48407619…` も再計算して一致した。

`sm90-flat-nvlink-barriers.patch` は全 geometry の rank 間手順を変え、W2 の tail と同じ箇所に触れ、GPU で一度も動いていないので series に入れていない（ファイルは残す）。

### 2.1 rebase が必要だったもの

指定順のままでは `combine-parallel` が `prefill-tail` の後に当たらない。tail は Prefill の combine を `if constexpr (kPairedCombine) { … } else {` で囲み、combine-parallel の hunk（旧 combine の loop）の文脈が変わるため。`git apply --3way` で tail・mainloop の上に当て直し、conflict は無かった（意味は同じ。Decode の `kParallelCombine` の hunk は tail の `else` 側に入る）。`decode-dynamic-schedule` は W2 の動的 FC2（`kDynamicL2`）と 4 か所で重なったので手で統合した。

- `kDynamicL2` は Prefill（`not kFP8SwapAB`）、`kDecodeTail` は SwapAB の Decode だけなので、条件は排他的。A/B loader と math の block 取得は `if constexpr (kDynamicL2) … else if constexpr (kDecodeTail) … else scheduler.for_each_block` の三分岐、cleanup の counter reset は両方を並べた。
- `kCombineSmemOffset` は W2（`kDynamicL2`）と W3（`kDecodeTail`）が別々に定義していた。一つにして `(kDynamicL2 or kDecodeTail) ? SMEM_EXPERT_COUNT_SIZE : 0` とし、W3 の「Combine buffers do not fit」の static assert も残した。
- workspace の slot は衝突しない。W2 の FC2 claim counter は grid-sync slot 3（slot 0/1 は dispatch/epilogue の grid sync）。W3 の block counter は L2 arrival-mask の `kNumPoolBlocks`（BLOCK_M=64 の 2208 個の pool block の次）で、Prefill の cleanup が 0 に戻すのは自分の pool block の範囲だけ。W3 の `kNumPoolBlocks + 1 <= kNumMaxPoolTokens / kMinCandidateBlockM` の static assert は 3 geometry すべてで compile を通った。

rebase した 2 本は `patches/deep-gemm/` の版を使う。W3 の元の版（8+drop の上）とは文脈だけが違う。

## 3. compile 結果（Firmus login、image の CUDA 13.0.88 nvcc、`build-sm90-megamoe.py` と同じ instantiation、Decode は wave 32 と wave 16）

| chain（8 本の上） | Prefill reg / spill st/ld | Decode / Decode-16 reg / spill | Decode SASS |
|---|---|---|---|
| image（8 本） | 96 / 40/40 | 168 / 0 | `8e2041e0…` / `e717f97d…` |
| + tail + mainloop（drop 無し） | 96 / 64/80 | 168 / 0 | image と同一 |
| + drop | 96 / 20/20 | 168 / 0 | `11e8d550…` / `5e6473d6…` |
| + drop + tail | 96 / 20/20 | 168 / 0 | drop と同一 |
| + drop + tail + mainloop | 96 / 44/60 | 168 / 0 | drop と同一 |
| + drop + tail + combine + dynamic（a-measured） | 96 / 20/20 | 168 / 0 | `f0939e5d…` / `d7c90330…` |
| 全 14 本（series） | 96 / 44/60 | 168 / 0 | `5bcd675b…` / `a0e029dd…` |
| 全 14 本 + timeline 計装 | 96 / 72/120 | 168 / 28/136 | — |

W2+W5 の組み合わせ（drop 無し）の Decode SASS は image と byte 単位で同一。drop が Decode から不要な load を除くので、drop を含む chain の Decode は image と変わる。

## 4. kernel bench の staging（image `7c6286e5…` の上に当たる overlay）

- `patches/deep-gemm-bench/a-measured/`：drop、tail、combine-parallel、dynamic-schedule（01–04）。各 patch は GPU で単独に計測済みだが、この 4 本の組み合わせは未計測。b が落ちたときの fallback。
- `patches/deep-gemm-bench/b-bundle/`：series の 9〜14 番（01–06）と `timeline`。
- `patches/sglang-bench/`：W4 の `10`〜`40`。W1 の scheduler patch は含めない。
- 候補 directory の patch は名前順に当たる。どの候補も image の header（`135a43be…`）に対して `git apply --check` を通した。a の結果 `3a202f47…`、b の結果 `ff7a58cc…`（series の最終 header と一致）。SGLang の 4 本は 15 patch の tree に通った。
- 標準の timeline patch（`patches/deep-gemm-timeline/`）は drop 以降の chain に当たらない。W2 の修正版 timeline は 8 本 + tail（drop 無し）向けで、b にも当たらない。そこで W3 の timeline（8+drop+combine+dynamic 向け）と W2 の修正版を b の最終 header へ 3-way merge して `b-bundle/timeline` を作った（conflict 3 + 1 を手で解決。`tl_combine_tokens` の宣言は Prefill 側の一つにまとめた）。`provision-firmus-kernel-bench.sh` は、空でない `timeline` をその候補専用の timeline patch として使う（空の marker は従来どおり標準版）。timeline 計装版も login node で compile を確認した（上表）。
- step 数：image bench 1 + SGLang check 2 + SGLang bench 2 + a（gate + bench）2 + b（gate + bench + timeline）3 = 10。上限 12 に対して余裕 2。image の timeline marker（`patches/deep-gemm-bench/timeline`）を足せば 11。

## 5. runtime 設定（serving run 1）

aligned chunks、async admission、DP tie rotation、idle-rank Prefill geometry をすべて 1 にした。`decode_polling_interval` は 1（async admission が代わりに効く）。`optimistic_prefill_attempts` は 2、Decode cap 136 と cuda-graph bs 34 は据え置き。worker の起動行は `shuffle`、`routing`、`aligned_chunks`、`rotate_ties`、`idle_prefill_geometry`、`async_admission` の値をすべて出す。
