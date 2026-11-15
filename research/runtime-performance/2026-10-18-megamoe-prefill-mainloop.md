# 2026-10-18 Prefill MegaMoE mainloop の再構成（FFMA 形 promote、次 K block の scale 先読み）

精度方針の変更（`task/1. Qwen3 AI Task Rules.md` Q4：浮動小数点の加算順序と FMA 縮約の違いは許容。FP8 未満への精度低下、高精度データの新たな FP8 化、計算の省略は不可）を受けて、geometry note（当時の資料 `2026-10-18-megamoe-geometry-and-redundancy.md`、現行ツリーには非同梱） §2.5 で保留していた Prefill MegaMoE の promote を FFMA 形（別 accumulator に積んで `final += s·acc`）に替える案と、mainloop の他の再構成を実装・計測した。対象は Firmus の serving image `39070273796.sqsh`（MegaMoE header sha256 `135a43be…`。v0.1.5 の header に main の `patches/deep-gemm/*.patch` 8 本を `containers/build-sm90-megamoe.py` の順で当てたものと一致）の Prefill geometry（BLOCK 128×256、math 4 warpgroup = 512 thread、3 stage、reuse accumulator、phase 特化）。

## 1. 結論

- **FFMA 形の promote は採らない。** image の Prefill は `sm90-prefill-rescale.patch` で既に「持ち越した scale を 1 要素 1 回の乗算で更新する」形になっており、FFMA 形は演算数が同じ（§2）。そのうえ別 accumulator 64 個が 96 register の上限に入らず、試作は 1856 B spill し、ptxas が全 WGMMA を直列化する（C7512）。FC2 の K 半分 2 group の統合も同じ理由で不可、warpgroup の epilogue ずらしは 3 stage の ring で 0.7 µs 分しかずらせない（§3）。
- **採用候補は 1 本：`patches/deep-gemm/sm90-prefill-mainloop-prefetch.patch`。** Prefill の FC1/FC2 loop で、次 K block の barrier 待ち・scale 読み込み・rescale 比の計算を現在の WGMMA の実行中に済ませ、descriptor を加算だけで進める（§4）。演算式と順序は変えていないので出力は image と bit 一致する見込み。Decode の SASS は byte 単位で不変、Prefill は 96 register・spill 44/56 B（image 40/40 B）。
- **GPU 計測は未実施（保留）。** 実験ブランチで 1 回投入したが、全 worker の候補を 1 本の kernel bench にまとめる方針になり、開始前に取り消された（§7）。期待効果は Prefill MegaMoE の GEMM 区間で 0〜6%（§5）で、統合 bench で確かめる。

## 2. 前提の訂正：image の Prefill は既に 1 要素 1 乗算

geometry note は upstream の reuse 経路（k-block ごとに 1/scale を掛ける → WGMMA → scale を掛ける、2 乗算）を見ていた。image には `sm90-prefill-rescale.patch`（58e29c5）が入っており、Prefill は大 hidden 用の「scale を持ち越す」経路を使う。

| 段 | k-block あたりの処理（warpgroup ごと） | accumulator 1 要素あたりの乗算 |
|---|---|---:|
| FC1 | full wait → SFA 2 個（LDS）と weight scale 2 個（LDG）→ 比 (前 scale / 今 scale) を 4 個（RCP 4 回）→ 64 要素に掛ける → WGMMA 4 本 → `wait<0>` → empty arrive | 1 |
| FC2 | 同様に比 2 個 → 掛ける → K 前半 WGMMA 2 本 → `wait<0>` → per-64 activation scale の比を掛ける → K 後半 WGMMA 2 本 → `wait<0>` → arrive | 2（K 半分ごとに 1） |

FFMA 形は 1 要素 1 FFMA で、**演算数は FC1・FC2 とも現状と同じ**。利点があるとすれば依存関係だけである。

## 3. 制約：96 register

- Prefill kernel は `__launch_bounds__(640, 1)` なので ptxas は 65536 / 640 → **96 register** で割り付ける（nsys の起動 metadata も 96）。`setmaxnreg` の 112 は効かない（Decode でも同じ現象、[Decode mainloop note](2026-10-18-megamoe-decode-mainloop-pipeline.md) §1.1-4）。image の時点で 40 B spill している。
- accumulator は 64×128 / 128 thread = 64 個。FFMA 形はもう 64 個の FP32 accumulator を常に持つ必要がある。
- 実測（image の CUDA 13.0 ptxas、`containers/build-sm90-megamoe.py` と同じ instantiation）：

| 版 | register | spill store / load | ptxas の警告 |
|---|---:|---|---|
| image | 96 | 40 / 40 B | なし |
| FFMA 形（既存の非 reuse 経路をこの geometry に使う） | 96 | 1856 / 1768 B | **C7512**（register 不足で全 WGMMA を直列化） |
| 先読み案（§4） | 96 | 44 / 56 B | なし |

- 同じ理由で **FC2 の K 半分 2 group を 1 つにまとめる案も成り立たない**。per-64 の activation scale が K 半分ごとに変わるので、まとめるには K 後半を別 accumulator に積む必要がある（64 register）。scale を FP8 の operand 側に掛けるのは再量子化で、精度規則に反する。
- **warpgroup の epilogue をずらす案**：4 warpgroup は同じ stage ring（3 stage）を共有し、各 stage はすべての warpgroup が読み終わるまで解放されない。先行できるのは最大 2 stage で、その 2 stage 分の WGMMA は 2 warpgroup 分の仕事（128×256×128 tile の半分）なので tensor core 時間で約 0.7 µs（1.55 GHz）にしかならない。FC1 の epilogue（SwiGLU、amax、FP8 化、TMA store と完了待ち）を隠すには足りず、ring を深くする SMEM も無い（225.6 KB / 227 KB）。実装しなかった。

## 4. 実装：次 K block の scale 先読みと descriptor の加算化

SASS を見ると、image の FC1 は full barrier の wait から最初の QGMMA までに約 166 命令がある。内訳は weight scale の LDG 2 本と SFA の LDS 2 本、それに依存する FMUL→MUFU.RCP→FMUL の鎖（4 本）、64 個の FMUL、そして `make_smem_desc` で 8 個の descriptor を pointer から作り直す約 70 命令（シフト・マスク・R2UR）。LDG（L1 hit でも約 30 cycle、miss なら数百）と RCP の待ちがそのまま k-block ごとの critical path に乗る。

`patches/deep-gemm/sm90-prefill-mainloop-prefetch.patch`（純粋な挿入 154 行）：

- `kQwenPrefillMainloop`（H4096 / I1536 / BLOCK 128×256 / math 512 thread / phase 特化）のときだけ、既存の rescale 経路の前に専用 loop を置く。それ以外の geometry と Decode は既存のまま。
- 現在の WGMMA group を発行した後、`wait<0>` の前に、次の stage の full barrier を待ち、次 k-block の SFA と weight scale を読み、rescale 比を **今と同じ式で** 計算しておく。WGMMA 間の critical path に残るのは accumulator への乗算だけになる。FC2 は K 後半の group の実行中に次 k-block の 2 種類の比（k-block 頭と K 半分の境目）を作る。最初の k-block は rescale が不要なので、その scale は WGMMA の実行中に読む。
- descriptor は warpgroup ごとの base を 1 回作り、下位 32 bit に stage と K の offset（16 byte 単位）を足すだけにした。SMEM は 256 KiB 未満で offset はすべて 16 byte の倍数なので、14 bit の address field から桁あふれしない。arrive → 最初の QGMMA は 54 → 36 命令。
- 演算式も順序も変えていないので、**出力は image と bit 一致する見込み**（§5 で確認）。
- Decode（wave 32 と wave 16）の SASS は image と byte 単位で同一（sha256 `3b7c22b8…` / `b97d51e8…`）。
- W2 の `sm90-prefill-tail.patch`・`tail-dyn.patch`・`drop-l1-weight.patch` とは前後どちらの順でも `git apply` が通る（触る行が離れている）。timeline patch も上に当たる。ただし timeline の FC1/FC2 full-wait counter と「最初の FC1 stage」事象は既存 loop の `full_barriers[...]->wait` を置き換える形なので、この loop では記録されない（block の busy 時間、per-CTA clock、kernel 全体は記録される）。

## 5. 期待効果と数値 gate（計測は保留）

期待効果の根拠：

- FC1 の 1 k-block は timeline から最大 1.13〜1.16 µs（1.54 GHz で約 1750 cycle）で、tensor core の理想は 4 warpgroup 合計 1024 cycle。warpgroup ごとの直列部（barrier → 最初の QGMMA）から外れるのは、LDG の待ち（L1 hit で約 30 cycle、miss なら数百）、LDS の待ち（約 30）、FMUL→RCP→FMUL の依存鎖（約 30〜60）、descriptor の約 18 命令で、合わせて 1 k-block あたり約 100〜150 cycle。warpgroup の直列部が tensor core の空きの原因なら FC1/FC2 の GEMM 区間で最大 5〜6%、tensor core 側（SMEM 帯域など）が律速なら 0% になる。命令の総数はほぼ変わらず（先読み分は WGMMA の実行中に移るだけ）、issue 帯域の律速は改善しない。
- 先例：Decode の SwapAB loop で weight scale の `__ldg` を full barrier の前に出した hoist は Decode を 2.4〜11.2% 短縮した（[Decode mainloop note](2026-10-18-megamoe-decode-mainloop-pipeline.md) §1.2）。Prefill は 4 warpgroup が tensor core を共有するので隠れる分が大きく、効果は小さい側に見込む。
- Prefill kernel 全体（約 1.7 ms）のうち GEMM 区間は約 1.5 ms なので、kernel 時間では 0〜5%、Prefill step（MegaMoE が 58%）では 0〜3%。

数値 gate（`scripts/check-firmus-megamoe.py`、閾値は不変：relative L2 < 0.025、max scaled error < 0.08）：

- 先読み案は rescale 比を同じ括弧付けの積（`(prev_a·prev_sf)·rcp(a·sf)`、FC2 の境目は `lo·rcp(hi)`）で作り、同じ乗算を同じ順で accumulator に掛け、同じ WGMMA を同じ operand で発行し、最後に同じ postscale lambda を呼ぶ。加算を含まないので FMA 縮約の余地もない。したがって出力は image と bit 一致し、gate の誤差も image と同じになるはず。
- image kernel の gate 値（39085655145 の注釈だけの候補 `a-comment`。kernel は image と同一）：Prefill 経路を通る `capacity-hot-experts`（4096 token/rank）が relative L2 0.01736 / max scaled 0.02148、全 case の最大が 0.01946 / 0.02326。先読み案でも同値を期待する。
- kernel bench の出力 hash（12 case）が image と一致すれば bit 一致の確認になる。Prefill の case で一致しなければ不具合として扱う。

FFMA 形の試作（`prefill-fma`）は gate は通る見込みだが、spill と C7512 で遅くなるのが確実なので統合 bench には入れない。

## 6. 配線（未実施、integrator 向け）

build への追加は行っていない。image に入れる場合：

1. `scripts/build-sglang-container.sh`：`sm90-dispatch-pull-overlap.patch` の cp の次に `patches/deep-gemm/sm90-prefill-mainloop-prefetch.patch` を context へ cp する。
2. `containers/Dockerfile`：`COPY sm90-dispatch-pull-overlap.patch …` の次に `COPY sm90-prefill-mainloop-prefetch.patch /opt/apac/sm90-prefill-mainloop-prefetch.patch`。
3. `containers/build-sm90-megamoe.py`：`sm90-dispatch-pull-overlap.patch` の適用の**後**に `--check` 付きで適用し（8 本を当てた header に対して作ってある）、`result["prefill_mainloop_patch_sha256"]` を記録する。compile する geometry は変えない。
4. `scripts/check-firmus-image.sh` と `qwen-experiment-metrics/tests/test_firmus_image.py`：新しい patch の sha256（`c297df4a…`）の照合を足し、`header_sha256` を `135a43be…` から `48407619e711f1aee4de0c7d170fe7e122afb1c7a7c69fb365b8b7d3215d418f` に更新する（W2 などの patch を併せて入れる場合は全部当てた後の値になる）。
5. kernel bench で測る場合は、同じ patch を `patches/deep-gemm-bench/<案名>/` にそのまま置けばよい（site-packages 基準で image に当たる）。timeline を付けるなら空ファイル `timeline` も置く。

他の worker の patch との関係：W2 の Prefill tail 系（`sm90-prefill-tail.patch`、`tail-dyn.patch`、`drop-l1-weight.patch`）とは前後どちらの順でも当たる。Decode 系は Decode の SASS が不変なので干渉しない。W2 の tail と組み合わせたときの register / spill は未確認なので、統合時に compile で見る必要がある。

## 7. 作業記録

- 計算環境：Firmus login node（`/tmp/claude-pm-20261018`）に image の CUDA 13.0 と `deep_gemm` を `unsquashfs` で取り出し、`containers/build-sm90-megamoe.py` と同じ template 引数で Prefill / Decode（wave 32）/ Decode（wave 16）を `--ptxas-options=-v,-warn-spills` で compile し、SASS を比較した。
- 実験ブランチ `experiment/firmus-prefill-mainloop-20261018`（`3326a3a`）で Actions [39094510670](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39094510670) を 1 回起動した（候補：先読み案 ×2 標本、FFMA 形）。Slurm 24468 は queue 待ちの間に coordinator が取り消し、GPU では一度も動いていない。後片付け step が失敗して残った Firmus の runtime directory は手で削除した。実験ブランチは `archive/experiment/firmus-prefill-mainloop-20261018` に退避して削除した。
- 計測は coordinator がまとめる統合 kernel bench で行う。
