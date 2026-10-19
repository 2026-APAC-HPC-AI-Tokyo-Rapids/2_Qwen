# 2026-10-19 Firmus Prefill bundle の統合 branch（`perf/firmus-prefill-20261019`）

origin/main（7999.37 構成 + ノート、起点 16bdbe1）から切った、次の Prefill bundle の統合先。`perf/firmus-max-20261018`（a7b9edd）にあって main に無い 3 変更のうち、Prefill に限定できるものだけを持ち込み、MegaMoE と attention の Codex 作業（`perf/firmus-prefill-megamoe-20261019` d9b1b2a、`perf/firmus-prefill-attn-20261019` 2e85ef7）は §5 のとおり merge commit で取り込んで配線した。runtime 設定は main のまま（Decode cap 128、graph bs 34）。GPU job・Actions は起動していない。

| 変更 | 扱い |
|---|---|
| (a) flat NVLink barrier | **持ち込まない**（Prefill 限定では相互運用できない。§1） |
| (b) fused 入力 RMSNorm + QKV quant | Prefill EXTEND かつ 2048 行以上に限定して持ち込んだ（§2） |
| (c) in-image の DeepGEMM host 再 build | 基盤と SwapAB-144 host patch・header guard を持ち込んだ（§3） |

## 1. flat NVLink barrier を Prefill に限定できない理由

[flat barrier note](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/blob/perf/firmus-max-20261018/research/runtime-performance/2026-10-18-megamoe-flat-nvlink-barriers.md)（max branch）は「異なる geometry が同じ collective に混ざることは許す」と書くが、これは**全 geometry が flat の場合**の話で、Prefill だけ flat にする場合は成り立たない。

- 現行（original）の `comm::nvlink_barrier` は 1 call に 3 回（pre-pull、pre-combine、cleanup 後）。SM0 が全 rank の `signal[phase]` に ±1 を `red.release.sys` し、自 rank の signal が `kNumRanks`（符号が反転する周期では 0）になるまで待つ。phase と sign は rank ローカルの `counter & 3` で、barrier を呼ぶたびに進む。つまり**全 rank が同じ回数・同じ順で呼ぶ**ことが前提で、待つのは他 rank 全員の signal。
- flat は 3 回の original barrier をすべて外し、別の counter（L2 arrival-mask 末尾 2 slot の 64-bit counter と、per-expert count sum の high word）に置き換える。1 call あたり全 CTA が全 rank に 1 回ずつ加算し、各 rank は 132 × 4 = 528 の到着を待つ。
- Prefill だけ flat にすると、Prefill rank は original signal を送らず、Decode geometry の rank は永遠に待つ（300 秒の timeout で trap）。逆に Prefill rank は、Decode geometry の rank が加算しない flat counter と count high word を待つ。双方向の deadlock で、original の phase counter も rank 間でずれる。
- geometry は host が**各 rank の局所 token 数**で選ぶ（256 token/rank 以上で Prefill の M128、未満は M64：128 以下が SwapAB、129〜255 が非 SwapAB）。DP attention の Prefill では rank ごとの token 数が独立なので、1 つの collective に混在し得る。idle rank は `megamoe-idle-prefill-geometry` で 256 行の masked row を通して Prefill geometry に揃えたが、256 token 未満の実 batch（短い prompt、端数 chunk）を持つ rank は Decode geometry になる。gate の `[7, 0, 65, 129]` も同じ状況を作っている。

回避策と判定。(i) Prefill が original signal も送る：barrier が減らず利得が無い。(ii) Decode も flat にする：Decode SASS が変わり、max branch の run 39119219065 で Decode MegaMoE が +3.3%。(iii) SGLang 側で Prefill サーバーの全 rank の geometry を揃える（DP 全 rank の token 数から、最大が 256 以上なら全 rank を 256 行以上に padding し、全 rank が 256 未満なら全 rank が Decode geometry）：barrier の protocol も揃うので Prefill 限定の flat が成立し得るが、小さい rank の padding 計算と mixed の gate 変更を伴い、GPU 検証が要る。見込みは MegaMoE の Prefill −1.2%（約 −20 µs/層 × 94 層 ≒ −1.9 ms/step、約 −0.7%）。今回は実装していない。

圧縮した結論：Decode の SASS を保ったまま Prefill だけ flat にする patch は作れるが、serving で geometry が混在すると hang するため、branch には入れていない。`patches/deep-gemm/sm90-flat-nvlink-barriers.patch`（main にある旧版、series 外）はそのまま。

series の 14 本は main のままで、Firmus login の CUDA 13.0.88 nvcc（image 39110496019 から `unsquashfs` で自分のディレクトリへ抽出、確認後に削除）で `build-sm90-megamoe.py` と同じ instantiation を compile した。

| chain | Prefill reg / spill store/load B | Decode / Decode-16 reg / spill | Decode SASS |
|---|---|---|---|
| main の 14 本（image 39110496019 の installed header と SHA-256 一致） | 96 / 44/60 | 168 / 0、168 / 0 | `5bcd675b848f5341` / `a0e029dd6a8078fc` |
| + `sm90-swap-ab-geometry-guard`（15 本、本 branch） | 96 / 44/60 | 168 / 0、168 / 0 | 同一（Prefill `a12f4c30a49f8420` も同一） |

## 2. fused 入力 RMSNorm + QKV quant を Prefill に限定

kernel は 1 CTA 2 行。run 39119219065 の profile：

| | flashinfer + group quant | fused |
|---|---:|---:|
| Prefill（約 3986 行） | 31.4 + 19.9 = 51.3 µs/層 | 46.4 µs/層 |
| Decode（約 126 行、63 CTA） | 3.2 + 2.0 = 5.2 µs/層 | 8.8 µs/層 |

2 点を通る直線は unfused が 3.70 µs + 11.94 ns/行、fused が 7.57 µs + 9.74 ns/行で、**約 1760 行で交差**する。fused は 89 register で 1 SM に 5 CTA、すなわち 660 CTA = 1320 行で 1 wave になり、約 2.4 TB/s 相当で帯域律速にも届いていない（unfused は約 3.4 TB/s）。ゆえに低行数側は線形でなく、交差点は 2 点からのモデルにすぎない。閾値は **2048 行**とした。交差の少し上で、そこでの利得は 1 µs 未満と予測されるので多少ずれても損失は小さく、request-aligned chunk の標準（3986〜4027 行）は 4.9 µs/層（約 0.46 ms/step、約 −0.16%）を得る。

- 経路に入る条件は `forward_batch.forward_mode == ForwardMode.EXTEND` かつ `hidden_states.shape[0] >= _INPUT_NORM_QUANT_MIN_ROWS`（2048）に、従来の構造条件（残差あり、aux hidden 取得層でない、all-reduce 融合・scattered input・batch-invariant でない、BF16 連続、DeepGEMM block-FP8 の QKV linear）を足したもの。DECODE、TARGET_VERIFY（Decode サーバーの verify）、MIXED、DRAFT_EXTEND、idle、小さな EXTEND は行数によらず従来の flashinfer + group quant。
- CPU の routing check（`scripts/check-firmus-latency-source.py`）はモード（DECODE / TARGET_VERIFY / MIXED は 4027 行でも不使用）と行数（4027、3986、閾値は使用。閾値 − 1、126、1、0 は不使用）を追加で検査する。
- GPU check（`scripts/check-firmus-decode-kernels.py`）は e8b9817 の基準のまま：residual は bitwise、scale は 1 BF16 ulp 以内、逆量子化値は E4M3 の 1 段以内、異なる FP8 code 数は記録のみ。行数に 2047、2048、2049 を足した。
- 次の kernel bench で閾値を実測で確かめられるよう、`scripts/bench-firmus-sglang-kernels.py` の norm + quant の行数を 4〜4096 の 15 点（1024、1536、2048、3072 を含む）にした。

## 3. host 再 build の基盤

DeepGEMM 0.1.5 の geometry 選択は wheel の `_C.so`（host C++）にあり、header patch の外にある。`containers/rebuild-deepgemm-host.py` が v0.1.5 の source（commit `1c8138a5…`、submodule 込み）に `patches/deep-gemm-host/series` の host patch を順に当て、`_C.so` だけを置き換える（installed の JIT header は保持）。記録 `/opt/apac/deep-gemm-host-build.json`（`sm90-megamoe-build.json` の `host_build` に取り込み）は patch ごとの SHA-256 を持ち、`check-firmus-image.sh` が image の実 `_C.so` の hash・patch 列・source commit・ABI を照合する。**host patch は 1 本以上が必須**（空の series は rebuild も image check も失敗する。patch 無しでは wheel の検証済み `_C.so` を同等のものに置き換えるだけで、build 時間と ABI のリスクだけが増えるため）。

- host patch 1：`patches/deep-gemm-host/sm90-swap-ab-144.patch`（max branch のまま）。SwapAB の選択を 144 token まで広げる。Decode cap 128 では 129〜144 token の call が無いので serving では使われない。
- header patch：`sm90-swap-ab-geometry-guard.patch` を series の 15 本目に追加（host と header が食い違って SwapAB を M128 や N256 に指定したら compile error にする静的検査で、演算は変えない）。header pin は `5644dfa107d08a166c1a439cd33bcef5d703670f3adc39554fc884a2031361b1`。
- gate：`scripts/check-firmus-megamoe.py` に `swapab-boundary-mixed [128,129,144,145]`、`decode-graph-swapab-144 [144,140,136,132]`（graph replay）、`swapab-hot-experts-144 [144]*4`、`mixed-geometry-graph [7,0,129,256]` を追加（許容値は不変）。
- 再 build の実行（link、ABI import、`_C.so` の置換）は、この branch でも main でもまだ一度も走っていない。run 39119219065 で max branch の同じ基盤が build と gate を通っている。

## 4. 検査

CPU のみ。pytest 全件、ruff、`bash -n`（scripts / platforms / containers / aws / submission）、行継続 grep、`check-kernel-experiment-tags.py` は通過。SGLang の 24 本（既存 23 + fused norm + quant）を fc275674 の full tree（sparse checkout 無効）に順に当て、`check-firmus-staging.py`、`-megamoe-source.py`、`-latency-source.py`、`-prefill-chunks.py`、`-tier2-source.py admission dp-ties idle-geometry` がすべて PASS。

## 5. Codex 2 branch の取り込みと最終構成

両 branch は merge commit で取り込んだ（由来を残す）。conflict は、DeepGEMM の series と header pin、`check-firmus-megamoe.py` の gate case、`check-firmus-decode-kernels.py` の check 関数と `__main__` の呼び出し。後者は両方の関数を残して両方を呼ぶ（fused norm + quant と contiguous Prefill attention）。attn branch の check script は CI の ruff 規則（B023、S102、C408、RUF059）に通らなかったので、行単位 noqa、file 単位の B023 noqa（入れ子の helper はそれを定義した loop の反復内で呼ばれる）、未使用変数の改名で通した。

最終構成（image は新規 build が要る。Decode cap 128 のまま）。

- DeepGEMM host series：`sm90-swap-ab-144.patch` → `sm90-prefill-pingpong.patch`（pristine v0.1.5 `1c8138a5` に順に `git apply` できることを確認）。
- DeepGEMM header series：main の 14 本 → `sm90-swap-ab-geometry-guard.patch` → `sm90-prefill-pingpong.patch`。pristine v0.1.5 の header から順に当てた最終 header は `0c9880757092e4713ab9ff13fb1388fbee2e198b6aff324d0c772118e8fafdb2`（worker 自身の pin `13664810…` は guard を含まない 15 本のもの）。
- `containers/build-sm90-megamoe.py` の Prefill compile 設定は block_m 128、block_n 128、epilogue_threads 256、stages 5、registers 168、reuse_accum False、arrival_counter True、full_sync False、swap_ab False。Decode は従来どおり。`scripts/check-firmus-image.sh` の geometry 照合と image test も同じ値にした。
- `scripts/check-firmus-megamoe-pingpong-build.py` は image 39110496019 の抽出 toolchain を要る offline の比較ツールで、build step ではない（image build 内の compile は `build-sm90-megamoe.py` が同じ geometry で行う）。guard を先に当てるよう直した。
- SGLang chain：既存 23 本（`add-rmsnorm-group-quant` を含め 24 本）→ `prefill-attention-contiguous-kv`（`gigatoken-encode` の後が最後）。manifest の target は rope.cuh、ops/attention/rope.py、qwen3_moe.py、flashattention_backend.py、prefill_contiguous.py。`build-sglang-container.sh` は `check-firmus-prefill-attention-source.py` も実行し、`containers/check-rope-kernel-build.py` は `QKNormRopeFP8Kernel<true/false>::run_prefill` を compile する。image の必須 patch 一覧とテストにも追加した。

Prefill MegaMoE の ping-pong は 128×128 / 256 math thread / 5 stage / 168 register に host と header の両方が変わるため、header だけの overlay の kernel bench では新経路に入らない。新しい `_C.so` と header の両方が要る。host 再 build の実行は本 branch ではまだ走っていない。

## 6. 最終 compile（Firmus login、image の CUDA 13.0.88、CPU のみ）

`build-sm90-megamoe.py` と同じ instantiation を、最終 header（16 本、上の pin）から compile した。

| geometry | register | spill store/load B | stack B | SASS（先頭 16 桁） |
|---|---:|---:|---:|---|
| Prefill 128×128 / 256 math thread / 5 stage（ping-pong） | 168 | 0 / 0 | 0 | `5ffea0804097b207` |
| Decode wave 32 | 168 | 0 / 0 | 8 | `5bcd675b848f5341` |
| Decode-16 | 168 | 0 / 0 | 8 | `a0e029dd6a8078fc` |

Decode の 2 つの SASS は、ping-pong を足す前の 15 本（guard まで）の compile と byte 単位で同一（Prefill は旧 96 register・44/60 B spill から変わった）。`check-firmus-megamoe-pingpong-build.py` も同じ toolchain で PASS（候補に spill と WGMMA 直列化警告が無く、Decode の base と候補が同一）。抽出した toolchain は削除した。

## 7. 検査

CPU のみ。pytest 全件、ruff、`bash -n`、行継続 grep、`check-kernel-experiment-tags.py` は通過。SGLang の 25 本を fc275674 の full tree に順に当て、`check-firmus-staging.py`、`-megamoe-source.py`、`-latency-source.py`、`-prefill-attention-source.py`、`-prefill-chunks.py`、`-tier2-source.py admission dp-ties idle-geometry` がすべて PASS。GPU の gate と性能は未実行。

## 8. 今後 patch を足すとき

- host patch：`patches/deep-gemm-host/` に置き、`series` の末尾に 1 行。
- header patch：`patches/deep-gemm/` に置き、`series` の末尾に 1 行、`header.sha256` を pristine v0.1.5 から全 series を当てた最終 header で更新。
- SGLang patch：`build-sglang-container.sh` の変数、`sglang_patches` の末尾、target の `case`、`check-firmus-image.sh` の必須一覧、`test_firmus_image.py` の `SGLANG_PATCHES`。
