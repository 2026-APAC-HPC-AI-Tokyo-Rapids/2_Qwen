# 2026-10-18 SM90 MegaMoE の kernel 単体 bench と kernel 内 timeline

対象は Firmus の serving image `e29dfde…`（`configs/firmus-framework.json` の `images.sglang`）に入っている DeepGEMM `sm90_fp8_mega_moe_impl`（dispatch + FC1 + SwiGLU/FP8 量子化 + FC2 + combine を NVLink 越しに融合、EP4、1 CTA/SM の persistent kernel）。これまで kernel 案は数時間かかる serving run でしか判定できず、nsys では kernel が一つの塊にしか見えなかった。本稿は、(A) image を作り直さずに数分で回る 1 ノード kernel bench 経路、(B) kernel 内の段階別 timeline、(C) その初回計測の結果をまとめる。

## 1. 起動方法

- commit message に `[cluster:firmus] [nodes:1] [kernel-bench]` を付けて push する（`[require-build]` は付けない）。`qwen.yml` の `firmus-kernel-bench` job が `validate` の後に `firmus-kernel-bench.yml` を呼び、AWS の container job と serving 用 `firmus` job は走らない。`check-kernel-experiment-tags.py` は他クラスタ・Dynamo・profile・build-only タグとの併用、`[nodes:1]` 欠落、`[require-build]` 付きを validate 段で拒否する（初回 push の本文にタグ文字列をそのまま書いて Dynamo 経路まで起動した事故への対策）。
- DeepGEMM の kernel 案は 1 案 1 ディレクトリで `patches/deep-gemm-bench/<案名>/*.patch` に置く（2026-10-18 に複数案対応へ移行。直下に `*.patch` を置く旧形式は配備段で拒否する）。案名は `[a-z0-9][a-z0-9._-]*` で `image` は予約。各案のパッチは名前順に site-packages を基準に `git apply` する（`patches/deep-gemm/*.patch` と同じ書式で、image に適用済みのパッチの上に重ねる）。案のパッチが当たらないと login node の配備段で止まり、Slurm には投入されない。`hpcai` partition と QoS `normal` の上限は 17 分のため、配備段で **合計 12 Slurm step まで**に制限する。image bench が 1、image timeline が要求時に 1、各案が gate + bench の 2（timeline 要求時はさらに 1）、SGLang 案が両 tree の check の 2（bench script があればさらに 2）。したがって timeline なしの DeepGEMM だけなら 5 案、全案に timeline を付けるなら 3 案まで。これは検証 run 39085655145 の 71〜77 秒/step に基づく通常時の予算で、失敗や長い JIT まで時間内終了を保証するものではない。kernel 案の実験はブランチ上で行い、勝った案は通常の `[require-build]` image build と serving 実行で確かめる。
- timeline（計装版）は案ごとの opt-in。案のディレクトリに空ファイル `timeline` を置くと、その案のパッチ + `patches/deep-gemm-timeline/sm90-mega-moe-timeline.patch` の overlay を作る。`patches/deep-gemm-bench/timeline` を置くと image 版の timeline も取る。timeline パッチが案の上で当たらないときは `logs/timeline-apply-<案名>.log` に残してその案の timeline だけ飛ばし、summary でその案を FAILED にする（job は続ける）。
- SGLang JIT kernel の案は `patches/sglang-bench/*.patch`（`patches/sglang/*.patch` と同じく `/opt/sglang/source` 基準の `python/sglang/...` パス）に置く。配備段で image の `python/sglang` を取り出してパッチを当て、Slurm job ではその tree を `PYTHONPATH` の先頭に置く。image tree と案の tree の両方で `scripts/check-firmus-decode-kernels.py`（`sglang.kernels.jit` の import 元を assert してから実行）を走らせ、`scripts/bench-firmus-sglang-kernels.py` があればそれも両方で走らせる。案の JIT は `TVM_FFI_CACHE_DIR=scratch/tvm-ffi-candidate` で必ず作り直し、image tree は HOME 下の既定 cache（MegaMoE gate と共有）を使う。
- `bench-firmus-sglang-kernels.py` の約束：`/opt/sglang/venv/bin/python bench-firmus-sglang-kernels.py --output <jsonl> --variant <image|candidate>` として image 内の単一 process で呼ばれる（4 GPU が見えるが `cuda:0` を使う）。どちらの tree かは `PYTHONPATH` で決まる。1 行 1 case の JSON で、`case`（両 variant で同じ名前）、`median_us`、`p10_us`、`p90_us`、`output_sha256`（同じ入力に対して決定的な出力 byte の SHA-256）を必須とする。JIT compile と正しさの確認は計時の外で行い、失敗は非 0 終了で返す。summary は case ごとに image と candidate の median 差と hash 一致を表にし、hash 不一致を FAILED にする。
- 配備（`scripts/provision-firmus-kernel-bench.sh`）は image の size と SHA-256 を確認し、`unsquashfs` で image の `deep_gemm`（SGLang 案があれば `python/sglang` も）を取り出し、MegaMoE header を image の build 記録と照合したうえで `overlay/candidates/<案名>`、`overlay/timeline/<image|案名>`、`overlay/sglang` を作り、案と timeline の状態を `results/variants.json` に書く。DeepGEMM は package ディレクトリの header から JIT し、cache key に include header を含むので、overlay を `PYTHONPATH` の先頭に置くだけで kernel が差し替わる。
- Slurm job（`platforms/firmus/kernel-bench/run.sbatch`、hpcai、1 ノード 4 GPU、上限 17 分）は image 内で (1) image 版の計時（job 内の基準。失敗したら job を止める）、(2) image 版の timeline（要求時）、(3) SGLang の image / 案の検査と計時（案があるとき）、(4) 案ごとに数値 gate `check-firmus-megamoe.py`、`scripts/bench-firmus-megamoe.py` による計時（出力 hash を image 版と照合）、要求時は timeline（出力がその案の計時と bit 単位で一致しなければその step を失敗にする）の順に実行する。各 step の終了コードと所要秒は `results/steps.tsv` に残し、gate が落ちた案はその案の残りだけ飛ばして次の案へ進む。案ごとの合否は summary script が `variants.json`・`steps.tsv`・結果から判定し、一つでも失敗があれば summary step を失敗にする。
- 計時の各 case で、各 rank が NVML（`torch.cuda.clock_rate` / `power_draw`）で SM clock と電力を計時ループの直前と直後に読む（CUDA event の区間外）。summary は各時点で最も低い GPU の clock を前 / 後で表示する。検証 run 39085655145 では同じ kernel・同程度の median でも瞬間 clock が 1545〜1935 MHz に変動したため、これを使った時間の正規化はしない。kernel 内の clock は timeline の per-CTA 値を参照する。Prefill は電力制限で clock が 1.47〜1.67 GHz に揺れるので、案の比較はこの列も見る。
- 所要時間は初回で push から archive まで 9 分 39 秒（validate 1 分、配備 1.5 分、queue 待ち約 2 分、Slurm job 2 分 18 秒）。Slurm 内の step は、image 版の計時が約 70 秒（DeepGEMM の JIT を含む）、案ごとに gate 約 70 秒、計時約 40 秒、timeline 約 35 秒（39066974393 の実測）。成果物は `qwen-firmus-kernel-bench-*` として自宅サーバーへ保管され、Actions summary に判定、案の横並び表、詳細表、timeline 表、SGLang 表が出る。手元での再集計は `uv run --no-project --with numpy python scripts/summarize-megamoe-kernel-bench.py <archive>/results/results`（単一案時代の archive は 332e30c 時点の script で読む）。

### 複数案対応の E2E 検証（39085655145）

Actions [39085655145](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39085655145) は成功終了。image bench、SGLang 両 tree の check/bench、コメントのみの DeepGEMM 2 案の gate/bench、1 案の timeline の計 10 step がすべて終了コード 0。数値・出力 hash・timeline の照合と成果物回収も成功した。raw 出力は `/srv/qwen-artifacts/39085655145/1/`（`SHA256SUMS` 付き）。実装は main `1a9bb76`、検証コミット `14c829c` は `archive/experiment/firmus-kernel-bench-multi-20261018` tag から追跡できる。

## 2. bench の条件

- 重みと scale は数値 gate と同じ Qwen EP4 形状（local expert 32、w13 [32,3072,4096] FP8、w2 [32,4096,1536] FP8、block128 FP32 scale）。symmetric buffer の token 上限は serving と同じ 4096/rank（`sglang.sbatch` の `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK`、Prefill/Decode 共通）。
- 1 iteration は serving と同じ `mega_moe_pre_dispatch_sm90` + `fp8_mega_moe`。warmup 5 回、計時 30 回で、iteration ごとの rank 間最大値の median / p10 / p90 を報告する。Decode は CUDA graph を replay する。
- routing は balanced（128 個の一様乱数の上位 8）、local（rank r の token が自 rank の expert 32r..32r+31 だけを選ぶ）、skew（expert j of rank r の logit を log f[r] − 0.25 log(j+1)、f = 1.0/1.1/1.2/1.3 とした Gumbel top-8。16384 token の事前較正で rank 負荷 1 : 1.11 : 1.21 : 1.31、expert 負荷 max/mean 約 2.0。実測負荷は case ごとに記録）。
- `DG_PRINT_CONFIGS=1` で host heuristic の選択を記録した。Prefill（1024/2048/4096）は BLOCK_M128/BLOCK_N256、math 512 thread、3 stage、wave あたり 32 expert。Decode は SwapAB BLOCK_M64/BLOCK_N128、math 256 thread、8 stage で、32〜128 token は wave あたり 32 expert、**4 と 16 token は wave あたり 16 expert**（別の template instantiation で 2 wave）。

## 3. 実行記録

| run | commit | 内容 | 結果 |
|---|---|---|---|
| 39053515799 | b46dccb | bench 経路の追加 | 本文中のタグ文字列で Dynamo 経路も起動したため validate 直後に自分で取り消し。Slurm 投入なし、配備途中の run directory は手で削除 |
| 39053820377 | b90cc52 | タグ検査の追加と再起動（image 版 + timeline） | 成功。Slurm 24368（g1） |
| 39054922367 | 9606b24（実験ブランチ、削除済み） | 中身のないコメント 1 行の候補パッチで candidate 経路を確認 | 成功。Slurm 24373（g4）。12 case 全ての出力 hash が 39053820377 と一致 |

ed0a9fb は集計 script の修正（4.4 節）で、GPU job は伴わない。

## 4. 結果

### 4.1 計時（39053820377、image 版。括弧内は 39054922367 の同一 kernel）

| case | median ms | p10 / p90 ms | rank 負荷 max/mean |
|---|---:|---:|---:|
| decode-balanced-4 | 0.1380（0.1379） | 0.136 / 0.139 | 1.16 |
| decode-balanced-16 | 0.1880（0.1880） | 0.186 / 0.189 | 1.05 |
| decode-balanced-32 | 0.1944（0.1948） | 0.193 / 0.197 | 1.05 |
| decode-balanced-64 | 0.2059（0.2076） | 0.205 / 0.209 | 1.02 |
| decode-balanced-96 | 0.2186（0.2169） | 0.217 / 0.220 | 1.05 |
| decode-balanced-128 | 0.2292（0.2305） | 0.227 / 0.231 | 1.05 |
| decode-local-128 | 0.2256（0.2265） | 0.225 / 0.226 | 1.00 |
| prefill-balanced-1024 | 0.5288（0.5186） | 0.526 / 0.535 | 1.02 |
| prefill-balanced-2048 | 0.9246（0.9069） | 0.917 / 0.938 | 1.02 |
| prefill-balanced-4096 | 1.6680（1.7427） | 1.653 / 1.774 | 1.00 |
| prefill-local-4096 | 1.6683（1.5964） | 1.576 / 1.772 | 1.00 |
| prefill-skew-4096 | 1.8661（1.8421） | 1.834 / 1.972 | 1.13 |

- decode-128 と prefill-4096 は serving 前の gate の記録（0.231 / 1.66 ms）と一致する。
- 同一 kernel の 2 run 間の差は Decode で ±1% 以内、Prefill で最大 ±4.5%。Prefill は 1.65 ms 付近と 1.75 ms 付近の二峰になり、これは後述の GPU clock 低下と連動する。Prefill の候補比較は 1 run の median 差 5% 未満では判定できない。
- local-only は Decode で −1.6%、Prefill で差なし（p10 は 5% 速い）。EP4 ノード内では NVLink 転送量そのものはほぼ効かない。
- skew（rank 負荷 max/mean 1.13）は Prefill で +12%（1.87 / 1.67）。最も重い rank の負荷にほぼ比例して時間が伸びる。expert 配置の均衡化（別作業）が kernel 時間に直接効く根拠になる。
- Decode は 4 token でも 0.138 ms かかる。37 pair で 32 expert のほぼ全てに触れ、重みを読み切るためで、token 数を減らしても下限は重み読み出しで決まる。

### 4.2 Decode-128 の timeline（計装版、代表 call、µs、CTA の median / max、4 rank）

| 段階 | rank0 | rank1 | rank2 | rank3 |
|---|---:|---:|---:|---:|
| 計数 + send-count atomic + 送信元 index 書き込み + grid sync | 3.3 | 3.1 | 3.2 | 3.3 |
| pre-pull NVLink barrier | 7.1 | 7.3 | 6.8 | 6.2 |
| barrier → 最初の token pull（warp 0） | 7.0 / 9.4 | 6.8 / 8.0 | 6.0 / 7.6 | 6.2 / 7.8 |
| 最初 → 最後の token pull（warp 0、4 token） | 15.2 / 22.2 | 14.8 / 18.0 | 14.3 / 18.5 | 14.5 / 18.0 |
| math 開始 → 最初の FC1 stage 到着 | 19.2 / 20.5 | 19.3 / 19.8 | 18.2 / 18.7 | 17.5 / 18.4 |
| FC1（6 block/CTA） | 138.7 / 141.4 | 136.9 / 141.0 | 136.4 / 138.9 | 137.2 / 140.4 |
| FC1 終了 → FC2 最初の stage | 0.8 | 0.8 | 0.8 | 0.8 |
| FC2（8 block/CTA） | 78.5 / 82.7 | 77.6 / 81.5 | 77.3 / 81.7 | 77.8 / 80.4 |
| combine 前 grid sync（rank 内の遅い CTA 待ち） | 8.7 / 27.5 | 7.7 / 35.7 | 7.0 / 32.6 | 6.8 / 27.6 |
| combine 前 NVLink barrier（rank 間待ち） | 1.8 | 4.8 | 6.4 | 6.1 |
| combine | 0.1 / 9.2 | 0.3 / 8.8 | 0.1 / 8.3 | 0.1 / 9.0 |
| 最終 grid sync + NVLink barrier | 2.4 / 5.3 | 2.3 / 4.8 | 2.4 / 5.0 | 2.5 / 4.2 |
| kernel 全体（最後の CTA 終了） | 254.7 | 255.0 | 254.5 | 254.7 |

per-CTA counter（median）：FC1 待ち（full barrier）21 µs のうち約 19 µs は最初の block の dispatch 待ちで、以後の待ちは約 2 µs。FC2 待ち 6.7 µs。TMA A warp の empty 待ち 150 µs、B warp 191 µs。SM clock は全 rank 1.80 GHz。

読み方：

- 起動から最初の FC1 までの約 19 µs（計装版 255 µs の 7.5%）は dispatch。barrier 通過までの 10 µs のうち 7 µs は pre-pull NVLink barrier 自体の往復で、その後も pull は warp あたり 4 token を直列に処理し、1 token 約 5 µs かかる。TMA 1D load の待ちは合計 0.1 µs しかないので、残りは remote SF の 32 float 読み、remote top-k 重みの読み、TMA store とその完了待ちという直列の NVLink 往復。Decode の dispatch は帯域ではなく遅延で律速している。
- GEMM 区間は約 220 µs。重み 604 MB/rank を読む HBM 下限（4.8 TB/s で 126 µs）に対し実効 2.7 TB/s（FC1 2.9、FC2 2.6）。一方で math の stage 待ちは少なく、TMA warp は空き stage を待ち続けている。8 stage の pipeline は常に詰まっており、律速は math 側の k-block ごとの直列処理（SwapAB WGMMA → `warpgroup_wait<0>` → FP32 promote → empty arrive）とみられる。ただし計装の負荷は主にこの loop に乗る（4.4 節）ので、比率には +16% 分の誤差がある。
- FC2 終了から kernel 終了まで 12〜17 µs。内訳は rank 内の遅い CTA 待ち（median 7〜9、最大 28〜36 µs）、rank 間 barrier 2〜6 µs、combine（1 warp 1 token でも最大 9 µs）、最終 barrier。

### 4.3 Prefill-4096 の timeline（計装版、代表 call、µs）

| 段階 | rank0 | rank1 | rank2 | rank3 |
|---|---:|---:|---:|---:|
| 計数 + atomic + 送信元 index 書き込み + grid sync | 11.0 | 8.9 | 9.7 | 9.7 |
| pre-pull NVLink barrier | 5.2 | 9.8 | 5.3 | 10.3 |
| 最初 → 最後の token pull（warp 0、124 token） | 604 / 641 | 595 / 628 | 622 / 657 | 585 / 611 |
| math 開始 → 最初の FC1 stage 到着 | 38 / 51 | 39 / 58 | 40 / 56 | 43 / 62 |
| FC1（25 block/CTA） | 926 / 936 | 868 / 882 | 945 / 958 | 865 / 879 |
| FC2（33 block/CTA） | 604 / 623 | 584 / 603 | 619 / 643 | 597 / 627 |
| combine 前 grid sync（rank 内の遅い CTA 待ち） | 27 / 61 | 32 / 64 | 22 / 38 | 40 / 75 |
| combine 前 NVLink barrier（rank 間待ち） | 43 | 119 | 2 | 95 |
| combine | 84 / 95 | 84 / 92 | 83 / 91 | 87 / 95 |
| 最終 grid sync + NVLink barrier | 8 / 17 | 11 / 16 | 7 / 17 | 9 / 17 |
| kernel 全体 | 1728.8 | 1728.0 | 1722.6 | 1728.3 |
| SM clock（GHz） | 1.54 | 1.64 | 1.50 | 1.64 |

per-CTA counter（median）：FC1 待ち 103〜116 µs（最初の stage まで 26〜40 µs を含む）、FC2 待ち 46〜51 µs、L1 到着待ち（TMA A）32〜37 µs、TMA A/B の empty 待ち 1030〜1280 µs、combine warp 0 の load 待ち 32〜37 µs。

読み方：

- GEMM は約 1.5 ms で 1.24 TFLOP/rank、約 800 TFLOPS。math は FC1/FC2 とも stage 待ちが少なく、TMA warp が空き stage を待っているので、計算律速。速い rank の FC1 は 0.825 TFLOP / 0.87 ms ≈ 950 TFLOPS で、clock 1.64 GHz 時の FP8 dense 理論値（1979 × 1.64/1.98 ≈ 1640）の約 58%。
- **Prefill 中は GPU が電力制限で 1.47〜1.67 GHz まで落ち、GPU ごとに差がある。** 両ノード（g1、g4）とも rank 2（GPU 2）が最も遅い（1.47〜1.50 GHz）。FC1/FC2 の所要時間は clock に反比例して rank 間で 8% ずれ、速い rank は combine 前の NVLink barrier で最大 119 µs（kernel の 7%）待つ。kernel 時間は一番遅い GPU の clock で決まり、同じ run の中でも GPU 2 が 1.60 GHz に戻った call は 1657 µs、1.50 GHz の call は 1729 µs だった。Decode は全 GPU 1.80 GHz で揃っている。
- dispatch の pull は 600 µs かけて FC1 と重なっており、L1 到着待ちは合計 30 µs 台で律速ではない。重ならない部分は、起動から最初の FC1 まで約 40 µs（2.3%）と、最後の FC2 から終了まで 95〜240 µs。後者は rank 内の遅い CTA 待ち、rank 間の clock 差による待ち、直列の combine 83〜87 µs（4096 token × top-8 × 8 KiB = 268 MB を 3.2 TB/s 程度で読む）、最終 barrier 約 8 µs から成る。

### 4.4 計装の overhead と注意点

- 出力は両 case とも計装なしと bit 単位で一致した（max abs diff 0）。数値 gate も image 版・候補版の双方で PASS。
- overhead（30 回の median 比）：Decode-128 は +16.0% と +15.4%。Prefill-4096 は +3.3% と −3.5% で、run 間の揺らぎ（±4.5%）に埋もれる。Decode の増分は主に math loop にかかる。k-block ごとの clock 読み、lead thread の SMEM 加算、timeline pointer の spill reload（CUDA 13.0 の ptxas で Decode の spill は 0 → 28/160 byte、Prefill は 40/40 → 72/72 byte）が原因。Decode の GEMM 区間の絶対値は約 15% 長めに出ている。dispatch・barrier の区間はほとんど計装されていない。
- dispatch warp と math warp の間の名前付き barrier（`barrier.sync 1`）は `BAR.SYNC.DEFER_BLOCKING` になる。その直後に置いた `%globaltimer` 読みは barrier 解放より前に retire する（math が dispatch 到着の 9 µs 前に handshake を「通過」していた）。このため集計では、この二つの handshake 事象で終わる段階を作らない。待ちは次の段階に含めて表示する（ed0a9fb）。grid sync と NVLink barrier の事象は spin する thread 自身が記録するので正確。
- CTA 内の時刻は clock64 差分を CTA ごとに実測した SM clock で換算し、CTA の起点は `%globaltimer`（刻み 256 ns）で揃えている。
- timeline パッチは header の文脈行に依存するので、候補パッチが同じ箇所を触ると適用に失敗する。配備段はその案の timeline だけを飛ばして記録する（1 節）。必要なら timeline パッチを候補の上で作り直す。

## 5. 想定外だった点

1. Prefill MegaMoE は電力制限で 1.5〜1.65 GHz に落ち、GPU 2 が両ノードで最も遅い。rank 間の待ちと run 間の二峰性の主因になっている。Prefill の候補比較では clock を記録しないと誤判定する（timeline の per-CTA clock 推定が使える）。
2. Decode の dispatch は 1 warp 4 token の直列 NVLink 往復で 1 token 約 5 µs かかり、起動から FC1 開始まで約 19 µs かかる。
3. Decode の GEMM は HBM 帯域の 6 割程度で、stage 待ちが少なく、math 側の k-block 直列処理で頭打ちになっているように見える（計装の割り増しに注意）。
4. 4 token と 16 token の Decode は wave あたり 16 expert の別 instantiation を使う。serving の Decode には二種類の kernel が混在する。
5. EP4 ノード内では local-only routing でも速くならず、通信量より負荷の偏り（skew で +12%）が効く。

## 6. 次の候補（この bench で数分で判定できるもの）

- Decode dispatch：remote SF と top-k 重みの読みを TMA 1D load と同じ mbarrier にまとめる、1 warp が複数 token の load を先行発行する。目標は起動 → FC1 の 19 µs の短縮。
- Decode math：SwapAB の k-block loop で promote と次 stage の WGMMA を重ねる（`warpgroup_wait<1>` の段数を増やす）。HBM 下限 126 µs との差 約 90 µs が対象。
- Prefill tail：combine を最後の FC2 と重ねるか、rank 内の遅い CTA を減らす block 割り当て。clock 差による rank 間待ちは kernel の外（電力・clock 設定）の問題として切り分ける。
