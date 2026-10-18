# 2026-10-18 SM90 MegaMoE Prefill の末尾短縮：FC2 block の動的取得と combine の連続 load

[kernel timeline](2026-10-18-megamoe-kernel-timeline.md) の 4.3 節によると、Prefill-4096 の MegaMoE は中央値の CTA が最後の FC2 を終えてから kernel 終了まで 95〜240 µs かかる。内訳は、rank 内で遅い CTA を待つ時間（median 22〜40 / max 38〜75 µs）、rank 間の NVLink barrier（主に GPU ごとの clock 差）、直列の combine 83〜87 µs、最終 barrier である。Prefill の GPU 時間の 58% は MegaMoE なので、この末尾が Prefill 供給を縛る（[計画](2026-10-18-firmus-maximum-performance-plan.md) の 2-1）。本稿では Prefill 専用の header patch を作り、1 ノードの kernel bench で 2 回測った。

結論：**`patches/deep-gemm/sm90-prefill-tail.patch`（FC2 block の動的取得 + 連続 load の combine）を採用候補とする。** 2 回の job とも、全 12 case の出力 hash が image と一致し、数値 gate も通った。電力制限の効かない Prefill-1024/2048 では image 比 −5.3〜−6.1%。Prefill-4096 は p10 で balanced −2.2% / −3.3%、skew −5.0% / −4.8%、local −5.4% / −5.2% だった。効果の大部分は FC2 の動的取得による。combine の変更は、clock の安定した case で追加 0.4〜1.0% を出した。Decode と Decode-16 の SASS は変わらない。serving image への組み込み（`[require-build]`）と serving run はまだ行っていない。

## 1. 実装

起点は image `e29dfde…` の header（`patches/deep-gemm/*.patch` の 8 本を build 順に当てたもの、sha256 `135a43be…`）。変更は `kSpecializePhase`（Qwen Prefill の BLOCK_M 128 / BLOCK_N 256 / math 512 thread 形状）かつ SwapAB でない場合に限った。Decode の instantiation は同じ SASS になる。

### 1.1 FC2 block の動的取得（`kDynamicL2`）

従来は、FC1 と FC2 の block 列を 1 本の stream にし、CTA c に c, c+132, … を静的に割り当てていた。Prefill-4096 では 1 rank あたり FC1 が約 3100 block、FC2 が約 4100 block ある。CTA ごとに FC1 の数が 1 block（約 37 µs）違い、FC2 の数も 1 block（約 18 µs）違う。さらに最初の FC1 は token pull の到着順に始まるので、CTA の開始時刻は最大約 25 µs ずれる。この差がそのまま、combine 前の遅い CTA 待ちになっていた。

変更点：

- FC1 は従来どおり静的に割り当てる。FC2 は rank ごとの counter から順番に取る。counter には workspace の未使用の grid-sync slot 3 を使い、既存の cleanup（combine 前 barrier の後、SM0）で 0 に戻す。
- 取るのは A loader（TMA A warp）である。FC2 に入る役割の中で最初に次の block を必要とするからである。relaxed な `atomicAdd` を 1 block 先に発行して、その遅延を現在の block の処理に隠す。取った番号は scheduler の expert 走査で decode する（expert、m block、n block、pool block offset、expert の token 数）。結果は 2 slot の ring で B loader と 16 個の math warp に渡す。ring は expert-count 領域（1 KiB）の末尾 64 byte に置き、full / empty の mbarrier 各 2 個を使う。受け手は scheduler の状態を FC2 の間保持しない。このため math warp の register 圧力は増えなかった。
- block の中身は従来とまったく同じ計算である。1 CTA が取る番号は単調増加し、FC2 が FC1 を待つ依存（L2 arrival counter）も従来どおりなので、新しい待ち合わせの循環は生じない。
- expert-count 領域の先頭 512 byte は dispatch の計数用、その直後の 96 byte は timeline 計装の counter 用である。combine の smem はこの 1 KiB の後ろから使うようにずらした。こうすると、B loader が最後の終端番号を読む前に combine がその領域を上書きする競合が、形式上もなくなる。

### 1.2 combine の連続 load（`kPairedCombine`）

従来の combine は、warp が 1 token の 2 KiB chunk ごとに top-k slot を 1 本ずつ TMA で読んでいた。2 段の double buffer なので in-flight は 2 本で、chunk の境目ごとに pipeline が空になる。Prefill では warp あたり約 2 token × 4 chunk × 8 slot を処理する。

変更後は、slot を 2 本ずつ 1 つの mbarrier に載せる（既存の warp あたり 2 個の barrier をそのまま使う）。in-flight は 4 本（8 KiB）になる。load は slot、chunk、token をまたぐ 1 本の列として発行し、次の chunk の最初の load が現在の chunk の最後の対の reduce と重なる。reduce は chunk ごとに従来と同じく slot の昇順で FP32 `fmaf` を積むので、出力は bit 単位で一致する。重みは token ごとに lane が 1 回ずつ読み、shuffle で配る。smem は load buffer 4 個 + store buffer 1 個 × 16 warp × 2 KiB（160 KiB）で、barrier 領域の手前に収まる（static assert あり）。

### 1.3 測ったが本 patch に含めないもの

- `drop`：dispatch 側の L1 top-k 重みの remote 読み出しを、Prefill の分岐だけで削除した（重みは late FP32 combine が local に読むので、書いた値は誰も読まない）。komoriyuta の `sm90-combine-parallel.patch` にあり、Decode worker が `sm90-dispatch-drop-l1-weight.patch` として別に統合する予定のものと同じ考え方である。build 順の中で 1 回だけ置くために、本 patch とは分けた。本 patch とは文脈が重ならず、どちらの順でも当たる。
- dispatch pull の SF 読みと重み読みの重ね（Decode の `sm90-dispatch-pull-overlap` の Prefill 版）：重み読み自体が不要と分かったので作らなかった。

### 1.4 register / spill / SASS

Firmus login で image の CUDA 13.0 を `unsquashfs` で取り出した（`/scratch/home/ACCOUNT/claude-prefill-tail`）。`containers/build-sm90-megamoe.py` と同じ Prefill/Decode の instantiation に、wave あたり 16 expert の Decode を加え、`--ptxas-options=-v,-warn-spills` と `cuobjdump -sass` で確認した。

| 版 | Prefill reg / spill (st/ld byte) | Prefill SASS sha256 | Decode / Decode-16 SASS sha256 |
|---|---|---|---|
| image（`135a43be…`） | 96 / 40 / 40 | `c1f5f67f…` | `bafabb98…` / `c2690cd6…`、168 reg、spill 0 |
| tail-dyn（1.1 のみ） | 96 / 40 / 40 | `41532a11…` | 同一 |
| **tail（1.1 + 1.2、本 patch）** | 96 / 40 / 40 | `371743b5…` | 同一 |
| drop | 96 / 20 / 20 | `fc516eee…` | 同一 |
| drop + tail | 96 / 20 / 20 | — | 同一 |
| tail + timeline 計装 | 96 / 68 / 72 | — | — |

最初の試作では ring で生の block 番号だけを渡し、受け手が各自 decode していた。この版は Prefill の spill が 128/200 byte に増えた（FC2 epilogue の scatter loop で不変量が退避された）。decode 済みの 2 word を渡す形に変えて 40/40 byte に戻した。

## 2. 実行記録

| run | commit | 内容 | 結果 |
|---|---|---|---|
| [39086777829](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39086777829) | `bb80858`（`experiment/firmus-prefill-tail-20261018`） | drop、tail-drop、tail-dyn、tail（ring は expert counter 直後） | 全 gate 成功、各 12/12 hash 一致。Slurm 24442 |
| [39093721942](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39093721942) | `01c67d1`（同、`6caea62` に rebase） | 同じ 4 案。ring を末尾へ移動。tail と tail-drop に timeline | 全 11 step 成功、各 12/12 hash 一致。Slurm 24462 |

2 回目は timeline patch を、この実験 branch の上でだけ本 patch に合わせて置き直した（6 hunk）。この版は math の per-block hook（`tl_end_block`）を、使われなくなった静的分割の分岐に置いたままだった。そのため FC1/FC2 の block 数・busy・最終 block の事象が 0 / 欠測になった。hook を動的分岐に移した修正版を `2026-10-18-megamoe-prefill-tail-timeline.patch` に置いた（コンパイル確認のみ、GPU 未実行）。待ち時間の counter、barrier、combine の事象は正しく取れている。

## 3. 結果

### 3.1 計時（median ms、括弧内は image 比。左が run 1、右が run 2）

| case | image | tail-dyn | **tail** | tail-drop | drop |
|---|---:|---:|---:|---:|---:|
| prefill-balanced-1024 | 0.5312 / 0.5317 | −5.4% / −5.2% | **−5.7% / −5.9%** | −6.3% / −6.3% | −0.7% / −0.5% |
| prefill-balanced-2048 | 0.9221 / 0.9244 | −5.2% / −5.1% | **−5.3% / −6.1%** | −5.9% / −6.4% | −0.5% / −0.4% |
| prefill-balanced-4096 | 1.6657 / 1.6852 | −5.0% / −5.3% | **−0.5% / −4.7%** | −3.5% / −6.0% | −0.2% / −0.9% |
| prefill-local-4096 | 1.6706 / 1.6521 | −6.3% / −7.4% | **−10.6% / −8.7%** | −8.8% / −9.5% | −5.0% / −2.7% |
| prefill-skew-4096 | 1.8697 / 1.8843 | −3.4% / −5.5% | **−4.9% / −7.4%** | −6.6% / −7.5% | −0.3% / −2.4% |

Prefill-4096 の median は電力制限による二峰性（同一 kernel で ±4.5%）に左右されるので、p10 も示す（validation run 39085655145 では、同一 kernel 3 本の p10 が ±0.3% に収まった）。

| case（p10 ms） | image | tail-dyn | **tail** | tail-drop |
|---|---:|---:|---:|---:|
| balanced-4096 | 1.6565 / 1.6569 | 1.5763 / 1.5802 | **1.6207 / 1.6016** | 1.5628 / 1.5628 |
| local-4096 | 1.5753 / 1.5764 | 1.4993 / 1.4969 | **1.4895 / 1.4948** | 1.4871 / 1.4894 |
| skew-4096 | 1.8297 / 1.8271 | 1.7468 / 1.7431 | **1.7376 / 1.7395** | 1.7290 / 1.7350 |

- Decode の各 case は全案で ±1% 以内（SASS 同一なので揺らぎ）。
- tail と tail-dyn の差（combine 変更の寄与）：1024/2048 では 2 回とも tail が 0.4〜1.0% 速く、local/skew の p10 でも 0.1〜0.7% 速い。balanced-4096 だけは 2 回とも tail-dyn が p10 で 1.4〜2.8% 速い。この case は 2 回とも、tail の計測が低 clock 側の山に入っている（NVML の終了時 clock は tail 1605 / 1800 MHz、tail-dyn 1860 MHz 前後）。clock の安定した case で一貫して tail が速いので、tail を採る。
- drop は 2 回とも小さく速い（1024/2048 で −0.4〜−0.7%）。tail-drop は全 case で最速か最速と同等で、二つの変更は足し合わさる。

### 3.2 timeline（run 2、tail、prefill-balanced-4096、代表 call、µs、CTA の median / max）

image の値は validation run 39085655145 の image kernel（コメント 1 行の候補）の timeline。

| 段階 | image（rank 0/1/2/3） | tail（rank 0/1/2/3） |
|---|---|---|
| combine 前 grid sync（rank 内の遅い CTA 待ち） | 62/114、66/128、27/39、38/69 | 20/33、13/25、18/34、17/30 |
| combine 前 NVLink barrier（rank 間） | 158、154、2、164 | 9、42、28、2 |
| combine | 85、86、82、85 | 86、87、88、86 |
| kernel 全体（最後の CTA 終了） | 1791.0（SM 1.670 GHz） | 1571.8（SM 1.611 GHz） |

- 遅い CTA 待ちは max で 38〜128 → 25〜34 µs、median で 27〜66 → 12〜20 µs に減った。残りは最後の FC2 block 1 個分（約 18 µs）とほぼ同じで、動的取得で揃えられる下限に近い。
- A loader の FC2 block 数は、image では CTA あたり 33 / 33〜34（静的割当）、tail では median 33 / max 35〜38 だった。速い CTA が多く取っている。FC1 は両方とも 25。
- combine は 86〜88 µs で、image（82〜86 µs）と変わらない。warp 0 の load 待ちは image の 35〜37 µs から 47〜49 µs に増えた（1 回の待ちで 2 本分を待つため）。in-flight を 2 倍にしても時間は縮まず、combine は HBM 帯域側で律速していると読める（4096 token × 8 × 8 KiB = 268 MB の読みと 33 MB の書き、約 3.5 TB/s）。1.2 の効果は combine 区間ではなく、chunk 境目の空白がなくなった分と、CTA 間のばらつきが減った分である。3.1 の差もこれと矛盾しない大きさである。
- kernel 時間 × SM clock で見ると、image 2991 k cycle に対して tail 2532 k cycle（−15%）。ただし image の call は rank 2 が遅く、rank 間 barrier で 158 µs 待った call である。clock は CTA の中央値なので、この比を性能の根拠にはしない。

### 3.3 数値 gate

`scripts/check-firmus-megamoe.py`（tolerance は変えていない）は全案 PASS。MegaMoE の 64 行の最大は relative L2 0.01946、max scaled error 0.02326 で、出力が bit 一致なので image と同じ値である。

## 4. 統合の手順（integrator 向け、本 branch では未配線）

- build 順：既存 8 本 → （Decode worker の `sm90-dispatch-drop-l1-weight.patch` を入れる場合はここ）→ `sm90-prefill-tail.patch` → Decode 側の tail patch。本 patch は dispatch の行に触れないので、drop の前後どちらでも当たる。
- 8 本 + 本 patch の header sha256 は `6b321026b7539995a1cfc4a3ddacdc6c52c5a8dc5cc0f08ceef4ed986f4346e4`。drop を先に当てると drop patch の中身で値が変わるので、その場合は統合時に計算し直す。
- 配線：`containers/build-sm90-megamoe.py` で他の patch と同じく `git apply --check` / `git apply`、`sm90-megamoe-build.json` に `prefill_tail_patch_sha256` を記録、Dockerfile の `/opt/apac` への COPY、`scripts/check-firmus-image.sh` の header sha と patch sha の照合、関連 test の期待値。
- workspace の grid-sync slot 3 を使う。Decode 側の patch が同じ slot を使う場合は、どちらかを slot 2 に移す。

## 5. 想定外だった点と残り

1. 動的取得は Prefill-1024 でも −5% 効いた。静的割当の不均衡は FC1/FC2 の block 数差だけでなく、dispatch 到着順による開始時刻のずれも含み、token 数によらず CTA の 1〜2 block 分残っていた。
2. combine の in-flight を 2 倍にしても combine 区間は縮まなかった。残りの 86 µs は帯域で決まっている。
3. 残る末尾は、rank 内待ち約 15 µs、rank 間の clock 差による待ち（kernel の外）、combine 86 µs、最終 barrier。combine を最後の FC2 と重ねる案（token ごとの到着判定）は、動的取得の後では遅い rank の combine をほとんど隠せない（遅い rank は FC2 の終わりまで全 CTA が埋まっている）。加えて hang の危険が大きいので作っていない。精度規則の緩和で「加算順の変更」は許されたが、bf16 の部分和や低精度化は禁止のままである。combine の読み出し量を減らす変更（同じ rank 内の expert 出力を FC2 epilogue で先に足す）には、block をまたぐ同期が要る。次の候補は FC2 の終わりと combine の重ね合わせだが、Prefill の GPU 時間に対する期待値は 1〜2% にとどまる。
