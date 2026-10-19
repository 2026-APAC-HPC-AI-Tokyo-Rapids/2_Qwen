# 2026-10-19 Prefill MegaMoE：2 consumer warpgroup と FP32 promote の ping-pong

**CPU コンパイルまで確認した統合候補。GPU の正しさ・速度は未確認で、採用済みではない。** GPU job と push は行っていない。現行 14-patch header の上に Prefill 専用の 15 本目と host patch を作り、融合 dispatch → FC1 → SwiGLU/FP8 → FC2 → combine、NVLink protocol、Decode のコードを維持した。build の配線は変更していない。

## 1. 根拠と対象

[bottleneck map](2026-10-19-firmus-prefill-bottleneck-map.md)、[mainloop](2026-10-18-megamoe-prefill-mainloop.md)、[timeline](2026-10-18-megamoe-kernel-timeline.md)、[tail](2026-10-18-megamoe-prefill-tail.md)、[FA3](2026-10-18-firmus-fa3-reconsideration.md)、[small kernels](2026-10-18-sglang-small-kernels.md)、既存 patch の commit 本文と利用可能な Codex 会話履歴を確認した。指定された `2026-10-18-megamoe-geometry-and-redundancy.md` と `task/1. Qwen3 AI Task Rules.md` はこの checkout にない。精度条件は今回の指示と mainloop note に従った。source は共有 v0.1.5 checkout の私有コピーで、共有 checkout は編集していない。SGLang は変更していない。

run 39190318626 の `probe/{gemm,fa3,rope}.jsonl` と run 39114949128 の `profile/profile/kernel-summary.json` は `/srv/qwen-artifacts/` の指定ファイルを直接読んだ。以下の GEMM 値は `probe/gemm.jsonl` の median：

| standalone grouped GEMM | FC1 | FC2 | 合計 |
|---|---:|---:|---:|
| 32768 useful/padded rows | 738.006 µs | 423.898 µs | 1161.904 µs |
| routed、32656 useful / 35072 padded rows | 811.434 µs | 447.402 µs | 1258.835 µs |

FP8 演算量は even で 1.23695 TFLOP。even の clock snapshot は FC1 1260 MHz、FC2 1350 MHz、電力約 693 W。融合版の GEMM 区間約 1.5 ms、serving の kernel 約 1.70 ms に対して、効率差を詰める余地がある。ただし standalone FC2 は **per-128 activation scale**、融合版 FC2 は **per-64 activation scale**。同じ FLOP 数でも scale/promotion の仕事は同一ではなく、standalone をそのまま呼ぶ置換は精度条件を満たさない。

## 2. 変更

`patches/deep-gemm/sm90-prefill-pingpong.patch` と `patches/deep-gemm-host/sm90-prefill-pingpong.patch`。

| 項目 | 現行 Prefill | 候補 |
|---|---|---|
| CTA tile | 128×256×128 | 128×128×128 |
| consumer | 4 WG、M/N を 2×2 に分割 | 2 WG、M を 2 分割 |
| dispatch / loader / math threads | 64 / 64 / 512 | 64 / 64 / 256 |
| total threads | 640 | 384 |
| stages | 3 | 5（既存 host の SMEM 式で自動選択） |
| dynamic shared memory | 225600 B | 211168 B |
| accumulator | reuse 64 FP32/thread | raw 64 + final 64 FP32/thread |
| math register allocation | 要求 112、実際 96 | 要求・実際とも 168 |

host が従来 Prefill を選ぶ条件（256 行以上）に加え、H4096/I1536、EP4、128 experts、top-8 のときだけ変更する。それ以外は従来の geometry と heuristic。Decode の SwapAB rollout 上限は変更しない。

各 WG は独立した FP32 raw accumulator に FC1 の K128、FC2 の K64 を積み、`final += (activation_scale * weight_scale) * raw` で promote する。FC2 は半 K ごとに raw を再利用し、常時 3 本の accumulator を必要としない。scale は従来の FP32 per-128 / per-64 のまま。FP8 の再量子化、追加の低精度 cast、計算の省略はない。最後の SwiGLU/amax/FP8 cast、FC2 BF16 store、late top-k FP32 combine は既存実装を使う。

consumer の turn を CTA 内 named barrier 7/8 で交互に渡す。WG0 は最初だけ待たず、WGMMA の完了後に WG1 の turn を release し、promote を行う。WG1 は対称に release する。最後の WG1 の release は次 tile の WG0 が消費するため、tile 間で turn を初期化しない。全 WG は valid M にかかわらず mainloop を完走し、invalid M の epilogue だけ従来どおり省く。最終 WGMMA と他 WG の epilogue、隣接 tile の mainloop と epilogue も重なる余地があるが、tile 全体を独立に割り当てる CUTLASS 型の完全な epilogue ping-pong ではない。

`cutlass::arch::NamedBarrier` の user ID は内部で 8 加算されるため使わず、既存の raw-ID `ptx::sync_aligned` と `bar.arrive` を使った。7/8 は既存 dispatch 0/1、full math 2、WG epilogue 3..6 と重ならない。descriptor は standalone と同様に base を一度作り、stage/K の 16-byte 単位 offset を下位 word に加算する。shared memory は 256 KiB 未満なので address field の桁あふれはない。

## 3. protocol と readiness

変更は CTA 内 geometry と math loop に閉じる。NVLink barrier/tag、grid-sync の CTA 数（132）、workspace、expert wave、symmetric pool、dispatch/combine の global address は変更しない。BLOCK_M は 128 のままで pool padding も同じ。

FC1 の出力は WG ごとに既存と同じ 64×64 の量子化 tile。L1 N tile 数は 12→24、split-N は 2→1 となるので、FC2 readiness の到着数は full M で `12×2×2 = 24×2×1 = 48`、tail M≤64 で 24 のまま。各 store 完了後の release-add と loader の acquire 判定は既存の式を使う。FC2 の dynamic claim / shared ring も継続する。

combine は旧 split-MN 用の 4 分割を候補にも明示した。新 geometry で既定の 1 分割を選ぶと paired-load buffer が足りず static assert が失敗するためで、combine の演算順序は変えていない。consumer warp 数は 16→8 になるので combine の並列度が減るリスクがある。

## 4. CPU コンパイルと SASS

Firmus login node で、`39110496019.sqsh` から `unsquashfs` で CUDA と DeepGEMM を `/scratch/home/ACCOUNT/claude-megamoe-pingpong/` に抽出した。CUDA は **13.0.88**。検査は `scripts/check-firmus-megamoe-pingpong-build.py` に保存し、image の include を読み取り専用にして、一時 overlay の header だけに patch を当てる。GPU import/初期化/実行はしない。

flags は `--cubin --std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -O3 --ptxas-options=-v,-warn-spills -gencode arch=compute_90a,code=sm_90a`。template 引数は既存 build と同じ max_tokens4224、pool141312、SF pool2260992、132 SM、EP4。Prefill 候補のみ表の geometry と reuse=false にした。

| specialization | registers | spill store/load | stack |
|---|---:|---:|---:|
| 現行 Prefill | 96 | 44 / 60 B | 24 B |
| 候補 Prefill | 168 | 0 / 0 B | 0 B |
| Decode 32-expert wave（両 header） | 168 | 0 / 0 B | 8 B |
| Decode 16-expert wave（両 header） | 168 | 0 / 0 B | 8 B |

**C7510/C7511/C7512/C7513/C7514 は全 compile でなし。** nvcc の既存 unused-variable #177-D は残る。現行 Prefill の spill warning 以外に ptxas warning はない。ログ正本と SASS SHA-256 は [compile.json](2026-10-19-megamoe-prefill-pingpong-compile.json)。Decode の両 wave は変更前後で `cuobjdump --dump-sass` の `code for sm_90a` 以降が文字列として同一。SHA-256 は 32-wave `35c619452ea3fa0721799b105dfa2ebe74c3109c6e532b5b4097c8a5782cd4e3`、16-wave `5c742311bd2aee433e20502ab0eff3e212308f00ea82029638ebafa4af07b1e2`。

ptxas はソース上の release より前へ一部 FFMA を移動する。最終候補の SASS では FC1 は `WARPGROUP.DEPBAR.LE` → 5 FFMA → `BAR.ARV` → 残りの promote、FC2 は各半 K とも wait → 1 FFMA → `BAR.ARV` → 残り、という順序を確認した。全 promote を終えてから handoff する配置にはなっていない。compiler の世代を変える場合は再確認する。

## 5. 数値 gate と未確認事項

既存 gate の relative L2 < 0.025 / max scaled error < 0.08 は変更していない。`scripts/check-firmus-megamoe.py` に次を追加した。

- Prefill/Decode/idle が混在する `[7,0,65,257]`。
- host threshold 前後と tail の `[255,256,257,513]`。
- `[256,257,512,1024]` の CUDA graph：4 replay で入力を変え、rank 2 は全 masked/live を交互に切り替える。
- Prefill の全ゼロ activation `[256,0,0,0]`。

元の `[7,0,65,129]`、4096 hot-experts、Decode graph と tile widths、idle Prefill 行数の case は残す。gate は既存 decode-kernels check も呼ぶ。py_compile、Ruff、15-patch の順次適用/hash 照合、bundle の host-144 patch の後への host patch 適用を CPU で確認した。**GPU gate、_C.so の実再ビルド/ABI import、serving の公式 parser は今回実施していない。** raw/final の分離は rescale と FP32 の加算順序を変えるので、出力 hash の一致を合格条件にせず、既存の数値 gate を使う。FC1 の丸め差は SwiGLU/FP8 の境界を越え得る。

## 6. 時間の見積もりとリスク

64×128×128 の WGMMA は WG あたり 4 命令、Hopper の SM あたり FP8 throughput を使うと約 256 Tensor Core cycle。FC2 の半 K は 2 命令、約 128 cycle。promote は 64 FFMA/thread で、1 WG の warp issue 量ならおおむね数十〜百 cycle。raw/final の分離と交互 handoff により、従来の rescale → WGMMA → rescale の依存鎖から promote を外し、他 WG の Tensor Core 実行と重ねる。reciprocal/rescale 比も不要になる。

standalone の snapshot clock を便宜的に 1.55 GHz に揃えると even は `0.738×1.26/1.55 + 0.424×1.35/1.55 ≈ 0.97 ms`。routed は約 1.03 ms。この値は条件を揃えた測定ではなく、**同じ clock で効率が近づいた場合の目安**にすぎない。候補には FC2 の per-64 promote と handoff、融合 epilogue が残るため、GEMM 区間の仮説は **1.1〜1.4 ms、中心約 1.25 ms**（現行約 1.5 ms）。固定の周辺費用約 0.2 ms を足すと kernel は 1.3〜1.6 ms、中心約 1.45 ms。中心値なら kernel 約15%、Prefill 約8%の短縮に相当する。

最大の不確実性は電力。standalone と同じ効率になって clock も 1.3 GHz に落ちれば、便益は大きく縮む。BLOCK_N 半減で tile 数・A の再読み込み・epilogue 回数が倍になる（standalone と同じメモリ効率になる保証はない）。turn barrier は新しい同期費用で、2 WG の compute を交互に直列化するため、producer 遅延や unbalanced tail の影響も要実測。8 warp combine の並列度低下もあり、速度向上を保証しない。

## 7. 正確な配線（integrator が実施）

1. header patch は `patches/deep-gemm/series` の最後に追加済み。15 本適用後の header SHA-256 は **`13664810d8d92ce592b14949aa8b29d60edfd9b3288e2d842397d35ee1aba14a`**（`header.sha256` を更新済み）。現行の directory COPY / series loop が取り込むので、新しい Dockerfile COPY は不要。
2. host 再ビルド経路は `origin/perf/firmus-max-20261018` の `containers/rebuild-deepgemm-host.py` と対応する Dockerfile/build-script 配線を再利用する。この checkout にはまだないため、本作業では取り込んでいない。source commit `1c8138a59b3df5dac50b76b730834ed57982fdfb` に `sm90-swap-ab-144.patch`、次に **`sm90-prefill-pingpong.patch`** を `git apply --check` / `git apply` で当ててから TVM-FFI `_C.so` を rebuild する。既存 rebuild script は host-144 の単一 patch を hard-code しているため、2 本目の明示的適用とその SHA の build record を足す。installed JIT header を上書きしない既存のチェックを保つ。
3. `scripts/build-sglang-container.sh` の host patch manifest（untargeted list）と `scripts/check-firmus-image.sh` の host build-record 照合に新 patch の SHA を追加する。host-144 の配線と同じ要領。image test の host record 側も同時に更新する。header の series/pin 照合は既存の仕組みを使う。
4. `containers/build-sm90-megamoe.py` の Prefill compile config を **block_m128, block_n128, epilogue_threads256, stages5, registers168, reuse_accum=False, arrival_counter=True, full_sync=False, swap_ab=False** に変更する。Decode config はそのまま。host ABI import、CPU compile の既存 build check は残す。
5. 統合 image の gate で拡張済み `check-firmus-megamoe.py` と既存 `check-firmus-decode-kernels.py` を実行する。host geometry が変わるので header overlay だけの旧 image kernel bench では新経路に入らない。新 `_C.so` と header の両方が必要。

統合 serving では official accuracy/parser の合格、混在 geometry / graph の完走、Prefill symbol が 128×128 / 5-stage / 256 math threads、Decode symbol と geometry が従来どおりであることを先に確認する。そのうえで同条件の反復比較を行い、Prefill MegaMoE の 1.70 ms が中心予想 1.45 ms 付近へ近づくかを見る。clock/power と combine を別に観測し、総 kernel 時間だけから mainloop 改善を断定しない。改善がなければこの候補を採用しない。

融合版で見込みが出なかった場合の代案は dispatch と combine の間を standalone 型 grouped GEMM + SwiGLU/quant に分離すること。1.16〜1.26 ms はその GEMM 部分の測定上の目標だが、FC2 は per-64 FP32 promote へ適合させる必要があり、既製の per-128 呼び出しを使うだけでは不可。追加 launch と中間 BF16/FP8/FP32 scale のメモリ往復を含めた E2E で判断する。

抽出 toolchain、私有 source コピー、診断 CU/cubin/SASS は検証後に削除済み。raw compiler log は本 note に隣接する compile.json に保存した。GPU/scheduler job、push、共有 source の更新は一切行っていない。
