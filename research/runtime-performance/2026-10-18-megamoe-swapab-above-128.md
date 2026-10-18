# 2026-10-18 MegaMoE SwapAB を 144 token まで使う host patch と Prefill geometry の再評価

結論：**128 は SwapAB kernel の token 容量ではなく host の選択閾値。129〜144 token に既存の Decode 最適化を使う候補を作った。** `patches/deep-gemm-host/sm90-swap-ab-144.patch` は sgl-project/DeepGEMM v0.1.5 の host heuristic だけを変更する。`sm90-swap-ab-geometry-guard.patch` は既存 14 本の header patch の後に適用する静的検査で、演算を変えない。`containers/rebuild-deepgemm-host.py` は `_C.so` だけを作り直し、installed package の header と Python ファイルを保持する。Dockerfile、build script、series、header pin、serving config は変更していない。

GPU job・push は実施していない。以下の「安全」は source の添字・容量上の判断で、GPU gate の合格を意味しない。性能値は明示した既存測定以外は推定である。浮動小数の加算順序・FMA 差を許す今回の規則に従い、データ型・量子化 granularity・計算内容・数値 gate の閾値は変更しない。

## 1. 対象と調査した履歴

- host：tag v0.1.5、commit `1c8138a59b3df5dac50b76b730834ed57982fdfb`。`csrc/jit_kernels/heuristics/sm90_mega_moe.hpp` と `impls/sm90_fp8_mega_moe.hpp`、`setup.py`、`build_sgl_deep_gemm.sh`、`sgl_deep_gemm/__init__.py`、`csrc/tvm_ffi_api.cpp` を確認した。共有の sparse clone は変更していない。
- kernel：指定された `scratchpad/rev` のコピー。現行 14 patch の header pin `ff7a58cca00aa42a90386d12b0437b5d8caf35eb0ad082a8a1a0b4b74001a2a1` と一致。scheduler、workspace、buffer は image 内の header も確認した。
- [timeline](2026-10-18-megamoe-kernel-timeline.md)、[Decode mainloop](2026-10-18-megamoe-decode-mainloop-pipeline.md)、[epilogue/dispatch](2026-10-18-megamoe-decode-epilogue-dispatch.md)、[Decode tail](2026-10-18-megamoe-decode-tail.md)、[Prefill tail](2026-10-18-megamoe-prefill-tail.md)、[Prefill mainloop](2026-10-18-megamoe-prefill-mainloop.md)、[bundle](2026-10-18-firmus-max-bundle.md)、[収支表](2026-10-18-firmus-maximum-performance-plan.md)、[小 kernel](2026-10-18-sglang-small-kernels.md) を参照した。geometry/redundancy note はこの worktree に無いため `git show f09f57e:research/runtime-performance/2026-10-18-megamoe-geometry-and-redundancy.md` で読んだ。commit `96031ab` と、その promote 演算数の訂正 `f09f57e` も確認した。
- Actions [39070273796](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39070273796) の terminal 状態は success、head `0374c551…`。過去の手元の bench/timeline summary と作業記録（scratchpad の `summary.md`、`sum-comb.md`、compile script）も照合した。
- **image と現行 header を区別する。** configs の image `/scratch/home/ACCOUNT/qwen/containers/39070273796.sqsh` から取り出した header は `135a43be…`（旧 8 patch）だった。今回のコンパイルには、その image の CUDA/CUTLASS と、上記の現行 14-patch header を組み合わせた。image の古い MegaMoE header を再び patch 起点にしたわけではない。

## 2. 上限は何か

Qwen は hidden 4096 / intermediate **1536** / top-8 / EP4 / local expert 32。`expected_tokens_per_expert = num_tokens / 4`。256 token で 64 に達して M128/N256/512 math threads に移る。それ未満は M64/256 math threads。

**前提の訂正：現行 Qwen の 129〜255 token の fallback は M64/N128、非 SwapAB。** host の `decode_use_block_n_256` は intermediate ≥3072 を要求する。Qwen の w13 の N=3072 と intermediate=1536 は別物である。M64/N256 はこの形状では host に選ばれない。将来その条件を変えた場合の参考として §3・4 で強制 instantiation も扱う。

| 項目 | 実際の制約／129〜144 への影響 |
|---|---|
| `kSwapABTokenChunks = BLOCK_M/8 = 8` | **1 pool block 内**の 64 token を転置して読むための chunk 数。全 rank の token 数を 64/128 に制限しない |
| WGMMA N_SWAP | `scheduler.get_valid_m()` は `min(expert_count - m_block_idx*64,64)`。その値だけを 8 に切り上げ、8…64 の specialization に入る。expert に 65 行以上あれば次の M block を処理する |
| expert ごとの block 数 | scheduler と dispatch はともに `ceil(expert_count/64)` を使い、累積 pool block offset を進める。1 expert が 64 行まで、という仮定は無い。144 token ×4 rank の hot expert=576 行でも 9 block |
| pool | template の上限 4224/rank では `align(4*4224*8 + 32*(192-1),384) = 141312` token。64 行 geometry は候補 BLOCK_M の一つで、任意の routing を含む容量式に入っている。144/rank の最大受信 pair は 4608、expert 境界の padding を足しても十分小さい |
| SF pool | `SF_BLOCK_M=align(64,128)=128`、添字は `pool_block_idx*128 + row`。L1 dispatch、TMA、L1 epilogue の L2 SF 書き出しともこの stride。必要量は `141312/64*128=282624`、確保値 2260992 は最小候補 M8 用でさらに大きい。static assert もある |
| SMEM | SwapAB の M64/N128、8 stage、384 threads のまま。230672 B。batch token 数によらず、同時に 1 block 分を再利用する。全 batch を smem に置かない |
| dynamic Decode scheduler | rank-wide counter と shared ring は次の **block 差分**を渡す。Prefill の packed 16-bit expert-count ring はこの経路では使わない。spare workspace slot と combine buffer の領域分離は既存 static assert の対象 |
| dispatch/source metadata | 入力は `kNumMaxTokensPerRank`、source metadata と BF16 combine area は pool／top-k／rank 上限から確保。num_tokens の loop は grid stride で、128 固定の配列・mask は無い |
| combine | 1 token を 8 chunk に分ける。`work_idx < num_tokens*8`、stride=132 CTA×8 warp=1056。132 token までは 1 周、**133 以降は一部 warp が 2 周**する。phase と load stage は loop 外で継続し、出力 store buffer は再使用前に wait。容量制限ではないが、約 1 chunk 分の latency 段差を見込む |

最大値は三つに分ける必要がある。

1. **現行 Qwen の geometry 選択を維持して広げられる最大値は 255 token/rank。** 256 では M128 になり SwapAB の template 条件を満たさない。精度・添字のために 128 を残す必要は見つからなかった。
2. M64 を別途強制するなら、この template／buffer の入力上限は **4224/rank**。expert の最大行数は 16896。kernel に独立の「全 batch は 128 以下」という制約は無い。ただし今回は host の M128 への切替を変えず、この広い範囲の性能・GPU 正しさも検証していない。
3. **本候補の採用範囲は 144/rank。** 次回の 34〜36 request ×4 verify token だけを覆い、145 以上を今のまま残す。255 まで一度に変更して、未測定の大きな Decode／短い Prefill に範囲を広げない。

追加の header guard は `kFP8SwapAB => kSwapABEligible` を assert する。host と header が食い違って M128 や N256 で SwapAB を指定した場合、黙って非 SwapAB の式で走らず compile error にする。pool/SF/SMEM については既存 guard で足り、新しい batch 固定配列は追加していない。

## 3. 132〜144 token の費用推定

既存の Decode-128 約 **196 µs/call**（今回与えられた 5 patch 込みの基準）を出発点にする。以下は均等 random routing のモデルで、serving の実際の route trace ではない。各 expert の count は Binomial(4T,1/16)、平均 T/4。8 行切り上げの期待値を取り、全 32 expert 分を足した。

| token/rank | 平均行/expert | SwapAB の実行行相当（全 expert、8 切上げ） | 非 SwapAB の行相当（64 切上げ） | tensor FLOP 比（非 Swap / Swap） |
|---:|---:|---:|---:|---:|
| 128 | 32 | 1136 | 2048 | 1.80 |
| 132 | 33 | 1168 | 2048 | 1.75 |
| 136 | 34 | 1200 | 2048 | 1.71 |
| 140 | 35 | 1232 | 2048 | 1.66 |
| 144 | 36 | 1264 | 2048 | 1.62 |

- balanced では M pool block はほぼ 32 のまま（144 でも期待値 32.00013）。SwapAB と非 SwapAB N128 は FC1 768 / FC2 1024 tile、132 SM で約 6 / 8 CTA wave 相当。SwapAB だけが tile 内の無効 token の WGMMA を省き、行数は 128 基準から +2.8〜11.3%。非 SwapAB は 64 行を常に計算し、tensor work は約 77.3 GFLOP/rank/call。SwapAB は 44.1〜47.7 GFLOP。これは tensor work の比であって **kernel latency の比ではない**。
- 両者の weight payload は FC1 402.7 MB + FC2 201.3 MB = **604.0 MB/rank**。H200 の 4.8 TB/s で 126 µs が共通の HBM 床（全 expert が 1 M block の場合）。M block が増える hot routing では重みを再読する。TMA による activation tile 読みは N128 で FC1 約201 MB、FC2 約101 MB だが、大半は同じ pool 行の L2 再使用であり、そのまま HBM bytes に加算しない。
- 強制 N256 は FC1 384 / FC2 512 tile、約 3 / 4 wave。1 tile の N と weight load が 2 倍で、weight bytes と full-64-row FLOP は減らない。activation の TMA bytes と tile 固定費は半減する一方、stage は 8→4、各 WG の accumulator は 32→64 要素、FC2 の low/high 一時 accumulator も増える。今回の CPU compile では **816/980 B spill**（§4）。したがって「wave 半減だから 2 倍速い」とは見積もれない。この値は IH1536 を強制した参考値で、IH3072 以上の正規 host 選択の性能を示さない。
- SwapAB は hoist、L1 quant、dispatch overlap、dynamic scheduling、parallel combine を引き続き使える。非 SwapAB はそれらの SwapAB 専用分岐を外れる。drop-l1-weight など全 geometry に効く変更まで外れるわけではない。

時間の推定は二つのモデルで挟む。196 µs のうち起動・dispatch・末尾を約30〜35 µs、残りを約160〜165 µsとする（既存 timeline からの近似）。SwapAB の残りを行相当で比例させると 132〜144 は約201〜215 µs、combine 2 周目による数 µs を加え **約0.20〜0.22 ms/call**。weight bandwidth 側に寄るほど増分は小さい。

非 SwapAB N128 は、weight bandwidth だけが律速なら同程度の **約0.20〜0.22 ms** でも不思議ではない。一方、既存の「loader は free slot 待ち、math の直列 loop が律速」という状態が続き、math 部を full-64-row work に比例させると **約0.32〜0.33 ms**。追加最適化の不適用もあり、計画上は **0.25〜0.33 ms 程度を警戒するが低確度**、という結論になる。強制 N256 は同じ約0.20〜0.33 msの計算／帯域モデルに、tile 固定費の軽減と spill の負担が逆向きに加わるため、この情報だけで N128 より速いとは順位を付けられない。

従って本 patch の期待は **128→132〜144 で最適化経路を失う cliff の回避**であり、数十〜百 µs の改善があり得るが、0% や逆転も排除しない。128 以下は host 判定も kernel も変わらない。EP の最遅 rank を全員が待つので、一つだけ閾値を超えた rank を守ることにも意味がある。score 改善率は算出しない。

## 4. CPU コンパイルと検査

Firmus **login node のみ**、専用 `/scratch/home/ACCOUNT/claude-swapab-20261018/` へ次の image 部分を `unsquashfs -processors 2` で抽出した。CUDA は **13.0.88**。GPU 計算・scheduler は実行していない。scratch は `stat -f` で fhgfs（共有 BeeGFS）、抽出・compile tree は約5.1 GBだった。

```sh
unsquashfs -processors 2 -no-progress -d /scratch/home/ACCOUNT/claude-swapab-20261018/img /scratch/home/ACCOUNT/qwen/containers/39070273796.sqsh usr/local/cuda-13.0 opt/sglang/venv/lib/python3.12/site-packages/deep_gemm
```

`deep_gemm` をさらに作業用にコピーし、impl header だけを現行 14-patch＋guard に差し替えた。共通 template 引数は `4224,4096,1536,128,8,wave,BM,BN,128,141312,2260992,stages,64,64,math_threads,132,4,infinity,true,reg,reuse,arrival,full,false,swap`。`num_tokens` は runtime 引数なので 128/132/144 は **同じ instantiation** を使う。

```cpp
#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>
extern "C" void* instantiate() { return reinterpret_cast<void*>(&deep_gemm::sm90_fp8_mega_moe_impl<4224,4096,1536,128,8,32,64,128,128,141312,2260992,8,64,64,256,132,4,cute::numeric_limits<float>::infinity(),true,0,false,false,true,false,true>); }
```

```sh
nvcc --cubin --std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -O3 --ptxas-options=-v,-warn-spills -gencode arch=compute_90a,code=sm_90a -I/scratch/home/ACCOUNT/claude-swapab-20261018/deep_gemm/include kernel.cu -o kernel.cubin
cuobjdump -sass kernel.cubin
```

| instantiation | wave / BM / BN / math / stage / reg指定 / reuse / arrival / full / swap | register | spill store/load B | stack B | cubin B |
|---|---|---:|---:|---:|---:|
| 対象 SwapAB | 32 / 64 / 128 / 256 / 8 / 0 / false / false / true / true | 168 | 0/0 | 8 | 125784 |
| 小 Decode 対照 | 16 / 64 / 128 / 256 / 8 / 0 / false / false / true / true | 168 | 0/0 | 8 | 125784 |
| 実際の fallback | 32 / 64 / 128 / 256 / 8 / 0 / false / false / true / false | 168 | 0/0 | 8 | 80600 |
| 強制 N256 参考 | 32 / 64 / 256 / 256 / 4 / 0 / false / false / true / false | 168 | 816/980 | 336 | 95960 |
| Prefill 対照 | 32 / 128 / 256 / 512 / 3 / 112 / true / true / false / false | 96 | 44/60 | 24 | 116440 |

WGMMA の C7510/12/14 直列化警告は全 instantiation で無し。未使用定数の nvcc warning と、表の Prefill/N256 の spill warning はある。未変更の 14-patch header でも SwapAB wave32・wave16・Prefill を再コンパイルし、guard 追加前後の `cuobjdump -sass` はそれぞれ byte 一致した。SASS SHA256 は順に `2c55a5ace3a685aa91380b9c92d9055067789bc7cc6f5b218d5a018560ecfd4b`、`4b320e91737d891f15cd54279274a1fe6fdad599ec3f9eefd362ff921448970a`、`1970d8f3f5c8a72b297f87fbdc55499abc48bdac56e862dc5761fbf78bda1a0a`。

他の CPU 検査：pristine v0.1.5 → 既存 series 14 本 → guard の順に `git apply --check` と適用が通り、指定された rev コピーとも一致。host patch は pristine csrc に当たり、**実際の patched 関数を C++ harness に読み出して** 0…4224 の閾値 sweep、Prefill geometry の除外、kill switch を検査して PASS。新しい Python script と gate は py_compile／Ruff PASS。既存 image fixture 33 件も PASS（新host buildを実行するfixtureではない）。

host `_C.so` 自体のリンク／ABI smoke は今回は実行していない。§6 の image build に含める。GPU gate は未実行であり、追加した case の PASS は主張しない。抽出 toolchain・作業用 package・cubin は調査終了後に専用 directory ごと削除した。

## 5. Prefill：緩和後も今回は geometry 変更を追わない

bit 一致の制約は外れたが、W5 の他の制約は残る。timeline の FC1 約58%／FC2 約44%、loader の free-slot 待ち、640-thread launch bound の **96-register cap** は丸めとは無関係である。

- **M64/N256、2 math WG**：launch は384 threadsで168 registerになり、M128の端数 padding も減る。ただし tensor work あたりの TMA activation 書込みは現行の約5/3、M tile 数は2倍。同じ64×128 WG tile の rescale/通常 promote は、訂正後の現行では FC1 1演算、FC2 2演算であり、FFMA への変更で演算数は減らない。別 accumulator を持つ非 reuse 形は今回も816/980 B spill。Prefill 専用 phase、dynamic FC2、paired combine、mainloop prefetch の条件も外れる。stage増加（3→4）が loader 待ちを解決する状況でもない。丸め差を許すだけでは具体的な勝ち筋が無い。
- **M128/N128、2 WG**：launch は384 threadsでregister余裕は増す。しかしtile数とepilogue回数が2倍、activation TMA bytes/FLOPは4/3。SMSPごとの隠蔽用warpも減り、元のphase専用経路を移植する作業が要る。96-register capを避けること自体は仕組みだが、現行44/60 B spillを消す利益だけでこの費用を超える根拠は無い。非reuse化ならFC2の複数accumulatorも増える。
- **M192/N128、3 WG**：128-register程度のlaunch上限に改善するが、Mの端数paddingは平均1024行/expertで約6→9%、warpは4→3、専用最適化からも外れる。数十byteのspill除去以外の改善機構は見つからない。
- **640 threadsのままregister指定だけ増やす**：`setmaxnreg` と ptxas のlaunch-bound割当は別。W5 の96-register/1856-1768 B spill/C7512の非reuse試作は精度緩和で解決しない。support roleを切るにもdispatch/loaderは各64 threadsが最小で、4 math WGのまま512-thread launchにはできない。
- stage追加やclusterだけの変更も、loaderが先行しsmemが225600 Bで埋まっている状況を改善しない。独立tileを2組のWGでping-pongする構造ならepilogueを隠す機構はあるが、CD bufferとstage/barrierの分離を伴う新kernelで、単なるgeometry／launch変更としては扱えない。

よって Prefill geometry patch は作らない。「丸めが変わるから不可」ではなく、**既にrescale済みで演算数削減が無く、register・tile固定費・データ再読の交換が有利という証拠が無い**ため。10〜15%短縮の見込みは引き続き裏付けられない。

## 6. integrator が行う配線と検証（未実施）

### 6.1 build context と Dockerfile

この作業では両ファイルを変更していない。追加する具体的内容：

1. `scripts/build-sglang-container.sh` の context 作成で、`containers/rebuild-deepgemm-host.py` を `${context}/containers/` へ、`patches/deep-gemm-host/` を `${context}/deep-gemm-host/` へコピーする。既存の組織 mirror は更新しない。build専用のrecursive cloneをcontextに用意する（下のコマンド。sourceの `.git` とsubmoduleを含める）。
2. guard を `patches/deep-gemm/` に移し、`series` の14本の**末尾**に追記する。`header.sha256` を **`5644dfa107d08a166c1a439cd33bcef5d703670f3adc39554fc884a2031361b1`** に更新する。host patch は csrc 基準なので、header の series に入れない。
3. Dockerfile の `install-accelerator-dependencies.sh` の直後、既存 `build-sm90-megamoe.py` の**前**に次の COPY/RUN を足す。この時点で torch、sgl-deep-gemm、CUDA、host build依存が入っている。**sgl wheel の host build は `tvm_ffi.cpp.build` による `csrc/tvm_ffi_api.cpp` の共有library**。過去geometry noteの `setup.py` / `csrc/python_api.cpp` は別のpybind11 ABIなので使えない。今回のscriptはv0.1.5の `build_sgl_deep_gemm.sh` と同じC++ flags・include・link設定（torch、TVM-FFI、CUDA13、cudart/nvrtc/cublasLt/cublas）で作る。`pip install` や wheel の再インストールを使うと既存headerを上書きするので、本scriptはextensionだけを置換する。

```sh
git clone --depth=1 --branch v0.1.5 --recurse-submodules --shallow-submodules https://github.com/sgl-project/DeepGEMM.git "${context}/deep-gemm-source"
git -C "${context}/deep-gemm-source" rev-parse HEAD
```

```Dockerfile
COPY deep-gemm-source /opt/apac/deep-gemm-source
COPY deep-gemm-host /opt/apac/deep-gemm-host
COPY containers/rebuild-deepgemm-host.py /opt/apac/rebuild-deepgemm-host.py
RUN CUDA_HOME=/usr/local/cuda /opt/sglang/venv/bin/python /opt/apac/rebuild-deepgemm-host.py /opt/apac/deep-gemm-source && rm -rf /opt/apac/deep-gemm-source
```

scriptはsource commitを固定確認し、private一時コピーにhost patchを適用、同じvenvのtorch／TVM-FFI ABIで `csrc/tvm_ffi_api.cpp` をbuild、installed `_C*.so` のみ置換、CUDA stubで新processのABI importを検査する。patch SHA256は `5ddffa4a18f8a3b53f4cd4c40631b4b4bd0ea5f6c58fb297ae2555953b7d01cb`。source commit・TVM-FFI entrypoint・patch hash・置換前後のextension hash・保持したheader hashを `/opt/apac/deep-gemm-host-build.json` に記録する。package名・versionのmetadataを新しいwheelで置き換えない。既存 `build-sm90-megamoe.py` は従来通りpristine headerを確認した後、seriesを適用・compileする。

4. `scripts/check-firmus-image.sh` の extraction に `opt/apac/deep-gemm-host-build.json` を足し、source commit、TVM-FFI ABI/entrypoint、host patch hash、extension hashを照合し、sidecarへ回収する。header のguardは既存series照合で検査できる。fixture `qwen-experiment-metrics/tests/test_firmus_image.py` にhost metadataの正常系と破損検出を足す。imageの `_C.so` 実体hashも照合すること（header hashだけではhostの旧128制限が残っていても見抜けない）。manifestにhost source commitとpatchを記録する。
5. 通常のBuildah chroot → Skopeo/Umoci検証 → Enrootの経路で新imageを作る。今回の変更はheader overlayだけでは有効にならない。configのimage path/hash/size/provenanceを、その新imageへ揃える。

### 6.2 numerical gate と性能

`check-firmus-megamoe.py` には配線済みの追加caseをコミットした：`[128,129,144,145]` のmixed geometry、`[144,140,136,132]` のCUDA graph（live/idleとmasked tailを変更して4 replay）、`[144]*4` をhot expertへ集中（64行を跨ぐ複数M block）。既存caseと `check-firmus-decode-kernels.py` は残し、relative L2 <0.025 / max scaled error <0.08 は不変。既存の数値gateからdecode-kernel gateも呼ばれる。

統合担当は次の順で検証する。

1. image buildのhost link/ABI smokeと上記CPU compileを完走させる。
2. Actions経由で既存gate＋追加caseを実行し、terminal状態と全rankの結果を確認する。特に133以上のcombine再使用、hot expert、145側の非SwapABとの混在を確認する。
3. `bench-firmus-megamoe.py` の既存CASES末尾にbalanced/localの132・136・140・144、および境界145を追加する（末尾なら既存caseの乱数seedは不変）。baselineとcandidateを同一入力・同一clock条件で各2標本以上比較し、`DG_PRINT_CONFIGS` で129〜144がSwapAB、145が非SwapABなのを確認する。任意の N256 geometry は別commitの比較条件にする。今回CPU compileした強制N256を本番hostのfallbackと取り違えない。
4. 新範囲のbaselineとcandidateは転置WGMMAの丸めによりhashが異なり得る。現行summaryのhash完全一致判定を、その比較caseだけ数値gateを正本とする扱いへ変更する。既存128以下のhash不変検査は残す。許容差を緩めない。gate未合格のhash差を情報扱いで通さない。
5. servingで34/rankなら現行cap136・graph bs34を使える。36/rankを試す条件commitではcap144、graph bs36を追加する。padding後の実際のkernel行数が144以下であることもログで確認する。条件を同時に変更してpatch効果と混ぜない。公式入力・result parser・精度scoreとSLA、artifactを回収し、同条件複数runで採否を決める。

この時点のdeliverableは未配線のbuild候補と検証caseであり、servingでの採用済み最適化ではない。
