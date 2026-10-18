# 2026-10-18 Firmus FA3 再検討：呼び出し、再ビルド不要の選択肢、次の一測定

結論：FA3 全体を「sgl-kernel 再ビルドが必要」として対象外にする理由はない。ただし、今回は速度改善 patch を採用しない。既定の packing と split は今回の形状にかなり合っており、DP の scheduler metadata 制限を外すだけの変更には具体的な不整合がある。最も判断価値が高い次の一測定は、**実 Prefill の同じ FP8 Q/KV を用いた paged 対 gather＋contiguous varlen の合計時間**である。これは既存の sgl-kernel API で測れる。GPU 実行、push、CUDA コンパイルは本作業では行っていない。

## 1. 正本と調査範囲

- branch 起点 `823b79f`。SGLang は `fc275674a97d78553b22597ee72c8fd48f5420f4`、`scripts/build-sglang-container.sh` の現行 **22 patch**を順番に適用した私有コピーで確認した。指定された `sgfix` は読み取りだけとし、さらに pinned commit から tree を作り直した。現行 chain は add-RMSNorm/group-quant を含まない。過去の [bundle](2026-10-18-firmus-max-bundle.md) の本数・順序をそのまま用いていない。
- [最大性能計画](2026-10-18-firmus-maximum-performance-plan.md)、[timeline](2026-10-18-megamoe-kernel-timeline.md)、[Decode tail](2026-10-18-megamoe-decode-tail.md)、[Prefill tail](2026-10-18-megamoe-prefill-tail.md)、[Prefill mainloop](2026-10-18-megamoe-prefill-mainloop.md)、[小 kernel](2026-10-18-sglang-small-kernels.md)、履歴、ローカル Codex 会話記録、Actions `39100197595` の失敗ログを参照した。進行中 Actions は操作していない。指定の `2026-10-18-megamoe-geometry-and-redundancy.md` と `task/1. Qwen3 AI Task Rules.md` はこの worktree には無い。精度条件は今回のユーザー指示を正本とする。
- A1 raw 正本 `/srv/qwen-artifacts/39070273796/1/qwen-firmus-pd-39070273796-1/profile/profile/kernel-summary.json` を直接読み、Prefill-0 は 6745 call、平均 **488.947 µs**、prepare は **3.839 µs**、Decode-0 の主 kernel は 50160 call、平均 **49.574 µs**、主要 prepare は **3.486 µs**を確認した。全 stream/GPU の kernel 時間の和であり、step critical path や TARGET_VERIFY だけの純粋な平均ではない。4k・4027 token/rank・545 TFLOPS・15.5%/12% は計画と今回の measured context に従う。raw を別ファイルへ複製していない。
- image は config の `39070273796.sqsh`（SHA `7c6286e5…`）。`ssh firmus` で Python interface だけを `unsquashfs` で自分の `/scratch/home/ACCOUNT/claude-fa3/` に抽出し、調査後削除した。image の `sgl_kernel/flash_attn.py` は pinned source と同じ SHA-256 `610747f2c68495c5f3ba67f1247f0a1b4e4f7df116e2f89cd8e8983f0271f9ac`。toolchain は使っておらず抽出していない。

FA3 C++ の参照は pinned SGLang の `python/sglang/kernels/aot/CMakeLists.txt` が指定する [sgl-attn f89bc230 / flash_api.cpp](https://github.com/sgl-project/sgl-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/flash_api.cpp)、[heuristics.h](https://github.com/sgl-project/sgl-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/heuristics.h)、[tile_size.h](https://github.com/sgl-project/sgl-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/tile_size.h)。URL archive の pin は SHA-256 `418b5681…`。image の native binary がこの archive と同一由来かは Python SHA だけでは証明できないが、以下の 128×160 / 64×96 geometry は A1 kernel 名とも一致する。upstream main の可変ソースから今回の挙動を推定していない。

## 2. 正確な呼び出し

対象は Qwen target の decoder self attention（94 層）。DP attention の attention TP=1 により **Q 64 head / KV 4 head、D=128、GQA 比16**。モデル全体の TP=4 と混同しない。CP、MLA、SWA、cross attention、cascade（topk>1）は使わない。

`RadixAttention` → `FlashAttentionBackend.forward_extend` が **EXTEND と TARGET_VERIFY の両方**を扱う。`forward_decode` の scheduler metadata 再利用箇所は TARGET_VERIFY には到達しない。backend が ctor で保存する `self.flash_attn_with_kvcache` と、実際に `forward_extend` が呼ぶ module-global の wrapper も区別が必要。実際は `sglang.kernels.ops.attention.flash_attention.flash_attn_with_kvcache(ver=3)` → `flash_attention_v3` loader → sgl-kernel または community implementation。`environ.py` の `SGLANG_USE_SGL_FA3_KERNEL=EnvBool(True)` が既定なので、環境で上書きしない今回の経路は sgl-kernel。今回照合した native API は image の `sgl_kernel.flash_attn.flash_attn_with_kvcache` → **`torch.ops.sgl_kernel.fwd.default`**。community kernel へ切り替える実験では binary/API の再確認が必要。

| 引数・metadata | Prefill EXTEND（request-aligned、prefixなし） | Decode TARGET_VERIFY（EAGLE3 topk=1） |
|---|---|---|
| Q | FP8 E4M3FN、contiguous view `[total_q,64,128]`、約4027行 | 同じdtype/layout、32要求なら128行 |
| K/V | pool の FP8 buffer を `[num_pages,64,4,128]` に view | 同じ |
| cache_seqlens | int32 の各要求の全長 `L_i` | int32 `seq_lens + 4`（verify窓を含む） |
| cu_seqlens_q | int32 `[0,cumsum(L_i)]`、no-prefixでは cu_k と alias | `[0,4,8,…,4B]` |
| cu_seqlens_k_new | int32 `[0,cumsum(L_i)]` | int32 `[0,cumsum(seq_lens+4)]` |
| max_seqlen_q | 要求長のhost max、約4k | 4 |
| page_table | req_to_token を要求行で gather、64 token ごとのslotを64で割った int32ページID | 同じ。CUDA graph replay では所定 buffer へ更新 |
| scale / mask | `128**-0.5`、causal=True、window=(-1,-1)、softcap=0 | 同じ。bottom-right causalで4候補の未来をmask |
| tuning | num_splits=0（deterministic modeでは1）、pack_gqa未指定、sm_margin=0 | 同じ |
| その他 | scheduler_metadata未指定、return_softmax_lse=False、ver=3 | 同じ |

`k,v`（append用）、`qv`、rotary tables、cache_batch_idx、leftpad、sinks は渡さない。`k_scale/v_scale=None` なので q/k/v descale も None。`out` は `_attn_output` があればその view、なければ None。native wrapper は `cu_seqlens_k=None`、`seqused_k=cache_seqlens`、`max_seqlen_k=None` を渡す。従って C++ 内の K 上限はページ表の幅×64であり、常に実際の最大cache長そのものではない。名前が `cu_seqlens_k_new` でも k_new/v_new=None なので KV を FA3 内で再追加する意味ではない。

`qknorm-rope-fp8-kv.patch` と後続 `qknorm-rope-fp8-token-warps.patch` の融合経路は、QK norm/RoPE後のFP8 Qと、`out_cache_loc` へのFP8 KV書き込みを行い、`_stored_fp8_kv_last_call=True` で backend の再storeを止める。EXTEND と TARGET_VERIFY/top1、CP無し、標準 contiguous MHA pool、scale無し、位置とslotのindex dtype一致が条件。**入力 K は未更新の BF16 のまま**（`23aabcc` の修正を含む）。raw `k` を contiguous FA3 へそのまま渡す案は、dtypeだけでなく norm/RoPE済みかも違うので誤り。FP8 Qの `.to(kv_cache_dtype)` は同じdtypeへの no-op で、新しいFP8量子化ではない。

## 3. geometry と量的な余地

固定 C++ の `use_one_mma_wg` は `q_len × (Hq/Hkv) <= 64` で有効。Decode は4×16=64なので **M64/N96、1 MMA warpgroup**、Prefill は **M128/N160**（paged non-TMA）。A1 の実 kernel 名にも両方が現れる。Prefill の K 軸4kには約25個のN160 tile、packed Q軸には約504個のM128 tile×4 KV headがあり、単純なSM不足ではない。Decodeは各要求あたりpacked M64が1個、32×4=128仕事。H200の **132 SM** に対する理想的な1波の充填率は128/132=97.0%。これはwarp occupancyや実HBM効率を測った値ではない。[NVIDIA H200構成図 p.4](https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/CUDA-Programming-and-Optimization.pdf)

帯域上限 **4.8 TB/s** は [NVIDIA H200仕様](https://www.nvidia.com/en-au/data-center/h200/) に従う。32×約4000×4×128×2 byteのKVは約131 MB（計画の129 MBは実平均長による）。129 MBを一度読むだけで26.9 µs、49.6 µsとの差22.7 µsが理想的な余地。stepに占める12%から、HBM下限まで縮んでも全stepは約 **5.5%短縮**。実際はQ/O、再読込、mask、softmax、起動、依存関係があり、この下限達成を予測しない。

Prefillは計画の545 TFLOPSを用いると約266.5 GFLOP/層。公開FA3のH100 FP8 **約1.2 PFLOPS** は別shapeの到達値で、今回の保証でも上限でもない。それを仮の目標にしただけでも489→222 µs、全Prefill stepでは15.5%×(1−545/1200)=**8.5%短縮**に留まる。[FA3原論文](https://arxiv.org/abs/2407.08608)。GQA/causal/page64でこの値が出るという根拠は無い。計画の700 TFLOPSなら約381 µs、全step約3.4%短縮という算術になる。

## 4. sgl-kernel再ビルド不要のレバー

| レバー | 機構、期待幅の根拠、判断 |
|---|---|
| num_splits | 0はheuristic、1は無分割、>1はKV分割＋FP32 partial/combine。今回の32要求では128仕事≥0.8×132なので固定ソースは既に1を選ぶ。Prefillも十分な仕事があり1。1を強制するだけでは主形状は変わらず、低batchで並列性を失う。2分割でも単純な波充填率は256/264=97%で同じ。1 split当たりDecodeのpartial Oだけで128×64×128×4=4.19 MB、追加write/readとcombineを払う。帯域並列性の改善はあり得るが未測定。`bench_fa3_verify` の既存 sweep が判定手段。異なるDP rankが異なるsplitを選ぶこと自体はFA3の通信不整合ではない（FA3はrank-local）。 |
| pack_gqa | Noneで今回のpaged non-TMAは既にTrue。C++ dispatchは `PackGQA_ || PagedKVNonTMA || Split` なのでFalse指定もunpacked kernelを選ぶ有効な比較にならない。prepare側だけFalseの解釈になる危険があり、今回は使用しない。単純なpacking有効化による利得は0。contiguousではTrue/Falseの意味が生じるが、4-token verifyは16 headを束ねて64行を埋めるのが自然。 |
| scheduler metadataをforward当たり1回 | prepareの上限節約はPrefill約3.84×93=357 µs/step、Decode約3.49×93=324 µs/step（94層を仮定、層間再利用）。それぞれ主attentionの約0.8%/7%、全stepの約0.1%/0.8%の規模。nativeはprecompute利用時のsemaphore resetも発行するのでこの全量は取れない。後述の整合性修正・graph確認が先。 |
| page64→128/256 | FP8 PrefillのTMA候補はM128/N224で、`page_size % 224 == 0` が必要。64/128/256はいずれも不成立。DecodeはTMA候補M64/N96だが、4×16=64が `> M64` を満たさないため、ページを変えてもTMAにならない。単に「page256ならTMA」は今回誤り。ページ表byte数はKV本体に比べ小さく、直接の帯域利得は小さい。割当・断片化・PD転送・staging・server-infoのpage64契約も変更するので採用しない。224の倍数はFA3側の候補に過ぎず、SGLang allocator等の対応を確認せず設定しない。 |
| no-prefix contiguous FP8 K/V | 最も検討価値がある。paged non-TMA→contiguous TMA、tile N160→N224という経路の変化がある。FP8の同じbyteをslot順にgatherすれば精度は変わらず、現行APIで実験できる。約4027×4×128×2=4.12 MBのKVに対するgather read+writeは8.25 MB、理想下限1.72 µs＋起動。attention本体の数µs以上の改善なら回収できる可能性があるが、545→700 TFLOPSの差108 µsを実現する証拠は無い。融合storeを省略するとPD転送が壊れるので維持する。将来の融合kernel dual-outputは別のJIT変更・compile・数値gateが必要。 |
| causal / varlen layout | no-prefixでもcausal=Falseは未来tokenへのattentionを加えるため不可。verifyも最後以外のqueryで未来候補をmaskする必要がある。全要求を一本に連結してcu_seqlensを消すことは要求間漏洩。長さが同じと実証された場合だけfixed-length/batched viewでvarlen prepareを省けるが、可変長の公式入力ではpaddingとmaskが必要。prepare約3.5–3.8 µsだけを目標に意味変更しない。 |
| CUDA graph metadata | TARGET_VERIFYは既にbucket別bufferをcapture/replayし、cache_seqlensとpage_tableを更新する。graph化だけでnativeのprepare kernelが消えるわけではない。Prefill full graph用metadata経路もsourceにはあるが、有効化はshape bucket/padding/DP replayまで別に検証が必要。削れるCPU launch時間はこのkernel-summaryでは定量化できない。scheduler結果をcapture時の長さで固定すると不正。 |
| out / sm_margin / FA3 loader | outは既存API/経路があり、追加導入のbyte-copy節約を主張する前に実runで `_attn_output` 利用を確認する。sm_margin>0は通信との同時実行にSMを譲るknobで、FA3単体は速くならない。通信重複の実測無しでは選ばない。community kernel選択は再ビルド無しでもbinaryと精度の変更なので、今回のpinの性能見積もりをそのまま適用できない。 |

KVをBF16へ上げること自体は精度条件に反しないが、DecodeのKV trafficがほぼ2倍になる。QだけBF16、K/VをFP8のままにする混在は今回のnative dtype契約では使えない。新たなFP8 cast、FP8未満、mask省略、context短縮、近似attentionは候補にしない。tile/pipeline/warpgroup数の直接変更はC++ instantiation変更とsgl-kernel rebuildを要するので本表のruntime候補と区別する。

### 4.1 DP precompute制限を外すだけでは使えない

sourceコメントは、distributed attentionでlive cache lengths/splitsが変わり、stale metadataによってsplit combineがOOBになり得るとしてDP/CPを無効化している。これは記録された防御理由であり、今回その元事故を再現したわけではない。現行sourceには、今回に直結する次の問題がある。

1. `_compute_scheduler_metadata` は `max_seqlen_q=1` 固定。verifyの4とは不一致。今回は双方ともone-MMAでM64なので直ちに全caseが壊れるとは断言しないが、query geometryを正確に表していない。
2. ctorのprecompute用head数は **model_runner.ps.tp_size=4** で割り、16 Q / 1 KV head。一方Qwenの実layerはattention TP=1なので64 Q / 4 KV。GQA比は同じでも仕事数・split heuristic・L2 head groupingが異なる。この不整合はCPUで確認した。
3. native fwdはページ表幅×pageをK上限に使う。precomputeの実max cache長とtile境界が違えばheuristicがずれ得る。split数、packing、Q/K上限、dtype、causal、head数、SM数をfwdと一致させる必要がある。
4. TARGET_VERIFYにはmetadata作成・消費の配線自体がない。通常decodeだけに配線されたfieldを有効化してもverifyのprepareは消えない。

将来実装するなら、各attention backend/rankのforward専用bufferに、実layerの64/4、q=4、同じpage表上限・実cache長で作り、各replay前に更新する。rank間や別batchで共有しない。最初はnum_splits=1を両側で固定すればdynamic split容量の食い違いを限定できるが、低batchでの性能損失を別に評価する。nativeはsemaphoreを変更しresetするので、単なる不変tensorとしてcapture時だけ作る設計は避ける。これらを修正するだけでDP安全性が証明されたことにはならない。元のDP guardは今回保持した。

## 5. 次の一測定と採用条件（統合担当向け）

**実official Prefill forwardの一層で、同じQとcache内容に対し、現行paged callと「`out_cache_loc`順のFP8 gather＋`flash_attn_varlen_func`」を同一scheduler割当内でpaired計測する。** no-prefixのhost条件、各要求のcu_seqlens、実4027前後のtoken数、scale=None、causal=Trueを固定する。gather後は `[total_k,4,128]` のFP8 K/V、cu_k=cu_q、両max長=実host max、num_splits=1、pack_gqa=None、同じscale/window/softcapを渡す。raw BF16 Kを使わない。paddingがあればdummyの独立segmentを付けて実要求を混ぜない。

計測対象はFA3だけでなく**gatherを含むCUDA event区間**。warmup後の反復を交互に測り、medianとtail、kernel名（paged M128/N160からcontiguous側へ変わったか）、GPU・binary pin・shapeを記録する。profilerは別pass。同一FP8 cache bytesを入力し、全実tokenのBF16出力差と非有限値を確認する。これがnet短縮を示さなければ、no-prefix用の新たなKV出力bufferやJIT融合には投資しない。net短縮すれば、その経路を最小限のSGLang patchにし、store/PD handoffを残して測る。今回この測定を実施したという意味ではない。

serving採用には条件ごとの追跡可能なcommit、CPU検証、必要な新image、Actions経由のscheduler実行を統合担当が行う。公式入力・設定を固定してbaseline/candidateをそれぞれ複数run、terminal状態まで確認し、公式result parser、TTFT/ITL/goodput、server-info、raw artifactを評価する。`check-firmus-megamoe.py` と `check-firmus-decode-kernels.py` は許容差を変えず通し、加えてattention出力差・graph bucket変更・rankごとの不均一長/idle・PD転送後の生成を確認する。既存gateだけでは新attention経路の数値検証を代用できない。bench/profile計装は通常benchへ残さない。

## 6. 今回の成果物、検証、配線

追加は本noteと `scripts/check-firmus-fa3-source.py`。CPU checkは実sourceの `forward_extend` をAST抽出してEXTEND/VERIFY両方の選択を実行し、FP8 cache、mask、引数、未指定knobを検査する。続けて実sgl-kernel Python wrapperを実行してnative opの引数位置まで確認し、DP/CP guard、q=1固定、model TP head数の問題を検査する。torch/CUDAはimportせず、数値検証を装うstubにはしていない。

`fc275674` へ現行22 patchを順に `git apply --check` / `git apply`、新CPU check、既存 `check-firmus-latency-source.py` がPASS。Python構文検査とdiff whitespace検査も実施。固定 `tile_size.h`（SHA-256 `49d112117594a6b319d7fcb8f26f753306bab74a952f8344e43d2289525dff85`）をローカル g++ 13.3.0 / C++17 で直接includeし、Prefill M128/N160、Decode M64/N96、TMA候補N224を5個のstatic_assertでcompile確認した。初回のwarnings-as-errorsはupstreamの未使用引数で停止し、`-Wno-unused-parameter` を追加して通過（header無変更）。これはCPU constexpr検証のみ。nvcc/ptxas、SASS、GPU numerical gate、CUDA graph実動作、serving性能は**未検証**。変更kernelが無いためCUDA compile結果は無い。

統合担当が今行う配線は任意で、patch済みsourceを引数に `python3 scripts/check-firmus-fa3-source.py <source_dir>` を既存CPU source checkの横に追加するだけ。本noteを採用するためのbuild/runtime設定変更は無く、imageを再ビルドする必要も無い。性能patch、flag、container manifest、config、既存数値gateは変更していない。不要なremote抽出物は削除し、GPUジョブも一時的な監視ジョブも作成していない。
