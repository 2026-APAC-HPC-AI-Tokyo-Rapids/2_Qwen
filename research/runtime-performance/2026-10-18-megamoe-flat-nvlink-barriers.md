# 2026-10-18 SM90 MegaMoE 全 geometry の flat NVLink barrier 再検討

結論：`sm90-flat-nvlink-barriers.patch` を全 14 patch の後へ載せ直し、series の 15 本目に追加した。**CPU compile 済み、GPU gate・性能は未確認の候補**である。以下の前提で通信手順は成立する。全 rank が同じ workspace を同じ集団呼び出し順に使い、各 rank 内では呼び出しを stream 順に直列実行すること、EP4 の各 rank が 132 CTA を起動すること、workspace の初期 zero が全 rank で完了していること。この前提は既存 MegaMoE の利用条件でもある。異なる geometry が同じ collective に混ざることは許す。

[Decode tail note](2026-10-18-megamoe-decode-tail.md) §1.5 の旧案から、コンパイラの memory clobber と、generic global memory を TMA で読む前の async proxy fence を明示した。FC1/FC2、量子化、scale、BF16 scatter、FP32 combine の演算・型は変えていない。GPU 仕事、job 投入、push は行っていない。

## 1. 起点と rebase

起点は `823b79f`、[bundle note](2026-10-18-firmus-max-bundle.md) の 14 本（header `ff7a58cca00aa42a90386d12b0437b5d8caf35eb0ad082a8a1a0b4b74001a2a1`）。提供された `scratchpad/rev` はコピーだけを使い、upstream v0.1.5 の header から全 series を順に `git apply --check` / `git apply` して再現した。旧 W3 patch の各 hunk は、現在の 14 本へは offset 調整で適用できた。その上で本稿の順序保証を明示した。Prefill tail の cleanup（grid-sync slot 3）と Decode の動的割り当て counter reset は両方残す。最終 header pin は `cb0c5238d4ac08546197e8c7644ea7bad0e2dbcc5077b2ed99a2373c9656fb6e`。

過去の根拠は上記 bundle / Decode tail、[timeline](2026-10-18-megamoe-kernel-timeline.md) §4.2、[Prefill tail](2026-10-18-megamoe-prefill-tail.md)、[Prefill mainloop](2026-10-18-megamoe-prefill-mainloop.md)、[予算表](2026-10-18-firmus-maximum-performance-plan.md)。`gh run view 39086770417` で旧実験が terminal failure であることも確認した。旧 gate の `[7,0,65,129]` 停止の記録は Decode tail §2.3。指定された geometry-and-redundancy note と task rules のファイルはこの worktree には無く、精度規則は今回の指示を正本とした。関連 Codex session も確認したが、上記の保存済みノート以上の flat barrier の追加検証結果は得られなかった。

## 2. 通信手順と workspace

| 段階 | 送信 | 待つ側 | 保護するデータ |
|---|---|---|---|
| count publish | 各 rank の SM0 が各 expert に `red.release.sys.add.u64(expert_status)` | 各 dispatch warp の既存 count spin + `fence.acq_rel.sys` | per-rank count、remote source index、pre-dispatch 入力 |
| pre-combine | 各 CTA の全 math thread が named barrier、代表 thread が各 rank の counter に release.sys add 1 | 各 CTA の math thread 0 が local counter を acquire.sys load、最後に named barrier | 全 rank から自 rank の combine buffer に scatter された BF16 |
| cleanup 完了 | 各 CTA の全 dispatch thread が named barrier、代表 thread が各 rank の別 counter に release.sys add 1 | SM0 の dispatch thread 0 だけが acquire.sys load、最後に dispatch named barrier | recv/send count、arrival、動的 scheduler counter の reset |

pre-combine / cleanup counter は `workspace.num_max_pool_blocks-1` / `-2` の L2 arrival-mask slot（uint64）。Qwen の固定容量 141312 token、最小候補 BLOCK_M=8 なので slot は 17663 / 17662。8 byte aligned な既存 uint64 配列内である。通常 arrival は BLOCK_M=64 で `[0,2208)`、128 で `[0,1104)`、Decode claim は slot 2208 の下位 uint32。Prefill claim は別領域の grid-sync slot 3。いずれも counter と重ならない。static assert は pool + Decode claim 1 slot + flat 2 slot が収まること、および signal 用 thread 数を検査する。

番地は rank ごとの token 数、BLOCK_M/N、wave、SwapAB に依存しない。`thread r` が `(rank_idx+r)%kNumRanks` を担当し、全 CTA が全 rank にちょうど 1 回ずつ加算する。EP4 / 132 CTA では各 counter の 1 call の増分は **528**。両 barrier の呼び出しは template 条件の外にあり、Prefill、SwapAB、非 SwapAB、wave 16/32、token 0 で分岐しない。旧 NVLink barrier はこの kernel からすべて外れ、旧 protocol の signal を待つ geometry は残らない。

## 3. メモリ順序の論証

仕様の根拠は CUDA 13.0 に対応する [PTX ISA 9.0](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#memory-consistency-model) §8.4–8.10。`.gpu` は同一 GPU、`.sys` は他 GPU も含む。volatile load は relaxed.sys 相当であり、それ自体に acquire は無い。release/acquire は RMW を介する observation order でも成立し、同期は累積する。これは以下のコードに対する論証であり、GPU 数値 gate の代用ではない。

### 3.1 pre-pull barrier を削除できる条件

各 dispatch CTA は per-expert count を atomic 集計し、remote source index を書いてから、既存の rank 内 dispatch `grid_sync` を通る。これは CTA named barrier → release.gpu atomic → acquire.gpu spin → named barrier の手順で、SM0 の publisher は全 dispatch CTA の書き込みを引き継ぐ。同一 GPU の集約には .gpu で足りるが、次の publish は必ず **release.sys** とする。元の relaxed な `atomic_add_sys` のままでは足りない。

SM0 は per-rank count を書いた後、その expert の sum に `(132<<32)|num_tokens` を release.sys で足す。token 0 の rank も high word 132 を足す。低位 token 合計は最大 4×4224 で uint32 の桁上がりは無い。既存 scheduler の各 lane は sum の high word が 528 になるまで `ld.volatile.global.b64` で待つ。最後の値を読めば全 rank の release RMW を observation order で覆う。dispatch は `fetch_expert_recv_count()` の warp sync 後、全 lane で `fence.acq_rel.sys` を実行する。各 expert の spin を担当した lane の観測は warp sync を通じて他 lane に伝わり、後続 system acquire fence と対になる。これで per-rank count と source index を読める。

remote FP8 token / FP32 SF / top-k は、その rank の前段 pre-dispatch kernel が完了してから当該 MegaMoE kernel が起動するため、同じ publish 以前のデータとなる。TMA による token pull の前には `fence.proxy.async.global` も実行する。通常の system acquire と generic→async の順序を混同しない。前段の型・量子化・padding は変えていない。

A/B loader と math も count spin を持つが、この値で決めるのはローカルの block 範囲である。remote source index と token を読むのは dispatch だけで、GEMM が読む local pool は dispatch の TMA store 完了と既存 arrival release/acquire によって別途保護される。loader に system fence を無条件追加する必要はない。

### 3.2 pre-combine と remote buffer の可視性

FC2 scatter は各 math thread の generic-proxy global store である。flat barrier の最初の named barrier は **全 epilogue thread** が参加し、scatter した全 thread の書き込みを signal thread に伝える。全 rank に送る release.sys add はその後に置く。counter の最終値を読んだ acquire.sys は、全 528 release を RMW の鎖を通じて受け取る。最後の named barrier でその観測を全 combine thread に伝える。コンパイラの移動を防ぐため named barrier の前後と acquire inline PTX に `memory` clobber を明示した。

combine が使う TMA load より前には `fence.proxy.async.global` を置く。scatter は generic、load は async なので、generic の acquire だけで異なる proxy の問題まで解決したとは扱わない。TMA load 完了の mbarrier wait、既存 TMA store の arrive/wait と shared-memory proxy fence はそのまま残る（[PTX Async Proxy](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#async-proxy)）。未完了の async store を flat barrier の release で代替する変更はしていない。

### 3.3 cleanup と次の call

pre-combine で全 math CTA が計算を終えた後、各 CTA は従来どおり dispatch+math の handshake を通る。dispatch の token pull も完了済みである。したがって、recv count / arrival / source metadata の読者が残る間に cleanup しない。Prefill FC2 claim と Decode claim の発行も、最終 block または終端の通知を math が読む前に済んでいるので、SM0 の counter reset と競合しない。最後の ring の B-loader 受信が残る場合にも combine は ring の後ろの smem を使う、という 14 本側の修正を維持する。

各 dispatch CTA の cleanup store → named barrier → release.sys → 各 rank の SM0 acquire → kernel 完了 → 同じ stream の次 call、の順で全 rank の reset が次 call に先行する。cleanup counter を待つのは SM0 だけでよい。SM0 以外が先に戻っても、その rank の kernel は SM0 を含む全 thread が終わるまで完了しない。

cleanup は combine と重なる。このため「全 rank の cleanup が終わった」だけから「全 rank の combine が終わった」とは推論しない。早い rank が次 call に入っても、次の token pull は全 rank の **次 call の** count publish を待つ。それぞれの publish はその rank の前 call の kernel 完了後である。従って次 call の FC2 scatter が、遅い rank の前 call の combine 読み出しを追い越す経路も無い。これは pre-pull barrier 省略時に必要な追加条件である。

## 4. epoch、graph、idle、geometry 変更

各 counter は一度 zero 初期化された後、call ごとに 528 ずつ単調増加し、通常 cleanup では reset しない。call n の開始値を `(n-1)×528` とする。CTA 自身の local add の返り値 `old` は `[(n-1)×528,n×528)` に入るため、待つ目標は `(old/528+1)×528 = n×528`。この local add も 528 個の一つなので、直前の epoch を誤って完了扱いしない。

次 call の到着がこの区間へ混ざらないことは次のように帰納できる。

- pre-combine：次 call に入れる rank は、前 call の final SM0 acquire を終えている。その acquire は全 rank の全 dispatch CTA の cleanup 到着を必要とする。各 cleanup 到着は対応する math CTA の pre-combine wait と handshake より後なので、全 rank の前 call の pre-combine counter はすでに n×528 へ到達済みである。
- final：次 call を始める rank の前 call kernel は、SM0 acquire だけでなく、他の全 local dispatch CTA の local/remote add 命令も終えている。さらにその rank の次 call の pre-combine は、全 rank が次 call の count を publish するまで成立しない。従って次 call の final add 以前に全 rank は前 call kernel を終え、各 rank の前 call final counter は n×528 に達している。

全ての CTA に「返り値から求めた同じ epoch」が割り当たる。uint64 の wrap は許さない（528 増分で約 3.49×10^16 call）。通常の process 寿命では問題にならないが、数学的に無限回 replay を保証するものではない。

CUDA graph は target や epoch を host で capture せず、各 replay の device atomic の返り値から計算する。capture に伴う warmup / 実行回数が異なっても、各 rank の collective 順序が同じならよい。graph replay の途中で counter を 0 にしない。buffer 全体の初期 zero、再作成時の collective zero は全 rank が quiescent な状態で行う。片 rank だけの reset、実行中の buffer zero、同一 workspace を独立 stream から同時使用することは対応しない。debug の `DG_COMM_KERNEL_DEBUG` による全 buffer zero も、全 rank の停止・再初期化が保証される場合だけであり、通常 serving のように off とする。

idle rank は全 top-k が -1 でも全 CTA を起動する。count high word、pre-combine 528 個、final 528 個の参加数は減らさない。処理対象が無い scheduler も pre-combine に到達する。combine warp の早期 return はこの barrier と dispatch handshake の後である。masked row は従来どおり zero を出し、remote の未生成 slot を読まない。

`[7,0,65,129]`、idle Prefill と busy Prefill、wave 16/32、連続する Prefill→Decode→非 SwapAB の call も、番地と到着数が同じなので同じ帰納法で扱える。geometry 内の pool partition と scheduler だけが変わり、その call の cleanup と次 call の count publish が境界を保護する。全 rank は同じコード版を使うことが必要で、旧版と新版の混在 deployment は不可。

## 5. CPU compile

`ssh firmus` の login CPU だけを使用。`configs/firmus-framework.json` の image `/scratch/home/ACCOUNT/qwen/containers/39070273796.sqsh`（SHA-256 `7c6286e5dcfcd8f5fba37d03d0c1ac0b4f6e93562a24a147afa1063f8214bc82`）から、自分の `/scratch/home/ACCOUNT/claude-barriers/image` へ `unsquashfs` で CUDA と wheel include を取り出した。CUDA 13.0 **V13.0.88**。既存の共有 checkout / 他 worker の toolchain は変更していない。

基本 flags は `containers/build-sm90-megamoe.py` と同一：

```sh
nvcc --cubin --std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -O3 --ptxas-options=-v,-warn-spills -gencode arch=compute_90a,code=sm_90a -I<headers> <kernel.cu> -o <kernel.cubin>
```

共通 template は `4224,4096,1536,128,8,wave,BM,BN,128,141312,2260992,stages,64,64,epilogue_threads,132,4,infinity,true,register_budget,reuse,arrival_counter,full_sync,false,swap`。Prefill は wave32 / 128×256 / stage3 / ep512 / budget112 / true,true,false,false。Decode は wave32 と16 / 64×128 / stage8 / ep256 / budget0 / false,false,true,true。指定の非 SwapAB は wave32 / 64×256 / stage4 / ep256 / budget0 / false,false,true,false（compile にだけ追加）。さらに、提供されたhost sourceの `decode_use_block_n_256` は intermediate_hidden>=3072 を条件にしており、H4096/I1536の129–255行ではBN128を選ぶ。今回の依頼のBN256という前提との差を隠さず、非SwapAB 64×128 / stage8 / 他条件同一も追加compileした。実際のimage hostの選択は統合時に確認する。wheel ABI import と GPU 実行は行っていない。

| geometry | 14 本 reg / spill store/load B | +flat reg / spill store/load B | stack B 前→後 |
|---|---|---|---|
| Prefill 128×256 | 96 / 44/60 | 96 / 20/20 | 24→24 |
| Decode SwapAB wave32 | 168 / 0/0 | 168 / 0/0 | 8→8 |
| Decode SwapAB wave16 | 168 / 0/0 | 168 / 0/0 | 8→8 |
| 非 SwapAB 64×256 wave32 | 168 / 816/980 | 168 / 816/980 | 336→336 |
| 非 SwapAB 64×128 wave32（追加確認） | 168 / 0/0 | 168 / 0/0 | 8→8 |

[compile transcript](2026-10-18-megamoe-flat-nvlink-barriers-compile.txt) に実際のnvccコマンドとptxas出力を保存した。前後5形状の全 cubin を生成し、`cuobjdump -sass` で読み取れることも確認した。spill は ptxas の静的な命令集計で、実行ごとの転送量ではない。非 SwapAB の大きな spill は起点にもあり、本 patch で解消するものではない。Prefill spill 減少の速度効果も未計測。未使用定数の compiler warning と上表の spill warning はあるが compile error は無い。

## 6. gate の範囲と残る確認

既存 `scripts/check-firmus-megamoe.py` の `cross-rank-empty-tail` は geometry 混在と token0、`capacity-hot-experts` は全 Prefill、`decode-graph` は変わる入力・mask・rank2 の idle/live 交替を4 replay、`decode-tile-*` は少数 token と他 rank の idle を検査する。idle geometry flag が on の `idle-prefill-rows` は 256 行すべて zero も検査する。bench 部分は warmup と20連続 replay / call を同一 buffer で実行する。

今回 `mixed-geometry-graph=[7,0,129,256]` を追加し、同じ4 replay の数値比較を行うようにした。SwapAB、非 SwapAB、Prefill と token0 の同居、および rank2 が129行のまま mask で idle/live 交替する場合を覆う。許容値 relative L2 < 0.025 / max scaled error < 0.08 と独立 FP32 参照は変更していない。先頭で呼ばれる `check-firmus-decode-kernels.py` も変更していない。

CPU でできたのは全15 patchの適用・pin照合、Python構文検査、image metadata test（33 passed）である。**追加caseを含むGPU数値gateは未実行**。特に、geometryを変える2 callを単一graphへ連結した多数replay、全rank idle、rankの起動をずらした多数反復は、統合時の追加 stress として価値がある。現行gateの各case間にはhost同期があるため、これらまで測定済みとはしない。

## 7. 期待効果と採否

Decode timeline の pre-pull 約7 µs、pre-combine grid-sync median 7–9 µs、final 2–5 µs を単純に足した16–21 µsは **短縮の見積もりにならない**。pre-combine の大半は最後の FC2 producer 待ちで、flat にしても全producer完了は必要である。新 count spin も全rankの publish、flat acquire も最後のscatterを待つ。動的割り当て後は rank 内待ち自体が median2–6 µsへ減っている。

消せるものは pre-pull の SM0 専用往復と後段grid伝播、および pre-combine / cleanup の集約・再伝播である。W3 の推定と同じ **Decode 4–7 µs/call（200 µsの2–3.5%）を検証目標**とする。94層なら0.38–0.66 ms/stepだが、serving throughputの増加率をこの値から直接断定しない。Prefillでも同種の数µsが効く可能性はあるが、1.6–1.8 msに対して0.2–0.4%程度で、run間の揺らぎより小さい。

逆方向の要因は atomic の集中である。各 barrier は rank ごとに528 arrival、EP全体2112 add、そのうちremote1584 addとなる。2 barrierで4224 add/call、全CTAのacquire spinも同じcache lineへ集中する。旧版より通信回数が多く、proxy fenceも必要なので、**改善ゼロや悪化もあり得る**。複数rankの同条件benchとtimelineで、最後のFC2→全combine開始、count publish→最初のFC1、cleanup完了→kernel終了を分けて判定する。gate timeout、誤差、prefill回帰があれば採用しない。

## 8. integrator の配線

- series 末尾の15本目と `header.sha256`、image metadata test は本commitで更新済み。現行buildはディレクトリ全体とseriesを読むためDockerfileやbuild scriptへのpatch名追加は不要。同じ15本を全rankのimageへ適用する。
- source変更なので新imageのbuildが必要。既存 `configs/firmus-framework.json` の旧imageのままservingしない。integratorがbuild後に実際のimage path/hash/provenanceへ更新する。GPU gateは上記スクリプトを使い、idle geometry on/offの双方で確認し、許容誤差を緩めない。
- kernel-bench overlayは現在8本適用済みimage用であり、本patch単独をそこへ当ててはいけない。既存 `b-bundle` の01–06（series9–14）に続けて本patchを07として置くか、15本入り新imageを基準にする。旧timeline overlayはそのまま当たると仮定しない。barrier計装位置を新helperに合わせ、計装無しのbenchと分ける。
- terminal gate成功、複数回の同条件bench、その後の公式task / parser / artifact回収までのE2E確認はintegratorが行う。本作業はローカルcommitだけであり、GPU検証済みという採用判断は行わない。

自分のFirmus抽出toolchain・compile directoryは確認後に削除し、GPU / scheduler jobは一つも作成していない。
