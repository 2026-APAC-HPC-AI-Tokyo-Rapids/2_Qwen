# Round 36：Prefill SASS の CPU 監査

追加の削減を CPU compile で確認できた小案は、dense GEMM の初回 promotion、Q/K norm・RoPE の固定 shape、onepass 入力の定数除算の3件だった。ただし、**現在の定常 Tensor Core 依存を大きく短縮する案は得られていない**。FC1 first-K の実測を仕事量で正規化した参考幅は、それぞれ QKV＋O 合計 **0〜4.9 µs/層**、writer **0〜1.7 µs/call**、onepass **0〜1.4 µs/call**。GPU の時間・byte 一致・通常 serving は未検証で、採用候補の成立を意味しない。FA3 の barrier ID 共通化は CPU 段階で命令が増えたため棄却する。

親 main `f3b48f7`、branch `research/sass-audit-r36-20261104`。変更は本 note と tracker の1段落のみ。SSH、Actions 操作、Slurm/PJM、GPU 実行、image build は行っていない。mirror、`~/.codex`、`/tmp/qwen-*`、他 worktree は変更していない。

## 対照と再現条件

AGENTS.md、tracker/model、r31、r32〜35、consumer review、r13 scale 移動、FA3 P/P3、r23 writer、r27 norm、r28〜30 dense の記録と関連 commit 本文、保存された Codex 会話・CPU proof を確認した。Actions の証拠はローカルに退避済みの成果物を読んだもので、新たな Actions アクセスではない。

- image は `3a47b757`、SGLang `fc275674a97d78553b22597ee72c8fd48f5420f4`、DeepGEMM `1c8138a59b3df5dac50b76b730834ed57982fdfb`、FA3 `f89bc2306632d1ec5f97b014dded4254f5b4a907`、CUTLASS `57e3cfb47a2d9e0d46eb6335c3dc411498efa198`。
- r32 proof の source/dependency を自分の scratch へ展開し、現14 overlay の FC1 header `75085f03…` を使用。**FC1 first-K は有効**。新規 compile の MegaMoE は 11320 命令・181120 byte、encoding SHA256 `f157a6dddb926f5944f751e46448afb69091eb9cfbcccd17e21fe852205e8c41` で r32/r35 と一致した。dense header は r32 source と r28 proof の双方で SHA256 `2cc49403b142424781c6d64e78b9f4855b49c0d6baf26b8c597b8087d9b9503f` が一致する。
- SGLang は image source fixture の hash を検査する既存契約に従い、現 config 順に全 runtime overlay を scratch の copy へ適用した。JIT dependency header は mirror から読み取り専用でコピーした。
- **FA3 は P3 まで監査した。** 保存 run `41767886175/1/results/results/prefill-fa3-prepack/selection.json` は `selected=P3`、82/84 positive cells、pooled 中央値 5.137 µs/P 比。serving env も variant 172256 に対応する。r24 の kernel 名も P3 を含む。過去の「P3 を閉じる」という途中の結論だけから、現 serving を P41184 と決めつけない。P41184 は gate の対照・選択結果に応じた経路、single は N192/variant224 として別に数えた。
- dense は QKV `(7988,9216,4096)` と O `(7988,4096,8192)`、BM256/BN128/BK128、3-stage、cluster2/A multicast。r30 の M tail 分離は不採用であり、入れない。入力は E4M3、scale/promotion FP32、出力 BF16。
- writer は `qknorm_rope_prefill_fp32_kernel<true,int64_t>`、Q64/KV4/D128、grid250×72、256 thread。int32 も確認。MoE 入力は採用中の `_onepass_moe_input`、BM64、8 warps、3 stages、32×K128。input `norm_quant` も含める。非採用の post-norm-only を serving の大項目に数えない。

CUDA は `/usr/local/cuda-13.4` の nvcc/ptxas **13.4.92**、sm_90a。C++ は O3/C++20、FA3 は既存の fast-math・lineinfo・CUTLASS 定義を追加した。Triton は GPU/driver を使わない ASTSource/GPUTarget(cuda,90,32)、**3.7.0**、全 pointer の実呼出し相当の16-byte alignment、norm は fusion false、onepass は既存どおり fusion true。生成先 PTX/cubin は sm_90a。image の CUDA13.0.88 と Triton/compiler の実機 SASS を検証したものではない。alignment 指定のない初期診断の spill や、r28 再現用 Triton3.6 の数値を採用構成の資源・命令表へ混ぜない。

r24 nsys-rep を自分の scratch へ CPU export し、r31 と同じ PP0/GPU0、内側窓 `1426516182..8442926294 ns`、7988-row pair、7989-row norm を除外した **53周期**を再現した。周期113.605 ms、MegaMoE63.561、dense23.470、FA3 18.945、onepass2.937、writer/pack2.148、norm1.538 ms/周期。同 run の既存 clock 照合は1.410 GHz。MegaMoE の床37.972 ms/効率59.7%、FA3 の床8.230 ms/効率43.4%は既存模型で、残差を削除可能時間にはしない。これは FC1 採用後の内部時間の新測定ではない。

## 命令内訳

以下は包括的な PC 範囲の**静的 warp 命令 slot**。1命令は16 byte。predicate false の slot も含め、poll の反復、cold wait へ分岐した先、Tensor Core 内部の実行回数・待ち時間は含めない。FC1 chunk は後半3回、dense chunk は QKV4回/O8回。FC2 の K12 は全展開済みなので「定常 back-edge」とは呼ばず、初回後11 block を数える。

| kernel／範囲 | QGMMA/HGMMA | FMUL/FFMA | LDS/LDG 系 | BAR/SYNCS 系 | 整数/address | その他 | 合計 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MegaMoE FC1、8×K128、`7a90..df30` | 64/0 | 1092/0 | 32 | 48 | 179 | 196 | 1611 |
| MegaMoE FC2、後11×K128、`12500..1ac90` | 88/0 | 1500/0 | 44 | 65 | 188 | 285 | 2170 |
| dense QKV、8×K128、`f00..6ed0` | 64/0 | 32/1024 | 40 | 48 | 197 | 129 | 1534 |
| dense O、8×K128、`d60..6d30` | 64/0 | 32/1024 | 40 | 48 | 197 | 129 | 1534 |
| FA3 P3、非対角 KV96 step、`4fe0..67a0` | 7/0 | 66/52 | 0 | 13 | 51 | 192 | 381 |
| FA3 P、同 step、`4fa0..6730` | 7/0 | 66/52 | 0 | 13 | 46 | 194 | 378 |
| FA3 single N192、非対角 step、`7670..9bd0` | 10/0 | 66/100 | 0 | 13 | 48 | 362 | 599 |
| onepass、K128 group、`df0..36e0` | 0/8 | 66/50 | 51 | 16 | 288 | 177 | 656 |
| norm_quant、展開済み main、`0..16f0` | 0/0 | 55/16 | 11 | 6 | 141 | 139 | 368 |
| Q/K writer、Q の通常経路、int64 | 0/0 | 50/33 | 9 | 0 | 98 | 68 | 258 |
| 同 K、valid nonzero cache slot | 0/0 | 50/33 | 10 | 0 | 117 | 87 | 297 |
| 同 V、同条件 | 0/0 | 0/0 | 3 | 0 | 86 | 75 | 164 |

LDS/LDG 系は LDSM、LDGSTS を含むが store は含めない。同期系は WARPGROUP arrive/dependency wait、DEPBAR、LDGDEPBAR、FENCE/MEMBAR/WARPSYNC も含む。整数/address は I/UI、LEA、SHF/LOP、PRMT、VIADD、SEL、pointer の R2UR 等。「その他」には MOV/UMOV、CS2R、control flow、store、変換、FADD/FMNMX、SFU を含む。列は排他的で、合計を機械照合した。

Q/K/V は K loop を持たない。表は valid・repair=false・cache slot非zero の source 分岐に対応する issue 経路で、全 byte の例外 repair/assert block を数えた全体は472 slot。int32 は Q256/K295/V164。norm は内部除算 helper と alignment padding を含む全体480 slotとは区別した。onepass の656には4個の除算 fallback block が含まれ、通常は各5 slotを飛ばすので **636 issued slot/group**。静的 span を動的本数と混同しない。

MegaMoE の FMUL は FC1 accumulator1024＋scale68、FC2 accumulator1408＋scale92。RCP はそれぞれ32/44本。FC1 の初回恒等演算は残らず、FC2 初回も r35 のとおり既に除去済み。各 half の `wait_group → accumulator FMUL → QGMMA` は実データ依存である。UMOV113本/chunk や descriptor 用 R2UR106本/chunk があっても、uniform descriptor の register lifetime を無視して消せるとはしない。

FA3 P3 の非対角 step は EX2 50、FMNMX52、FADD48、FP8変換24。QK の完了から max/EX2/sum/pack を経て次の PV に入り、PV完了から O rescale を経て次の PV に入る依存がある。P3 の spill4/4 byte は producer の tile 準備側で、表の KV step では LDL/STL0。single の既存 spill84/116 byteも、定常 step の削除対象と同一視しない。単純な max/sum の木、half2/polynomial exp、V load 順序変更、slot 追加を再提案しない。

## 候補の順位と理論見積もり

FC1 の入口192命令と16〜25 µs/callの比は、GPUの1命令 latencyではない。historical を代表とすると、入口の動的 warp 命令は `192×(68288/64)×12×4 = 9833472`。各候補の**削る issue 量をこの分母へ正規化**した。kernel、tile、資源、隠蔽率が異なるので因果的な速度予測ではなく、実装優先度の参考模型である。下端は0、上端はこの経験係数の大きい側。悪化する可能性も残る。clockを合成した換算や公式 score 予測はしない。

| 順位 | 候補 | 消える／変わる命令 | 参考短縮幅 µs/呼出し | build 要否 | kernel patch の規模 |
|---|---|---|---:|---|---|
| 1 | dense 初回 promotion へ +0 を吸収 | QKV68/O67の入口初期化を除去、定常1534→1533。code全体は増加 | QKV＋O **0〜4.9**、24層で0〜0.118 ms | 新規 compile 必須 | 約40〜70行＋対象限定・検証 |
| 2 | Q64/KV4/stride18432 の writer 特殊化 | 全体472→464、通常 Q−4/K−9/V−11 | **0〜1.7**、24層で0〜0.041 ms | JIT再compile 必須 | 約15〜30行＋host guard |
| 3 | onepass の amax/448 の定数補正を除去 | loop656→627、通常636→619、whole4128→4104 | **0〜1.4**、24層で0〜0.034 ms | Triton再compile 必須 | 約15〜25行＋証明・数値検査 |
| — | FA3 scheduler barrier ID の外出し | P3 step381→383、whole4480→4488 | **正の見込みなし** | 実装継続しない | CPU probe 約8行、棄却 |
| — | MegaMoE/その他 | byte保持と正の削減を両立する追加案なし | 算定できる正の利得なし | 変更しない | 0 |

今回の文書には build は不要。将来3案を実装する場合の新規 build/compile は必要だが、既存 `stage-qwen-overlays.sh` の runtime mount だけで配れる header/Python/JIT の範囲なら image rebuild は不要である。host extension/AOT、コンテナ内容や依存を変えるなら fresh image が必要。FA3 の source/AOT 変更には fresh image が必要だが、今回の棄却案では行わない。

3案の参考上端を足しても約0.19 ms/24層である。同時の利得を証明した値ではなく、3 ms/周期を支持しない。大きい roofline gap をこの表で説明できたとも扱わない。

### 1．dense 初回 promotion の設計

対象 source は `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh`、現 overlay 後の259行の `final_accum`、285〜286行の K loop、345〜349行の promotion。QKV/O の上記2 template、Normal、BM256/BN128/BK128、2 math WG、3 stage、BF16出力だけに constexpr guard を付ける利用側 overlay を追加する設計とする。

最初の **8 K128** を `chunk<first=true>(0)` に展開し、その後を `chunk<false>(8...)` とする。chunk境界、各Kで2 M wave、stage/phaseの更新、scale読取とempty到着を維持する。firstのu=0だけ、promotionを **`fma.rn(scale, accum, +0)`** として最初からdestinationへ書く。残りは従来の `fma.rn(scale, accum, final)`。単なる積へ置き換える設計にはせず、+0を加える丸め・signed zeroの意味を保持する。sourceのゼロ宣言は保ち、compilerによる不要化に任せる。

scratch probeでは初回の128個のdestinationを追い、QKVのCS2R60＋MOV8、OのCS2R61＋MOV6が0になった。初回FFMAはRZを第3入力にし、全K積・promotionの順序を保つ。定常chunkは双方1534→1533、launch168 registers、spill0、C75xx0。全体はQKV2376→3888 slot、38016→62208 byte、O2336→3840 slot、37376→61440 byteに増えた。**codeが約64%増えるリスクは利得に対して大きい**。

動的削減は入口だけで `68×8×2304 + 67×8×1024 = 1802240 warp命令/層`。chunkの−1が残りQKV3/O7回にも効く分は112640。本数比で入口2.93〜4.58 µs、定常0.18〜0.29 µs、合計参考上端4.87 µs。入口ゼロはload/最初のWGMMAと重なり得る。最初のFFMAの入力依存は消すが、32/64回の promotion本体やTensor待ちは消さず、**kernel全体のcritical pathの短縮は未証明**。

有効FLOPはQKV603073806336、O536065605632/層で不変。論理一回通過I/Oも218734080/166482944 byte/層で不変、実HBM/NVLink byteは未測定。旧dense staggerやr28〜30のtail分離とは異なり、geometry・launch・paddingを変えない。実装前には他templateのSASS不変、全stage境界、byte一致、同一residentのQKV/O両順計時が必要。CPU probeは配備用patchではない。

### 2．Q/K norm・RoPE writer の固定 shape 設計

対象は overlay 後の `python/sglang/kernels/jit/csrc/elementwise/rope.cuh`、615行の `qknorm_rope_prefill_fp32_kernel`、618〜624行のhead/stride、1129〜1132行のfp32 dispatch。template引数 `FixedQwenShape=false` を追加し、trueのときQ64/KV4と入力3本のbyte stride18432をconstexprにする。hostではこの5条件とD128・BF16をすべて確認して分岐する。pointerそのもの、Kのpre-offset、token数、positions/cache_locのindex型は固定しない。対象外は既存実体へ送る。

int32/int64両probeで全体472→464、48 registers・spill0を維持。int64の通常Q258→254、K297→288、V164→153。int32は256→252、295→286、164→153。Q/KのFFMA33・FMUL50、reduction、RoPE、SATFINITE/NOSAT repair、Q/contiguous/pagedのstore本数は不変。arithmeticsを粗くする案でも、head4やpaired-round、V writer/prepack融合の再提出でもない。

grid250×72のissue差は `250×8×(64×4+4×9+4×11)=672000 warp命令/call`。末尾masked行も含むissue模型で参考1.09〜1.71 µs。runtime head/strideのload・選択・wide address chainの一部を削るため、先頭LDGと後段STGの依存には位置する。ただしLDG latencyや独立演算で隠れる部分は未確定。**定常K loopは存在せず、各CTAの通常経路の小削減**である。

FP操作・データ型・byte数を変えず、guard成立時のaddress式は同値。主要論理QKV read＋Q/contiguous/paged writeは約234.032 MB/call、trig/weight/metadataと別kernelのV prepackは別。scalar FLOPの厳密再集計と実HBM/NVLink byteは未確定で、0とはしない。GPUの全byteは未確認なので、実装時はQ/K、paged KV、contiguous V、pack後native V、attention/scale/Oまで現P3/Pとの一致を要求する。単独writerだけの短縮でserving採用を決めない。

### 3．onepass の定数除算の設計

対象は `python/sglang/kernels/ops/moe/prefill_moe_input.py` の170行 `_onepass_moe_input`、188〜194行のamax、194行の `tl.div_rn(absmax,448.0)`。**gain側の448/amaxは変更しない**。amaxは既存のmax/maximumでNaNを数値側へ落とし、`1e-10以上の正規数または+Inf`になる。この限定された入力について、scaleを次の同じnormal-pathの3 FMAで作り、+Infだけ+Infを選ぶ小helperを設計した。

```text
r = FP32 bit pattern 0x3b124925
q0 = fma.rn(amax, r, +0)
e  = fma.rn(q0, -448, amax)
q  = fma.rn(r, e, q0)
scale = (amax == +Inf) ? +Inf : q
```

現SASSの `0x2990/0x29a0` 等で毎group行う、定数448と定数reciprocalの2段のrefinementは、最終的に元と同じ `0x3b124925` を返す。ここを既知の定数にしてgeneric divideのrange check/fallbackも外す。rstd前後のscale、quant、BF16 gate operand、HGMMA、router、pad slotの演算順・同期は保持する。近似reciprocalを1回掛けるだけの変更ではない。

CPUのhardware FMA＋IEEE除算との全数比較で、FP32 `1e-10..max finite` の **1352931585 patternすべてがbyte一致**、refinement前後のreciprocalも一致。+Infは明示選択する。任意の負数・ゼロ・NaNへこのhelperを汎用化しない。GPUのFTZ/NaN、compilerの下流変換、融合全出力の一致をCPU証拠で代替しない。

aligned Triton probeは同じ132 registers、dynamic shared172544 byte、LOCAL0、loop656→627。通常pathは636→619で **−17/group**、静的−29の全量が毎回消えるわけではない。全体4128→4104、code66048→65664 byte。gain側2個のgeneric除算は残り、HGMMA8/group、BAR7/group、LDS/LDGSTS51/groupは変わらない。125 active CTA×8 warp×32 groupで544000 warp命令/層、参考0.89〜1.38 µs。

scale作成→shared一時値→scale store/barrierの依存にあり、定数refinementの直列段も減るが、gate MMAや次groupのprefetchに隠れる程度は不明。gate有効FLOP8.376025088 GFLOP/層とnorm/routerの有効仕事・payloadは不変、実HBM/NVLink byteは未測定。r7のshared reciprocalの耐久失敗とは、共有scaleのownershipや読み書きの順序を一切変えず、同じRN除算結果を保つ点が異なる。r27 square-FMAのようなreduction丸めの変更でもない。byte検証とchanged-state耐久を省略する根拠にはしない。

## 見送りと採否の限界

FA3 `mainloop_fwd_sm90_tma_gmma_ws.hpp` の1212/1219行は毎step `threadIdx.x` からscheduler barrier IDを計算する。mma_pipeの1793行付近の既存WG indexを使い、sync/arrive IDを外へ出すprobeを作った。必要なBARの削除や順序変更はしない。しかしP3の定常 **381→383**、全体4480→4488、spill4/4維持。S2Rは残り、addressの削減を別の整数処理が打ち消した。正の削減条件を満たさず、期待短縮は算定しない。これはVのload前倒しやslot数の再掃引ではなく、今回新たに棄却した案である。

MegaMoEのscaleはKごとに異なり、FMUL128/blockは恒等でない。既存compilerが合成scaleを再利用している。scale先読み/持越し、inline PTX集約、LDSM、independent WG、FC2分割、tail-pair、direct FC2、A gather、polling backoff、combine融合、padding-onlyゼロ削除は既存の否定証拠を維持する。leanで残ったrelease/acquire、FC1 arrival、FC2 sharedのwrite/read境界、grid/self-signalを「余分な同期」とする新証拠もない。poll反復やfenceの待ちを静的slot数から推定しない。

norm_quantではgain、正規化、BF16丸め、group128 max、FP8変換が残る。`minimum(q,448)`をSATFINITEに任せるだけではNaNの扱いが変わり得るためbyte保持案に数えない。r27 square-FMA、r28 residual pair store、combine+norm融合の再試験もしない。register移動は用途・live rangeを追い、単にMOV/PRMTが多いことを削除可能の根拠にしなかった。

GPUを追加する判断を変える不確実性は、上位3件の中間・最終byte一致、追加code/lifetimeの影響、通常kernelで露出する時間である。今回は解いていない。将来進める場合はsource/条件ごとにcommitを分け、既存Actions→buildまたはruntime overlay→scheduler→同resident gate→ready前準備→通常公式parser→成果物回収の経路を維持する。別runやinstrumented時間を組み合わせて採用しない。

## 検証・証拠

scratchは `codex-scratch/r36/`。compile command、source、cubin、encoded SASS、opcode/PC範囲、path集計、全数除算比較、53周期の集計を `r36-cpu-proof-20261104.tar.gz`（4407237 byte、SHA256 `650178e521c1fc5f23b290e5590f49a1053aae6640fc8452a289e0e9052894e8`）へまとめた。archive 内18本のSASS encodingを全照合し、gzip検査も通過した。補助scriptは追跡対象へ追加していない。文書だけの変更なのでpytestは行わない。指定のPATH設定後のRuff、`git diff --check`、集計の列和・hash・資源検査を完了した。compilerとCPU診断はすべてterminalで、作成したGPU/Actions/scheduler jobはない。展開した旧proof/source/SQLiteと一時診断は整理し、再現用archiveと説明・logだけをscratchへ残した。
