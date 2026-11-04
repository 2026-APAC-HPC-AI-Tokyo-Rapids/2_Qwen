# Round 42：MegaMoE scale software pipeline の依存調査

**NO-GO。phase 1 で停止する。精度を保って 30 µs/層以上を回収する設計を支持できない。** r41 の実測上限 77.160954 µs/層は有効だが、そのうち先読みだけで回収可能な時間は未同定である。短縮見積もりを 0 と測定した、あるいは 30 µs が物理的に不可能と証明した、という判定ではない。候補 overlay、host flag、serving gate、採用 env は実装しない。

## 入力と再現範囲

親 `ede08bd`、指定 branch `perf/pp4-r40-megamoe-mainloop-20261104`。AGENTS.md、r40/r41 の全文と r41 結果節、tracker/model と元 Prefill profile、下記の棄却 note、関連 commit 本文、保存 r40/r41 Codex prompt/final と会話記録、保存 Actions/Slurm 結果を確認した。元 model/profile は旧 EP4 の記録であり、今回の EP1 時間へ混ぜない。

対象は EP1、T7988=2×3994、E128/top8/H4096/I1536、FP8 E4M3 operand、K128 block scale、FP32 accumulator、FC2 BF16 partial。FC1 first-K、2 math WG、M128/N256/K128、4 stage、lean、split accumulator、132 CTA、384 thread を維持する。image は `2fb619c4a293a01b98ecd178d92414ab5cf493e2584acd945a1a8910ba3b4467`、SGLang `fc275674a97d78553b22597ee72c8fd48f5420f4`、DeepGEMM `1c8138a59b3df5dac50b76b730834ed57982fdfb`。

raw 正本は `/srv/qwen-artifacts/41875058470/1/results/results/megamoe-rescale-probe/`。12 rank の JSON と production JIT source/cubin を読み、production source と ELF の kernel text が全 rank で一致することを確認した。ELF 全体の hash は rank 間で異なるので、ELF metadata を命令差と解釈していない。

r40 proof の source/依存 header を一時 directory に展開し、記録済み build/runtime patch 全本の現在の hash と順序を照合した。その production header に r41 overlay を実適用した最終 SHA256 は `1821ba235eddd2c1162a05418e282d22efd9095e4045731291a1e0c7699d81f3`。12 rank の runtime header hash と一致する。source mirror は変更していない。

| 証拠 | run の production cubin | 今回 fresh production compile |
|---|---|---|
| compiler | CUDA 13.0.88、cubin の tkinfo で確認 | `/usr/local/cuda-13.4`、13.4.92、sm_90a、O3/C++20 |
| encoded text SHA256 | `92b487c8ca8e22d5268f42771c851f6535afc0f9ac9719b411cc0ad1b11810a3` | `f157a6dddb926f5944f751e46448afb69091eb9cfbcccd17e21fe852205e8c41` |
| encoded byte | 178560 | 181120、既存 r40/r41 proof と一致 |
| launch register / 最大参照 R | 168 / R205 | 168 / R206 |
| stack / local byte、LDL/STL | 0 / 0、なし | 0 / 0、なし。spill store/load も 0/0 |

逆アセンブラは両方ともローカル CUDA 13.4 の nvdisasm。以下では **実 run の PC** を主に用い、13.4 の PC を混同しない。全 unrolled K-block・各 half の wait、FMUL 範囲、ratio の reaching definition、WGMMA、stage release を隣接 proof に収録した。FC1 peeled 8 block、反復 chunk 8 block、FC2 12 block の計28静的 block を調べた。FC1 の反復 chunk は動的には K8–15、16–23、24–31 に対応する。chunk の backedge は別に扱う。

## 定常 K-block の正確な依存

`S[r,c,k] = fl(sa[r,k] * sb[c,k])`、`rho[r,c,k] = fl(S[r,c,k-1] * rcp(S[r,c,k]))` と書く。production は各要素を `C <- fl(C*rho)` とし、4本の K32 WGMMA でその K128 の積を加える。最後に最終 S を掛ける。FMUL64本は独立な accumulator 更新であり、64段の依存鎖ではない。ただし m64n128 WGMMA は64個すべての更新結果を必要とする。

各 WG の half0 は R88–R151、half1 は R24–R87。FC1 は両 half とも gate/up の8列 interleave を含み、4 factor を共有する。FC2 は half ごとに別の N128 weight scale を持ち、2行×2 half で4 factor が必要になる。

### FC1：反復 chunk の u=4、動的 K12/20/28

| 処理 | 実 run PC と operand / 順序 |
|---|---|
| この K の weight scale LDG | `a160: R203 <- sb_gate[k]`、`a170: R205 <- sb_up[k]`。前 K の half0 rescale 中に発行済み |
| 前 K の half0 / half1 最後の WGMMA | `a710` / `ac50` |
| stage k の full wait、SFA load | `ac60 TRYWAIT` → cold 分岐 → `ac80 LDS R193`、`ac90 LDS R194` |
| 前 K half0 の完了保証 | `acb0 DEPBAR.LE 1` |
| denominator 4個 | `ad30 R195=R203*R193`、`ad40 R193=R205*R193`、`ad50 R192=R205*R194`、`ad60 R194=R203*R194` |
| reciprocal 4個 | `ad70 R202=rcp(R195)`、`add0 R204=rcp(R193)`、`ade0 R205=rcp(R192)`、`ae30 R203=rcp(R194)` |
| 前 S と掛けて ratio | `adf0 R199`、`ae40 R197`、`aec0 R196`、`af40 R198`。前 S は `a180..a1b0` の denominator を再利用 |
| half0 accumulator FMUL64 | `ae00..b230`。最後の ratio の生成とも interleave |
| fence / half0 WGMMA4 | `b240 ARRIVE` → `b250,b2a0,b310,b360`。最後に group commit を表す `gsb0` |
| 前 K half1 の完了保証 | `b390 DEPBAR.LE 1` |
| half1 accumulator FMUL64 | `b3a0..b790`。half0 と同じ4 ratio を使用 |
| 前 stage k−1 の empty arrive | `b7a0`、lane0/warp 条件付き |
| fence / half1 WGMMA4 | `b7d0 ARRIVE` → `b7e0,b830,b880,b8d0` |
| 次 K の weight scale LDG | `ada0` と `b300` の offset `0x14` は **k+1 用**。上記 denominator の入力ではない |

13.4 の同じ u=4 は full wait `ad00`、LDS `ad20/ad60`、half0 wait `ad70`、RCP `ae50/ae60/ae70/aec0`、FMUL `ae90..b2c0`、half0 WGMMA `b2e0..b370`、half1 wait `b380`、FMUL `b390..b780`、release `b790`、half1 WGMMA `b7b0..b8a0`。同じ依存構造だが PC と descriptor 発行列は異なる。

### FC2：K=4

| 処理 | 実 run PC と operand / 順序 |
|---|---|
| half0 weight scale | `14bd0 LDG R190`、前 K の half1 WGMMA 発行中 |
| 前 K half0 / half1 最後の WGMMA | `14760` / `14cc0` |
| stage k full wait / SFA | `14cd0` → `14cf0 LDS R195`、`14d00 LDS R193` |
| half0 wait / denominator | `14d20 DEPBAR.LE 1` → `14d90 R191=R190*R195`、`14da0 R190=R190*R193` |
| half0 RCP / ratio | `14db0 R194`、`14dc0 R199` → `14dd0 R189`、`14df0 R188` |
| half0 FMUL64 / fence / WGMMA4 | `14de0..151e0` → `15200 ARRIVE` → `15210,15280,152e0,15390` |
| half1 weight scale | `14d80 LDG R192`、half0 rescale 前に発行 |
| half1 denominator / RCP / ratio | `152f0/15300` → `15310 R194/15330 R192` → `15370 R187/15380 R185`。half0 の3本目と4本目の WGMMA の間 |
| half1 wait / FMUL64 | `153b0 DEPBAR.LE 1` → `153c0..157b0` |
| 前 stage release / fence / WGMMA4 | `157c0` → `157e0 ARRIVE` → `15810,15890,158e0,15930` |
| 次 K 用 weight scale | half0 は `15840`、half1 は次 block の `15a20`。この K の scale と取り違えない |

13.4 の対応は full wait `14bf0`、LDS `14c10/14c20`、half0 wait `14c30`、RCP `14cb0/14cc0`、FMUL `14ce0..150e0`、WGMMA `15100..15230`、half1 RCP `151a0/151b0`、wait `15240`、FMUL `15250..15640`、release `15650`、WGMMA `15670..15790`。

### TMA と chunk 境界

full barrier は A/SFA producer と B producer の2 arrival、および期待転送 byte の完了を待つ。empty barrier は8 math warp の arrival を待つ。実 run の代表 B producer は `1e980 empty wait -> 1eab0 UTMALDG -> 1eac0 arrive/expect 0x8000 byte`。A/SFA producer は `21670 A TMA -> 21680 SFA TMA -> 21690 arrive/expect 0x4200 byte`。stage0 の full は `+0x37c10`、empty は `+0x37c30`、stage3 はそれぞれ `+0x37c28/+0x37c48`。SFA だけ先に読める独立した完了通知は現行にない。

前 stage の release は両旧 half の完了保証後。例えば FC1 `b390 -> b3a0..b790 -> b7a0`。release を見た producer が同じ ring slot の次世代 TMA を始められる。現在 stage を現在 half0 の完了だけで解放することはできない。

FC1 K0 は first-K peel で rescale がない。K8 の入口では `7890 full wait -> 78b0/78d0 LDS -> 78e0 wait_group 0 -> 78f0 release -> 7900..79b0 ratio`。half0 FMUL は `79d0..7df0`。**この入口だけは half1 FMUL `7ef0..8370` が half0 の1本目 `7eb0` と2本目 `83f0` の間へ既に移動している。** half1 用の `8480 wait_group 1` より前だが、旧 half1 は `78e0 wait_group 0` で完了済みなので合法である。

K16/24 の入口は同じ PC に backedge するが、ratio は chunk 末の `de00 full wait -> deb0/dec0 LDG -> ded0/dee0 LDS -> def0..df20 denominator -> df30..df60 RCP -> df70..dfa0 ratio` で作られ、`dfc0 wait_group 0 -> dfd0 release -> dff0 branch -> 79c0` となる。`7900..79b0` は再実行しない。この経路では **ratio が drain より前に既に配置されている**。最後 K31 では不要な次 scale prefetch を飛ばして drain する。FC2 最後は `1ae80 WGMMA -> 1ae90 wait_group 0 -> 1aea0 release`。chunk の最初・最後を定常 interior と同じ露出に数えない。

## critical path と隠蔽の判定

確定するのは partial order である。interior の half0 は「旧 half0 完了と stage full と ratio ready の join → FMUL64 → fence → 次 half0 WGMMA」、half1 は「旧 half1 完了と ratio ready の join → FMUL64 → fence → 次 half1 WGMMA」となる。scale 系の denominator/RCP/ratio には accumulator 完了からのデータ依存はないが、SASS の命令順で wait 後に残る部分がある。

一方、half0 の rescale は旧 half1 の WGMMA と、half1 の rescale は現在 half0 の WGMMA と重なる機会を **既に** 持つ。FC2 half1 の RCP も現在 half0 の実行中にある。FC1 の4 ratio は半分ごとに再計算していない。chunk 入口の合法な half1 FMUL 前倒しと backedge の ratio 前倒しも既存である。

この「重なる機会」は、全 FMUL/RCP latency が実時間で完全に隠れたという証明ではない。同じ SM の2 WG が Tensor Core と scalar scheduler を共有し、stage full 待ち・WGMMA issue の間隔次第で overlap の末尾が露出する。r41 には per-PC stall/group 完了時刻がないため、**77.16 µs のうち、どの load/RCP/FMUL が何 µs 露出したかの数値分解はできない**。静的 PC 差や命令本数比をその分解へ変換しない。

同じ half の accumulator を、その旧 WGMMA の完了を保証する wait より前に読み書きすることはできない。[PTX ISA の wait_group 規定](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group) と一致する。wait_group 1 は最新1 group 以外の完了を保証する。chunk 入口のように先行 wait_group 0 がある例外と、定常部を区別する。

## 検討した設計と 30 µs を支持できない理由

| 設計 | 実現できること、残る依存と費用 | phase 1 判定 |
|---|---|---|
| weight scale のさらに早い LDG | immutable weight は先読み可能。ただし production で既に前 K 中に発行され、FC1 の例では使用まで他 half の WGMMA と64 FMULが挟まる。load 命令の新規削減はない | 新しい短縮量を定められない |
| SFA/RCP の1–2 block lookahead を register に保持 | accumulator を待つ必要はなく、4 ratio/追加 lookahead を置く余地はある。SFA は該当 full barrier 完了が必要。4 stage は全 future stage の ready を保証せず、先の full wait を現在 half1 の issue より前へ移すと逆に発行を止める。ring 再利用後の SFA 上書き前に register へ読む必要もある | 可能性は否定しないが、両 half の周期を縮める根拠なし |
| FMUL を他 half の WGMMA 中へ移す | 定常部は既にその順序。同じ half の完了前へは移せない。chunk 入口で安全な移動も既存 compiler が実施 | 新しい64本の隠蔽窓を作れていない |
| combined S/rho を K ごとに1回、両 half で共有 | FC1 は既に4 factor を共有し、前 denominator を次 numerator として再利用。FC2 は half 間で weight group が違うので2 factorへ統合不能 | 既存 consumer review と同じ重複除去の失敗 |
| row ratio×column ratio に因数分解 | 2 row reciprocal＋2 column reciprocalで RCP は4本のまま。4個の部分 ratio と4個の合成 factorで scale FMUL も8本のまま。accumulator FMUL128本は残り、lifetime と FP32 の丸め/overflow/underflow条件は変わる | 命令数・完了依存の削減根拠なし。近似 reciprocal 化は提案しない |
| 独立 product＋sum の FFMA、狭 N への retiling | FP順序変更として検討対象にはできる。現 N256 で両方を完全保持すると accumulator だけで256 register。halfごとの一時product64個でも現在の128に追加し、補助 register の配置を新たに解く必要がある。N128なら容量は緩むが N item と A TMA/scale/epilogue/publish が増える | zero-spill と正味30 µsの両方を支持できず、実装には進まない |

r13-M は scale(k+1) を half1 発行前へ移す late/early を既に compile している。zero-spill でも、FC1 half0 の post-wait span 107→65 に対し half1 の WGMMA 発行 span は13→42/30命令へ増え、chunk全体も減らなかった。full wait を移動先へ運ぶだけでは、この失敗を解決しない。今回の実 run SASS も同じ阻害条件を持つ。

register の **168 は launch metadata**、math role の予算は **224/thread**。`64×48 + 64×40 + 256×224 = 62976`、source の総枠は64512。8刻みの次の232なら65024となり現 source 契約を超える。物理65536との区別が必要で、224が GPU の絶対上限という意味ではない。R206までの使用に対する17個の番号上の余地は、17個すべてが全 PC で free という liveness 証明ではない。少数 scale の lookahead は register 数だけでは棄却しないが、追加64/128 accumulator は同じ判断にできない。

### 過去の失敗との対応

| 過去案 | 維持する証拠 |
|---|---|
| FFMA promote | 旧4WG/96枠で1856/1768 B spill、C7512。現在2WG/224とは別構成なのでこれだけで新設計不可能とはしない。上記容量と追加仕事を別に評価した。`2026-10-18-megamoe-prefill-mainloop.md` |
| shared-RCP | 約0.3%速くても全GPU・全4routingの600回目が自身の初回と不一致。単発誤差合格を耐久合格へ昇格しない。`2026-10-30-pp4-round7-timeout.md` |
| split-LDSM / chunk stagger | mainloop 模型の利得は完全kernelで+1.6%へ反転。stagger +2.3–4.3%、FC2分割+2.9%。H100 模型を今回のsavingへ代入しない。`2026-10-30-prefill-megamoe-split-ldsm.md` |
| tail pair / swap tail | lean-pair は約+10%で不採用。swap-r26 は MMA約3.8%削減でも historical 約±0.6%、issue/wait増。`2026-10-30-pp4-round3-9668.md`、`2026-11-03-prefill-megamoe-swap-tail-r26.md` |
| direct FC2 / A gather | 完全kernel +1.6% / +44%。除去した転送と新しいcritical path/要求数を両方数える。`2026-10-30-prefill-megamoe-ep1-accounting.md` |
| sched-r20 / FC1 assignment order | 全12GPU・全routingで+38–166 µs/層。割当計算とB再利用の変化が利得を上回った。`2026-11-01-prefill-megamoe-sched-r20.md` |
| poll backoff | historical +1.10–2.26%。到着検知の遅れ。`2026-11-03-prefill-backoff-normfma-r27.md` |
| scale前倒し / 合成scale持越し / descriptor集約 | r13は依存周期短縮未証明、consumer reviewは既存再利用または60/80 B spill・188→204命令。`2026-11-01-prefill-megamoe-dep-r13.md`、`2026-11-02-prefill-megamoe-consumer-review.md` |

## 定量判断と仕事量

r41 の対応測定は全12GPUで complete、中央値77.160954 µs/層、GPU別65.970063–82.997799 µs。これは scale load、108 RCP、3552 FMULとその依存等をまとめて除いた **全scale経路上限**。消えたFMULにはscale算術も含む。FMULだけ、RCPだけ、stage waitだけの上限分配ではない。production 約2.70 ms、通常 serving 公式10072.13と、この合成入力のkernel差を区別する。

30 µsは上限の38.8798%で、要求の「約40%」に当たる。同じ probe の historical routing に対する定常292358 CTA-K128相当と、各GPUの同窓平均 clock を使う条件付き逆算では、30 µsに **平均17.03–19.40 clock/CTA-K128の正味露出削減**が必要になる。これは必要量であり、SASSの局所spanから得た短縮予測ではない。追加のwait・issue遅延・register費用を差し引いたこの規模の正の削減を支持する設計は得られなかった。

採用候補の推定短縮量・予測誤差は **未同定**。0–77.160954 µsという全経路の会計枠は confidence interval でも attainable range でもない。RCP4本やFMUL64本の静的spanへ固定IPCを掛けて30 µsを作らない。過去runのclock・計装timelineを合成せず、WGMMAと重なる仕事も二重に足さない。

有効GEMM仕事は FC1 1.608196816896、FC2 0.804098408448、計2.412295225344 TFLOP/層。検討した先読み/因数分解はこの仕事、元の128単位scale、モデル精度を変更しない。主要論理I/Oは既存r40/r41のFC1 47.571849216、FC2 34.325483520 GB/24層。新しいHBM/L2/NVLink実byte、cache挙動、phase時間は未測定で0にしない。EP1 expert dispatch/combineにrank間通信は不要だがPP/PD転送は別である。

## 停止・検証・後片付け

phase 1 の停止規則により、phase 2 の候補compile、full synthetic tensor数値比較、500回耐久、12GPU正逆gate、採用・prewarm/dispatch配線は実施しない。r41診断の廃止も「gateに置き換える」phase 2 に属するため今回変更しない。production、host/device patch、serving設定、sbatch、image pin、submissionは変更なし。この研究記録のsubmissionに **require-build は不要**。

指定の全 pytest と Ruff は `export PATH=$HOME/.local/bin:$PATH` 後に実行し、pytest起動前に継承SIGHUPをSIG_DFLへ戻した。**指定の全 pytest は100%完走して終了0、Ruffも終了0（All checks passed）。** 最終結果は隣接 CPU-validation JSON と prompt directory の `pytest.log`、`ruff.log` を正本とする。phase 1 は文書・証拠のみなので新しい実装用CPU testは追加しない。履歴非依存の既存全suiteを実行する。

proofは `megamoe-scale-pipeline-r42-cpu-proof-20261105.tar.gz`。新しいscalar依存抽出、全block表、実run12rankのidentity、fresh compile log/command、代表SASS excerptを保持し、既存のraw cubin/sourceを不必要に複製しない。展開source/build directoryは削除する。SSH、GPU実行、Actions/Slurm/PJM操作、image build、commit/pushは一切行っておらず、この作業によるremote残留jobはない。
