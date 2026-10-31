# Round 13-V: native V の一度書きと lifetime を保持した sum-chain

2026-11-01。基点 `31d9b10`、作業 branch `perf/pp4-r13-fa3v-20261101`。この作業はローカル CPU 検証と実装だけで、SSH、Actions、scheduler、GPU 実行、image build は行っていない。`prefill.fa3_vsum_gate=false`。旧 image pin は変更していない。

## 段階1の判定

**native V と4-chain sum は、以下の合成形で CPU admission に合格。** GPU 正しさ、速度、耐久、通常 serving の合格を意味しない。CUDA **13.4.92 / sm_90a** は image の **13.0.88** に対する代替検証にすぎない。

単純な4-chain sum は棄却した。加算依存は短くなるが、PV の最初の operand 上書きが N192 では EX2 の36個後から34個後、N96 では26個後から19個後へ前進する。採用実装は、変換後 P 全体の checksum に依存する runtime identity PRMT を加えた形だけ。全24 word / 12 word の古い P をそれぞれ98個 / 50個の EX2 後まで保持する。これは `e3e7744` の lifetime 手法を pipe mainloop へ適用したもので、負けた tree-max の再試行ではない。

正本は `fa3-prefill-vsum-cpu-validation-20261101.json`。pin と archive 検証、patch 順、入力 hash、正確な compile command、ptxas 全文、encoded SASS hash、加算依存の各 FADD と P 上書き PC を保存した。初期 `v-n*` は transpose 除去の診断で、仮の varlen address を使っているため実行候補ではない。合否と出荷対象は専用 bit を持つ最終 `combined-n*` のみ。

| row | launch registers | spill store/load B | stack B | shared B | 全体の静的命令 | 定常 consumer 命令 | FADD 依存深さ |
|---|---:|---:|---:|---:|---:|---:|---:|
| N192 control | 168 | 84 / 116 | 40 | 166,912 | 5,504 | 587 | 47 |
| N192 composed | 168 | 48 / 36 | 24 | 117,760 | 5,560 | 624 | 13 |
| M192/N96 control | 128 | 40 / 28 | 16 | 124,928 | 4,336 | 369 | 23 |
| M192/N96 composed | 128 | 0 / 0 | 0 | 100,352 | 4,464 | 388 | 7 |

全候補で C75xx なし。加算依存深さは SASS の FADD dataflow の数え上げであり、cycle 実測ではない。N192 の FADD 96、EX2 98、max 100、QK/PV QGMMA 4/6、N96 の FADD 48、EX2 50、max 52、QK/PV 4/3 は control と同数。max と EX2 のアルゴリズム、全 causal 項、FP8 quantization は変えない。

lifetime 保持の費用は隠さない。定常 consumer は N192 で37命令、N96 で19命令増える。全 P word の最後の read/上書きは EX2 後だが、最終 WGMMA wait 自体は一部の EX2 より前に残る。WGMMA 全体を softmax の最後まで隠した、あるいは発行依存が何 cycle 速くなったとは主張しない。

既存 control の **全 encoded instruction/control word** が一致する SHA-256：

- N192：`55084cdb742e96b50168b582fce6205dc08a17202994161898438e92de46eba3`
- M192/N96：`cd7d899e84278e57bffeba2593dbc114b510adf8324b0794131868a86ca5a555`

QK writer は元の kernel 定義・signature を残し、新 kernel を追加した。完全な JIT header/host entry を C++20 で実際に instantiate し、旧 FP32 writer の int32/int64 両方の encoded SASS が before/after 一致。新 writer 4 instantiation はすべて48 registers、stack/spill 0、C75xx なし。FA3 wrapper と独立 GPU dump TU も compile 成功。resource query は非 null resource pointer で CUDA query/launch 前に return する経路のみ。リンク用 scheduler shim は呼ばれれば abort する一時診断で、production には含めない。

## 対象、仕事量と配置

対象は FP8 E4M3、Q64/KV4/D128、GQA16、独立した二つの causal sequence。基準 L3994×2 の有用 QK+PV は **522.847 GFLOP**。M192/N96 の padded work **535.881 GFLOP** は V/sum を加えても不変。N192 single は有用 **261.424 GFLOP**、padded **274.811 GFLOP**。Tensor/EX2 床は重ねられるので足し算しない。

主対象は steady pair の **M192/N96**。M192 を取り消して大きな N192 transpose 削減量を得たことにはしない。single は現行 M128/N192 の native V/sum 行を用意し、同じ厳しい gate を課す。

| L3994 | N192 single | M192/N96 pair |
|---|---:|---:|
| KV visits | 21,840 | 56,784 |
| 訪問ごとの V transpose SMEM read＋write | 49,152 B | 24,576 B |
| 除去する論理 SMEM byte/call | 1,073,479,680 | 1,395,523,584 |
| producer thread/visit の除去命令 | LDSM 12 + PRMT 48 + STSM 12 | LDSM 6 + PRMT 24 + STSM 6 |
| 一度書く native V image | 2,064,384 B | 4,128,768 B |
| 旧連続 V の有効 byte | 2,044,928 B | 4,089,856 B |

N192 **pair** なら旧 transpose は2,146,959,360 Bであり、M192/N96 pair の削減量へ流用できない。表は役割ごとの命令で、consumer と producer の warp 数も異なる。consumer の追加命令と writer の追加 store を差し引いた全体の instruction 削減率とは呼ばない。実 HBM traffic、NVLink contention、電力/clock 応答は未測定。paged KV と PD payload は構造上変更しないが、未知の実転送 byte をゼロで埋めない。

source の FP32 QK post は連続 K/V と paged K/V に別々の store を持つ。新 writer は **連続 V の代わりに** native image を書き、paged cache の slot、byte、page layout を保持する。連続 V tensor は既存 API の shape 用に残るが、native attention はその内容を読まない。control と一般 fallback は旧 writer のまま。

image は `(request, KV head, ceil(maxlen/N), N*128)`。要求ごとに tile を独立に配置し、短い要求の末尾から共通 max block 数までゼロを書く。writer の V 側は `max_blocks*N` の固定 stride で request と local token を割り出す。Q/K は従来の連結 token 順。attention の bulk load は request/head stride と block offset を使う。KV cache を image と共用しない。native の bulk stride を未使用 V/V_new TMA descriptor へ流用せず、descriptor だけは旧 row/head stride 512/128 で生成する。実行用 `params.stride_V` は image stride のまま保持する。実際の host lambda を g++ で動かす回帰を追加した。

native offset は control の CuTe LDSM.T/PRMT/STSM の **cross-lane bit layout** から生成し、N96 の12,288 byte、N192 の24,576 byteと2 pipeline stageを全検査した。N192 の offset4096 は32へ移るため、旧 N224 の32-token局所式を流用してはいけない。全単射であっても GPU 正しさの証明ではない。追加 CPU 回帰は production の式を g++ で実行し、独立 CuTe map の全 byte hash と比較する。

writer の有効 FP8 値はそのまま。従来の連続 V は thread あたり64-bit store 2本、新 image は byte store 16本となる。paged store 2本は従来通り。address 計算、request lookup、padding 書込みと余分な Q/K CTA を払う。L3994 pair の padding は38,912 B、single は19,456 B。この費用を attention 単体の改善から無視しない。

過去の棄却との差：`e3e7744` の旧 thread-local map は27,776/28,672 byteが異なり、cross-lane routingの修復が必要だった。以前のGPU失敗はdump照合より前のFP8 gateで起きていたため、今回はdumpを最初に置く。N224 VByteImage の CPU-only map 判定を再利用せず、N96/N192 と varlen 要求境界を独立に扱う。tree-max、half2/poly exp、lazy rescale、対角 trim、Q double-buffer、early claim、staged Q は追加していない。

## 実装と gate

新 ID は末尾追加のみ：

- `n192-direct-o-varlen-pipe-vsum`：49376、N192、単一要求3584–4096。
- `m192n96-direct-o-varlen-pipe-vsum`：57568、N96、exactly 2要求で各3900–4096。

variant の sum bit は16384、native V bit は32768。既存 ID、M192 bit8192、旧 N224 bit4 の address は保持する。複合 env 値は `n192-vsum+m192n96-pair-vsum`。host 上に既にある length だけで dispatch し、CUDA tensor の値を判断に使わない。他 shape は旧行。旧 selector から新行を除き、この専用 gate だけが採用する。

変更経路：`prefill-vsum.patch` と native table/build、独立 `prefill_native_image.cu` と wrapper op、`prefill-fa3-vsum.patch` の JIT writer/model/dispatch、専用 selector、config、sbatch/provision/prepare の独立ブロック。image context は `kernels/fa3-verify` 全体をコピーし、build は新 patch と TU を明示的に取り込む。native のため **fresh image build が必須**。

round-12 pair gate の後に3 worker×4 GPUを並行で実行し、12 GPU 全員の round-12 成功、同じ UUID/worker/rank、同じ Slurm allocation と新 run ID を集約で照合する。round-12 が不合格なら新 GPU screen を実行せず、env を保持する。再割当等で GPU identity が変わっても採用しない。

対象は7 pair：3994+3994、3986+4027、3986+3986、4027+4027、4096+4096、3900+4096、4096+3901。single：3994、4096、短い fixed-path の3584。3584未満など gate 外の fallback は変更しない。

各 shape の最初、attention を呼ぶ前に native writer を実行し、元 writer の FP8 V から実際の LDSM/PRMT/STSM を使った独立 GPU dump と image 全 byte を比較する。request padding も含む。同時に Q/K と paged KV を旧 writer と byte 比較する。続いて sampled FP32 causal reference（round-12 と同じ行選択）、relative L2 ≤0.05、max scaled error ≤0.25、入力変更 replay 2回を control/candidate とも確認する。

計時は **QK post＋FA3＋実形状 O GEMM**。O は4096×8192、BF16出力、serving の `compiled_dims="nk"` と同じ API。各 graph sample は12 call、正順/逆順を交互に10 round。各 GPU×各 shape で正順・逆順・pooled の全中央値がそれぞれ **5 µs/call以上**短縮していることを要求する。capture の closure、出力、scale、metadata、weight は graph lifetime 中保持する。

全 screen 合格後、3994+3994 の producer＋`prefill_fp8.attention`＋O を1200回連続で実行する。初回は direct native dispatch と byte 比較・FP32確認、終了後に FP32 再確認と自身の初回 FP8 attention出力/FP32 scales/BF16 O projection の byte 一致を要求する。control の異なる sum 順序との bit 一致は要求しない。

欠落、stale/別run、UUID不一致、nonfinite、数値、dump、producer byte、margin、endurance、timeout は拒否。全員合格だけが env の該当1行を atomic rename し、他行を保持する。timeout は job failure にせず、round-12 の選択済み control で通常 serving を継続する。shell fixture で継続を検証した。raw fragment は共有 run directory の gate results 配下へ残し、既存の artifact 回収を使う。

## 時間上限

追加 wall 見積もりは **45–65秒**、未実測。目安は旧 QKV gate の GPU record 36.18–36.24秒（run41166990387）、FA3 が AOT 済みであること、10 shape の graph計時、独立 dump/reference、writer JIT、1200回の経路を加える費用。10 shape×2 order×10 round×12 call×2 modes の kernel call は4800回/GPUで、仮に合成1.3 msなら計時本体6.24秒。これは条件計算であって H200 実測ではない。

srun 全体を **`timeout --kill-after=2 100`**、rank elapsed100秒未満、aggregate deadline開始+105秒。100秒は65秒の1.5倍以上。cold JIT/launch を含み、超過なら採用しない。

17分枠の加算予算：round-10 の9:25 =565秒に pair152秒、新gate102秒、MegaMoE152秒、各集約に3秒ずつを置くと **980秒 =16:20**。round-11 の10:39 =639秒は今回再実行しない QKV gate を含む。同run fragment の最短36.177秒だけを控除した保守的な残り602.823秒に同じ415秒を加えると **1017.823秒 =16:57.823**。差し引かず全部を積むと17分を超えることを明記する。別runの kernel clock/timeline を合成してはいない。これは既存 job wall を使った配分計算で、通常処理の変動や新 JIT compile に対する完走保証ではなく、round-11側の余裕は小さい。MegaMoE 側の上限を150秒より増やさないこと。

## Claude が serving commit で行う手順

1. この branch の実装を MegaMoE と統合し、`configs/firmus-prefill-pp4.json` の **`prefill.fa3_vsum_gate` だけ false→true**。`prefill.fa3_pair_gate=true` は保持する。native overlay と build 入力も保持する。
2. serving commit の先頭を `[require-build] [cluster:firmus] [nodes:2] [prefill-pp4]` とし、fresh image を build する。この作業自身の commit は CI skip のみ。
3. `images.sglang.*` と `container_*` は予測値で変更しない。provision が成功した build の実値で runtime config を更新する既存経路を使う。後日 reuse pin を更新するなら、実際の build run、source、SHA、size、provenance を一式で取得してから行う。
4. env を手で native 行へ固定しない。prepare は旧 control を初期値にし、pair gate と本 gate の成功だけに複合値の publication を任せる。
5. terminal 状態、image CUDA13.0.88 の compile log、`results/prefill-fa3-pair/selection.json` と `results/prefill-fa3-vsum/selection.json`、12 UUID fragment、最終 worker env、公式 parser、raw artifact の回収を確認する。新 gate 不合格でも round-12 選択で serving を続ける。

## 見込みと未確認事項

新しい H200 数値・時間・通常 score は **未取得**。V の shared byte 除去と sum の依存短縮を加算して速度へ換算しない。lifetime 保持で consumer 命令が増え、writer も散在 store と padding を払うため、負の効果もあり得る。remaining 0.8–1.0 ms/24層を得るには、M192上の合成 call で約33–42 µsの追加短縮が必要であり、5 µs gate の合格だけでこの目的や10k達成とはしない。gate拒否なら本機能による利得は0。既存 round-12 の0.43–0.65 ms投影と別に扱う。

## 完了した CPU 検証

- 全体 suite：1307 passed、70 skipped、25 subtests passed、550.21秒。最終の JSON overflow 拒否を含む契約回帰：107 passed、3.15秒。
- CUDA13.4 の最終4行、旧/new QK writer、独立 GPU dump TU、Torch wrapper のコンパイル、ptxas と encoded SASS 比較を完了。
- 保存済み image manifest の baked patch hash 23本を照合。upstream source `fc275674a97d78553b22597ee72c8fd48f5420f4` を持つローカル prepared-image checkpoint を読み取り専用で利用し、runtime overlay 31本の適用、実 PP4 CPU source checks、実 config の prepare を確認した。SquashFS 実体の照合・新 image build と provisioning の E2E 検証を代替するものではない。
- 全体 suite の Unix socket 3テストだけは絶対パスが AF_UNIX の上限を超えるため、同じ許可済み scratch 内を指す `/proc/self/cwd` 相対 alias を一時 fixture に使用した。production と repository のテスト本体は変更していない。

GPU 数値・byte image・速度・1200回耐久、image CUDA13.0.88、通常 serving と公式 parser は未実行。commit/push と片付けの最終状態は完了時の日本語レポートに記載する。CPU evidence はこの note と同じ directory の JSON を正本とする。
