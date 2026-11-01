# Round 14：旧 writer＋V prepack と sum 単独の選別

2026-11-01。基点 `c01b1f5`、branch `perf/pp4-r14-fa3vp-20261101`。採用候補は P と S だけ。GPU、SSH、Actions 操作、scheduler 投入、image build は行っていない。`prefill.fa3_prepack_gate=false`。round 12 の pin と通常 serving 設定は保持する。CPU 正本は `fa3-prefill-prepack-cpu-validation-20261101.json`。

## 判断の根拠

round 12 は公式9958.43、同じ構成の round 13 / run41253426202 は9947.50。round 13 の native writer＋native FA3＋sum は全12 GPUで writer＋FA3＋O が46〜52µs遅く、最初の3994+3994で不採用だった。残り6pair、single、耐久の合格証拠はない。この合成差から writer と consumer と sum の実時間を割り振らない。

設計は `scratchpad/codex-scratch/astra2/report.md` §3.2/3.3/3.6/5 に従う。旧 native writer は2本の64-bit V storeを16本のbyte storeにし、warp命令境界のsector要求が約8倍となった。一方 native FA3 は転置を除くと同時に、VのloadをK acquireの後ろへ移す。今回このconsumer scheduleを変更しない。Pでwriter scatterを回避してもconsumer自体が負ける可能性を分解計時で調べる。

`AGENTS.md`、指定のresearchとtheory model/profileの位置づけ、git log80件、e3e7744・3c8dfec・round12/13の本文、保存済みgate raw JSON、astraのderived/analyze/count_paths、前回Codex会話ログを確認した。e3e7744で失敗したthread-local V mapをoracleとして再利用しない。round7のgate chain timeoutを踏まえ、追加群全体にwall上限を置く。

## 実装とbyteの流れ

- **P**：旧 FP32 writer → contiguous FP8 V → production prepack → native-only FA3 variant41184 →既存O。新規table ID15。pair専用native FA3は現在のvarlen addressで再コンパイルした。旧CPU証拠の中間 `v-n96` binaryは使わない。
- **S**：旧 FP32 writer / contiguous V → sum-only＋identity PRMT lifetime保持 variant24800 →既存O。新規ID16。
- **診断のみV/VS**：現native writer＋41184、現native writer＋57568。3994+3994だけで計時し、採用対象にはしない。既存ID14は診断用として保持する。退役済みN192 native ID13はbuildしない。P＋Sは作らない。

`kernels/fa3-verify/prefill_prepack.cu` は2×4×ceil(maxlen/96) CTA、128 threads。旧Vを16-byte単位でshared contiguous arrayへ読み、逆byte mapでnative shared arrayへ配置し、aligned uint4で連続global storeする。逆mapは配置だけの整数計算で、FP8の変換・近似・項の省略はない。要求末尾paddingは全byteゼロを書き、paged K/Vには触れない。

packのoracleはpack自身にしない。旧writerのcontiguous Vから、独立したforward mapで全byteの期待imageを作る。さらに既存 `prefill_native_image.cu` の実LDSM/PRMT/STSM dump、現native writerのimageを照合する。CPUでは旧producerのinstruction-bit-layout simulatorをN96へ具体化し、両stageの12,288座標を再生成した。旧writer mapのhashと一致し、production逆mapと16-byte gather式の全座標を検査した。GPUの3経路照合はgateに実装したが未実行。

`patches/sglang/prefill-fa3-vsum.patch` のdiagnostic writerは保持し、serving配線をP/Sへ置換した。hostの長さがちょうど2個、各3900〜4096、かつ既存fixed-path条件を満たす場合だけPまたはSを選ぶ。Pでは旧writerを1回だけ呼びpackを追加する。最初のpairでもopが登録されるよう、pack前に既存companionの `load()` を呼ぶ。それ以外はround12と同じN192行と旧writer。singleや範囲外でnative imageを確保しない。wrapper、build入力、provision、PP4設定の来歴を更新した。既存generic selectorからSも除き、専用gateだけが採用する。

## 仕事量：3994＋3994、Q64/KV4/D128

FP8 E4M3、FP32 accumulate、全causal項。QK＋PVの有効FLOPは522,847,416,320、tile padding込み535,881,056,256で、P/Sとも削減しない。Oも同じM7988/N4096/K8192、同じweight/scale、`compiled_dims="nk"`。

| 項目 | Pの変化 |
|---|---:|
| FA3 transposeのKV visit | 56,784回を除去 |
| FA3 logical shared read＋write | −1,395,523,584 B |
| pack unique tile | 336 CTA |
| pack global read | ＋4,089,856 B |
| pack global write | ＋4,128,768 B |
| pack global read＋write | ＋8,218,624 B |
| pack logical shared traffic | ＋16,515,072 B（staging、gather、native staging、readout） |
| launch | ＋1 |

残るV global tile loadは約697.762 MB、Kも合わせ約1.396 GBのlogical load。transpose削減をHBM削減と呼ばない。実HBM、NVLink、cache再利用、待ち時間は未測定。packの8.219MBを仮に4.8TB/sで割った1.71µsは帯域床の参考で、pack latencyではない。

## CPU/SASS証拠

**CUDA13.4.92、sm_90aのCPUコンパイルによる代替証拠。実imageのCUDA13.0.88のcubinではない。** 新規imageの正しさやH200速度を保証しない。

| 行 | registers | spill store/load | shared bytes |
|---|---:|---:|---:|
| C M192/N96 | 128 | 40/28 B | 124,928 |
| P/V native41184 | 128 | 0/0 B | 100,352 |
| S24800 | 128 | 40/28 B | 124,928 |
| VS57568 | 128 | 0/0 B | 100,352 |
| pack | 35 | 0/0 B | 24,576 |

全対象でC75xxなし。既存FA3 N96/N192のencoded SASSはround13最終controlとbyte一致。旧FP32 writerのi32/i64両方も、旧source、今回のdiagnostic writer追加後、保存済みround13のencodingが一致した。比較は命令wordとscheduling word双方を含む。

packは静的424命令、global storeは6本の `STG.E.128`。1命令の32 lane×16 Bは連続512 Bで、32 B sectorは16個。4warp×6命令＝384 sector要求/CTA、336 CTAで129,024要求、4,128,768 Bのimage全体を覆う。sector数はsource/SASSからの計数で、実DRAM transaction測定ではない。coalescing以外のgather/barrier/address費用も計時に含める。

Sは以前にlifetimeを証明したsum-onlyの**全encoded SASSと一致**し、50 EX2、48 FADD、加算依存23→7、定常369→391命令（＋22）を保つ。全12個の新Pのchecksumに依存するselectorとidentity PRMTで、全12 wordの旧Pの最初の上書きを50 EX2後まで遅らせる。合成版の＋19と混同しない。P lifetime保持がEX2/WGMMAの有利な重なりを保証するとは主張しない。

独立mapのCPU実行、ATen wrapper、23本のbuild patchと31本のruntime overlayの順序適用、実PP4 CPU source checksも確認した。完全なimage build/provision/GPU/公式parserのE2Eは後続servingで確認する。

## 一つのgate群

新scriptは `scripts/select-firmus-prefill-fa3-prepack.py`。round12 pair gate成功後だけ、同じ12 H200で1群を起動。HEADの2workerとTAILの1workerはSlurmの各4GPU割当で並行、各GPU内は逐次。import、common input、weight、output、graphのwarmupを共有・再利用する。

最初の3994+3994で次を測る。CUDA event→graph replay→event、forward/reverse各10 sample、12 call/sample。raw sample、shape、variant、GPU UUID、binary実hash、source hash、image hashをrank fragmentへ保存する。clockは独立samplerを設けずnullと理由を記録し、他runのclockで補わない。

1. writer-only：旧、native、旧＋pack、pack単独。
2. FA3-only：事前生成した固定入力のC/S/V/VS。
3. 合成writer＋FA3＋O：C/S/V/VS/P。
4. 固定attention出力を使うO-only。
5. FA3-onlyと合成の `I=T_VS−T_V−T_S+T_C` を両順・pooledで保存。

V/VSの診断結果を採否には使わない。P/Sの最初の合成測定は完全gateの最初の形状にも再利用し、5µs margin未達ならその候補を直ちに打ち切る。その後Pを完全検証し、残りwallが候補一式の45秒を予約できる場合だけSへ進む。45秒は保守的な設計配分で実測保証ではなく、超過すれば採用しない。

完全gateは全12GPU、7pair `(3994,3994), (3986,4027), (3986,3986), (4027,4027), (4096,4096), (3900,4096), (4096,3901)`。全形状でwriter＋FA3＋O（Pはpack込み）のforward/reverse/pooled中央値がCより各5µs以上短いこと。byte照合を数値比較より先に行い、FP32 causal参照でrelative L2≤0.05、max scaled error≤0.25。参照Qは従来と同じ各requestの9代表位置、その各位置の全causal keyを計算する。元入力と変更入力2回のgraph replayを検証し、controlとの誤差差分も保存する。

single3994/4096/3584は両候補envで旧dispatch・旧writer、Q/K/V/paged KVとround12 attention/scaleのbyte一致を検査し、速度判定しない。実writer→Pならpack→serving attention→Oをeagerで1200回連続実行し、初回のattention/scale/O bytesと一致させる。

数値・速度・byte不足はその候補だけ拒否。CUDA RuntimeError、process異常は非ゼロexitで群全体を拒否し、shellは自分が起動した残りのgate子プロセスだけ終了する。hangは群timeoutで拒否。異常contextで次候補を走らせず、servingは従来どおりfresh process。

集約はsuccessful pair gateとの同一allocation、worker/rank/UUID、12 distinct H200 UUID、run ID、image/binary hash、候補row/variant、全shape/single/endurance、期限を検査。Pが12GPUすべてで合格ならP、そうでなくSだけが全合格ならS。混在合格やP＋Sは採らない。envの既存FA3行がround12 controlであることを確認し、全結果がそろった場合だけ同じfilesystem上のrenameでatomicに置換する。それ以外はround12選択のまま。

## wall上限と17分枠

群開始＋175秒までの残り秒を各srunの `timeout --kill-after=2 N` に渡し、worker起動のずれで群全体の期限が延びないようにする。rank deadlineも群開始＋175秒。集約deadlineは開始＋180秒（N＋5秒）。集約コマンド自体も残りwallでtimeoutし、失敗してもservingを継続する。round12 pair gateの150秒＋kill2秒、155秒deadlineは変更しない。

round13のvsum rank elapsed約28秒をwallの代わりにしない。保存fragmentでは起動から最後のfragmentまで62.017秒以上、pairは57.084秒以上。実行終了・集約を含む旧wallの下限だけを差し引くので以下は保守的置換になる。

| 基準 | 計算 | 予算 | 1020秒まで |
|---|---|---:|---:|
| round13、job741秒 | 741−62.017＋180＋(155−57.084) | 956.899秒（15:56.9） | 63.101秒 |
| round12、job652秒 | 652−80.463＋155＋180 | 906.537秒（15:06.5） | 113.463秒 |

起動35秒、診断20秒、P45秒、S45秒、集約5秒、余裕30秒という設計配分。実測した各段階の所要時間ではない。cold import・filesystem・model stagingの変動は保証できない。warmup60秒・公式測定120秒は削らず、buildはallocation外。round7の長いgate chainを再導入しない。

## 後続serving commit

Claudeはこのbranch上で `configs/firmus-prefill-pp4.json` の `prefill.fa3_prepack_gate` だけをfalse→trueにする。`fa3_pair_gate=true` と全pinを保持し、FA3 envを手動選択しない。native op追加なので、通常serving用commitの先頭にbuild・Firmus・2node・prefill-pp4の既存タグを付けてfresh imageをbuildする（実装commitのskip-ciタグはserving commitに付けない）。commit本文には角括弧を入れない。runtime provisionが今回buildのimage hashへ置換し、12GPU結果からgateがenvを決定する。

この作業ではserving commit、Actions起動、GPU試験は実施していない。後続はterminal状態、gate fragments/selection、実13.0.88 cubin/SASS、通常serving公式parserのeligible/score、artifact回収を確認し、結果を台帳へ追記する。

## 条件付きの見込みと未確認事項

native FA3が単独20〜40µs短くpackが5〜15µsなら、相互作用を無視した感度例は正味5〜35µs/層。予測区間ではない。K acquire後のV load待ち、cache、writer/gather、Oとの相互作用が利得を消す可能性がある。Sは＋22命令とchecksum joinも払い、正の実測利得はない。PとSの利得を加算しない。

全worker周期へ全24層分が転写される容量模型では、10kに必要な短縮はround12基準約19.96µs/層、round13再現基準約25.20µs/層。5µs gate合格だけでは10k条件を満たさない。GPU正しさ、全7pairの速度、実clock、image13.0.88、1200回耐久、通常servingと公式スコアはすべて未確認。同条件repeatなしに単回の性能優位を断定しない。

## 検証と片付け

全体suite：Python3.12 / uv no-project、1374 passed、78 skipped、25 subtests passed、543.14秒。AF_UNIXの3件だけ同じ許可scratchを `/proc/self/cwd` から短く参照する一時fixtureを使用し、production sourceは変更していない。Torch2.11.0を含むP/S・pair・pipe・varlenの契約182件は6.51秒で合格。最新prepack契約113件、Ruff、shell syntax、diff検査も合格。

CPU raw証拠は許可scratchの `r14/cpu-sass-proof.tar.gz` に保存し、SHA256は `1c7100a8961a94ab579c9c8dcadacabfd03d6f3264cafed615ea5b3536c44cc9`。新規sourceコピー、object、生成TU、テスト用directory、socket alias等の不要な一時物を削除し、archiveと最終検証logを残した。元からあったexec.logは保持。SSH、Actionsの起動・停止・dispatch、GPU/scheduler投入、他branchへのpushは行っていない。今回作成した残留GPU/PJMジョブはない。
