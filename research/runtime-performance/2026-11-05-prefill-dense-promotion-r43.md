# Round 43：Prefill dense の promotion pipeline 依存調査

**NO-GO、phase 1 で停止する。** N64 の二重 temporary は依存上は構成できるが、追加の発行・SMEM 読出し・stage release・drain を差し引いて平均約30 clock/critical-wave K128を短縮する根拠を得られなかった。QKV＋O の正味25 µs/層、望ましい50 µs/層は支持できない。推定短縮量と予測誤差は**未同定**であり、短縮を0と実測した、物理的に不可能と証明したという意味ではない。候補の実装・GPU数値比較・serving gateには進まない。

## 対照、来歴、証拠の限界

指定 branch `perf/pp4-r40-megamoe-mainloop-20261104`、親 `045ca44`。AGENTS.md、astra43 report全文、指定のdense調査・r28 roofline・r36 SASS監査・r11・r30、r37 device/host patchとSGLang呼出し、dense tracker行、theory modelと元profile JSON、r24 summary、r37/r42記録・proof、関連commit本文、保存r37/r42 Codex prompt/final/会話、保存Actions/Slurm結果を確認した。元model/profileは旧EP4、r24は旧image `c7804efb` のprofile runであり、現imageの内部時間とはしない。

対象は `deep_gemm.fp8_gemm_nt((a, sa), (weight, weight_scale_inv), out, compiled_dims="nk")` の `Normal`。QKV `(7988,9216,4096)`、O `(7988,4096,8192)`、BM256/BN128/BK128、2 math WG＋128 TMA threads、3 stage、cluster2/A multicast、132 persistent CTA、E4M3 operand、K128単位のFP32 activation scaleと128×128 weight block scale、FP32 promotion、BF16出力。r37 first-Kは有効。tail分離・QKV epilogue融合は使わない。

最新raw正本は `/srv/qwen-artifacts/41875058470/1/`。image `2fb619c4a293a01b98ecd178d92414ab5cf493e2584acd945a1a8910ba3b4467`、DeepGEMM `1c8138a59b3df5dac50b76b730834ed57982fdfb`、SGLang `fc275674a97d78553b22597ee72c8fd48f5420f4`。保存accountingは `28739|COMPLETED|0:0|00:14:44|2`、公式score `10072.12600396729`、eligible/official validation valid、error 0。これは対照runの結果で、今回の候補の結果ではない。現configのDeepGEMM 16本・SGLang 36本のoverlayを保存sourceと全byte照合した。

r32 proofのdense header `2cc49403…` へ現在のr37 patchを実適用し、r37 production proofおよびr40 production sourceと同じ最終header SHA256 `915bd4d3ee23e0ce6da735cd6b273700caa0d94a15a6917fb6eed93a4ac72350` を得た。r41 overlayはMegaMoE headerだけを変える。source mirrorは変更していない。

**実CUDA13.0のdense cubinは取得できなかった。** 最新rawを列挙した37 cubinに1d2dはなく、主な保存cacheはr41 MegaMoE probeのものである。`ssh firmus` で読み取り専用にrun一覧、残存 `pd-41853888382-1.ownQvtHbCg` と `home/deep-gemm` を調べたが、最新run directory/cacheは残存していない。workerの `SGLANG_DG_CACHE_DIR=${SLURM_SUBMIT_DIR}/home/deep-gemm` という設定を根拠に確認したもので、compute nodeへのSSHやGPU処理はしていない。以下のPCはすべて**ローカルCUDA13.4.92の新規compile**であり、現imageの実行命令と同一とは主張しない。

| production instantiation | QKV | O |
|---|---:|---:|
| compiler/target | CUDA13.4.92 / sm_90a | 同左 |
| launch register / math role budget | 168 / 232 | 168 / 232 |
| 最大参照GPR | R202 | R202 |
| stack / spill store / spill load byte | 0 / 0 / 0 | 0 / 0 / 0 |
| LDL/STL / C75xx warning | なし / なし | なし / なし |
| encoded SASS byte / slot | 62208 / 3888 | 61440 / 3840 |
| encoded SHA256 | `9630609edf7b108d9ccdfd2fcaf5387a3a31f86ca79095394d043d1a2cdcfc35` | `11f685d47e665bb3c16cf1ebfa42cbd568e2fc963b5e10587c977ebe3f7a371b` |
| r37 first-K encodingとの照合 | 全一致 | 全一致 |

compileは `/usr/local/cuda-13.4/bin/nvcc`、O3/C++20、extended lambda/relaxed constexpr、`DG_SM90_PREFILL_DENSE_FIRST_K=1`。通常の174-D expression-has-no-effect警告は残るが、WGMMA直列化警告ではない。168はlaunch metadataで、math threadの上限ではない。role再配分は `128×40 + 256×232 = 64512` registers。

## 現行の正確なK128依存

WG indexをg、local waveをhとすると、行は `64g+128h`、hは0と1。各WGはN128を処理し、**同じ64-register temporary R24–R87**を二つのM64 waveで再利用する。finalは二つのwave分の128 registersで、定常部のdestination集合はR3–R196内の128個。残るR159/R160などはaddress/controlであり、番号の隙間を常時freeとは解釈しない。

各K128で、waveごとに `P_h,k = WGMMA(A_h,k,B_k)` をK32×4で作り、`F_h = fma_rn(fl(sa_h,k*sb_k),P_h,k,F_h)` を64個行う。FFMA64本は64段の依存鎖ではないが、同じtemporary全体を次のWGMMAへ渡す前に、64個を読み終える必要がある。

代表は反復chunkのu=4、動的K12/20/28（Oではさらに36/44/52/60）。

| 処理 | QKV PC | O PC |
|---|---|---|
| sb[k]のshared load、前Kのwave1 promotion中 | `96b0 LDS R197` | `9530` |
| full stageのTRYWAIT、cold分岐は前K promotion末尾 | `9720` → `9a00` | `95a0` → `9880` |
| wave0 saの2行load | `9ac0 R202` / `9af0 R200` | `9940` / `9970` |
| fence / wave0 QGMMA4 | `9b00` → `9b50,9c20,9c50,9c80` | `9980` → `99d0,9aa0,9ad0,9b00` |
| wave0 scale FMUL2、発行中に配置済み | `9c00 R199` / `9c10 R200` | `9a80` / `9a90` |
| wave0 wait_group 0 / FFMA64 | `9cb0` → `9cc0..a0b0` | `9b30` → `9b40..9f30` |
| wave1 sa load / fence | `a0c0,a0d0` / `a0e0` | `9f40,9f50` / `9f60` |
| wave1 QGMMA4 | `a0f0,a120,a130,a140` | `9f70,9fa0,9fb0,9fc0` |
| wave1 scale FMUL2、発行中 | `a100 R198` / `a110 R199` | `9f80` / `9f90` |
| wave1 wait_group 0 / FFMA64 | `a1a0` → `a1b0..a680` | `a020` → `a030..a500` |
| empty arrive、wave1完了後、promotion中 | `a290` | `a110` |
| sb[k+1]のload、次のfull TRYWAIT | `a2a0` / `a310` | `a120` / `a190` |

QGMMAの最初はRZ入力、続く3本は同じtemporaryへ加算、最後の `gsb0` はgroup commitを表す。`warpgroup_fence_operand` はcompiler向けの空asm制約で、要素ごとの機械fence命令ではない。`WARPGROUP.ARRIVE` はwgmma fenceであり、TMAのempty arrivalとは別である。

両compileでpeeled8＋steady8の全16静的K-block×2waveを抽出し、全32waveで **QGMMA4 → WARPGROUP.DEPBAR.LE 0 → FFMA64 → 同じR24–R87への次QGMMA** を確認した。次のWGMMAが前のpromotion途中へ入る例はない。一方、scale FMULは既にQGMMA発行中、次のfull TRYWAITやsb loadは前Kのpromotion途中へ移動している。source行順だけから全scale/waitを露出時間と数えるのも誤りである。

反復chunkはQKV `6a30..c9f0`、O `68b0..c870`、backedgeを含め各1533 slot、QGMMA64・FFMA1024・wait0 16・empty arrive8。chunkはQKVで3回、Oで7回反復する。peeled chunkのK0だけFFMAの第3入力がRZで、初期化削減は既採用の利益である。反復入口はfull waitとscale loadを通り、前chunkのtemporaryは完了・promotion済み。最後もwait0・最後のpromotion後にBF16変換/SMEM store/TMA storeへ進む。chunk/tile境界を定常interiorと同じ露出へ数えない。

### full/emptyとmulticast

denseはMegaMoEと異なり**一つのTMA producer**がA 32768 B、B 16384 B、SFA 1024 Bを発行し、各stageのfull barrierはarrival count1、期待50176 B。B scaleはtile開始時にmath threadsがglobalからsharedへ置き、math-only named barrierで公開する。各Kのsb shared loadはfull waitと独立、saは対応stageのfull完了が必要である。

emptyは各stageで `2 CTA × 8 math warp = 16 arrival`。各warpのlane0/1がpeer/selfのemptyへ到着し、peerが終了済みなら両laneをselfへ向け、countを保つ。現在は最後のM waveのwait0後に一度だけ行う。各CTAの二つのWG、両CTAのreaderが完了するまでproducerはring slotを再利用できない。producerは終了時にも3stageのemptyを待って分散barrierを安全に片付ける。pipelineのstage/phaseはpersistent tile間でも引き継ぐ。今回、この契約を変更していない。

QKVのproducerは `d990 empty TRYWAIT → dab0 A multicast → dc00 SFA multicast → dd60 B TMA → dd70 full arrive/expect`。期待byteは `dd30 MOV R8,0xc400`、fullはstage0 `+0x34c80`、emptyは `+0x34c98`。後続のproducer unrollと終了waitもfresh SASSに収録した。

## N64二重temporaryの設計と追加費用

一つのKを `A=(M0,Nlo), B=(M0,Nhi), C=(M1,Nlo), D=(M1,Nhi)` の4productへ分ける。各productはm64n64k32×4、temporaryはT0/T1の32 registersずつ。概念順序は次のとおり。

```text
full(k); load scales; issue+commit A(T0); issue+commit B(T1)
wait_group 1; promote A; fence T0; issue+commit C(T0)
wait_group 1; promote B; fence T1; issue+commit D(T1)
wait_group 1; promote C
full(k+1); load next scales; fence T0; issue+commit A_next(T0)
wait_group 1; release stage(k); promote D; fence T1; issue+commit B_next(T1)
...
tile end: wait_group 0; release last stage; promote last product; epilogue
```

これなら最も古いgroupの完了をwait1で保証してから、そのtemporaryを読み、もう一方のproductがin-flight中にpromotionできる。[PTX ISAのwait_group規定](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)に従い、最新groupのtemporaryには触れない。compilerのoperand fenceはtemporaryごとに限定し、pendingな他方へ空asmのread/write制約を掛けない必要がある。各出力要素のK128順と最初の `fma_rn(scale,P,+0)` は保てる。ただし新layoutの数値一致を検証したわけではない。

| K128あたりの仕事 | 現行 / WG | N64案 / WG | 現行 / CTA → N64案 / CTA |
|---|---:|---:|---:|
| QGMMA collective operation | 8 | 16 | 16 → 32 |
| commit group / wgmma fence / steady wait | 各2 | 各4 | 各4 → 各8 |
| promotion FFMA / math thread | 128 | 128 | 不変 |
| operand fenceの要素制約、pre/post合計 | 256 | 256 | 不変。group境界は増える |
| QGMMAが参照するA/B descriptor対 | 8 | 16 | 16 → 32 |
| distinct A lower descriptor / WG | 8 | 8 | N間で再利用可能 |
| distinct B lower descriptor / WG | 4 | 8 | Nhi用4値が追加 |
| WGMMA operand SMEM read | 48 KiB | 64 KiB | 96 → 128 KiB |
| TMA A/B/SFA fill | — | — | 49 KiB → 49 KiB |
| full wait / empty arrival数 | — | — | Kごとに据置き、位置は変更 |

descriptor対の参照倍増はdescriptor演算すべての倍増を意味しない。AをN間、BをM間で共有できる一方、Nhiのbase/offset、追加groupのdescriptor bind、fence/waitの発行は残る。物理SASSのUMOV/UIADD/ULEA本数とその露出clockは未compileなので未確定であり、0として利益を作らない。WGMMA発行数を倍にしてもFLOPは増えないが、発行仕事は増える。

SMEM readはCTAの4つのM64について、現行 `4×(64×128 + 128×128)=98304 B`、N64案 `4×2×(64×128 + 64×128)=131072 B`。Aを二度読むため**+33.33%**となる。QKVの73728 CTA-K128で **7247757312 → 9663676416 B、+2415919104 B**、Oの65536 CTA-K128で **6442450944 → 8589934592 B、+2147483648 B**。これはoperand読出し模型であり、TMA/L2 fetchや実HBM counterではない。hardwareの内部再利用でどこまで軽くなるかは未測定。

### lifetimeとrelease/drain

final128＋T0/T1各32＝192 registers/threadで、配列の容量だけは現行と同じ。40の残枠からscale・pointer・descriptor・scheduler・phaseを出す必要がある。A/Bがpendingまたはpromotion待ちの間、それぞれのrow scale factorを保存する。Nlo/Nhiは同じweight block scaleを共有するので、同じMの2factorを共有できる。B→C/Dの移行では旧M0の2factorとM1の2factorが同時にlive、さらにsbとsa loadのscratchが必要。cross-Kでは旧M1 factorを次Kのsb/factorで上書きしてはいけない。scale用に最大8 scalarの枠を保守的に予約すると残32でaddress/control等を賄う設計になるが、実register割当・spill0の証明ではない。descriptorの多くは現行SASSでURにあり、URをGPRへ単純加算しない。現行R202との差をfree register数とも呼ばない。

releaseはDの完了確認後まで遅らせる。上記順序ではA_next発行後のwait1でDまで完了するので、旧stage address/phaseを保持してから旧emptyへ到着する。stageをadvanceしてから現在stageへ到着すると誤releaseになる。full(k+1)を待つ間、Dは発行済みだが、Cのpromotion後に次fullが遅いとその隠蔽窓を失い、旧stage解放も遅れる。3stageのproducerを止めない保証は得られていない。Dを先にwait0でdrain/releaseする安全な形は、この境界のoverlapを失う。

emptyの16 arrival、fullの50176 B、multicast peer/self条件は据え置き、最終productの完了後に一度だけreleaseする。chunk8ごとのdrainを入れるならそのfill/drain費を計上する。跨ぐならpending productのidentity、旧stage/phase、旧scaleをchunk境界で維持する。最後のtileではA_nextを発行せずwait0と最後のpromotionが必要。WG間named-barrier staggerは追加しない。

## 数量による停止判定

必要量の単位を区別する。QKV18wave×32K＋O8wave×64K＝**1088 critical-wave K128/層**。r24と同じ1.410 GHzという**換算scenario**では、30 clockは23.15 µs/層で、astra43のscore約+50の必要量に対応する。今回の厳密な25 µsには **32.40 clock**、望ましい50 µsには **64.80 clock** が必要。これは現imageのclockを測った予測ではない。

最も好意的にも、正味短縮は「新たに隠せる露出promotion H」から「追加issue、増えたoperand serviceの露出、release遅延、fill/drain」を引く必要がある。HはFFMA本数でも静的PC幅でもない。FP32 promotion自体の仕事量は減らず、同SMの2 WGで既に隠れている部分は新利得に加えられない。

追加SMEMを無料にしないため、実効operand service能力をW B/clockとする感度模型を置く。Tensor仕事の床1024 clockとoperand床は重ねて `max(1024,B/W)` とし、足さない。TMA fill・scale loadとの競合は含めないので楽観的である。

| 仮定W、実測帯域ではない | 現行のサービス床 | N64案のサービス床 | 床の増加clock |
|---:|---:|---:|---:|
| 96 | 1024 | 1365.33 | 341.33 |
| 112 | 1024 | 1170.29 | 146.29 |
| 128 | 1024 | 1024 | 0 |
| 256 | 1024 | 1024 | 0 |

この表の0は「追加読出しをTensor床内へ完全に隠せる仮定」の差であり、追加読出しの実費が0という測定ではない。例えばW112なら、30 clockを残すには **H ≥176.29＋追加issue/release/drain clock** が必要になる。W128以上の楽観仮定でも **H ≥30＋追加issue/release/drain clock** が必要である。group/fence/wait・QGMMA倍増を静的本数から固定IPCでclockへ換算して、この不等式を埋めることはできない。

r24の二点fitの約1214 clock/K128と1024の差約190 clockは、promotionだけの測定ではない。上記W112の感度ではその大半を追加readが消し得る。一方W128以上、十分なH、安いissue/drainという仮定なら正の余地は残る。**どちらの仮定を選べばよいかを決めるdenseのper-PC stall、group完了時刻、operand帯域の証拠がない。** 旧r41 MegaMoE上限や別runのclockを借りて埋めない。追加仕事込みで必要量を支持するモデルに至らず、指定の停止規則に従いNO-GOとする。

### QGMMAを倍増しない簡単な案

- **同じtemporaryでM waveのpromotion順を交互にする：** 現行は確かに直列だが、次waveのm64n128は64個すべてのtemporaryへ書く。前waveのFFMA64がその同じ64個を読み終える前に発行できない。wave順を変えてもこのedgeは残る。wait0をwait1へ置換するだけでは、唯一のpending groupの完了を保証できず、不正なregister accessになる。
- **M waveごとに64-register temporaryを持つ：** final128＋64×2＝256で、232 budgetを24超え、補助registerを含める前にthread当たり255というHopperの上限も超える。[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)の上限とも整合する。片方のM waveのfinalをSMEMへ置けば64 KiB/tileが必要だが、mainloop中の既存D buffer 64 KiBを再利用する余地はあるため、shared容量だけで不可能とはしない。その場合はKごとにFP32 finalのread/write **128 KiB/CTA-K128** とload/store発行・依存が増え、N64案の追加32 KiBより大きい仕事になる。register内だけの簡単な順序変更ではなく、正味必要量を支持する証拠もない。今回は実装しない。

stagger、tail skip、body/tail split、QKV epilogue融合、BM192/BM128 retiling、stage4の過去の棄却を維持する。r2/r9の+12%/+6.3%、r11の+32〜34%、r30 prewarm後もmain同等、spill/転送増の証拠を覆す測定は今回ない。

## 仕事量、検証、後片付け

有効FLOPはQKV `603073806336`、O `536065605632`、計 `1139139411968` /層。FP8入力/weight、BF16出力とFP32 scaleを各1回通す論理I/OはQKV `218734080`、O `166482944` B/層。検討案も全K128積と元scale粒度を保つ。実HBM/L2/NVLink byteは未測定で0にしない。EP1 denseにexpert all-to-allはないが、PP/PD転送は別である。

phase 2のhost env flag、variant、full-shape synthetic/real checkpoint数値比較、500-call耐久、12 GPU正逆gate、prewarm/dispatchは実施しない。r41 probeの置換とdiagnostic host/overlay廃止はそのphaseの作業であり、今回は変更しない。production、config、image pin、submissionは不変。この研究記録に **require-build は不要**。

CPU proof `prefill-dense-promotion-r43-cpu-proof-20261105.tar.gz` にfresh compileのsource/command/log/cubin/encoded SASS、全waveの依存表、仕事量感度、保存run照合を収録する。資源・命令・byte計算を機械照合した。履歴非依存の検証にgitの過去blobを使わず、既存proofとrawを入力にした。診断scriptはproof内に限定し、runtimeへ追加していない。

指定の全pytestとRuffをPATH設定後に実行した。pytestの起動では継承SIGHUPをSIG_DFLへ戻して同じ `python -m pytest -q -p no:cacheprovider qwen-experiment-metrics/tests` をexecし、**100%まで完走、終了0**。Ruffも **終了0、All checks passed**。最終exit code、log hashとarchive検査は隣接CPU-validation JSONに記録した。GPU数値・速度・公式品質の証明とはしない。展開用work/proofを削除後、archiveだけを一時展開して全25 member hash、両cubin/SASS encoding、資源・全64 waveを再検証し終了0、一時展開も削除済み。再現証拠とlogを残す。tracked差分と新規note/JSONの空白検査、commit draftの形式検査も合格。今回のSSHは上記の読み取り専用確認だけで、commit/push、GPU実行、Actions/scheduler投入・停止、image buildは行っていない。この作業で作成したremote jobはない。
