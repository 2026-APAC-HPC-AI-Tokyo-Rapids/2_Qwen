# Round 32：MegaMoE FC1 初回 K128 の入口特殊化

CPU gate は通過した。実装、全12 Prefill GPU用の in-job gate、serving process自身の ready 前準備までを作成し、**GPU投入・fresh image buildの前で停止**する。GPUでのbyte一致、実時間、通常servingの公式scoreは未測定であり、採用済みとは扱わない。

親は main `f7875802a420105a16a0a1b0bc8516f0b368bbd6`、作業branchは `perf/pp4-r32-fc1-firstk-20261104`。AGENTS.md、r31の設計・CPU gate、r26/r27/r30のnoteと保存Actions由来の予算・結果、関連コミット本文、scratchのCodex会話記録、tracker/model/profile JSONを確認した。SSH、Actions操作、Slurm投入はしていない。

## 対象と変えない仕事

対象は Prefill EP1、T7988、E128、top8、H4096、I1536、M128/N256/K128、4 stage、lean、split accumulator。BF16入力・最終出力、FP8 E4M3の積とFC1出力、FP32 scale・accumulator、FC2のBF16 partial、FP32 weighted combineを維持する。FC2の初回は対象外。

有効仕事はr31と同じで、1層あたりFC1 `4×63904×4096×1536 = 1.608196816896 TFLOP`、FC2 `0.804098408448 TFLOP`、合計 `2.412295225344 TFLOP`。historicalの実行行数68288も変わらない。HBM・NVLinkの転送を削る変更ではなく、weight、pool、TMA、SwiGLU、group128 quant、partial、combineはそのまま。r31のFC1主要論理I/O模型47.572 GB／24層も変わらず、実HBM byteは未測定。EP1のexpert間NVLink通信はなく、PP/KV転送は別の仕事である。

r31の0.211 ms／周期は初回FMULだけの条件付き命令サービス量、ゼロ初期化を含む約0.42 msは楽観的な命令量換算である。実時間の予測や3 ms／周期の根拠にはしない。r26のswap-tail、過去のscale先読み・scale合成、FC2最適化の利得を加算しない。

## OverlayとCPU gate

`patches/deep-gemm-overlay/sm90-prefill-fc1-first-k.patch` をPP4 configの `deep_gemm_overlay_patches` の最後へ追加した。`DG_SM90_PREFILL_FC1_FIRST_K=1` のときだけ、FC1 splitの最初の8×K128 chunkを外へ出す。最初のK128の各halfの最初のK32だけscale-d=0にし、残りのK32と31個のK128は既存の順序でaccumulate・rescaleする。chunk境界のdrainは8/16/24/32のまま。最初のchunk全体を外へ展開したのは、WGMMA groupをloop edge越しに持ち越さず、既存ptxasの追跡条件を維持するためである。

元の `final_accum = {}` は保持した。最初のQGMMAへ入る有効行の経路ではcompilerがゼロ初期化を消し、padding-only WGに必要なゼロは残す。調査途中にpadding専用のゼロをFC1入口の残存と誤認し、手動初期化を試したが、R218／R208への増加と他geometryの変化を招いたため撤去した。**それらを最終候補のregister数・SASS証拠へ混ぜない。** 最終候補は元の初期化宣言を保つ版である。

CUDA `/usr/local/cuda-13.4`、nvcc/ptxas 13.4.92、sm_90a、C++20/O3、r26/r27と同じ24代表instantiationをcontrol・flagなし・flagありの3通り、計72 compileで比較した。全Prefill geometryに加えて既存Decode geometryも含む。

| 項目 | control | 最終候補 | 差 |
|---|---:|---:|---:|
| FC1最初のK128のaccumulator FMUL | 128 | 0 | −128 |
| 有効FC1入口のaccumulatorゼロ初期化命令 | 64 | 0 | −64 |
| 定常8×K128のSASS範囲の命令数 | 1619 | 1611 | −8 |
| kernel全体の静的命令数 | 9856 | 11320 | ＋1464 |
| SASS encoding byte | 157696 | 181120 | ＋23424 |
| ptxas launch register | 168 | 168 | 0 |
| 最大参照register | R206 | R206 | 0 |
| math setmaxnreg枠 | 224 | 224 | 0 |
| stack／spill store／spill load | 0／0／0 | 0／0／0 | 0 |

flagなしは **24/24の命令・制御wordが全byte一致**。flagありでも対象以外の23構成は全byte一致した。候補のLOCALアクセスは0、C75xx警告はなく、dynamic shared 228560 byteも維持する。既存のlegacy geometryには従来からspillがあるが、そのbyte一致を「全geometry spill 0」と読み替えない。

controlの初回は `0x2090..0x2cf0` の範囲にFMULがあり、候補の初回QGMMAは `0x1db0` と `0x1f30` で `RZ, !UPT`。定常chunkの範囲はcontrol `0x2090..0x85b0`、候補 `0x7a90..0xdf30`。表の定常命令数はback-edgeまでの静的範囲で、waitの反復回数や実issue時間ではない。コード量は約14.9%増えるので、命令cacheの実害はGPU gateで判断する。

control header SHA256は `fbfaa6be207b26dd7127aae406204b44668b4341ff5ce9e2928510c932e57852`、候補は `75085f03b05c8ab0cb2dd072994b27b1584225456928d49e66e67820ed1dee56`。control SASSはr31と同じ `d582cf2059b10cd17523ecf1bcb70135ca5ca7174defa29caf9ad8345f6015ad`、候補は `f157a6dddb926f5944f751e46448afb69091eb9cfbcccd17e21fe852205e8c41`。正本は `prefill-megamoe-fc1-first-k-r32-cpu-validation-20261104.json`。

実headerから抽出した元と候補のC++ lambdaを記録用CPU実装で実行し、開始4 stage×2 phase×lane 0/1の16条件でevent列を比較した。32 K128、256 K32積、初期化QGMMAは2本、half rescaleは64→62、prefetch31、lane0のempty release32、他laneのrelease0、absolute K128 index 2での遅延publish hook1回を確認した。FC2、paddingのstage消費、epilogueのarrival処理のsourceは保持し、pool block当たり `12×ceil(valid/64)×4` のarrivalを変えない。これはGPU上のraceやTensor Coreのsigned zero・NaN・FTZを証明するものではない。

host overlay `sm90-prefill-fc1-first-k.patch` は既存series末尾へ追加し、該当するEP1 split/H4096/I1536/E128だけでmacroをJIT sourceへ発行する。native hostの112条件ではflagなしの生成内容が一致し、macro発行は該当2条件だけだった。14本のheader overlayの順次clean適用と新host overlayのclean適用も確認した。実imageのCUDA 13.0.88でのbuild/link/loadは今回未実施で、13.4の結果をその代替成功とは扱わない。

## 将来のin-job gateと予算

既存 `select-firmus-prefill-megamoe.py` のprepare・独立FP32参照を再利用し、r26のsnapshot・UUID照合・同resident processの仕組みを `firmus_prefill_fc1_first_k.py` へ適用した。計時は計装なしの完全MegaMoE kernelで、snapshotや参照計算は計時外に置く。

- pair/prepackとgreedyが終わってから開始する。Pの全12 UUIDを照合し、3 worker×4 GPUの同じallocation・image・header・routingを要求する。
- balanced/hot/historical、全境界、zero/wide、保存36 routingの44 caseで、最終BF16、FC1 FP8、FC1 scale、FC2 BF16 partialの全byteを比較する。pool順はmetadataでtoken/top-k順へ復元する。
- 各caseの32 tokenを独立FP32参照と比較し、control・candidate双方でrelative L2 `<0.025`、max scaled error `<0.08`、finiteを要求する。TF32は無効。Tensor Core内部のbit一致は上のbyte比較で別に判定する。
- balanced/hot/historicalの各GPU・各routing・両順について12 roundを交互に計時し、それぞれの中央値で `max(5 µs, controlの0.5%)` 以上の短縮を要求する。集計の総和だけでは採用しない。
- 6 routing×2入力×2 bufferの全24 cellを50周、計1200 call実行する。600/1200 callで全cellの出力byteを比較し、最後は元のhistorical入力へ両bufferを戻して全4成分を再比較する。
- timeout、欠測、不一致、精度・速度不足は不採用。envは一時ファイル上で判定し、全成功後のrenameだけでservingへ公開する。serving用flagの初期値は0、gate keyはtrueである。

予算の根拠はr27予算JSONが保存するr26 run41598151128の実測で、group wall **52.3206秒**、rank内 **17.2849〜17.8880秒**。r27はworker90秒・集約100秒を実際の投入設定とし、通常job910秒／READY614秒で終了した。今回はFP32参照44 caseと600 callの中間確認が増えるため、52秒を保証値とはしない。同じ **worker90秒＋kill grace2秒、集約まで100秒** に制限し、増分が収まらなければ候補を採用しない。

開始条件は **`SECONDS + 100 + 660 <= 1020`**。job開始260秒までしか開始せず、r25のgreedy後222〜236秒なら残り24〜38秒を確保する。greedyを飛ばしたり、serving660秒予約・公式窓・耐久回数を削ったりしない。r27で観測したBeeGFSのTriton launcher ENOENTを避けるため、gateのDeepGEMM/Triton/TVM cacheはjob/worker別のnode-local `/tmp/sglang-r32-*` とする。新gateのwall timeは未測定。

## ready前の準備とdispatch回収

imageのSGLang sourceは `fc275674a97d78553b22597ee72c8fd48f5420f4`。実sourceのDeepGEMM `_BaseWarmupExecutor.create` は通常／grouped GEMMの6種類で、native MegaMoEは含まない。`pp_parallel_deep_gemm_warmup` はdummy forwardの仕組みであり、全Prefill processの候補・7988行の初回実行完了を保証する経路とは確認できない。`model_loader/loader.py` のload後処理が `quant_method.process_weights_after_loading` を呼び、`layers/quantization/fp8.py` が `build_sm90_mega_moe_experts_weights` を実行するので、変換済みweightは今回のhookより前に利用可能になる。r30と同じく、gateの別process/cacheをready前準備の代用にはしない。

`prefill-megamoe-fc1-first-k-r32.patch` をSGLang overlay末尾へ追加した。`ModelRunner.initialize` のpost-load transforms／KV dtype設定後、pool・graph・readyより前に、採用flag1のtarget Prefill processだけでhelperを実行する。最初の対応Qwen3 MoE層の実変換済みweight、servingと同じEP1 group・cap8192の共有buffer key、BF16の7988×4096入力、native pre-dispatch／fp8_mega_moeを一度使い、device synchronize後に完了を記録する。入力は専用のzero、routingは決定的な全128 expert分布、top-k weightは1/8で、RNG・model forward・KV・samplingを呼ばない。

実行後だけ `_r32_prewarmed` を立てる。別のonce-per-process logは実servingの `fp8_mega_moe` 呼出し後に出し、prewarm完了と混同しない。`results/prefill-r32-dispatch-{0,1,2}.txt` にprewarmとcandidate/controlの行を回収する。今回は未投入なので実dispatch記録はない。fixtureに保存したimage sourceからconfig順の全overlayを適用し、起動順、同じweight・buffer・shape、二度呼んでも一回、Decode/flag0除外、sync失敗時の完了log抑止をCPUで検証する。testはgit履歴・mirror・networkを参照しない。

## 検証・保管と次段階

`export PATH=$HOME/.local/bin:$PATH` の後に指定の `uv run --project qwen-experiment-metrics --with 'numpy>=1.26,<3' --with pytest python -m pytest -q -p no:cacheprovider qwen-experiment-metrics/tests` を実行し、**終了コード0、1983 passed／90 skipped、計2073件**を確認した。CPython3.13.12／NumPy2.5.4／pytest9.1.1。`uv run --project qwen-experiment-metrics ruff check qwen-experiment-metrics scripts containers` も**終了コード0**。shell構文、whitespace、native host、source由来schedule、overlay適用、shell fixtureの6条件、保存source SHA照合も成功した。初回完走はcommon build contextへコピーされるPP4専用patchを期待値に含めていなかった1件で終了コード1となり、共通configでは未適用であることを保ったまま期待値を修正して全体を再実行した。CPU調査中／配線中に中断した全体pytestと、その一時的な不採用判定を最終成功へ含めない。raw logと終了コード、件数の集約 `test-results.json` は指定scratchに保持する。skipをGPU成功とは扱わない。

CPUのcubin/SASS/log/source・再現手順は指定scratch `codex-scratch/r32` で作成した。worktreeとscratchは同じローカルext4上、tomokazu所有・0700である。`r32-cpu-proof-20261104.tar.gz`（19162092 byte、SHA256 `e2faa2a495c306729bf41f711bd28e5cdceb1a206454ef4df04f7ceaf21f8015`）へ72本のSASSとcompile入力・依存header・再現scriptを保管し、gzip検査とhash照合を行った。patchの空行表記を正規化した後も、適用後のC++ sourceが同じbyteであることを再確認した。展開用の重複source/cubin・診断中間物は削除し、archiveと検証logを保持する。今回のGPU job・残留scheduler jobはなく、既存jobには触れない。

このbranchのcommitはすべて先頭 `[skip ci]` で、push自体から計算を開始しない。**投入するならsourceとhost拡張が変わるためfresh buildが必要**で、将来の投入commit tagは `[require-build] [cluster:firmus] [nodes:2]`、`[skip ci]` は外す。既存image `a001f901` のpinは今回変更していない。将来のbuildがruntime configのimage情報を更新した上でgate→ready前準備→通常serving→公式parser→成果物回収を完走する必要がある。CPU通過をGPU採用や通常servingの改善へ読み替えず、公式scoreの優位性は同条件の複数runで判断する。

## Round 32 の実測と round 33 の準備（2026-11-04）

本節は上の投入前記録を更新する。round 32 submit は `3f9124fd8afb28e34c71288b6dc31e5c490c7f60`、Actions **41729881283**、Slurm **28626**。正本は `/srv/qwen-artifacts/41729881283/1/qwen-firmus-pd-41729881283-1/` の `logs/logs`、`results/results`、`source/source`。今回これらは読み取りのみで調査した。関連 commit 本文、r32 の Codex 会話、tracker/model と保存 profile も確認した。

全12 UUIDで44 case×4成分の byte 一致、独立 FP32 閾値、1200 call、600/1200 の全24 cell、両bufferへの元入力復帰が通過した。しかし selector は `AssertionError: failed workers or expired budget` で不採用。通常 serving は first-K flag 0 の control、公式 parser は eligible/validation valid、6312 requests、request error 0、score **10015.98524540721 output tok/s**（従来表記10015.98）。READY 606秒、Slurm COMPLETED/0:0、全体14:49。main r25 の10020〜10025程度との差と flagなしSASS byte一致を踏まえ、候補の serving 効果や退行の証拠とは扱わない。候補を使った通常 serving の結果はまだない。

### 終了不良の根拠と修正範囲

rank JSON の `finished_at`、`elapsed_seconds`、worker log の先頭 torchrun 時刻と末尾を照合した。以下はすべて同じ allocation の UTC+08 時刻であり、telemetry の UTC と混ぜていない。

| worker | torchrun 最初の log | rank 本体所要秒 | 最後の結果時刻 | 終了の観測 |
|---|---|---:|---|---|
| HEAD 0 | 07:46:54.056 | 18.954〜19.164 | 07:47:15.854 | 07:47:51、step 28626.9 killed |
| HEAD 1 | 07:46:54.184 | 18.998〜19.043 | 07:47:16.004 | 07:47:51、step 28626.10 killed |
| TAIL 2 | 07:46:34.272 | 18.053〜18.205 | 07:46:53.836 | srun failure の記録なし |

全HEAD rankが `healthy: true` をprintした後に約35秒停止している。`--fc1-first-k` は `firmus_prefill_fc1_first_k.run` を直接呼び、print後の明示処理は2 buffer の `destroy()` と `dist.destroy_process_group()` だけで、計算、追加barrier、独自threadのjoinはない。したがって測定本体でなく**後始末〜process終了区間の停止**がログで特定できる。`Gracefully skipping multicast initialization` は計算前のwarningで、TAILにもあり、その後全byte/耐久が通るため最初の致命的エラーとはしない。

source上の寿命管理の欠陥は、`prepare(1)` が eager NCCL WORLD と rank別singleton EP1 groupを作る一方、終了時はEP1を明示して揃って破棄せず一括WORLD破棄へ進み、さらに sgl-deep-gemm **0.1.5** の `SM90SymmBuffer.destroy()` が `handle/buffer/group/x/x_sf` だけをclearして `topk_idx/topk_weights/l1_acts/l1_acts_sf/l2_acts/l2_acts_sf` を残すこと。build記録のpackage versionと、ローカルcacheにある同版wheelのsourceを読み取りで照合した。[PyTorch 2.11 shutdown source](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1418) はNCCL finalize、watchdog join、communicator destroyを行い、[symmetric allocation解放](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu#L41)にもdevice synchronizeが入る。停止したC++ threadのstack、NCCL DEBUG、各destroy前後のlogは保存されていない。従って、**特定のNCCL finalize、symmetric handle destructor、thread joinのどれが待っていたかは確定できない**。別issueやwarningだけを根拠にNCCLの既知bugと断定しない。

round 33ではこの寿命管理を限定して直した。最終CUDA処理の完了と全rankの合流後、残る6 viewをclearして両bufferを破棄し、保持した当該EP1 groupを明示破棄、WORLD barrierで他rankのEP1解放完了を待ってからWORLDを破棄する。JSONの `healthy/elapsed_seconds/finished_at` 更新と公開はこの全後始末の後に移した。これにより、計算が終わっただけのJSONをprocess正常終了の証拠として扱わない。retry、fallback、強制正常exit、thread pollingは追加しない。CPUテストでは実finish関数を呼び、view解放・EP1/WORLD順序と、cleanup例外時にはhealthy JSONを公開しないことを確認する。これはGPUでの終了不良解消の実証ではなく、**修正後のGPU実行は今回禁止されており未確認**である。

### データ観測後に変更する pooled 規則

**この変更はround 32のデータを見た後の事後的な規則変更である。事前登録した規則の成功や独立再現とは扱わない。** 各routingにつき、GPUごとにforward/reverse各12回のcontrolを24個へ、candidateも24個へ結合し、それぞれのmedianの差 `d_i` と `d_i/control_i` を作る。12 GPUのmedian節約は5µs以上かつmedian節約率は0.5%以上とする。単位の異なるmax条件をこの2閾値として実装し、GPU間で全sampleを混ぜない。forwardとreverseもそれぞれGPU内medianの差を作り、その12 GPU medianが厳密に正であること、pooledの `d_i>0` が9 GPU以上、全GPUのpooled節約率が−1%以上を要求する。0は正のGPUへ数えない。byte/FP32/耐久/UUID/source/image/健全性/worker終了・deadlineは全12 GPU必須のまま。

| routing | pooled median節約 µs | median節約率 | forward median µs | reverse median µs | 正のGPU | 最小pooled µs | 最小節約率 | 新速度規則 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| balanced | 24.455965 | 0.966855% | 19.831955 | 24.559975 | 12/12 | 8.431911 | 0.335166% | 合格 |
| hot | 16.911983 | 0.650233% | 18.016040 | 17.416060 | 12/12 | 8.880019 | 0.349064% | 合格 |
| historical | 16.552091 | 0.640463% | 16.575992 | 17.216027 | 11/12 | −3.056049 | −0.122525% | 合格 |

参考としてpooledで各GPUに旧marginを要求すると9/12、9/12、10/12だけが通る。実際の旧コードはさらにforward/reverseの両方を各GPUで要求するため、pooledだけのこの数を旧selector全体の合格数と取り違えない。

必要な計時数値とrank出典hash、時刻だけを **37760 byte** の `qwen-experiment-metrics/tests/fixtures/prefill-r32-timing.json` へ抽出した。testはarchive、git履歴、mirror、GPUを参照せず、実sampleによる3 routingの集約採用と上表の数値を再現する。正しさ検査は既存fixtureで全12 GPUの欠測/不一致/健全性等を引き続き拒否する。実runの `workers_ok=false` をtrueに書き換えた記録は作らず、速度の再判定が合格してもround 32のselection失敗はそのまま残す。

### 起動遅延を含めた予算

Slurm最初のlogは **07:42:11+08**、gateの `run_id` に入る開始epochは **1793749581 = 07:46:21+08**、差は**250秒**である。最初のlog以前の短いshell初期化は記録されておらず、250秒はSECONDSの推定下限。元の `SECONDS+100+660<=1020` が実際にgateを許したことから、真のSECONDSは最大260秒だったことも確認できる。さらに同じHEAD上のrouter logで起動は23:52:15 UTC（07:52:15+08）、READY直後の最初のget_server_infoは07:52:17+08。sourceではrouter readiness後、get_server_info前に `READY after 606s` を出すため、READY時刻はこの間にある。606秒を引くとshell開始は07:42:09〜11付近で、gateのSECONDSは250〜252秒程度と推定できる。router起動の下限とgateのepochに各1秒の丸め、SECONDSにさらに1秒の余裕を取る保守上限は `1793749581+1-1793749935+606+2=255秒`。別runのclockやtimelineは使っていない。

HEAD2 worker同時起動はgateから**33.056/33.184秒**でtorchrun log、rank本体開始推定は**35.690〜36.960秒**、結果完了は**54.645〜55.004秒**。TAILは13.272秒でtorchrun、rank本体開始14.630〜14.632秒、結果完了32.684〜32.836秒。これらの差はsrun配置・container起動・Python/torchrun起動等を含む合成遅延であり、Enroot単体の時間に分解できない。

round 33はworker deadlineをgate開始から **95秒**、kill grace **2秒**、aggregateまでの予算を **105秒**とする。HEADの約37秒の本体起動待ち＋約20秒の本体を見込み、残り約38秒を後始末とworker終了に確保する。95+2秒後から105秒まで約8秒を集約に残す。serving予約**660秒**と**1020秒**窓を保ち、実行時は必ず **`SECONDS + 105 + 660 <= 1020`**、すなわちSECONDS≤255の場合だけ開始する。round32の記録時刻に基づく推定では **250+105+660=1015秒≤1020秒**、HEAD同時起動遅延も105秒の中に含まれる。上記の保守上限でも **255+105+660=1020秒** と成立することを、READY前後の同一run時刻とsource順序から確認した。fixtureへこの時刻も入れて境界を検査する。次回も実行時のSECONDS判定を正本にし、条件が満たされなければ既存の不採用経路を使う。READY606秒や全体889秒から逆算してserving予約を削らない。

### Round 32 image の一式pin

build artifact `qwen-aws-container-build-41729881283-1/build-logs.tar.zst` の `logs/oci-manifest.json` のSHA256は `70bf5f49c01a97f376223252c66499ae78efc20e2c92dae94d64223971a0772c`、serving artifactの `container-sidecar.json` と一致。`image-manifest.json`、`sglang-image.sha256`、`provenance.json`、sidecarのsourceとimage情報も照合した。

- image path: `/scratch/home/ACCOUNT/qwen/containers/41729881283.sqsh`
- image SHA256: **`3a47b757bb88f6bc14ad5f151e8405f645007f25f381f631d88046d9bd1b7b04`**
- size: **12296638464 byte**
- provenance: `/scratch/home/ACCOUNT/qwen/containers/41729881283.container-sidecar.json`
- build run: **41729881283**、build source: **`3f9124fd8afb28e34c71288b6dc31e5c490c7f60`**
- SGLang source: `fc275674a97d78553b22597ee72c8fd48f5420f4`、DeepGEMM host source: `1c8138a59b3df5dac50b76b730834ed57982fdfb`

過去pin commit `f821771` の範囲を参考に、configのimages.sglang全4項目とcontainer全4項目、CPU checkerのPINNED_IMAGE、FA3 prepack/pairのimage prefixとbuild run期待値を同期更新した。圧縮されたprewarm source fixtureのimage SHAとimage-manifest SHAも更新した。保存24 sourceはすべてr32 manifestのtarget hash、または同manifestの変更なしSGLang commitのsourceと一致し、source本文の変更は不要だった。image-manifest SHA256は `271d61f36f062b3a73770bcf5c1d01b04d0465500d15216ff7ddb3c249c69032`。過去の研究noteのa001f901は当時の事実として保持する。変更はhost上へstageするselector/集約/実行予算と文書のみで、image内のsource/toolchain/host拡張/patchを変えない。**次回はrequire-buildを付けずこのimageを再利用する**。今回のcommitは先頭skip-ci、submit commitは作らず、指定round33 branchにだけpushする。SSH、Actions/Slurm操作・GPU実行・image build、mirrorや他worktreeの変更はしない。


### Round 33 のローカル検証

指定のPATH設定後、指定どおりの全体pytestコマンドは最終再実行で**終了コード0**。最終収集数は2088件。`uv run --project qwen-experiment-metrics ruff check qwen-experiment-metrics scripts containers` も**終了コード0**。更新対象gate/prewarmの56件、shell構文、diff whitespace、全24保存source hashとimage/manifestの照合も通過した。初回全体pytestは圧縮prewarm fixtureに旧image SHAが残っていたため6 setup errorで終了コード1となった。この参照漏れを修正し、対象56件の成功後、指定の全体コマンドを再実行して0を確認した。実行中ジョブの取消・GPU再実行による確認は行っていない。新規の診断scriptや展開用一時sourceは残さず、意図した計時fixtureと更新した既存圧縮fixtureだけを追跡対象とする。
