# Prefill NCCL symmetric memoryの検証

ユーザーが[Stas Bekman投稿](https://x.com/StasBekman/status/2106978810247917610)の検証を指示した。公開embed APIで原投稿本文を取得し、そのリンク先[著者のprimary README](https://github.com/stas00/ml-engineering/blob/master/network/README.md#symmetric-memory)を確認。small/medium payloadのregistered NCCL symmetric bufferで速度向上を見込む説明だが、示されたH200/B200・TP8 micro測定をH100TP4/Qwen E2Eへ転用しない。NCCL2.27+/PyTorch2.9+を要し、実kernel使用はINFO/TUNINGの`AllReduce [Symmetric]`ログで確認する説明である。

同image182fbb6dのread-only CPU診断でtorch2.11.0+cu130/NCCL2.28.9を確認。最初のlogin Enroot診断はGPUdriverhook（NVIDIA_VISIBLE_DEVICES=none）がlibnvidia-ml無しで失敗し、voidによるGPU非要求診断で成功した。scheduler GPU障害やnewbuild理由ではない。sourcefc275674のfp8.py2413–2443はCUTLASS最終出力をuse_symmetric_memoryで割り当ててoutput引数へ渡し、routingpatchもoutputを維持する。DPattention無効ではallocation symmetric条件が成立する。DeepSeekV4向け[PR29007](https://github.com/sgl-project/sglang/pull/29007)のDeepGemm/Triton修正をこのCUTLASS試験へ持ち込まない。

既P-only NVLS37145086734（+0.264%単pair未採用）とは、NCCL allocator/register/graph同期経路の追加条件が異なる。現CustomAR共有buffer23.01MB/multicastTrueとも別。symm有効時通常allreduceはpyncclをcustomARより先に選ぶが、torch.compile枝は迂回して適合customAR優先を残す（parallel_state.py668–705/878–884）。4GiB/GPU target preallocationと登録費用を含め、flagのみで実適用や改善を断定しない。

## 通常性能条件

parent e484c760（routingon公式control）から`f229f661de9f4a1ebbfda59def7f43fa97497450` / [Actions37604857808](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37604857808) / PBS217308を投入した。runtime `pd-37604857808-1.GsyhWW9BWK`。P-only `--enable-symm-mem`を追加し、baseline実pairP030/D032へselectorを固定した。D/c48/P2TP4EP1/D2TP4/Pshuffle1/Proute1/FA3/FP8KV/EAGLE3BF16/profilefalse維持、customARdisable/NVLS/algoprotooverride無し、newcast無し。新sourcebuild不要。

51CPU tests（argv/orchestration/monitor/contracts）とbash/actionlint/diff通過。ASPIRE実sidecar/runtimeprovenanceでimage182fbb6d/size12231192576/buildsource9c5a5a3/sglangfc275674/commitf229f661/profilefalse、EC2/mirror/S3buildskippedを照合。PBSQ/aiq2/mem不足、030032各112CPU8GPU1880GB/excl/30minを確認。旧routingfixed216741/216810は取消せずLuna監視維持。

通常benchにはdebug/profilerを追加しない。terminal時parser/latency/精度診断/rawhash/actualargv/UUID/cleanupを監査し、baseline37576508318の2779.551953との一対比較は暫定とする。既定ログで実symmetric kernel証拠が得られなければ、別追跡profile条件のINFO/TUNINGで確認し、そのscoreを通常性能値として使わない。公式結果・実適用・主催者完全accuracy・採用は現時点未確認。

## 3本の未実行終端と接続監視修正（10月8日）

old routing off37319361445/216741、on37389659143/216810、symm37604857808/217308はActions failure/PBS F。最初のfatalはOct7 14:08:27/14:12:04/14:09:35Zのmonitor SSHtimeout124で、laststateQの連続接続障害が1800秒を超えた後、workflow cleanupが自身の未実行Qを取消した。mem不足は待機理由であり終了fatalではなく、5日Actions期限でもない。run-success欠損exit2は二次障害。全3PBSにstime/resources_usedwalltime/exechost/Exitstatusが無くGPU実行無し/公式score無し、各正本9hash(計27)一致、全runtime消去を独立確認した。

観測に基づく最小main修正acf2427はlaststateQ確定時だけ観測failure累積でfatalにしない。個別SSH180秒・Runknown300秒・実terminal判定・Actions上限を維持。elapsed1801秒fixtureでQ継続/Runknownfatalを確認、terminalsuccess/failure既fixture含む23tests/Ruff/bash/diff通過。GitHubpushが500を返す一時障害を観測したが、後続pushは成功しmainへ配備済み。

固定030032はread-only現物で他jobが各全資源占有、即時空き/ETA確約無し。symm通常比較だけをfree同selectorへ再準備し、routingoncontrol999a5a3（parente484c760）とP-onlysymmon2b04aa2（parentf229f661）にmonitor修正を適用。各52CPUtests/bash/actionlint/diff通過、sameimage182fbb6d/buildsource9c5a5a3/no-sourcebuild/resource削減無し。独立allocationのactualpair一致を照合し、別pairなら結論を制限する。routingfree一対は既監査完了のため、未実行旧fixedrouting2本の再投入を無条件には追加しない。symm性能・実kernel使用・精度・terminal全監査は未完了。

再投入はcontrol [37642502611](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37642502611) / symmon [37642503212](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37642503212)。Oct7 15:24Z時点で両Actions validate成功、ASPIRE既存成果物選択中、PBS未確定。同image来歴・実資源・実適用・公式結果・精度・全rawhash・cleanupの終端監査を続ける。

Oct7 23:37SGT実物照合: control37642502611→PBS217355 / runtime `pd-37642502611-1.IwVyAW9cat`、symmon37642503212→PBS217356 / `pd-37642503212-1.vW71S4Y8D4`。両Q/aiq2/profilefalse、submit自由select2×112CPU8GPU1500gb、admission各1880GB/scatterexcl/30min。sidecar両image182fbb6d/size12231192576/buildsource9c5a5a3/sglangfc275674一致、新EC2/mirror/S3buildskip。mem不足で待機、control予測Oct8 08:52:33SGT/013019は非保証・actualnodeではなく、onETA無し。Lunaへ両handles引継ぎ済み。

## NCCL symm候補の観測障害とraw欠損（10月8日）

候補37642503212/PBS217356は同013/019の16H100で09:17:52–09:29:48SGT、PBSF/Exit0/Stageout1まで計算した。しかしActionsは01:22:03ZにlaststateRのSSH124を観測し、接続障害300秒予算で01:27:33Zにmonitorがexit124。真のPBS終端前に回収を開始したため01:29:20Zに`tar: logs/router.log: file changed as we read it`でexit1、その後workflow自身cleanupがruntimeを削除した。時計差ではなく終端を確認できないまま回収へ進んだ二次障害である。Actions failure、正本37642503212/1は17logsのみで全17hash一致、results/source/run-success/greedyは欠損。

終端前のread-only runtimeでcontrol2775.546377213522、候補2843.0536912143675 outtok/s（+2.432217114%、goodfraction .9912739965/.9920454545、both eligible/error0）を観測したが、候補のrepository parser再計算・8greedy・全来歴監査が不能なので性能比較は参考値へ格下げし、採用根拠としない。flagtrueをNCCL symmetric kernel使用の証明にせず、通常runとは別のINFO/TUNING診断を必要とする。

実観測に基づく最小monitor修正はQ/R/unknownのすべてでSSH124/255の累積時間だけをscheduler失敗とみなさない。個別SSH180秒、PBSwall30分、Actions上限、実terminal/成果物checkerの失敗を維持する。接続復旧後に確定終端を確認してから回収し、完走成果物を観測障害だけで破棄しない。Q/R/unknown・elapsed1801秒の障害後にterminal0/1をそのまま返すfixture、既非接続fatalを含む23tests/Ruff/bash/diffPASS。tar専用fallbackは追加しない。control終端監査後、必要候補のみ同対照pair・同imageで再投入し別適用診断を行う。

### 必要候補と別適用診断の投入

main修正e758ee1のvalidate37714579426成功後、通常再試験`9d64fae44b26d2594fa85a477059a759ca6fdf6b` / [37714793539](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37714793539)と、別適用診断`2928a4c24b13c9d5febc446a9cfb33b6a0c50741` / [37714798160](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37714798160)をpushした。failed2b04aa26親の条件別branchにmonitor最小修正を適用し、両013/019を実対照pairへ固定した。各52tests/Ruff/bash/actionlint/diffPASS、同image182fbb6d/16H100/serving/precision維持、新build無し。通常runにdebug/profiler無し、診断だけP/profiletrueのNCCLINFO/TUNINGで`AllReduce [Symmetric]`の実kernel選択を確認する。profile scoreは比較へ混ぜない。control37642502611は取消せず回収継続、終端監査前でも長いqueue待ちを避けるため必要条件を先に投入した。PBS/runtime/provenanceとterminal/raw/greedy/hash/cleanup監査を続ける。

### Control終端監査と再試験来歴

control37642502611/PBS217355はActions success、F/Exit0、P013/D019、wall12:05（09:05:41–09:17:52SGT）。正本52rawhash、公開規則準拠repository parser再計算=保存score、公式60/120/30command、model/tokenizer/同image/source、4worker argv/server settings、16uniqueH10080GB、8greedy各64token/finiteを独立確認。score2775.546377213522、goodfraction.9912739965095986、1719req、cancel/error0、TTFT>3s15/ITL>20ms0、meanTTFT1402.9647734141943ms/ITL10.000490851648756ms。runtime-removedを独立SSH確認。主催者完全accuracyではなく、有効対照1runとして保持する。失敗候補217356の16physicalUUIDは対照とすべて一致するが、欠損rawの精度/score再計算は復元できない。

新通常37714793539→PBS217431/runtime `pd-37714793539-1.JA2mjieF8Z`、診断37714798160→217432/`pd-37714798160-1.RjCnUeShnd`。両実sidecar/provenanceでimage182fbb6d/size12231192576/buildsource9c5a5a3/sglangfc275674・条件commit・通常profilefalse/診断true一致を確認し、EC2/mirror/S3buildskipでnewbuild無し。両Q/aiq2・mem不足、013019pin、admission各8GPU112CPU1880GB/scatterexcl/wall30min。runtime/sourceの条件が異なる場合は比較しない。Lunaへexacthandlesを渡し、root方針でQ30分原則（ETA目前のみ短縮）/R60秒監視へ調整。Solの重複定期pollを止め、R/terminal通知で実kernel/公式raw/精度/hash/cleanupを監査する。


### 通常候補・別適用診断の完走監査（2026-10-08）

通常37714793539/PBS217431はActions success（04:15:42UTC）、F/Exit0、実P013/D019、wall12:00。正本52全hash、公開規則準拠repository parser再計算=保存score、同16physicalGPU UUID/image/source/model/config、4実argvの差がPの`--enable-symm-mem`のみ、8greedy×64token finite、runtime削除を確認した。公式入力/60秒warmup・120秒測定・30秒cooldownも一致し、monitor修正後のActions→PBS→公式結果→全raw回収→cleanupが完走した。

|通常条件|score out tok/s|good fraction|requests|mean TTFT ms|mean ITL ms|
|---|---:|---:|---:|---:|---:|
|control37642502611|2775.546377|0.991273997|1719|1402.964773|10.000491|
|P symm37714793539|2851.604172|0.992063492|1764|1268.981276|10.224663|

有効な単一pairではscore+2.740282%、TTFT-9.550026%、ITL+2.241616%、cancel/error0、TTFT>3sは15→14、ITL>20msは双方0。通常候補のP queueログ標本平均は0.68565、D accept len2.22214。標本は非時間加重であり、isolated NCCL kernel削減幅や因果帰属を表さない。欠損旧候補の+2.43%とprofile scoreは性能反復へ混ぜない。

8greedyの生成tokenは3/8一致、5/8の最初分岐は0-based17/14/21/39/25。canonical rawの同prefix max|ΔlogP|はprompt順0.37387/0.21768/0.22215/0.14166/0.49705/0.49346/0.00016011/0.05034。top_logprobsは全て空なので、最初分岐の同token候補順位/top2margin・全分布距離は未取得。別モデルで選択された異なるtokenのlogP差をmarginとみなさず、分岐後のlogP差も同入力の数値誤差とみなさない。NaN/invalidはなく、分岐だけでaccuracy劣化としない。一方、合成8promptは主催者accuracy評価ではなく、精度維持を断定しない。

診断37714798160/PBS217432もActions success（04:17:48UTC）、F/Exit0、wall11:35、同pair/16UUID。正本64全hash、公式parser再計算一致、8×64finite、4profile invocationとnode0/1 telemetry、PのみINFO/TUNING・D debug無し、同image/source/設定、runtime削除PASS。両P rank0で各9885件の`AllReduce [Symmetric]`選択を記録し、67108864 Bytesの`AllReduce_RSxLDMC_AGxSTMC`と8192 Bytesの`AllReduce_AGxLLMC_R`を確認した。INFOは選択/capture時のログであり、後続のbenchphaseでは8192-token `cuda graph: True`を確認。実SGLang fc275674のbreakable backendはcaptureしたShapeKeyのgraphをreplayし、同shape benchでそのcaptured symmetric kernelが使われることをこの範囲で裏付ける。実backendはbreakableでありfullではない。perkernel GPU時刻、isolated削減幅、全collectiveへの適用は証明していない。診断scoreは比較対象から除外する。

固定NCCL2.28.9-1一次sourceではBF16 small symmetric sumのaccumulatorはFP32、large multimem BF16 sumもCUDA≥12.2の`.acc::f32`で、出力BF16へ戻す。SGLang fc275674のpyncclはtensor.dtypeをそのままNCCLへ渡し、symm outputもempty_likeであり、新しい低精度castを追加していない。総和順/丸め変化と整合するが、個々の生成分岐の原因を唯一に特定せず、installed binaryの全disassemblyや分布全比較は未取得。[NCCL symmetric primitives](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/device/symmetric/primitives.cuh)、[all_reduce](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/device/symmetric/all_reduce.cuh)、[reduce_kernel](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/device/reduce_kernel.h)を根拠とする。

反復は通常instrumentation無しでcandidate→control順に各追跡commitをpushした。candidate`c65c22c38e5dbcd3352fdf36a2cf50596dfea174`/[37727083061](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37727083061)、control`67e2730d9040c9587114aab957c16152932c8324`/[37727107336](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37727107336)。同013019/image/settingsでP symmのみを変え、新build/profile無し。各25tests+9subtests/bash/Ruff/actionlint/diff PASS。push順からscheduler順を断定せず、PBS accountingの実開始順を記録し全raw/parser/finite/cleanupまで監査する。反復前のmain採用は行わない。

canonical rawはhome server `/srv/qwen-artifacts/RUN/1`、独立監査の派生証拠はlocal `/tmp/qwen-routing-audit-20261005/{RUN-audit.json,RUN-runtime-pair-audit.json,37714798160-application-audit.json,37714793539-canonical-prefix-diagnostic.json,nccl-normal-pair-comparison.json}`。一時diagnostic source取得物は一次source照合の証拠で、計算ノードへ注入していない。

反復control37727107336はPBS投入前のvalidateで失敗。最初fatalは`test_worker_argv_matches_archived_base_with_graph_candidate[aspire2ap-2-False]`で、実controlからsymm flagを除いたのに継承integration fixtureのP flag期待1行が残った。事前の選択25testsではこのintegrationを含めず検証範囲が不足していた。条件fixture1行のみ除去し、Actionsと同じ全repository tests/Ruff、bash/diff PASS後、`cc6cecb1e8c5c337eadadc41cb2a2e186fcc3c21`/[37727321620](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37727321620)を再push。image/source/serving変更無し、失敗runにPBS/GPU投入・raw損失無し、正常candidate37727083061は継続。実scheduler順は終端accountingで確認する。

反復確定来歴: candidate37727083061/PBS217513/runtime `pd-37727083061-1.5W8v4PPCxj`、control37727321620/PBS217516/runtime `pd-37727321620-1.UJNol23fHz`。qtimeは12:25:34→12:28:58SGTで投入順candidate→control、実開始順は未確定。双方Q/aiq2、013019各8GPU112CPU1880GB/scatterexcl/wall30min、通常profilefalse。runtime provenance/sidecarで各条件commit・image182fbb6d/size12231192576/source9c5a5a3/sglangfc275674を確認。selector1500gbに対しscheduler admissionは1880GBで、前pairと同じ。Luna単独Q30分原則/ETA前短縮/R60秒監視、Solは実R/terminalで監査再開する。

### 反復QのETAとActions期限（2026-10-08 read-only観測）

candidate217513はQのままPBS推定開始がOct13 13:19:23SGTへ後退、control217516はQでETAなし。ETAはschedulerの非保証推定である。workflowのself-hosted実行job timeoutは7200分、[GitHub公式上限](https://docs.github.com/en/actions/reference/limits)もjob5日。candidateの実PDjob開始Oct8 04:25:13UTCからの期限はOct13 04:25:13UTC/12:25:13SGTであり、現在ETAは54分10秒後。controlはOct8 04:28:43UTC開始→Oct13 12:28:43SGT期限。run全体createdAtではなく、待機するPDjob開始を基準にした。現在ETAのままならcandidateは開始前にActions上限へ達する可能性があるが、ETA変動を考慮し既正常Qを取り下げない。

pbsnodes全40DGXは各7–8GPU割当済で完全空き2nodeなし（39台8GPU/1台7GPU）。013はstatefree表示でも8GPU/64CPU/512GiB割当、019は8GPU/112CPU/1880GB割当であるため、stateだけから空きを判定しない。nodepinは選択を013/019へ制限し、全体混雑も観測される。他userの占有job詳細はaccount visibilityで取得できず終了見込みは不明。free候補/control追加には他pairの短backfillを拾う価値があるが即開始も同actualpairも保証しない。実pairが異なればhardware交絡として分離し、現在の同013019比較と直接poolしない。追加push前にrootへ状況・判断根拠を共有した。read-only証拠は`/tmp/qwen-routing-audit-20261005/20261008-{pinned-eta-and-nodes.txt,dgx-availability-summary.json,pinned-occupiers-and-scheduler.txt}`。

root判断でfree normal反復を2本だけ追加: candidate`7b5478ae4633e51a3c399bb615f3cedfdd314698`/[37741311522](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37741311522)を先、control`a6f12df00578ae55d189584fb0e0ae803cec53e8`/[37741366327](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37741366327)を後でpush。同image/source/serving/資源量/excl30min、候補P symmのみ、profilefalse/newbuild無し。各Actionsと同じ全repository tests/fixtures/Ruff＋actionlint/bash/diff PASS。既pin217513/217516は取消せず維持し、これ以上追加しない。全4本Luna単独Q30分監視、R/terminalイベントでSol監査。free両run実actualpairが同じ場合のみ反復根拠とし、異なる場合はhardware交絡を明記して採用を断定しない。投入順から実順序を断定せずPBS accountingで確認する。

free反復来歴確定: candidate37741311522→PBS217588/runtime `pd-37741311522-1.OeVlGj8Pcd`、control37741366327→PBS217589/runtime `pd-37741366327-1.ep3HTj6Yal`。両qtimeOct8 15:13:54SGT、Q/aiq2、select2×112CPU/8GPU/1880GB・host制約無し/scatterexcl/wall30min/profilefalse。各runtime provenance/sidecarで条件commit・同image182fbb6d/size12231192576/buildsource9c5a5a3/sglangfc275674を確認。candidate先pushだがPBS投入は同秒、実scheduler順とactual physical pairは未確定。pin217513/217516とfree217588/217589の計4本をLuna単独Q30分原則で監視し、イベント時Sol監査へ。これ以上追加しない。root checkoutのdirtyファイルは未変更、残留計算jobを勝手に取り下げていない。
