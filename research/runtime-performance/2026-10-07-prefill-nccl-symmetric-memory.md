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
