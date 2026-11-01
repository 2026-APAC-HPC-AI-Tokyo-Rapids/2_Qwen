# Round 15：native FA3 の V load 先行化 P2 と gate 整理

2026-11-01。基点 main `51b77d8`、作業 branch `perf/pp4-r15-fa3vload-20261101`。**P2 は CPU admission 合格、GPU 採用は未判定**。`prefill.fa3_prepack_gate=true` を維持し、新しい `prefill.fa3_vload_gate=false` で渡す。image pin は round 14 の `5a3df8c485a5ac939761b07a9ca82d7c38d864200250591ec6d9485dba778bb2` のまま。今回の table/source 変更には fresh image build が必要であり、pin の image で新しい P2 が動くとは主張しない。

## 根拠と変更範囲

AGENTS.md、round 14 の結果・prepack 設計・CPU JSON、astra2/report.md §3.3、theory tracker/model と参照 profile JSON、round 13 V/sum、round 7 timeout、指定の FA3/pack/writer/dispatch/gate/sbatch/prepare/config、展開済み r6int-fa3 の load()、git log 60 件と 85636aa・5f3164d・051f427・e3e7744 の本文を読んだ。過去の会話は保存済み r14/exec.log と final.md、research に残る経緯を確認し、新たな非公開会話を取得したとは主張しない。過去 Actions は保存済み round 14 note と gate-41264009819 の12 rank raw JSONを根拠にした。SSH、Actions操作、scheduler投入、GPU実行は行わない。

round 14 / 41264009819 の公式10009.54は単回で、10kとの差約0.1%は既知の約0.15%のrun差より小さい。同runの分解測定は native consumer が−34.0 µs、sum が＋39.4 µs、散在native writer が＋17.4 µs、旧writer＋pack が＋5.2 µs。S/VSを再試験する理由はなく、Pのnative consumerで後ろへ移ったV loadだけを進める。

`prefill-vload.patch` を `prefill-vsum.patch` の直後に適用し、bit65536を追加した。P2は **variant106720 = 41184 | 65536、table ID17**、row `m192n96-direct-o-varlen-pipe-native-vload`。既存ID0〜16は位置・variantを維持する。SのID16、VSのID14、既にretiredのID13はnull dispatch/resource0の予約枠にし、TUを生成しない。sumのtemplateコードは過去patch中に残るが、sum bitのactive行はなくbuildされない。

pack op、旧FP32 writer、native byte map、consumerの算術・mask・FP8変換・Oは変更していない。native writerはtiming/servingには使わず、round 14 と同じ独立byte oracleとしてだけ残す。別のcoalesced native writer案、komoriyutaのbranch、mirrorは変更していない。SGLang mirrorからの `git archive fc27567` は読み取りのみで、自分のscratchへ展開した。

## shape・仕事量・順序

対象はexactly2要求、各3900〜4096 token、Q64/KV4/D128、GQA16、causal M192/N96、FP8 E4M3、FP32 accumulate。基準3994＋3994の有効QK＋PVは522,847,416,320 FLOP、tile padding込み535,881,056,256 FLOP。全causal項とVの全FP8 byteを保持する。**Pに対してFLOP、logical global/shared byte、pack/launch数は不変**。Vのlogical global tile load697,761,792 B、K＋V1,395,523,584 B、packのread4,089,856 B/write4,128,768 Bも同じ。実HBM/NVLink byte、cache hit率、V待ち時間、clockは未測定でnullとする。

| producer load() | P、native41184 | P2 |
|---|---|---|
| prologue | K(n) acquire/load → Q、V(n)なし | V(n) acquire/bulk → K(n) acquire/load → Q |
| loop | K(next) acquire/load → V(previous) acquire/bulk | V(next) acquire/bulk → K(next) acquire/load |
| tail | 最後のVをload | 既にload済みなので追加Vなし |
| loop unroll | 2 | 2（controlの1へは変えない） |
| V buffer/stage | native imageをVの2 slotへ直接読む | 同じ2 slot、stage2 |

controlはVtの2 slotへ先読みしてから別のVの2 slotへ転置する。P2はcontrolの発行順に近づけるが、Vt bufferを復活させない。発行を早めて減らす対象は露出したV待ちであり、transposeの追加削減や新しい帯域削減ではない。

## barrier契約と進捗

各load_Vは既存の `pipeline_v.producer_acquire(state)` を必ず通す。nativeではPipelineTmaAsyncであり、acquireは同slotのempty barrierをphase付きでwaitし、leaderがfull barrierへ12,288 Bの `arrive_and_expect_tx` を置く。bulk copyは同じfull barrierを `mbarrier::complete_tx::bytes` で完了させる。consumerは同stateのfull wait → PV WGMMA → `warpgroup_wait<0>()` → `pipeline_v.consumer_release`。全384 math threadの既存release契約、phase、state増分、最終PVのrelease、producer tailを変えない。store先は `smem_v + slot*12288`。PとP2はsmem100,352 B、regs128で同じ。

producer順のblock番号をiとするとslotはi mod 2。V(i)のacquireはV(i−2)のPV完了/releaseを待つため、使用中slotを上書きできない。進捗にも循環待ちは増えない。PV(i−2)をreleaseするstepが必要とするKはK(i−1)までであり、V(i)より前に既に発行済みである。tile境界も最後の旧QKでQueryEmptyが到着し、旧末尾PVは次tileの最初のQKと重なる。新tileの最初のVが再利用するのは旧tileの末尾から2個目のslotであり、そのreleaseに新tileのK/Qは要らない。次のVの再利用に必要な新tileの最初のK/Qも、その時点では発行済みになる。block1個のtile、odd/even切替、最後のtileも既存最終PV releaseを保持する。

補助CPU依存DAGで隣接tileのblock数1〜43を全組、1/2/42/43 blockの4tile組、最終1/43 blockの計2107 case・420436 eventを検査し、循環なしとslot再利用前のPV完了を確認した。これはsource/barrier契約を検討する補助模型であり、GPU race/hangの証明には代えない。実GPUの全7pair、変更入力、1200回耐久が必須である。

## cache hint と CPU/SASS admission

固定CUTLASSの `SM90_BULK_COPY_G2S::copy` はcache policy引数を持たない。一方PTXのbulk G2SにはL2 cache hint形式があるため、P2だけ同じshared::cluster/global/complete_tx::bytesに `.L2::cache_hint` と `TMA::CacheHintSm90::EVICT_LAST` を渡す。値はcontrolと同じ0x14f0000000000000。scope、payload、barrier、orderingは元のCuTe呼出しと同じで、memory clobberも保持する。Pのhintなし命令は変更しない。[NVIDIA PTX cp.async.bulk仕様](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)。hintはcache挙動の指定であり、hit率や短縮を保証しない。順序とhintを同じP2に含めるため、GPU差を順序単独の寄与とは呼ばない。

**CPUはCUDA13.4.92 / sm_90aの代替証拠。実imageのCUDA13.0.88のcubinではない。** GPUの正しさ・速度・通常servingを保証しない。FA3/CUTLASS source archiveはbuild.pyのpinとSHAを検査し、全16patchを順に適用した。

| row | registers | spill store/load B | smem B | round14 encoded SASS |
|---|---:|---:|---:|---|
| C M192/N96、8416 | 128 | 40/28 | 124,928 | 一致 |
| single M128/N192、224 | 168 | 84/116 | 166,912 | 一致 |
| P、41184 | 128 | 0/0 | 100,352 | 一致 |
| P2、106720 | 128 | 0/0 | 100,352 | 新行 |
| old FP32 writer i32/i64 | 各48 | 各0/0 | 0 | 両方一致 |
| pack | 35 | 0/0 | 24,576 | 一致 |

全compileでC75xxなし。encoded比較は命令wordとscheduling wordの両方を全て含む。旧writer headerもround14とSHA一致し、現overlayをcompileしてi32/i64のSASS一致を確認した。resource値はfresh FA3 objectと保存済み同sourceのscheduler objectをlinkし、resource pointer分岐だけをCPUで呼んだ値で、CUDA device初期化・launchはしない。

P→P2の静的命令は4424→4408（−16）、padding NOPは双方28。bulk4 siteとtensor TMA4 siteは各々不変で、source上の動的load数も不変。QGMMA、MUFU.EX2、FADD/FFMA/FMUL/FMNMX、F2FP、FSELの1814命令はencoded word＋scheduling wordが順番も含め一致。producerの整数除算用MUFU.RCPは別に扱う。opcodeの増減全体はCPU JSONに保存し、静的−16を動的命令削減率や時間へ換算しない。

| P2の展開領域 | V empty acquire | V bulk | K empty acquire | K tensor load | V→K acquire / load の命令距離 |
|---|---|---|---|---|---|
| prologue | daf0 | dca0 | dcb0 | dde0 | 1 / 20 |
| loop展開1 | eb70 | edc0 | ede0 | ef60 | 2 / 26 |
| loop展開2 | f1f0 | f400 | f420 | f5b0 | 2 / 27 |
| 残余loop | f800 | fa20 | fa40 | fbc0 | 2 / 26 |

PはprologueのK acquire dac0/load dc30の後、loopでもK next acquire e9d0/load eb40の後にV previous bulk eda0。次の展開はf070/f1e0→f410、残余はf660/f7e0→fa10、最後のV bulkはfdf0。P2では4つのbulk全てがdescriptor付き `UBLKCP.S.G ... desc[...]` になり、0x14f00000のpolicy high wordを確認した。表の距離はSASS上の静的発行位置であり、stall cycleやruntime latencyではない。

## gateの整理と採用規則

削除：writer-only、固定入力FA3-only C/S/V/VS、合成診断C/S/V/VS/P、O-only、interaction、診断専用のnumerics出力。採用候補はPとP2だけ。CはPの完全gateの比較対象に残す。native writerはbyte oracle専用。S/VSのenv routeは削除し、table ID13/14/16をretiredにしてbuild/直接dispatchから拒否する。

全12 H200を3 worker×4 GPUで並行し、各GPU内はP完全gate→optional P2完全gateの順。P不合格のrankではP2へ進まない。新boolean falseならP2用output/graphを作らず、Pだけ検証する。cold import、入力、weight、timing/validation graphを同一群内で共有する。各候補開始時には45秒の完全検証予約を要求し、時間不足でshapeや耐久を短縮しない。

- 全7pair：3994+3994、3986+4027、3986+3986、4027+4027、4096+4096、3900+4096、4096+3901。
- Pはround12 Cに対し、writer＋pack＋FA3＋Oの正順・逆順・pooledの全中央値で各5 µs以上短縮する。P2は**同じ時点で再計時したPに対して**同じ条件を求める。Pの過去runや今回P gateのC時間から差し引かない。10 sample/order、12 call/sample、両順を交互に実行する。
- byte imageは旧V→独立forward map、実LDSM/PRMT/STSM dump、pack、native writerで照合し、Q/K/V/paged KVを数値より先に検査する。FP32 causal参照は既存の各request9代表Q位置と、その全causal keyを使用し、relative L2≤0.05、max scaled error≤0.25。原入力＋変更入力2回をcaptured compositeで検証する。
- P2は全pair・全replayでPのattention FP8、FP32 scale、BF16 O出力とbyte一致も必須。不一致は候補拒否し、理由をfragmentに残す。CPU故障注入でattention/scale/Oの各byteを原入力・両変更入力で壊し、この拒否を確認した。
- singles3994/4096/3584は旧N192行・旧writer、producerとattention/scaleのbyte一致・FP32・変更入力を要求し、速度判定なし。範囲外・fixed外も旧routeを保持するdispatch/model実行fixtureをP/P2双方に適用する。
- 1200回は実serving dispatchでwriter→pack→attention→Oをeager連続実行し、初回direct dispatchと一致、最終attention/scale/Oと初回byte一致、初回/最終FP32参照を要求する。

集約はschema v2で、旧v1を受け付けない。同じpair gate成功allocation・worker/rank/UUID、12 distinct H200、run ID、image/binary hash、gate boolean、期限、全shape/single/enduranceを検査する。Pが全GPUで合格して初めて採用可能。P2も全GPUで合格ならP2、PだけならP、P不合格ならround12の選択。GPUごとの混在採用やP2によるP不合格の救済はしない。候補の数値・速度・byte拒否はPへ戻れるが、CUDA RuntimeError/process異常/hangは群全体拒否とする。

envのFA3行はround12の値を確認した上で、全結果が揃ったときだけ同一filesystemのatomic renameで置換する。timeoutはworker段階・集約段階の両方をshell fixtureで模擬し、envをbyte不変に保ってservingの続行markerまで到達した。群がtimeoutしたときの「直前の選択」は先行pair gateのround12であり、途中経過のPをpublication済みと扱わない。既存shellが起動した群の子process以外を停止するコードは追加しない。

## 時間上限と17分枠

raw round14はrank elapsed35.176〜36.313秒、群開始から各rank処理開始まで16.247〜32.600秒、最後のfragment完了まで68.813秒。rank時間を群wallと混同しない。

設計見積もりは起動32.60＋P36.32＋P2追加25＋終了/集約5＝**98.92秒、100秒に丸める**。P側には診断削除前の36.32秒を丸ごと残し、診断削除による短縮を過大に差し引かない。P2の25秒は既存入力/weight/P graph再利用下での追加予算という条件付き見積もりで、H200実測ではない。7pairのtimingは候補1つ当たり3360 call、約1.2 ms/callなら4.03秒、1200回耐久はGPU約1.44秒。削除した診断のcapture/初回入力/FA3/VS/Sの準備費を再度負担しないが、残り約19.5秒は新P2のcapture・再参照・byte照合・single等への配分である。45秒の候補予約は25秒見積もりに余裕を持たせたもので、測定時間の短縮はしない。

**srun timeout150秒 = 1.5×100秒、180秒以下**。全workerに群開始＋150秒までの残り秒を渡し、workerの起動ずれで期限を延ばさない。kill grace2秒、集約も残りwallでtimeoutし、群開始＋155秒をpublication期限とする。先行round12 pair gateのtimeout150/総155秒は保持する。

round14 job12:22＝742秒、READY449秒には旧gate群が既に含まれる。68.813秒は最後のfragmentまでのwall下限なので、それだけを差し引く保守的置換を使う。

| 予算 | 計算 | 合計 | 1020秒までの余裕 |
|---|---|---:|---:|
| 通常のgate置換 | 742−68.813＋155 | 828.187秒＝13:48.2 | 191.813秒 |
| 先行pairも最大枠を追加計上 | 742−68.813＋155＋155 | **983.187秒＝16:23.2** | **36.813秒** |

後者は旧pair wallを一切差し引かず、さらに全155秒を足すので二重計上を含む保守的上限配分。READY側も449−68.813＋155＋155＝690.187秒、以降742−449＝293秒を保持して983.187秒となる。60秒warmup・120秒公式測定を削らない。buildはallocation外。別runのclock/timelineを合成した速度予測ではなく、今回のH200 wallやcold import/model staging変動を保証しない。round7のserial gate chainは再導入しない。

変更ファイルはFA3の `prefill-vload.patch`・`build.py`・`prefill-candidates.json`、SGLangの `prefill-fa3-vsum.patch`、selector・prepare・provision、Firmusの `sglang.sbatch`、PP4 config、`test_prefill_fa3_prepack.py`・`test_fa3_prefill_direct_o.py`、本note・CPU JSON・tracker/model。provisionはstagingの古いP/SコメントをP/P2へ更新した。

## Claudeのserving commit

このbranchで `configs/firmus-prefill-pp4.json` の **`prefill.fa3_vload_gate` だけfalse→true**。`fa3_prepack_gate=true`、`fa3_pair_gate=true`、image pin、通常設定は維持し、envを手でP2へ固定しない。新行とretired build/table変更があるためfresh image buildが必要。serving commitの先頭に既存のrequire-build・Firmus・2node・prefill-pp4タグを付け、skip-ciは付けない。本文には角括弧を含めない。provisionがbuildの実hashでruntime pinを更新する経路を使う。

後続ではimage CUDA13.0.88のcompile/SASS、terminal状態、12 GPUのfragment/selection、served env、通常servingの公式eligible/score、artifact回収を確認して台帳へ追記する。P2が拒否されたときはPが再検証済みならPを採る。

## 条件付き見込み・未確認

待ち削減の正の実測はまだない。仮にP比5〜15 µs/callが全workerの通常servingへ同じだけ転写すれば、24層stageで0.12〜0.36 ms/周期。round14の114.5 msを基準とした供給律速・good fraction固定の感度計算なら約＋0.10〜0.31%、10009.54から約10020〜10041 tok/sの規模に相当する。予測区間でも10k維持の保証でもない。V先行acquireがKの供給を逆に遅らせる可能性、cache競合、電力/clock、Decode供給、run差で利得が消える可能性が残る。FLOP/byte削減の理論床から短縮µsを捏造しない。

GPU byte一致、全12GPU×7shape速度、single、耐久、実13.0.88、公式serving scoreは未確認。CPU合格だけで採用または10k確実とはしない。

## 検証・証拠・片付け

CPU正本は `fa3-prefill-vload-cpu-validation-20261101.json`。raw command、ptxas/SASS、展開header、resource、source overlay、barrier modelは指定r15 scratchのCPU archiveへ保存する。最終全体suiteはPython3.12 / uv no-project、指定のpytest・pyyaml・numpy・rfc8785・editable構成で **1386 passed、90 skipped、25 subtests passed、544.41秒、exit0**。AF_UNIXの3fixtureだけ同じ許可scratchを `/proc/self/cwd` の短いaliasで参照した。Torch2.11.0追加環境のP/P2・pair・pipe・varlen契約は **206 passed、10.09秒**。direct-O/reference属性契約3件も合格。初回全体suiteはbitwise_referenceの行名を4件に固定した旧fixtureだけ失敗（他1385 passed）し、P2の明示的なbit/geometry契約を追加して上記全体suiteを再完走した。productionやSASSの変更で回避していない。

23 build patch＋31 runtime overlayの順序適用、実PP4 CPU source checks、prepareのP2 false/true、独立barrier DAG、resource pointer、旧writer/pack/controlsの再compile、Ruff、両shellの構文、diff検査も合格。image本体のbuild/provision/GPU/公式parser E2Eは未実行である。

CPU raw archiveは指定scratchの `r15/cpu-sass-proof.tar.gz`、1,702,261 bytes、SHA256 **ce2afd4f323b304d5ff91306f2287ba9bdb2a5c7c822b5eea03ade33ddb06b0c**。内部48 entryのmanifest/hashを検査した。不要なsourceコピー・source archive・object・生成TU・テストdirectory・socket alias plugin・repositoryのPython/pytest/Ruff cacheは削除した。archive、最終/初回suite log、Torch log、元のexec.log、提供されたround14 gate raw dataは残す。今回起動したSSH・Actions・scheduler/GPU jobはなく、今回作成した残留PJM/Slurm jobもない。他の作業のjobやbranchには触れていない。

最後のstaged diff検査で新規patchの空context行の末尾空白を検出し、除去した。GNU patchのfuzz0適用後headerは保存済みCPU compile sourceとbyte一致し、SASS証拠は不変。最終のP/P2・pair・pipe・varlen・direct-O契約はTorch2.11.0環境で209 passed、6.99秒。関連一時物も再削除した。
