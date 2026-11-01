# round 18：現行 split MegaMoE の診断専用 in-job 計装

目的は追加約3.4%の改善案を選ぶための不確実性の測定である。性能改善の提案・採用ではない。基準は main beb5640 の P：旧 FP32 writer → V prepack → native-only FA3、公式10008.67〜10020.58の3 run。全workerで2.01〜2.15 ms/周期、84〜90 µs/層を削る必要があるという astra §2/4/5の容量模型と、現在の内部実測がないことを出発点とする。

実装時に AGENTS、astra3/report、tracker/modelとそのhistorical profile JSON、EP1 accounting、round4 split、round5 LDSM、round13-M、旧EP4 timeline、round14/17、80件の履歴と関連commit本文を確認した。過去のActionsは保存済みnote/raw解析記録を参照した。SSH、Actions操作、scheduler/GPU投入はしていない。

## 測る問いと入力

1. splitの両half・chunk境界でscale準備とaccumulator待ちが露出しているか。
2. FC2 epilogue、tile終了からgrid待ち、combineが終了経路を占めるか。
3. expert tailのあるCTAが最後のmath・warp終了を決めるか。

T=7988=2×3994、EP1、H4096/I1536、128 experts/top8、FP8積・FP32 scale/accumulate・BF16 partial・FP32 combine。有効積は2,412,295,225,344 FLOP/層を維持する。診断は計算量・scale・K順序を省略しない。HBM/NVLinkの実byte、露出時間、削減可能な仕事は未確定であり、ゼロや既測定の別run値で埋めない。EP1なのでexpert間のNVLink転送を測る試験ではない。

balancedはtop8が重複しない499/500行の分布。historical-routingは既存 `megamoe-local-ep1-routing.npz` のsample2つ、layer index0/rank0の各4001行を各3994行に切り詰めて連結する。edgesはexpert0が空、expert1..6を1/64/65/127/128/129行に固定し、残りをdistinct top8で埋める。全GPUで同じ入力seedを使い、出力をcontrolとbyte比較する。

**fixtureは現在のPのroutingではない。true M64の最終的な速度上限は確定しない。** 今回serving routingの新規captureは足していない。既存capture/parserは94層・physical140・EP4の完全標本とprofile後半のactive markerを要求する。PP4/EP1のstage別24/24/24/22層、128 expert、最初のmicrobatchを同じABIで扱えず、通常bench窓の外での境界・コピー完了を保証するにはscheduler側の変更も要る。今回はserving経路を保ち、古いcaptureをcurrentと呼ばない。別のprofile passを追加して17分枠を広げることもしない。

## 隔離と記録点

通常header、通常overlay inventory、通常dispatchの生成CUDA textを維持する。host seriesの専用patchはデフォルトfalseのC++ templateと明示的なTVM-FFI `fp8_mega_moe_diag` APIを追加する。通常APIはfalseで、環境変数から診断を選ぶ経路はない。診断APIだけがincludeとkernel symbolを `_diag` へ変える。EP1/T7988/split以外は拒否し、専用tailの確保もhost側で確認する。

image build中、baked headerにPP4の13 overlayを適用した隔離treeから、診断patchと `trace.cuh` で別headerを生成する。packageに追加するのは診断header2個だけで、元のbaked headerは書き換えない。通常servingにはこれまで通りprovision時にoverlayが適用される。source mirrorは更新しない。

旧 `patches/deep-gemm-timeline` とbenchの設計から、combine buffer後方のpage-aligned tail、globaltimer、独立WGの考え方を再利用した。旧ABIを変更せずprivate ABIとする。12 warp×132 CTA、各warpに32-word headerと最大2048件の8-word eventを割り当てる。追加allocationは208,027,648 byte（正確には4096+132×12×16416×8）。overflowは欠損扱いとし、切り捨てたtimelineを解析しない。

全warpのlane0でentry/exit、SM ID、各math WG leaderで全tileのphase/valid rows/expert/pool block/N blockとmainloop開始・終了・epilogue終了を記録する。padding WGの早期returnも終了を残す。各WGのFC1/FC2それぞれ最初のfull tileと最初のtail tileだけを詳細標本にし、両halfのwait前後・rescale終了・issue終了、各chunk開始・drain前後、最初と次Kのscale準備前後、内包full待ち前後を記録する。full/empty stage待ちは既存waitを包み、phase別の累積nsと回数を残す。TMA A/Bとmathのroleを混ぜない。

tile列終了後、precombine grid/self-signal待ち開始・終了、既存dispatch handshake後のcombine開始、各warp終了を記録する。新しい同期・精度変更・no-rescaleは追加しない。異なるSMのclock64は使用せず、全stampはglobaltimer nsである。warp leaderのexit最大はCTA終了の観測proxyであり、最後の非leader命令や非同期store完了までを厳密に測るものではない。完全kernel時間は別のCUDA eventを正本にする。旧timelineで観測されたnamed barrierのDEFER問題もあるため、handshakeを単独の正確な同期完了点と見なさない。

## 計装影響とCPU証拠

CUDA13.4.92、sm_90a、現行PP4 host template容量8448、4stage/lean/splitでCPUコンパイルした。imageのCUDAは13.0.88であり、これは代替証拠である。完全ELF一致やGPU数値一致を主張しない。機械可読証拠は `megamoe-prefill-diag-r18-cpu-validation-20261101.json`。

| 実体 | SASS encoding byte | launch表示register / 最大参照R | spill store/load byte | QGMMA / DEPBAR | C75xx |
|---|---:|---:|---:|---:|---|
| main相当serving control | 157696 | 168 / 206 | 0/0 | 160 / 40 | なし |
| 診断追加後の通常kernel | 157696 | 168 / 206 | 0/0 | 160 / 40 | なし |
| 診断kernel | 356864 | 168 / 221 | 28/44 | 160 / 40 | なし |

static命令数は9856→22304、globaltimer readは0→338、LDL/STLは0/0→11/7。全static DEPBARから次QGMMAまでの命令span中央値は73→167（最後のDEPBARは次QGMMAがないため除外）で、全spanを証拠JSONに残した。phase間を含む静的配置の比較であり、K blockの動的cycleや削減可能時間ではない。stampの分岐・storeがwait後から次の発行までを広げていることが分かり、計装scheduleの影響を無視できない。

通常SASS encoding SHA256は前後とも `d582cf2059b10cd17523ecf1bcb70135ca5ca7174defa29caf9ad8345f6015ad`。round13-Mのmain相当controlとも同じであり、157696 byte全体を比較した。表示168とmath roleのsetmaxnreg224を区別する。診断ではstatic codeが増え、register/spill/scheduleが変わる。WGMMAとDEPBARの本数が同じでも実schedule不変の証明ではない。

通常と診断を同じbuffer・入力で2回ずつwarmupし、C,D / D,Cの順序を交互に6 pair測定する。pre-dispatchはkernel計時外、CUDA eventは完全MegaMoEのみを測る。全pairでBF16出力のuint8 viewを照合し、finiteも確認する。別の1 callからraw traceを取得する。これをGPUごとの3 routingで行い、順序別・pooled・pair別の計装差を保存する。

**現在のCPU証拠にはspillがあるため、この診断版の細分値を改善の採否へ直結させない。** 集計は影響絶対値2%超、spill、C75xxのいずれかがあればfine_intervals_usable_for_adoptionをfalseにする。2%は診断の保守的な警告線で、性能候補の採用marginではない。総時間差が小さくても局所区間が無影響とは言えない。大きな区間が見えた場合も、削除可能な全量や90 µsの利益とは扱わず、まずstampをさらに限定した再計装で再確認する。

## in-job wiringと17分枠

config `prefill.megamoe_diag=false` でcommitする。既存のpair/P/S gate、7 pair shape、12 GPU、byte oracle、FP32、single、1200回はmainのまま。P3等は含めない。

Pの集約・env publicationが完了した直後に、headの2組とtailの1組を各srun/torchrun4 GPUで一度だけ同時起動する。結果は `results/prefill-megamoe-diag/prefill-W/rank-R.json`、logは既存logs、集計は同directoryのsummary.json。採用envやP selection JSONへの書込み経路はない。診断scriptとfixtureをprovisionし、imageに追加した専用API/header/build recordを利用する。

独立stepの起動込み最大45秒、kill猶予2秒、集約最大3秒。それに加えて**mainのprepack群開始から180秒という既存envelopeの残り**に収める。worker起動時に終了猶予と集約の5秒を予約し、残りでtimeoutを短くする。枠が尽きた場合やprepack群無効時はstatus.jsonに未完了を記録する。計装timeout/失敗/集約失敗でも、確定済みPの選択を変えずservingへ進む。GPUの再選択は行わない。

round14のjob742秒・READY449秒、repeat732秒・READY444秒、round17 READY434秒、gate rank約34〜36秒を区別する。旧group wall68.813秒には起動最大32.600秒が含まれ、rank時間をwallとして扱わない。buildはallocation外。

今回はmainの175/180秒上限を変更しない。独立診断をそのenvelopeに共有する保守的な配分は、round17と同様に旧pair実wallを差し引かず先行pair最大155秒を丸ごと加えて、

`742 - 68.813 + 180 + 155 = 1008.187秒 = 16分48.2秒`

となり、17分まで約11.8秒である。診断50秒をさらに足さない。READY側は `449 - 68.813 + 180 + 155 = 715.187秒`、後段293秒を保持する。これは条件付き予算で、cold start/staging変動を保証しない。READY_SECS=600はworker起動後の待ち期限なので、先行gate時間をこの600秒と混同しない。60秒warmup・120秒公式窓は削らない。Pの既存180秒上限が全部使われた場合、今回の診断は未完了になり得ることを許容する。

## 集計と次の意思決定

`scripts/summarize-prefill-megamoe-diag.py` は同じrunの12 UUIDと3 routing、完全なtile端点、単調なns、交互計時、byte一致を確認する。欠損/失敗はcomplete=false、adoption=nullであり、候補の採否を出さない。起動中の3秒集約が間に合わなければ、回収済みJSONに対して同じscriptを後から実行できる。

出力はphase別CTA/WG合計の分布、mainloop内のfull待ち割合、producer別empty待ち、half別wait/rescale/issue、chunk/drain、scale準備とその内包full待ち、FC2 epilogue、grid/handshake/combine、最後のmath tileと全warp終了の関係を含む。中央値を足してkernel総時間を作らない。stage待ちやsampled区間をmainloopへ加算しない。tailとfullの終了差・tailから最終warp終了までの距離・最後のCTAへの一致は関連の観測であり、反実仮想のM64利益ではない。inactive WGの訪問数とactive行数も区別する。

影響が小さい追加計装で依存待ちが大きいと確認できれば、half/chunkを跨ぐ依存案を再設計する。scale準備の露出が小さければ、round13-Mの単なる先読み移動は再提出しない。FC2 epilogueやtile-to-gridが大きい場合は、coalesced store/combineのlayout契約とwork分配を先に検討し、旧direct-FC2のstore要求倍増を避ける根拠を要求する。tailが最後のmathを決めず、全warp終了との隔たりも小さくない場合、true M64を主力としない。tailが最終CTAに一致してもcurrent routing取得と専用full-tile不変の設計ができるまで上限やspeedupを確定しない。見えた対象が20 µs程度なら、その部分から90 µsを取る案は閉じるが、spillを含む今回の細分値だけではその棄却も行わない。

## Claude側のserving commitと未確認事項

新API/headerを含むimage buildが必要。image pinと既存 `container_*` はこのbranchでは変更しない。Claudeが変更するruntime箇所は `prefill.megamoe_diag` をtrueにする1箇所で、既存PとMegaMoEのpinは維持する。起動commit先頭のrequire-build/clusterタグは既存運用に従う。今回のskip-ci commitだけを再利用して古いimageで診断を実行しない。

未確認はCUDA13.0.88の実build、TVM-FFI ABIロード、H200でのbyte一致・完全時間・計装影響、45秒内の完了、12 GPUのraw JSON、現在routing、通常serving公式結果である。CPUの通常SASS不変と、診断のGPU数値合格を混同しない。

## ローカル検証と片付け

全体suiteは指定のPython3.12/pytest/pyyaml/numpy/rfc8785/editable構成で **1415 passed、78 skipped、25 subtests passed、exit 0**（1493件収集）。関連fixture群は223 passed/10 skipped。通常host 230条件と明示diag→通常のJIT text復帰、非7988拒否、CUDA3コンパイル、変更PythonのRuff、shell構文、差分のwhitespace検査が通った。正規化したpatchを再適用し、コンパイル済み診断headerとのSHA256一致も確認した。

初回suiteで見つけたhost seriesの末尾とprepack shell抽出境界のテストは、新しい診断追加に合わせて修正した。最終suiteの一時ファイルは指定scratch内に置いた。長いAF_UNIX名を避けるため3テストだけ同じ物理scratchを指す短いdirectory-fd pathを使い、product/test sourceの動作は変更していない。全fixtureへの短縮を試した中間実行は、provenanceの文字列比較と衝突したため打ち切り、この限定方式で全体を完走した。

task所有の展開source、作業header、native checker中間binary、pytest runtime、不要なprototype生成scriptと初期の外部pytest temporary directoryは削除する。最終compiler/SASS/cubin、検証log、最小のscratch再実行補助だけを保持する。clusterジョブを作っていないので、この作業由来の残留Slurm/PJMジョブはない。
