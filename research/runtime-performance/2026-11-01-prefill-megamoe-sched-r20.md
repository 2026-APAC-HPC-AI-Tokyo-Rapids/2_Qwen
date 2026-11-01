# Round 20：Prefill MegaMoE FC1 の expert 順貪欲割当

CPU 事前条件は合格。L は GPU の採用ゲート待ちで、serving には未採用。`megamoe_fc1_balance_gate`、`megamoe_timing`、`routing_capture`、旧 `megamoe_diag` は全て false で保存する。P の pair/prepack gate は true のまま。image pin と対応する来歴を main の round14 image、SHA256 `5a3df8c485a5ac939761b07a9ca82d7c38d864200250591ec6d9485dba778bb2` に戻した。L を試すには fresh image build が必要である。

今回 SSH、Actions の参照・操作、scheduler 投入、GPU 実行はしていない。Actions の過去の経過は保存済み research、round18/19 の結果 JSON、commit 本文、round18 Codex final の記録から確認した。組織 SGLang source は指定 commit の読み取りと scratch への複製・patch 検査だけに用い、更新していない。

## Astra 分析の要点と今回解く不確実性

消える scratch に依存しないよう、提供された astra-routing-skew の判断材料をここに残す。

- 現行経路は PP4/EP1、T7988、H4096/I1536、128 expert、top8、M128/N256/K128、4 stage、2 math WG、split accumulator、tail-pair 無効、132 CTA。round19 の通常 header は SHA256 `fbfaa6be207b26dd7127aae406204b44668b4341ff5ce9e2928510c932e57852`。
- 有効 routed row は 63,904、GEMM の有効仕事は 2.412295225344 TFLOP/層。balanced/historical/edges の M128 block は 512/571/565、half tail は 0/75/72。half-WG skip 後の実行 MMA は balanced 比 historical +4.20%、edges +3.32%。historical の追加 padding は完全時間を配賦した単価で約98–99 µs相当だが、純粋な MMA latency の測定ではない。
- FC1 番号は `12*(sum(B[:expert])+m)+n`、CTA c は c,c+132,…。MMA量の最大/平均は 1.010/1.031/1.040。FC2 は A loader の atomic counter 取得で、固定 fixture だけから CTA 担当を確定できない。FC1 の不均衡を FC2 が吸収すれば完全時間の短縮は 0 もあり得る。
- 0.5 重みの M block LPT では historical/edges の FC1 最大が50→48.5、expert順貪欲なら49/48.5。ただし LPT は初期 expert の小さい tail を後回しにし、FC2 readiness を悪化させ得る。half tile は MMA 半分でも A load、stage、barrier、scale 準備が残る。旧非split条件の0.64–0.70という時間比を、今回の実測値とは扱わない。
- round19 の2334–2363/2464–2500/2509–2546 µs は prefill-2 の4 GPUだけで、routing順も固定。計装版は約4.5倍遅く、細分値は採否に使えない。historical は現 serving routing ではない。
- 次に解く対象は、順序効果を除いた12 GPUの完全時間、現在の serving routing、そして GPU 割当計算込みで L が利益を残すかである。別runの clock や timeline を合成しない。

## L の設計と FC2 readiness

`patches/deep-gemm-sched/expert-greedy.patch` は、通常の最終 PP4 header から別名 header/symbol `_sched` を作る専用 patch。通常 overlay に混ぜず、通常 header を変更しない。変更は FC1 の task 列と kernel symbol のみで、FC1/FC2 mainloop、scale、BF16 partial、top-k combine の算術・順序は source 上そのまま。

expert 昇順、各 expert の M block 昇順で、12 個の N tile を一組にし、現在の累積コストが最小の11群へ割り当てる。同コストなら群番号が小さいものを選ぶ。CTA は `group=c/12, n=c%12`。full は整数10、valid M≤64 は7。0.5だけに最適化すると固定費を無視するため0.7を選んだが、この重みが現splitの実費と一致するという主張はしない。

各 warp の lane0–10 が群負荷を1整数ずつ持ち、`__reduce_min_sync` で勝者を決める。A/B loader と全 math warp は、それぞれ既存の GPU expert counter の snapshot から同じ決定を再現する。host 計画、D2H、追加 launch、global metadata、追加 global barrier はない。毎層の expert/M walk と整数 reduction は kernel 内にあり、完全時間の event span と CUDA graph replay に含まれる。最大571 blockの判断を各 role が繰り返す費用と、同じNを担当することによる cache の変化は未測定。

FC2 の source は、poolごとの L2 arrival counter が `N tile数 × active M WG数 × warp arrival parts` に達するまで A loader が acquire 待ちし、2WGでは続いて `fence.proxy.async.global` を実行する。math warp の FC1 stores は次 FC1 tile の K block2、または最初のFC2取得前に publish される。このため full-first の全体ソートは readiness を遅らせ得る。L は expert/M の元の相対順を各群で維持し、初期tailを後続expertのfullより後ろへ移さない。ただし群ごとの進行速度、最後の遅延publish、FC2の動的取得は残るので、readiness時間の改善は保証しない。既存のFC2取得・publish・arrival契約を変更していない。

host は12 GPUゲートが追加する `DG_SM90_PREFILL_FC1_BALANCED=1` がある場合にだけ、EP1・split・実T7988を `_sched` へ dispatch する。他の行数、Decode、診断APIはcontrolのまま。通常servingの `run_sm90_mega_routed → deep_gemm.fp8_mega_moe` 経路から到達する。include置換後の候補headerも hash に含め、通常headerの hash だけで候補JIT cacheを使い回さない。

## CPU 証拠

正本は [CPU検証JSON](megamoe-prefill-sched-r20-cpu-validation-20261101.json)。CUDA13.4.92、sm_90a、同じ最終PP4 overlay、同じhost templateで本番用builderを実行した。

| 指標 | control 追加前/後 | L |
|---|---:|---:|
| encoded SASS byte | 157,696 / 157,696 | 156,160 |
| encoded SASS SHA256 | 両方 `d582cf2059b10cd17523ecf1bcb70135ca5ca7174defa29caf9ad8345f6015ad` | JSONに保存 |
| spill store/load byte | 0/0 | 0/0 |
| max referenced register | 206 | 214 |
| QGMMA / WARPGROUP.DEPBAR | 160/40 | 160/40 |
| C75xx | なし | なし |

builderは image のCUDA13.0.88でも同じ検査を必須にする。ローカル13.4の合格で代用しない。native host auditは230条件を通し、L→診断→controlのJIT text復帰、非7988でのcontrol維持を実行して確認した。実CUDA kernelをGPUで動かした検査ではない。

以下は各CTAの分布。表記は「仕事量:CTA数」。整数7/10の決定は同じで、報告時の評価重みだけを変えている。全tileのexact-once、pool/expert/M/N対応、群内expert/M順もfixtureで検査する。

| routing | control MMA重み0.5 | L MMA重み0.5 | L コスト重み0.7 |
|---|---|---|---|
| balanced | 46:60, 47:72 | 46:60, 47:72 | 46:60, 47:72 |
| historical | 47.5:36, 48:12, 48.5:36, 49:24, 49.5:12, 50:12 | 47.5:12, 48:24, 48.5:48, 49:48 | 49.5:12, 49.6:12, 49.7:12, 49.8:36, 49.9:24, 50:12, 50.2:12, 50.3:12 |
| edges | 47:36, 47.5:24, 48:24, 48.5:12, 49:12, 49.5:12, 50:12 | 47.5:36, 48:48, 48.5:36, 49:12 | 48.9:12, 49.1:12, 49.2:24, 49.3:12, 49.5:24, 49.6:24, 49.7:12, 49.8:12 |

MMA平均は46.54545/48.5/48.09091で不変。0.7重みのcontrol最大は47/50.8/50.8、L最大は47/50.3/49.8。tile個数はbalanced46–47、historical51–53、edges50–53で、仕事量をそろえるとtile個数がそろうわけではない。全128 expert行数・全分布はJSONと `scripts/model-prefill-megamoe-schedule.py` で再現可能。

sourceで算術の変更がないこと、通常SASSの不変と、GPU出力のbit一致は別の証拠である。GPU bit一致は未確認であり、採用には必須とする。

## 同じ常駐プロセスでの計装なし計時と gate

`run-prefill-p-gates.py` を3 worker×4 rankで並列起動する。Pの全7 pair shape、3 single shape、独立FP32参照、producer/attentionのbyte検査、1200回耐久を完走してfragmentを保存してから、同じPID/CUDA contextでMegaMoEへ進む。追加実験の時間を確保するため、combined経路では旧S/V/VS factorial診断とS候補の探索を行わない。Pの検査回数と採用marginは削っていない。従来のP/S経路はcombined無効時に残す。

MegaMoEは、計装headerを使わず、通常splitとLだけを測る。balanced/historical/edgesを8 round、routing列はrotateと逆順を含める。各roundでcontrol→L、L→controlを測る。4連続完全callをCUDA graphで実行し、per-layer µsへ割る。predispatchによるfixture入力準備は外、MegaMoE内部のdispatch・GPU割当計算・FC1/FC2・combineは全て内。

固定seedの通常・40倍入力を各routingでcontrolの独立FP32参照と照合し、Lとcontrolのuint8 viewを比較する。採用条件は以下の全て。

- pair/Pゲートと同じUUID集合12個が全て存在する。worker/rank位置の一致ではなく集合で照合する。run ID、image、header、期限、H200を検査する。
- 各GPU・各routingで、正順・逆順・pooledの中央値が全て10 µs以上短い。
- 全replay・timed pairの出力がcontrolとbit一致しfinite。controlの参照誤差は既存 `.025/.08` 以下。
- 600回以上、3routingと2入力を切り替え、同じworkspaceを再利用した後の出力・finite・完走を検査する。最後だけの同一入力検査にはしない。
- 欠測、NaN/Inf、順序不正、UUID重複/相違、遅着、spill増、C75xx、byte不一致、耐久不足は拒否する。

合格時だけ既存P envへL switchを追加する。不合格ならPの通常splitが残る。後半の失敗を含むsrun終了コードだけではPを取り消さず、保存済みP fragmentの全条件・期限をPの集約器が確認する。shellのfailureおよび実timeout fixtureで、完成済みP採用、L拒否、serving継続を確認する。P自身が未完了なら従来のpair/controlへ戻る。旧trace stepはfalseで実行しない。

## serving routing の別pass

新しい `prefill-routing-capture-pp4.patch` を使用し、旧EP4/140expert patchを流用しない。onepass経路を許可し続け、`_decode_routing_capture` にcapture objectを入れない。

公式clientと公式parserが終了してscore.jsonを保存した後、15秒上限のcontrollerを起動する。全4 endpointの既存flush_cacheでPP microbatchと待機・転送をdrainし、全DP rankのload snapshotでもrunning/waitingと関連disaggregation queueが0であることを確認する。cacheはこの診断passだけのためにflushしたと来歴へ記録する。その後、各Prefill workerへ4件の3994入力IDを送り、同じbootstrap host/port/roomを付けてDecodeにも対を直接投入する。通常workerのままで、新しいmodel/torchrunは起動しない。診断requestのIDだけ `r20-routing-` prefixを持つ。

target ModelRunnerがこのprefixを確認してcaptureを有効にし、onepassが書いた有効行のtop8を、同じstreamでbuffer再利用前に専用int32 tensorへcopyする。各PP stageが担当する24/24/24/22層だけを2microbatch分保存し、forward終了後にD2H・圧縮保存する。通常計測中はprefixが一致せず、capture stateの作成もGPU copy・histogram・同期・filesystemアクセス/pollingも行わない。各層では非活性boolean分岐のみ。CPU tensor fixtureは原tensorを変更しないことと、後で元bufferを上書きしても保存copyが変わらないことを検査する。

保存先は `results/prefill-routing-capture/worker/pp-rank-sample.npz`。worker、PP rank、GPU UUID、global layer列、target forward ID、要求ID、要求別extend/prefix長、実token数、source/image、P、serving envを付ける。inputs.jsonも残す。通常は3994×2 pairを狙うが、schedulerがsingleを形成した場合も実要求境界を保存し、7988と偽らない。集約は94層のstage分割、2sample、12 UUID集合、stage間の要求境界一致を要求する。現在の構成の別pass標本であり、公式窓全体のroutingの代表性は保証しない。

15秒の外側timeoutとcontrollerの絶対期限はstartupも含む。失敗はstatus/部分npz/logへ残し、公式score・終了status・P envを変更しない。正常計測の途中で有効化するmarker pollingは追加しない。

## 時間上限と17分枠

astraの45+36+30+6+5=122秒にLの交互計時・準備10秒を足し、45+36+40+6+5=132秒を条件付き計画値とする。起動変動用18秒を含めて実行期限150秒、kill猶予2秒を含む集約期限155秒。Pにはstart+81までを与え、残り69秒を先に予約する。期限のためにshapeやround数、耐久回数を減らして合格扱いにはしない。

round14実742秒、旧prepack群wall下限68.813秒、先行pairの保守枠155秒を使うと、

`742 - 68.813 + 155 + 155 + 15 = 998.187秒`

すなわち16分38.187秒、17分まで21.813秒。これはcold-startを保証する余裕ではなく、実測前の保守的な予算。L追加分を含めても150秒の新群期限を延ばさないため、上限式は変えない。60秒warmup・120秒公式計測を短縮していない。

## 見込みと Claude の次の serving commit

Lが減らすのはFC1最大担当量とその露出だけで、FLOP/論理byte数は不変。HBM実転送・NVLink・cache変化は未確定。MMA0.5模型の最大差1 full tileを旧完全時間配賦単価で換算すると、全て露出する場合約33–34 µs/層という感度例になる。0.7重みの最大差はhistorical0.5、edges1 full tile。固定費込みの単価は未測定なので予測値とはしない。FC2に吸収される、scheduler費用が勝る、あるいはbalancedで悪化する可能性がある。balancedの仕事量は変わらないため、全routing10µs条件は厳しい。10.2kに必要な84–90µs/層へ達する根拠ではない。

Claudeが試す場合はこのbranchのserving commitで `megamoe_timing=true`、`routing_capture=true`、Lも評価するなら `megamoe_fc1_balance_gate=true` とし、`megamoe_diag=false` とpair/prepack trueを維持する。先頭require-buildとFirmus/nodes/prefill-pp4の既存起動タグを使ってfresh buildへ進め、pinを別の古いimageへ差し替えない。こちらのcommitはskip ciで、ジョブを起動しない。

未確認は実image CUDA13.0.88/host ABI、12 H200でのbit一致・速度・600耐久、時間内の全worker完了、routing別pass、通常servingの公式parser/eligibility/scoreと成果物回収である。これらの成功を今回のCPU検証から推定しない。

## 検証と片付け

指定Python3.12/pytest/pyyaml/numpy/rfc8785/editable構成の全体suiteは1615件収集、1537 passed、78 skipped、25 subtests passed、exit 0。最終関連suiteも83件合格。新しいgateの拒否条件、UUID集合、bit oracleのsigned zero検出、600回の検査配線、同一process配線、P保存後のfailure/実timeout、capture通常窓での非活性、専用copy後のbuffer再利用、全DP peerのdrain確認、build入力を含む。64通りのboolean組合せで再選択フラグを切り替える。

CUDA13.4の3コンパイルと230条件のnative host auditが成功した。指定SGLang commitと全overlayをscratchで再構成し、新patchの適用/逆適用検査とPython ASTを確認した。正規化後のpatchもコンパイル済みheaderと対応する。変更PythonのRuff、3 shellファイルのbash構文、Git差分のwhitespaceを検査した。

全体suiteにはround18/19と同様、AF_UNIX名が長くなる3 fixtureだけを同じscratchのdirectory-fd aliasへ向ける補助pluginを使用した。検査件数・待機・動作は削っていない。初回全体suiteで露呈したoverlay位置の既存assert、stagingの静的inventory、凍結EP4 workerとの比較例外を新しい明示的配線に合わせ、最終全体を再実行した。

不要なscratch source、展開archive、pytest fixture、native中間binary、編集用scriptと補助pluginを削除し、最終compiler/SASS/header/CPU検証・test logと提供済みastra入力を保持した。最初の関連suiteで指定外に作られた自分のpytest一時directoryも削除した。クラスタjobは作成していないので、この作業由来の残留Slurm/PJMジョブはない。他者のjobには触れていない。

## serving 結果：L は 12 GPU 全てで遅く棄却、P は gate の時間予約で未採用、9959.45（run 41317543745、2026-11-02）

- 最初の serving commit c798d7b の run 41315801020 は validate で止まった（build・GPU 未使用）。CI は project の CPython 3.13 で test を回すため、偽 torch の `__spec__` 参照と、thread pool の送信順に依存した 2 test が落ちた。test だけを直した c6605ff で再投入した。
- Slurm job 28126、READY 477 s、PP0 周期 113.9 / 114.3 / 117.5 ms。gate の結果：
  - **候補 L：棄却。** 計装なしの完全 MegaMoE（T=7988、3 routing を 8 round 回転・逆順、control と交互）で、12 GPU 全て・3 routing 全てで L が遅かった。出力は byte 一致。

| routing | control（µs、12 GPU の中央値の範囲） | L − control（µs） |
|---|---:|---:|
| balanced | 2507〜2603 | +102〜+166 |
| historical | 2698〜2797 | +78〜+113 |
| edges | 2645〜2747 | +38〜+111 |

    各 warp が GPU の counter から割り当てを毎回計算し直す費用と、expert・M block 順の割り当てで B の再利用が崩れたことが、FC1 の最大負荷の低下（50 → 49 単位）を上回ったと考えられる。FC1 の静的な割り当て順を変える方向は閉じる。
  - **12 GPU の計装なし計時は全て完了**（round 19 の時間切れを解消）。ただし control の値は round 19（balanced 2334〜2363 µs）より高い。この run では P の gate の直後に同じ process で続けて測っており、clock・電力の状態が違う。round 19 と直接比べない。
  - **P：未採用**（`no candidate passed all twelve GPUs`）。prefill-2 では P は合格したが、起動の遅い prefill-0/1 で、MegaMoE 用に先に予約した時間を除くと残りが P の完全 gate に足りず、`insufficient time for complete candidate stage` で P を始めなかった。round 20 の gate 群の作りの欠陥で、main の gate（round 14 の形）にはない。
  - **routing capture：未完了**（`pass did not finish within 15 seconds`）。
- **serving：** round 12 の選択で公式 **9959.45**、eligible、validation valid、6273 requests。round 12 の構成の再現（9947.50〜9958.43）と同じ範囲。
- **今後：** この branch の gate 群（P と MegaMoE 計時を同じ process で続ける形、時間の予約）は main に入れない。もし MegaMoE の計時を再び使うなら、P の完全 gate を先に必ず終え、その後の残り時間だけを MegaMoE に使う。
