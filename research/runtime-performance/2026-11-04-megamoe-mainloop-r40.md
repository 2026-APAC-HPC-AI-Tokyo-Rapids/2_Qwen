# Round 40：Prefill MegaMoE 定常 mainloop の CPU 調査

**NO-GO。段階1の本番 source/SASS 再現は成功したが、段階2の cycle model を完全 kernel 時間へ校正する条件を満たせず停止した。** 50 µs/層以上の短縮を支持する、誤差付きの予測は得られていない。これは新設計が物理的に不可能という判定ではない。指定の「条件が失敗したら即興で進めず停止」に従い、段階3の候補 overlay・候補 compile・FP8数値検査へ進んでいない。新候補を compile して棄却した、という意味の NO-GO ではない。

## 対象・来歴

親 `a3aae9032d5eb70d501205dde72109552c4e471c`、branch `perf/pp4-r40-megamoe-mainloop-20261104`。AGENTS.md の後、指定の `codex-scratch/r40/astra-report.md` の MegaMoE 節と手順2–3を最初に確認した。r31/r36/r32/r35、以前の MegaMoE note と CPU-validation JSON、tracker/model、consumer review、probe source、関連 commit 本文と保存 Codex log、保存 Actions 由来の r24 集計を確認した。入力 hash は proof 内 `inputs.json` に保存した。tracker/model の先頭にある旧 EP4 条件を現 EP1 と合成していない。

対象は **T7988=2×3994、E128、top8、H4096、I1536、EP1**。FP8 E4M3 operand、元の K128 weight/activation group scale、FP32 積和・scale、FC2 BF16 partial と FP32 combine を維持する。現行 FC1 first-K は有効。run の有効 GEMM 仕事は FC1 **1.608196816896 TFLOP/層**、FC2 **0.804098408448 TFLOP/層**、合計 **2.412295225344 TFLOP/層**。

現 config の image は `5018d322ff7b5884520300052af6f3cd1d5594e210bee85e62cd0495f417d94a`、SGLang は `fc275674a97d78553b22597ee72c8fd48f5420f4`、DeepGEMM は `1c8138a59b3df5dac50b76b730834ed57982fdfb`。今回の CUDA は **ローカル `/usr/local/cuda-13.4`、nvcc/ptxas 13.4.92、sm_90a、O3/C++20**。Firmus image の **CUDA 13.0.88** の機械語・実行時間を検証したものではない。

## 段階1：source の再構成と fresh compile

mirror は使用・変更せず、既存 r32 proof にある source と依存 header をこの worktree の一時 directory に展開した。archive の MegaMoE header から runtime overlay を逆適用して image header、さらに build series を逆適用して pristine header に戻し、現在の順序で再適用した。すべて実際の patch 適用であり、header を候補へ単に差し替えた比較ではない。再構成 script と全 patch hash は archive の `reconstruct.py`、`provenance.json` にある。

| 段階 | MegaMoE header SHA256 |
|---|---|
| DeepGEMM pristine | `a53d83cad76109588b3d5053be02877a0c2a47c38b27472269dc1319df62676d` |
| image の `patches/deep-gemm/series` 14本適用後 | `ff7a58cca00aa42a90386d12b0437b5d8caf35eb0ad082a8a1a0b4b74001a2a1` |
| 現 config の runtime overlay 15本適用後 | `75085f03b05c8ab0cb2dd072994b27b1584225456928d49e66e67820ed1dee56` |

`build-sglang-container.sh` は v0.1.5 を clone して revision を照合し、host extension は `rebuild-deepgemm-host.py` が `patches/deep-gemm-host/series` の12本を source copy へ順に適用して構築する。host rebuild は package の JIT header を保持する。device header は `build-sm90-megamoe.py` が base series の14本を適用する。実行時は `stage-qwen-overlays.sh` が image package の copy へ config の15本を順に適用する。最後の dense-r37 overlay は別 header を変更するため MegaMoE header hash は変わらない。host の split 選択は末尾 template `true,false,true`、FC1 first-K は生成 source の macro であり、単なる header 存在を有効化と誤認していない。

実際の compile template は次のとおり。先頭8448は buffer capacity の template 引数で、実 token 数7988とは別。これを T8448 の実演算と数えない。

```text
8448,4096,1536,128,8,128,128,256,128,92160,1474560,4,
64,64,256,132,1,cute::numeric_limits<float>::infinity(),true,0,
true,true,false,false,false,true,false,true
DG_SM90_PREFILL_FC1_FIRST_K=1
```

| CPU compile の資源 | 現行 FC1 first-K 有効 | r24 対応 control、first-K 無効 |
|---|---:|---:|
| launch register / 最大参照 register | 168 / R206 | 168 / R206 |
| math role の setmaxnreg 予算 | 224 | 224 |
| spill store / load | 0 / 0 B | 0 / 0 B |
| stack / local memory | 0 / 0 B | 0 / 0 B |
| LDL/STL / C75xx | 0 / なし | 0 / なし |
| 静的 SASS slot | 11320 | 9856 |
| encoding byte | 181120 | 157696 |
| encoding SHA256 | `f157a6dddb926f5944f751e46448afb69091eb9cfbcccd17e21fe852205e8c41` | `d582cf2059b10cd17523ecf1bcb70135ca5ca7174defa29caf9ad8345f6015ad` |

両 control を新規 compile し、以前の保存 encoding と一致した。hash は `nvdisasm -hex` の命令・制御wordから復元した機械語で、ELF metadataやSASS文字列のhashではない。`cuobjdump --dump-resource-usage` も収録した。ローカルの静的 SHARED は1024 B、runtime の dynamic SMEM は228560 B。保存 r24 profile の staticSharedMemory は0であり、異なる compiler の resource metadata を同一と主張しない。

### tile、役割、accumulator と scale

CTA は **M128/N256/K128、384 thread、132 CTA、4 stage**。warp0–1の64 threadが dispatch、warp2–3の64 threadが A/SFA と B の loader、warp4–11の256 threadが2 math WG。dispatch/loaderのregister予算は48/40。math は `64×48 + 64×40 + 256×224 = 62976` register の分配で、224という予算とlaunch metadataの168は別の値である。

各 math WG は **M64/N256**、thread当たり128個のFP32 accumulatorを持つ。N方向の **m64n128を2 half** に分け、各halfは64 register。現SASSでは half0がR88–R151、half1がR24–R87。laneの行は `warp_in_wg×16 + lane/4` とその8行下、列は8列fragment内の `2×(lane mod 4)` のpair。FC1のgate/upは8列単位のinterleaveで、half0がgate全体、half1がup全体という配置ではない。

各 K128 の4個のK32積をFP32で蓄積する。carried accumulator を `Ck = Cprev × (s_prev/s_k) + Pk` と更新し、tile最後に元の合成scaleを掛ける。FC1はgate/up別、FC2はN128 weight group別の元scaleを使う。**各Kのscaleは異なるため最終scale一個にまとめられない。** 現行定常部のaccumulator処理は1要素1 FMULで、FFMAへの単純置換では命令本数は減らない。

### PC で確定した依存列

下表は現 first-K 有効 SASS の代表定常 block。いずれもactive math warp当たりの静的slotで、反復するcold barrier waitの時間は含めない。

| 範囲 | slot | QGMMA | accumulator FMUL | scale FMUL | RCP | WGMMA wait / arrive |
|---|---:|---:|---:|---:|---:|---:|
| FC1 interior `ad00..b8a0` | 187 | 8 | 128 | 8 | 4 | 2 / 2 |
| FC2 interior `14bf0..15790` | 187 | 8 | 128 | 8 | 4 | 2 / 2 |
| FC1 chunk全体 `7a90..df30`、8 K128 | 1611 | 64 | 1024 | 68 | 32 | 境界を含む |
| FC2初回後 `12500..1ac90`、11 K128 | 2170 | 88 | 1408 | 92 | 44 | 境界を含む |

187は最後のQGMMAまでの包括範囲。次のpipeline更新のMOVを含める旧consumer表の188と切り方が違う。boundaryに近いFC1の `8630..9320` は208 slot、FC2の `12500..13200` は209 slotであり、全Kを187に置き換えない。

| half | 前の同halfの最後QGMMA → wait | 64個のFMUL | 次の4本のQGMMA | さらに次Kの先頭QGMMA |
|---|---|---|---|---|
| FC1 half0 | `8530 → 86c0` | `8910..8d40` | `8d60,8d90,8da0,8dc0` | `9980` |
| FC1 half1 | `8620 → 8dd0` | `8de0..91d0` | `9260,92c0,92f0,9320` | `9f30` |
| FC2 half0 | `12410 → 12560` | `12640..12a40` | `12a60,12b10,12bc0,12cc0` | `13770` |
| FC2 half1 | `124d0 → 12cd0` | `12ce0..130d0` | `13100,13180,131d0,13200` | `13de0` |

各行は `前group完了 → wait_group 1 → 64個の独立FMUL → fence/arrive → 同halfの次QGMMA`。64段の乗算直列鎖ではないが、QGMMAは64個すべての結果を必要とする。scale自体にも LDS/LDG→FMUL→RCP→FMUL の依存があり、compilerは一部を現在のQGMMAと重ねる。`dependencies.json` に各FMULのPC・operandを全数保存した。

`wait_group 1` は最新1group以外の完了を待つ。accumulatorの読取り前に該当groupの完了が必要であることは [NVIDIA PTX ISA の wait_group 定義](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group) と整合する。ISAの同期規則は、その命令の動的待機clock数を与えない。

### arrival / release

共有ringのfull barrierはA/SFA・Bの2 producer arrivalと必要byteの完了を待つ。empty barrierは8 math warpのarrivalを数える。定常部はhalf0の旧group完了を待ち、次half0を発行、half1の旧group完了を待ってから前stageをreleaseし、次half1を発行する。sourceでreleaseがrescaleより前でも、SASSでは独立なFMULを先に置ける。例はFC1 `8dd0 → 8de0..91d0 → 91e0 release → 9260 QGMMA`、FC2 `12cd0 → 12ce0..130d0 → 130e0 release → 13100 QGMMA`。

その後に次stageのfull wait・scale先読み、stage/phase更新。FC1は32 K128を8ずつ4 chunk、FC2は12 K128を1 chunkとして展開する。chunk末尾は `wait_group 0` でdrain後に最後のstageをreleaseする。FC1の例は `df00 wait0 → df10 release`。FC1 first-Kは最初のchunkだけ入口をpeelし、最初の各halfのK32をscale-d=0にする。定常・chunk境界は保持される。FC2初回は既にcompilerが同様のゼロを吸収済み。

FC1 epilogueのFP8・scale storeとwarp内同期後、arrivalを次のFC1 itemのK block2、または最初のFC2 item取得前にrelease公開する。FC2 loaderのacquireを経て、全N itemの完了を確認してからFC2へ進む。ring stage releaseとFC1 output-ready公開は別の同期であり、いずれも今回除去していない。

## 段階2：cycle model と校正できない理由

2 math WG×2 halfでK128当たり4 group。1 halfの仕事は `2×64×128×128 = 2097152 FLOP`。理想Tensorサービス量は256 clock/half、CTA全体1024 clock/K128。これはissue-to-completion latencyの測定値ではない。

cycle modelのpartial orderは、WGをg、K128をk、halfをhとして次を要求する。

```text
issue(g,k,h) >= max(complete(g,k-1,h), stage_ready(g,k), scalar_issue_available) + rescale_span(g,k,h)
complete(g,k,h) = shared Tensor service schedule + completion tail
```

両WGは同じTensorサービスとscalar schedulerを共有し、各schedulerにmath warpは2本ある。64 FMULを1warp・1cycle/issueと仮定すると最終結果まで `63 + Lmul` clock。2本のmath warpを交互発行する仮定では `127 + Lmul` clock。Lmulを4/8/16と置く感度値は67/71/79、または131/135/143 clock。**これらは仮定の感度であってH200実測latencyではない。** producerも同じschedulerのissueを使うため、実際のslot間隔はさらに変わる。FFMAのlatency、QGMMA完了tail、stage到着、WG間の位相、chunk drainの露出は未確定のまま残す。

重要なのはrescale区間の長さではなく、同じ他条件でscale依存を変えた時の**完全kernelのcritical path差**。他half・他WG・TMA・epilogueに隠れる仕事を足し算して予測しない。

校正対象はr24 **run41423181527**、PP0/GPU0、内側窓 `1426516182..8442926294 ns`、7988行pair53周期/1272call。保存r31再集計は **63.560586 ms/24層、同runのclock1410 MHz**。新たなtrace取得ではなく、その正本の `analysis.json` をhash付きでproofへ保存した。同runの有効Tensor床は **37.9715837524 ms/24層**。1層では次の会計になる。

```text
観測完全kernel       2648.357750 us
有効Tensor床         1582.149323 us
未配賦残差           1066.208427 us
```

r24はFC1 first-K採用**前**。今回のcurrent SASSをr24の全体時間へそのまま当てないため、first-K無効のr24対応controlもfresh compileした。定常依存は照合できても、採用後の完全kernel内部profileや入口短縮の同run補正は無い。r32–34の別runの16–25 µsをr24から引く処理はしていない。

r24には正常kernelの内部phase時間・本番expert count・per-PC stallが無い。したがって、既知の合計一個からscale露出と他の露出を分離できない。次の3会計は**同じ観測へ誤差0でfit**する。

| scaleに仮配賦する露出 us/層 | 他の露出・padding等 us/層 | 有効Tensor床との合計 us/層 |
|---:|---:|---:|
| 0 | 1066.208427 | 2648.357750 |
| 50 | 1016.208427 | 2648.357750 |
| 100 | 966.208427 | 2648.357750 |

これは実scheduleを再現した3モデルではなく、合計だけへのfitが**依存コストを同定しない反例**である。fit残差0を校正誤差0とはできない。scale依存の正味寄与は未確定。固定clock・同じ有効Tensor仕事の除去可能時間に対する粗い会計枠は0〜1066.208427 µs/層で、confidence intervalではない。この枠全体をscaleが使っているとも主張しない。少なくとも0と100の配賦を排除できず、50 µsの採否を判定する予測誤差は拘束できない。

### historical routing の条件付き感度

本番r24 routingの代入には使わず、保存fixtureの68288実行行について閾値だけを数える。M64実行をfull CTA相当へ換算すると `68288/128 = 533.5`。FC1はN item12×K32、FC2はN item16×K12で、全307296 K128 CTA相当/層。初回accumulator rescaleを両FCで除いた定常分は **292358**。平均で全定常K128からb clockの**露出**を追加費用なしに消せる仮定では、

```text
saving = b × 292358 / (132×1410) = b × 1.570803783 us/層
50 us/層には b >= 31.830838 clock が必要
```

これは別runの内部clockの移植ではなく、明記したhistorical仕事量にr24周波数を仮定した感度計算。r24の実際のtile数やWG位相を再現していない。accumulator FMUL全体のissueサービス量256 clock/CTA相当を同じ式へ入れると402.125768 µs/層だが、その何割が露出するか不明で、これを予測savingにしない。

旧H100 probeの1272–1278 clockやno-rescaleとの差177–217 clockは、別GPU・別workloadの測定である。r24のH200へ代入して露出を定めない。r19計装版の4.4–4.7倍の攪乱したphase時間も校正に使用しない。

## 段階3以降と設計仮説の扱い

段階2で停止したため、`patches/deep-gemm-bench/r40-*` は作成せず、候補compile、FP8数値エミュレーション、追加TMA/SMEM/descriptorの実装・計数は未実施。比較表のafterは「未評価」であり0ではない。

| 項目 | before 現行 | after 狭N＋独立product/sum仮説 |
|---|---|---|
| per-K依存 | 同half完了→64 FMUL→次QGMMA、halfごと | 目標はproduct WGMMAとrunning-sum FFMAを分離。実SASS未評価 |
| launch / math register | 168 / 224枠 | 未確定 |
| spill / local memory | 0 / 0 B | 未確定 |
| 追加TMA/SMEM/descriptor | 現行から変更なし | 未確定 |
| 完全kernelの予測saving / 誤差 | 新測定なし | 未確定 / 拘束不能 |

現N256でproductとrunning sumを各128registerにする単純形は、accumulatorだけで256registerとなり224枠に収まらない。N128で各64registerとするなら算術上の必要量は128へ下がるが、それだけでWG/tile全体が実装可能とは証明できない。FC1は元のpost-SwiGLU group128 scaleを保持する必要があり、Nを半分にした時のN item数、A再転送、scale共有、epilogue/arrivalの契約を閉じる必要がある。この仮説を「旧FFMAのregister失敗と同一なので不可能」とも「registerが空くので50 µs速い」とも判定しない。

過去証拠を維持する。

| 既存案 | 過去の否定証拠・現状 |
|---|---|
| FFMA別accumulator | 旧4WG/96registerで1856/1768 B spill、C7512。現2WG/224枠とは別構成 |
| scale先読みr13、合成scale持越し | 依存短縮なし、またはcompilerが既に再利用 |
| inline PTX descriptor集約 | 60/80 B spill、代表block188→204命令 |
| split-LDSM、chunk staggering | probe利得が完全kernelで+1.6%へ反転。staggerは+2.3–4.3% |
| independent WG | 旧EP4で+22–36%、stage減とB TMA重複 |
| lean、4stage、split、FC1 first-K | 既採用。今回の新規savingへ再計上しない |
| tail-pair、swap-tail-r26 | pairing不採用。swapはMMA約3.8%削減でもhistorical約±0.6%、issue/wait増 |
| direct-FC2、A gather | 完全kernel+1.6%、+44% |
| shared-RCP | 速度約0.3%でも600回目の耐久不一致 |
| sched-r20、backoff | FC1割当は全12GPUで+38–166 µs/層、backoff+1.10–2.26% |

**GO条件の候補zero-spill・clean compile・校正誤差付き50 µs以上という3点を満たした候補は無い。** 本作業から主張できる予測savingは「未同定」であって0 µsの測定値ではない。新しいGPU測定をこの作業で提案・投入しない。再開時に解くべき不確実性は、現first-K採用済み同H200完全kernelでの定常scale依存の露出と、新tileの追加費用が31.83clock相当を上回るかどうか。

HBM実traffic・L2 hit・NVLink実byteは未確定。EP1のexpert間通信は不要だがPP/KVの通信とは別。r31の一回通過論理量はFC1 47.571849216 GB/24層、FC2 34.325483520 GB/24層で、本作業では不変。削除したmodel work、GEMM FLOP、転送byteは無い。

## 検証・証拠保管

指定の全pytestとRuffを `export PATH=$HOME/.local/bin:$PATH` の後に実行した。初回全pytestはBasic cleanupのSIGHUP testだけがtimeoutで失敗した。shellの `trap - HUP` では継承されたSIG_IGNを解除できなかったため、PythonでSIGHUPをSIG_DFLへ戻してから同じuv argvをexecした。実際の状態は1から0へ変わり、失敗したtest単独は終了0。source/test/submissionは変更せず全suiteを再実行した。**SIGHUP修正後の指定全pytestは終了コード0、指定Ruffも終了コード0（All checks passed）。** skipをGPU検証成功とは扱わない。ログは指定 `codex-scratch/r40/pytest.log`、`pytest-initial-sighup.log`、`pytest-sighup-targeted.log`、`ruff.log`。

raw compile log、SASS、cubin、encoding、資源表、再構成source/依存header、PC集計、partial-order証拠、未同定のmodel JSONを `megamoe-mainloop-r40-cpu-proof-20261104.tar.gz` に保存した（3916838 byte、SHA256は隣接CPU-validation JSONを正本とする）。gzip検査とarchive内1089ファイルの全hash照合を通過した。production overlay、host selector、config、sbatch、submission、mirrorは変更していない。SSH・GPU・ジョブ投入・停止・commit・pushは一切行っていない。
