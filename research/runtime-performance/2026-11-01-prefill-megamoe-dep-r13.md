# Prefill EP1 MegaMoE round 13-M：half 間への scale 先読み移動は段階1で保留・棄却

## 結論

**2案とも段階1の条件4を満たせず、段階2へ進めない。** CUDA 13.4 / sm_90a では spill 0、C75xx なし、候補を無効にした現行 serving split の SASS encoding 全体が byte 一致した。しかし、half 0 の wait 後の局所 span 短縮と引き換えに、次 stage の待ち・ratio 計算が half 1 の発行経路へ移った。両 half と chunk 境界を含む依存周期について、正の clk 短縮を根拠付きで算定できなかった。

これは GPU での速度劣化を観測した判定ではない。速度の符号は未確認であり、今回指定された事前証明の基準で採用を止めた。新しい mainloop、host flag、in-job gate、config boolean は登録していない。実行設定・image pin・既存 FA3 pair gate は base `31d9b10` のまま。

機械可読の証拠は [CPU validation JSON](megamoe-prefill-dep-r13-cpu-validation-20261101.json)。prototype の差分、全 template 引数、27本の patch manifest、compiler 条件、命令 encoding の hash、half ごとの span を収めた。未確定の clock・速度・実 HBM byte は `null` にした。

## 対象と来歴

- base は round 12 serving commit `31d9b10`、作業 branch は `perf/pp4-r13-megamoe-20261101`。
- DeepGEMM の source は `1c8138a59b3df5dac50b76b730834ed57982fdfb`。既存 checkout は読み取りのみ。Git archive を指定 scratch 内へ取り出し、`patches/deep-gemm/series` 14本と PP4 config の overlay 13本を順番に `git apply --check` / `git apply` した。
- 再構成 header SHA256 は `fbfaa6be207b26dd7127aae406204b44668b4341ff5ce9e2928510c932e57852`。round 4 の split header と一致した。CUTLASS/cute は既存 `codex-scratch/db/control/deep_gemm/include` を読み取り参照し、DeepGEMM の他 include は archive から再構成したものを優先した。
- control は `DG_SM90_PREFILL_4STAGE=1`、`DG_SM90_PREFILL_LEAN=1`、`DG_SM90_PREFILL_SPLIT_ACCUM=1`。paired tail は無効。math role の `setmaxnreg` は224のまま。
- PP4 / EP1、T=7988、capacity=8448、H=4096、I=1536、128 experts、top-k=8。tile は128×256×128、2 math warpgroup、各 warpgroup は m64n128 の2 half。FC1 は32 K block / U8、FC2 は12 K block / U12。
- A/B は FP8 E4M3、scale / accumulator は FP32、expert partial と最終出力は BF16。tiling、stage数、A/B読出し方式、K順序、chunk境界、全scale・全積を変えていない。

CUDA は `/usr/local/cuda-13.4` の nvcc/ptxas、sm_90a、C++20、O3、relaxed constexpr / extended lambda、ptxas verbose / spill warning を使用した。**image は CUDA 13.0.88 であり、今回の CPU コンパイルは代替検証である。image 上の compile、GPU の出力・耐久・速度を保証しない。**

履歴は AGENTS、theory tracker/model と元 profile JSON、round 3～7 の note / CPU validation JSON、現行 patch chain / host checker / selector / mainloop probe、Astra report §2.5・候補1/4、round 12 FA3 pair gate と round 11 QKV gate、直近80 commit と関連本文を参照した。過去 Actions の観測は保存済み note / probe log を使った。新しい Actions 取得・操作や非公開 Codex 会話履歴へのアクセスはしていない。

## 仕事量：削る対象は byte や FLOP ではない

T=7988 の1層あたり有効 FC1 は `4*T*topk*H*I` = 1,608,196,816,896 FLOP、FC2 はその半分の804,098,408,448 FLOP。合計2,412,295,225,344 FLOPで、両案とも削減0。

| 論理 buffer 契約 | byte / 層 |
|---|---:|
| 全128 expert FP8 weight | 2,415,919,104 |
| weight FP32 scale | 589,824 |
| unique FP8 input + BF16 final output | 98,156,544 |
| dispatch pool write + FC1 read | 523,501,568 |
| FC1 activation write + FC2 read | 196,313,088 |
| BF16 expert partial write + combine read | 1,047,003,136 |

これは tile 再読出し・padding・cache hit を除く論理量で、実 HBM/L2 traffic ではない。EP1 の層内 dispatch/combine NVLink payload は構造上0 byte。PP/PD 転送は別に残る。今回の2案はこの量も変えず、未測定の traffic 削減を見込まない。旧 routing fixture は8002 tokenであり、現在7988 tokenの routing を GPU から採取したとは扱わない。

## 現行 SASS と register lifetime

serving template では half 0 が `R88..R151`、half 1 が `R24..R87`。各 half の64個の FP32 値は、前 K の4本の QGMMA から次 K の rescale と4本の QGMMAまで生存する。1本の m64n128 QGMMA は64個全部を accumulator operand とするので、その half の rescale を32個だけ済ませても発行できない。

FC1 の chunk 内 K=1 の代表箇所：

```text
0x2cf0  QGMMA ... R24 ... gsb0          前 K half 1 の最終発行
0x2d80  WARPGROUP.DEPBAR.LE gsb0, 0x1   前 K half 0 の完了を保証
0x2ec0  FMUL R156, R20, R153.reuse      次 ratio の分母側
0x2f10  MUFU.RCP R23, R156
0x2ff0  FMUL R88, R88, R23.reuse       half 0 の最初の rescale
...     half 0 の accumulator FMUL 計64本
0x3420  FMUL R151, R151, R20
0x3430  WARPGROUP.ARRIVE
0x3440  QGMMA ... R88 ...              今 K half 0 の最初の発行
0x34a0  QGMMA ... R88 ... gsb0         同 half の最終発行
0x34b0  WARPGROUP.DEPBAR.LE gsb0, 0x1   前 K half 1 の完了を保証
0x34c0  FMUL R24, R24, R23.reuse
...     half 1 の accumulator FMUL 計64本
0x38b0  FMUL R87, R87, R20
0x3940  QGMMA ... R24 ...              今 K half 1 の最初の発行
```

source では ratio を WGMMA 後の prefetch に置いていても、この実体の FC1 SASS では4本の RCP が half 0 の DEPBAR 後へスケジュールされていた。従って ratio の位置を調べる意味はあった。一方、64 FMUL は64段の直列 FP 依存鎖ではない。異なる accumulator register への独立した乗算群で、全64個が揃うまで QGMMA を出せないという発行量と join の依存である。64命令をそのまま64 clkの削減可能量として扱わない。

stage の寿命も維持する必要がある。前 stage の empty arrive は前 K の両 half が完了してから、現在 stage は現在 K の両 half が完了してから解放する。今回の prototype は既存 arrive 条件を保持した。

## 2案の契約と過去案との違い

以下の `scale(k+1)` は、次 full barrier の確認、SFA/weight scale の読出し、既存と同じ括弧付けの ratio 計算をまとめたもの。

```text
control: wait h0; rescale h0; issue h0; wait h1; rescale h1; issue h1; scale(k+1)
案1 late: wait h0; rescale h0; issue h0; wait h1; rescale h1; scale(k+1); issue h1
案2 early: 旧ratioを保持; wait h0; rescale h0; issue h0; scale(k+1); wait h1; 旧ratioでrescale h1; issue h1
```

案1は両 half の rescale が終わった後に ratio を更新するため、旧ratioの追加 latch が不要。案2は FC1 の4個、FC2 の half 1 用2個の旧ratioを保持してから prefetch し、half 1 には必ず旧ratioを掛ける。両案とも同じ各要素へ同じ ratio と4個の WGMMA product groupを同じ K 順で適用し、同じ final scale を掛ける。ratio=1 の場合に rescale を省く分岐はない。

round 5 の split-pre は chunk 末で次 chunk の half 0 を rescale した。今回は **全 K block の scale 先読みと half 1 発行の順序**だけを変え、chunk 末の特別な pre-rescale は行わない。U16/U32化、LDSM/RS、独立 promotion 用の別 accumulator、WG間 stagger/independent WG、FC2 interleave、tail split/pair、4分割への変更も行わない。全 accumulator の一部だけを別命令 tile として発行する4分割案の再提案でもない。

CPU の scalar schedule model は K=12/32、異なる入力2 replay、各128要素で、per-element の rescale / product-group順序と FP32 出力 byte の一致を8件確認した。4個の FP32 partial を WGMMA の代わりに用いた順序模型であり、Hopper の WGMMA 数値や approximate reciprocal の emulation ではない。GPU 数値 gate の合格として数えない。

## ptxas / SASS の結果

| 実体 | spill store/load B | C75xx | 最大参照 R | 静的 SASS 命令数 | encoding SHA256 |
|---|---:|---|---:|---:|---|
| original serving split | 0/0 | なし | 206 | 9856 | `d582cf2059b10cd17523ecf1bcb70135ca5ca7174defa29caf9ad8345f6015ad` |
| prototype default off | 0/0 | なし | 206 | 9856 | original と同一 |
| 案1 late | 0/0 | なし | 207 | 9864 | `be36ff07ecb25e92b2aaa3aa0f42cadf66691cc9a7c77568ca3210853bc9b41e` |
| 案2 early | 0/0 | なし | 205 | 9904 | `9b8299191d425e4bd9eb9a25473bb6d0c5bf51dce10977d212611d31b84d6891` |

4実体とも ptxas の launch register 表示は168、math role の拡張budgetは224。表示168と最大R206等は別の値である。全実体で QGMMA=160、FMUL=3110、WARPGROUP命令=80。全role・補助関数・paddingを含む静的コードの計数で、1回の動的呼出しの命令数ではない。

byte 比較は `cuobjdump -sass` の各命令の2個の64-bit encoding wordを連結して行い、original と default off の157,696 byteが一致した。ELF全体の一致やCUDA13.0での一致は主張しない。今回 fresh compile したのは現行 serving split の template 1個の4実体である。他の22 instantiationを再コンパイルして不変確認したという主張もしない。production sourceには変更を入れていない。

### 局所 span と発行列を両方数える

FC1 chunk 内 K=1について、DEPBARの次から最初のQGMMA直前まで：

| 実体 | half 0 の span / accumulator FMUL / RCP | half 1 の span / accumulator FMUL / RCP | half 1 の最初～最後のQGMMA span / RCP |
|---|---|---|---|
| control | 107 / 64 / 4 | 72 / 64 / 0 | 13 / 0 |
| late | 65 / 64 / 0 | 79 / 64 / 0 | 42 / 4 |
| early | 65 / 64 / 0 | 76 / 64 / 0 | 30 / 4 |

FC2 K=1の half 0 は79→65/65、half 1 は66→80/75。half 1 のQGMMA発行 span は17→18/13だが、ここにもRCPが0→2/1へ移る。FC2の別Kでは分布が異なるため、この1箇所だけを全12 Kへ外挿しない。全Kの値をJSONに保持した。

half 0 の107→65だけを42 clk短縮とは言えない。late/earlyとも次stageのfull待ちを half 1 発行前に行う。さらに ptxas は ratio を half 1 の4本のQGMMAの間にも配置し、最終QGMMAとcommitの時刻を遅らせ得る。FC1の8 K blockを含む後方分岐spanは control/late が1619命令、earlyが1639命令で、chunk全体のinline命令数は減っていない。固定の wait→64 FMUL→全64 registerを読むQGMMA という accumulator依存も両halfに残った。

FC1 K=2の `publish_l1_rows` も両half発行後という既存条件を守るため、新順序では次scaleのprefetch後になる。前tileのFC2-ready公開がこの待ちで遅れる可能性もあり、half 0だけの局所差を完全MegaMoEの短縮へ置き換えられない。

**条件4の判定：clk短縮は未算定。** 静的命令spanの局所差は示せたが、ready時刻、実行時のbarrier反復、QGMMAのgroup完了、両WGのtensor利用を含む周期の短縮へ変換する根拠が足りない。命令数差を固定IPCでclkへ変換したり、片側のspanの最大だけで両halfの周期を推定したりすると、移動先の待ちと発行遅延を落とす。その見積りを採用根拠にしない。条件1/2と現行controlについての条件3の合格だけで段階2へ進めない。

## 見込み、未確認事項、Claude側の手順

旧round 3の24層 MegaMoE 65.462659 msに追加1.2～1.5%を掛ければ、0.786～0.982 ms/周期、約33～41 µs/層という開発目標になる。これは現行splitの実測予測ではない。H100模型の約180 clk差は今回のkernelで削れると証明された量ではなく、無scale診断の床との差でもある。今回2案の採用時speedupは未算定で、FA3 M192との合算値や10k達成予測は作らない。

今回のcommitは研究記録のみなので image build は不要。header/hostのproduction変更はなく、`patches/deep-gemm/series`、host series、overlay inventory、`images.sglang.*`、`container_*`を変更しない。gate追加・12 GPU UUID/run照合・timeout shell fixture・17分枠の追加時間見積りは、段階2を実施しないため対象外。

**Claudeがserving commitで変更する箇所はない。** `prefill.megamoe_dep_gate`やprototypeのdefineを追加・有効化しない。この研究commitを取り込んでも、MegaMoEの3本の既存pinを維持する。将来この2案を再検討するなら、まず両halfとchunk境界を含む正の依存短縮の新しい証拠を揃え、その後に独立の実装・12 GPU完全MegaMoE gateを作る。今回のscratch用defineをservingへ渡す経路は存在しない。

再開時の採否を変える未確定事項は、ratio移動で早まるhalf 0の開始が、遅くなるhalf 1の最終発行と次full待ちを上回るかである。SASS上でRCPが消えたことや半分のspanが短いことだけでは解けない。GPU測定を承認済みと扱わない。

## 検証と後片付け

- CUDA13.4の4 compile、spill/C75xx検査、SASSの全encoding hash、default-offのbyte一致を確認。JSONに埋めたprototype patchを再適用し、candidate header hashが一致することも確認した。
- 現行 `check-sm90-prefill-local-host.py` は230 native host条件でPASS。host/layoutのCPU検査であり、追加のCUDA compileは実行していない。
- 全体suiteはPython3.12で実行し、1266 passed、70 skipped、25 subtests passed、3 failed（565.14秒）。3失敗はすべて指定scratchの長い絶対パスに対する `AF_UNIX path too long`。同じ物理scratchへの短い `/proc/<pytest-pid>/fd/<directory-fd>/...` 絶対パスをtmp fixtureから返して、その3件を再実行し、3件ともPASSした。production/test sourceの変更はない。したがって全体で1269件を検証したが、全suiteの単一実行が無修正のfixtureで成功したという意味ではない。
- 全体suiteのコマンドは `uv run --no-project --python 3.12 --with pytest --with pytest-subtests --with numpy==2.2.6 --with-editable ./qwen-experiment-metrics python -m pytest qwen-experiment-metrics/tests --basetemp=<指定scratch>/pytest-tmp -p no:cacheprovider`。3件の再検証も同じPython/dependencyで行った。
- `git diff --check`はPASS。rendererのRuffは既存のEXE001が1件、C408が3件で、baseの同じ4指摘と一致し、追加指摘0。profile accounting、今回の台帳行の生成、既存手動追記の保持を確認した。baseから存在するtracker/rendererの他の差分は今回も保持しており、ledger全体の `--check` 成功とは扱わない。
- raw SASS・compiler log・再現に必要な最小scriptとtest logは指定scratch内へ保持する。archive展開、重複header/binary、失敗した検証の中間directory、pytestの一時run directoryは削除する。SSH、Actions操作、scheduler投入は行っていないため、この作業が作成したcluster jobやremote runtimeはない。

## tracker 用の文面

2026-11-01 round 13-M：現行PP4/EP1 splitのscale先読みをhalf間へ移す2案をCUDA13.4/sm_90aでCPU検証した。両案spill0・C75xxなし、default offのserving split SASSはbyte一致。FC1 K=1のhalf 0のpost-wait spanは107→65命令だが、half 1の発行spanは13→42/30命令へ拡大し、両halfのwait→64 FMUL→QGMMA依存は残る。chunk全体のinline命令数も減らず、依存周期の正のclk短縮を示せないため段階1の条件4で保留・棄却。GPU未実行、速度未算定、production変更・gate追加なし。詳細は2026-11-01-prefill-megamoe-dep-r13.mdとmegamoe-prefill-dep-r13-cpu-validation-20261101.json。
