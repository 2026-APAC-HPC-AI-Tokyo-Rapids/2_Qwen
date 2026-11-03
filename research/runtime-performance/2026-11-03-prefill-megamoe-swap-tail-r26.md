**Round 26: Prefill MegaMoE の限定 swap tail。CPU admission 通過、GPU 未測定、既定 off。**

基準は main `e64cb6d`、image `a001f90142b571380894e9532e467c57b831397c9e2e9d613d9e04004c87def9`、DeepGEMM `1c8138a59b3df5dac50b76b730834ed57982fdfb`。PP4/EP1、代表 T7988、128 expert、top8、H4096/I1536、FP8 積・FP32 carried scale/accumulator・BF16 partial。4STAGE/LEAN/SPLIT_ACCUM を control とする。Astra26/27 の提供済み最終評価、tracker/model/profile、EP1 accounting、TAIL_PAIR `2dc907e`、tail split、静的割当 L、inline PTX consumer の記録・コミット本文と保存会話を確認した。既存の割当変更を再開する案ではなく、padding の積自体を除く案である。

**設計と接続**

`patches/deep-gemm/series` に `sm90-prefill-swap-tail.patch`、host series の BM192 patch 直前に同名 host patch を追加した。host は既存 four-stage lean split の EP1/128 expert/H4096/I1536 に限り、`DG_SM90_PREFILL_SWAP_TAIL=1` に応じて生成 CUDA source の include 前へ同名 macro を挿入する。通常の template 引数と `generate_impl` の source 本体は flag なしでは同じ。include hash は新しい header を反映し、macro のある source とない source は既存 JIT signature で区別される。新しい cache、queue、環境変数の中継は追加しない。

最後の M128 pool block の `valid_m=1..16` を N16、`17..32` を N32 へ分岐する。`33..128`、前半64行＋後半端数の block、Decode は従来のまま。pool の容量・行割当、TMA box、scheduler の item 列、FC2 claim、combine は変更していない。分岐は通常 accumulator の生存開始前、math tile の入口にあり、通常 K loop は書き換えていない。

WG0 が weight-as-A の m64nN を4 fragment、各 K128 内を K32 昇順4回で処理する。各要素の再スケールは `(prev_sX * prev_sW) * rcp(sX * sW)`、初期 accumulator への倍率は1、最後の倍率は `sX * sW`。部分和を別に作って足す方式へ変更しない。単純な初版として fragment ごとに commit/wait する。通常 split の待ちを隠す利点まで保てたとは扱わない。

FC1 は8行ごとの gate/up の対応を保ち、現行と同じ EX2/FTZ、RCP/FTZ、`(g * sig) * u` を使う。SwiGLU を register に保持し、各 warp の32特徴の最大値を共有して、token ごとの **128特徴** の amax を得る。既存 helper で `max(amax,1e-10)`、FP32 `1/448` 乗算、逆数、FP8 変換を行い、token-major pool と従来の SF slot へ書く。per-64 化はしていない。

FC1 の FP8 staging は N32 でも4,608 B、amax exchange は512 B。FC2 は8 token × 528 B = 4,224 B の strip を使い、最終 scale の後に BF16 RNE へ丸め、source metadata の同じ token/top-k partial slot へ128-bit storeする。いずれも WG0 の既存9,216 B slice 内に収まる。WG1 が直前の通常 tile を終えている間も、その別 slice を上書きしない。入口の WG barrier は WG0 の直前の staging read を待つ。次 tile へ戻る前にも staging read 完了を待つ。

1 K128 あたり、producer は従来どおり1 stage、full arrival は A/SFA と B の2回、empty release は両 WG の各4 warp、計8回。WG1 は padding-only の既存経路を通る。FC1→FC2 は `12 * ceil(valid_m/64) * 4` のまま。前 tile の遅延 publish は次 FC1 の K block 2、または最初の FC2 fetch の前に行う。空 WG でも publish を省かない。

**CPU の証拠と限界**

正本は `prefill-megamoe-swap-tail-r26-cpu-validation-20261103.json` と `scripts/check-prefill-megamoe-swap-tail.py`。

- 保存 routing の全36分布は、24単要求＋12二要求の構成を維持した。409,808 item、8,430,336 stage use を再生。対象 pool block は2,090個、計算行削減の合計は92,144行。
- `0,r,64+r,128+r` を含む全 r=1..64 の64分布も再生。229,376 item、4,718,592 stage use。通常／swap の全 valid=1..128 で出力座標の重複・欠落がない。token-major からの独立 inverse と fragment forward map を照合した。
- gate/up、weight SF、activation SF、128B swizzle の425,984基底座標を独立の論理座標・sector modelと照合。各 role の stage/phase、全warpの release、各pool blockのarrival、遅延publishを照合した。再生は合法な item 列の数え上げであり、実 GPU の全 interleaving を形式証明したものではない。
- FP8 で表現できる整数入力、独立 scale、zero/通常/wide、全 r=1..32、FC1 packed scale と FC2 scale の384ケース。K256の境界probeに加え、FC1 K4096とFC2 K1536を全rで使い、K128ごとの各K32×4をFP32で書き下し、weight-as-A の4 fragment計算と通常の carried-scale 計算がCPU上でbyte一致。別の NumPy 参照は入力を先に dequantize して matmul し、最大絶対差は `3.5762786865234375e-7`。全128列 amax、scale乗算→逆数→FP8 RNE、最終scale後BF16 RNEも照合した。

Tensor Core 内部の累積順、GPU の EX2/RCP の近似値、CUDA 13.0.88 での丸め、race の不在は CPU では証明できない。CPU の byte 一致を GPU の bit 一致へ昇格させない。

snapshot helper は通常 kernel が完了してから既存 buffer を読む。pool の行順は atomic の割当順で変わるので、source token/top-k metadata を使って canonical 順へ戻し、padding を比較対象から除く。metadata offset は実際の `SM90Workspace` をCPU C++でコンパイルして照合した（capacity8448、pool92160、offset4,466,720 B）。別途 CPU Torch 2.11.0 で helper 自体を動かし、pool permutation の復元、FP8・scale・BF16 partial・符号付きzeroを含む出力byte差の検出を確認した。正本は snapshot validation JSON。

**CUDA 13.4.92 / sm_90a の代替証拠**

元 serving header SHA256 `fbfaa6be207b26dd7127aae406204b44668b4341ff5ce9e2928510c932e57852` を再現した。変更後の全 PP4 overlay 適用済み header は `d701aeeb4ee96a57033eefdeb4f109a86c08176b6bea326698e6876d9dcdf197`。base series の pin は `4a671fae836c79ab64b0d4a61af01a51a6060c8191c01c371fbdf460d3aaaa69`。

| 対象 | SASS命令数 | code byte | spill store/load | launch REG / 最大参照R | N16/N32/N128 QGMMA静的本数 |
|---|---:|---:|---:|---:|---:|
| control | 9,856 | 157,696 | 0/0 B | 168 / 206 | 0 / 0 / 160 |
| flagなし | 9,856 | 157,696 | 0/0 B | 168 / 206 | 0 / 0 / 160 |
| N16のみの計数用probe | 11,560 | 184,960 | 0/0 B | 168 / 210 | 32 / 0 / 160 |
| N32のみの計数用probe | 12,520 | 200,320 | 0/0 B | 168 / 208 | 0 / 32 / 160 |
| production N16＋N32 | 13,800 | 220,800 | 0/0 B | 168 / 213 | 32 / 32 / 160 |

各Nのみのprobeは片方へのdispatchへ置換した計数用で、本番の別候補ではない。表のQGMMAはFC1/FC2両bodyを含む静的本数。動的にはswapの各active WG/K128が16本、controlは8本。candidate/controlとも C75xx、STL、LDLなし。math の setmaxnreg は224を維持する。ptxasのnamed barrier使用数は3→4。hostのdynamic sharedは228,560 B、cuobjdumpのSHARED欄は両者とも1,024 B、CD領域は18,432 Bで不変。

実host selectorをCPUでコンパイルした112設定に加え、過去のnon-swap Decodeを含む24 instantiationを変更前／flagなしでコンパイルし、**instruction/control wordの全byteが一致**した。通常servingのencoding SHA256は従来と同じ `d582cf2059b10cd17523ecf1bcb70135ca5ca7174defa29caf9ad8345f6015ad`。候補は `4222d4c69124d5f89aaf8cf4540ccf945b4b956f3b1e6bab910af3a566da4201`。legacy instantiationに元からあるspillは増えていないが、legacy全体がspillゼロという意味ではない。

compile includeのCUTE/CUTLASS全802fileは、使用したcached wheelとimage側の0.1.5 wheelで全byte一致を確認した。nvcc dependency scanでも、DeepGEMM本体のheaderがfallback includeから混ざっていない。比較manifestはproof archiveに保持する。

image の CUDA は13.0.88である。今回の13.4コンパイルは代替証拠であり、実imageのfresh build・候補load・GPU byte/速度・公式servingは未確認。正本は CUDA validation JSON、compiler入力・ログ・cubin・SASSは作業scratchのproof archive。

**in-job gate**

既存 `select-firmus-prefill-megamoe.py --swap-tail` から `firmus_prefill_swap_tail.py` を呼ぶ。sbatch は既存 pair、完全P/P3、greedy gate の後、model load の前に、3 worker × 4 GPU を並列に動かす。`prefill.megamoe_swap_tail_gate=false` を既定とし、image pinは変更していない。

対象は44ケース：balanced/hot/historical、1..64・65..128の全境界、129/191/192/193/255/256、空expert、zero/wide、保存routing全36分布。最終BF16、FC1 FP8、FC1 FP32 scale、FC2 BF16 partialの4成分を **uint8 viewでbyte比較**する。全12GPUのUUID集合は同じallocationの先行pair gateの集合と一致させ、worker間のGPU再割当は許すが欠測・重複は拒否する。source/image/routing hashと期限も照合する。

同じresident process・非計装の完全MegaMoEを、balanced/hot/historicalそれぞれ12 round、control→candidateとcandidate→controlを交互に計時する。predispatchは計時の外、MegaMoE内部のdispatch/FC1/FC2/combineは計時の内。**各GPU・各routing・両順とも、中央値の削減が `max(5us, control * 0.005)` 以上**でなければ不採用とする。旧selectorの全体合算marginより強い条件である。

続いて6 routing、2 input version、2 symmetric bufferを変えながら1200連続callし、最後の各routingの出力をcontrolと照合する。最初に保存したhistoricalの4成分を基準に、両bufferで元入力へ戻した再現も確認する。途中不一致・timeout・欠測・margin未達ではflagをserving envへ書かず、現行4STAGE/LEAN/SPLITでservingする。採用時のenv更新は既存gateと同じ一時ファイル→renameとする。

**時間の予算**

Actionsの操作は禁止されているため、live Actionsにはアクセスせず、指定archiveのActions由来Slurm/rankログと保存履歴を読んだ。正本は budget JSON。`/srv/qwen-artifacts/41573354672/1/qwen-firmus-pd-41573354672-1/logs/logs/` の集計は次のとおり。

| gate | rank内elapsedの範囲 | groupの時刻から分かる壁時計 |
|---|---:|---:|
| pair | 23.740〜25.211 s | startから次prepack startまで97 s |
| P/P3 | 39.304〜40.049 s | startから次greedy startまで97 s |
| greedy | 11.911〜12.003 s | startから最後のrank完了まで28.052 s、後処理は別 |

jobは802秒、READYは519秒。最後のgreedy rank完了はjob開始から222.052秒で、その後完了まで579.948秒だった。rank内timerをcontainer起動・import・srun teardown込みの時間と取り違えない。

追加gateの見積はgroup75〜100秒だが実測ではない。worker120秒＋kill grace2秒、集約をstart+125秒までに制限し、開始条件を `1020 - SECONDS >= 125 + 660` とした。既存serving/回収の660秒は維持する。観測802秒へ上限125秒を足す参考値は927秒。最後のgreedy rank完了を起点とする保守的計画は約 `222 + 125 + 660 = 1007秒` で13秒の余裕だが、teardown次第で開始条件に届かず省略され得る。将来の起動時間を保証した値ではない。

超えそうな場合に削るのは **追加gate全体** で、既存P gate、44ケース、1200回、serving窓を短縮して合格扱いにはしない。gate未実行も不採用。短い診断の別投入、retry、kernel benchは追加していない。

**見込みと次の投入**

historicalは有効63,904行、現行計算68,288行。該当はN16の45 expert、N32の13 expert、計58。削減2,576行、`0.097240743936 TFLOP/層`、現行MMAの3.772%に相当する。有効計算 `2.412295225344 TFLOP/層` は不変。全体WGMMA発行数は約5.436%増え、各swap tileのcommit/waitも2→4へ増える。TMA box、必要weight、combine trafficは減らない。N16の論理operand readはK128あたり48→40 KiB、N32は48 KiBのまま。実HBM/NVLink byte、clock/電力効果、正味時間は未測定。

CPUだけでは勝敗を断定できない。コード40%増、共有amaxと並べ替え、fragmentごとのwaitが主な危険である。1〜2 ms/24層を目指すなら完全MegaMoEで平均42〜83 us/層の実測短縮が必要で、selector marginを越えたことだけでその目標を達成したとは扱わない。

Claudeが投入用commitを作る際は、まずこのbranchのsource/hostをfresh imageへビルドし、実CUDA13.0.88で候補のcompile/loadを確認する。新imageの検証済みpath/hash/provenanceをpinして `prefill.megamoe_swap_tail_gate=true` にする。serving envへflagを直接pinせず、12GPU gateに決定させる。投入tag付きcommit、build、pushによる計算起動、SSH、Slurm投入、image pin変更は本作業では行っていない。

実装checkpointは `cae926c`、証拠・gate記録は `271e9e2`。すべて指定branch `perf/pp4-r26-megamoe-swap-tail-20261103` だけへ通常pushし、commitの先頭をskip ciとした。

**検証結果と片付け**

指定の全体検証をCPython 3.13.12、TMPDIR既定のまま実行し、**2,098 passed・90 skipped、exit 0**。初回の2失敗はhost seriesの末尾・順序の契約で、BM192を末尾に保ち、新patchを含む期待列へ更新して解消した。関連190件も成功。最後にCPU参照のzeroケースを完全ゼロ入力へ明確化し、実寸K4096/K1536も含む384ケースへ拡張して、独立数値検証1件を再実行した。

```sh
uv run --project qwen-experiment-metrics --with 'numpy>=1.26,<3' --with pytest python -m pytest -q -p no:cacheprovider qwen-experiment-metrics/tests
uv run --project qwen-experiment-metrics ruff check qwen-experiment-metrics scripts containers
```

Ruffは全件成功、変更shellの `bash -n` と `git diff --check` も成功。GPU実行、実CUDA13.0.88のbuild/load、公式parserによる新serving結果は未実施であり、採用済みとはしていない。

証拠archiveは指定scratchの `r26/r26-cpu-proof-20261103.tar.gz`（12,882,910 B、SHA256 `eff5d9832a137ee0c2aafd4a80e3314ee726a27bd4dd13fea6394d5c228e5082`）。入力source、compiler command/helper、raw compile log、cubin/SASS、CPU snapshot proof、依存manifest、最終test logを収め、SHA256SUMSとともに保持した。展開source、loose compiler生成物、編集用script、今回分のpytest一時directory、worktreeのvenv/cacheは削除した。本作業由来のscheduler jobは存在せず、既存Actions・他者のjobには触れていない。

## GPU結果（run 41598151128、Slurm 28515、commit 1537090）

- **image：** fresh build `87f2c06956480a95bddcd16070668bf31f96b60dec11365885d328c50738ee8a`（未pin）。実CUDA 13.0.88でcompile・loadでき、12 GPUのUUIDはすべて別だった。
- **正しさ：** 全12 GPU・44 caseで、最終出力・FC1 FP8・FC1 scale・FC2 BF16 partialがbyte一致した。1200回の耐久と状態の復帰も全GPUで合格し、各workerは約17.5秒で終わった。
- **速度：** 不採用（`AssertionError: below selector margin`）。両順の中央値で見た候補/現行の差は次のとおり：

| case | 12 GPUの範囲 |
|---|---|
| balanced | −1.1〜−3.9% |
| hot | −1.2〜+2.8%（rank 0は+2.5〜+2.8%） |
| historical | −0.6〜+0.6% |

  historicalは実servingの分布に最も近いが、MMAを3.8%減らしても時間はほぼ変わらない。WGMMA発行の増加（約5.4%）と、tileごとのcommit/waitの増加が、padding分の削減と相殺した。hotでは遅くなる。
- **serving：** 現行splitのままで、dispatchの行はない。公式score **10005.78**（eligible、good 98.13%）、job 844秒。
- **予算の副作用：** r26 gateをDecode greedy gateの前に置いたので、greedy gateは予算不足でskipした（`insufficient remaining serving budget`）。このrunのDecodeは従来のgreedy経路で、10005.78はround 25の10021.60と同じ構成ではない。差はround 22〜25のscoreの幅の内側である。
- **結論：** swap-AB tailのpadding削減は、限定版でも実時間の短縮にならない。branchはremoteに残し、mainには文書だけを記録する。expertごとにweightが異なるので、M tileはexpertの境界をまたげない。再開するなら、WGMMA発行とwaitを増やさずにpaddingを削る構造が必要だが、現時点では候補がない。
