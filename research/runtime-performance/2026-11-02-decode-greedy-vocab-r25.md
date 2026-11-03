# Round 25: Decode greedy verify の実モデル語彙幅への接続

2026-11-03 作業。base は main 38b1e6a、作業 branch は perf/pp4-r25-greedy-vocab-20261102。SSH、Actions 操作、Slurm 投入、GPU 実行は行っていない。以下は CPU 検証と投入前の実装であり、serving 実行の証拠ではない。

## 原因の全数調査

round 22 の合成 gate は 152064 列で成功したが、実モデル config の vocab_size は 151936。round 24 の run 41423181527 の trace と訂正記録に従い、10025.16/10024.51 を融合の serving 結果として扱わない。関連 patch、kernel/wrapper、gate/test、build/container、config、prepare-firmus-pp4-settings.py、sbatch と Git 履歴を検索した。

| 範囲 | 変更前の前提と原因 |
|---|---|
| decode-greedy-verify.patch / logits processor | self.vocab_size == 152064。実モデルでは BF16 経路に入らず FP32 cast。padding は直前の logits[:, :self.vocab_size] で除かれるが行 stride は元の幅を保つ。 |
| 同 patch / Decode graph | model_config.vocab_size == 152064 と torch.zeros の固定152064列。実モデルでは共有 FP32 buffer。 |
| 同 patch / eligible | logits.shape[1] == 152064。正しい BF16 が届いても実モデルの151936列を拒否。 |
| wrapper.cpp / greedy_argmax | vocab == 152064。上位条件だけ直しても例外になる。 |
| greedy_verify.cu / math.cuh | 語彙の決め打ちはない。vocab と stride を引数で受け、chunks=ceil(vocab/4096)、範囲外を除外。Q4 と B1..48 は対応範囲として維持。151936 の末尾384列、152064の末尾512列を検査する。 |
| select-firmus-decode-greedy.py / Case | 入力、info、processor、graph buffers が152064、dense stop storageが152065、stop sentinelが152064、penalty sliceも固定。実モデルconfigへの配線なし。151643/151645も合成stopの固定index。paddingは入力生成時点でsliceされ、processorのslice自体を検査していなかった。 |
| test_decode_greedy_verify.py | dispatch mock、NumPy/C++入力、sentinelが152064のみ。B1/32/44/46/48のみ。実語彙幅のdispatch拒否を検出できなかった。 |
| build.py / build-sglang-container.sh / containers | 語彙の決め打ちなし。greedy TUのみfast-mathを外しftz=false/fmad=false。sourceとwrapperをimageへ含めるのでfresh buildが必要。 |
| configs/firmus-prefill-pp4.json | 語彙指定なし。decode_greedy_verify_gate=true と現行image pinを維持。 |
| prepare-firmus-pp4-settings.py | 語彙指定なし。gateフラグを転記し、gate前のDecode envを0にする。採用は集約後。変更不要。 |
| sglang.sbatch | 語彙指定なし。既存MODELと/scratch mountはあるがgateへ渡していない。model pathだけ追加する。90秒＋660秒は維持。 |

build・config・sbatch・他のtestにあるgate名の参照も確認した。歴史的researchや既存CPU証拠の152064は当時の条件なので書き換えない。

## 修正

processor は model config 由来の self.vocab_size までsliceし、BF16・語彙一致・内側stride1・int32で表せる行strideを検査する。graph bufferはmodel_config.vocab_size列、samplerはinfo.vocab_sizeと実shapeを照合する。wrapperはB1..48/Q4、正の語彙、int32のindex/stride上限を検査する。CUDA kernelと共有mathは変更しない。

process内で最初のfallback理由をwarning、実際のtorch op呼び出し成功後にgreedy_tiles/greedy_finishの使用をinfoで各1回記録する。capture/eagerでの呼び出しの記録であり、graph replay完走やGPUの正しさを保証するlogではない。追加の同期や毎stepのlogはない。

gateは --model-path のconfig.jsonから語彙を読み、torchrunへ同じpathを渡す。workerと集約はvocab_sizeを照合するv2 recordを使い、旧152064合成recordは採用しない。合成stopは0・vocab-2・vocab-1・padding sentinel vocab。padding列を+Infにし、processorでのsliceとbufferへのcopy、直接sliceした行stride付き入力のargmaxを検査する。既存の全case一致、状態変更/復帰、sanitize両方、penaltyなし、10000回耐久、全GPU・両順各blockの50µs閾値は維持する。

## CPU 数値検証（2026-11-03）

引き継いだ未commit testは、正負のsubnormal・±0・warp/tile境界のtieを実際の勝者にする16ケース。これを含めた `test_decode_greedy_verify.py` は CPython 3.12.12 / NumPy 2.5.4 / pytest 9.2.0、g++ 13.3.0 の O2・fno-fast-math で **514 passed、skip/failure 0、307.159秒、exit 0**。

共有 `greedy_verify_math.cuh` をそのままCPU compileし、4096列tileを逆順・laneを逆順にreduceした結果を、独立したNumPyのBF16 bit展開→FP32 sanitize→dense penalty→argmaxと比較した。151936/152064 × B1〜48 × Q4 × paddingなし/64列 × sanitize両方の384ケースで、initial・min-token境界・状態変更・復帰・penaltyなしを照合した。主matrixのargmax行は188160。端数tileの1/384/4095/4096/4097列、vocab末尾のstop、sentinel、lengthの境界前/等値/後、全NaN/全−Inf、勝者となる正負subnormal、±0、warp/tile間の最小index tieを補助ケースで検査し、合計2006回のC++対NumPy比較・188568行のindexが完全一致した。paddingは+InfまたはNaNで汚染し、sliceした非連続行strideをそのまま渡す。

CPUのchain helperでもaccept index/length/bonusと復帰を比較したが、公式GPU verifyの実行ではない。実patchのdispatch・各1回のfallback/使用log、model config読込、gate集約/拒否、shell timeout/予算、ordered overlayとLM head/verify不変も514件に含まれる。数値正本は `decode-greedy-vocab-r25-cpu-validation-20261103.json`、rawのXML/logは後掲のarchiveに保存した。

## CUDA 13.4 compile / SASS

`/usr/local/cuda-13.4` の nvcc 13.4.92、sm_90aで、base 38b1e6a と実装4b40ae4のFA3 companionをそれぞれ **全34 CUDA translation unit** compileした。build.pyと同じ生成表・CUDA flags・固定FA3/CUTLASS archive（SHA照合済み）・patch順序を使い、container contextと同じようにprobeのfp8-output.patchを配置した。CUDA source/header・build条件など36入力はbaseとbyte一致、変更された入力はhost wrapperだけ。

cuobjdumpの識別用source絶対パスheaderだけを正規化し、全34 TUのSASS本文・命令encoding・制御語が完全一致した。融合以外は **33 TU / 39 function**。省略したspecializationはない。全compile logにC75xx診断なし。greedyはtile31 / finish32 registers、両方stack・spill load/store 0、ptxas静的shared64 bytes。fast-mathを外しftz=false/fmad=falseを維持、SASSにFADD.FTZ・FFMAなし。spill 0の主張はgreedyの2kernelについてであり、既存attentionの資源状態と混同しない。

wrapper.cpp全体も、Torch 2.11.0+cu130 headers・CUDA13.4 headers・生成candidate header・patch済み依存を使い、g++13.3.0のfsyntax-onlyで成功した。libraryのlink/loadは実施していない。ローカルCPU検証だけ最大4 compile並列にし、完了したTUの成果物を保持して継続した。image builderの最大2並列は変更していない。

flags・各TUのsource/SASS/encoded-word hash・資源数の正本は同じ `decode-greedy-vocab-r25-cpu-validation-20261103.json`。再現用source、各compile command/log、object、SASSは後掲のraw archiveへ保存した。命令encoding・制御語は合計290224 wordを比較した。

imageのCUDAは13.0.88。CPUのCUDA13.4 compileは代わりの証拠にすぎず、image build・link/load・実GPUの精度/性能・通常servingの公式結果は未確認。

## CI と同じ全体 suite と ruff

repo直下、CPython 3.13.12、NumPy 2.5.4、pytest 9.1.1、TMPDIR未設定（既定 `/tmp`）、一時pluginなしで次を実行し、**1939 passed、90 skipped、exit 0**。quiet出力の2029個の結果markerと別の収集一覧2029件を照合した。514件のDecode greedy検査は全件実行され、skipしていない。

`uv run --project qwen-experiment-metrics --with 'numpy>=1.26,<3' --with pytest python -m pytest -q -p no:cacheprovider qwen-experiment-metrics/tests`

`uv run --project qwen-experiment-metrics ruff check qwen-experiment-metrics scripts containers` は `All checks passed!`、exit 0。変更したsbatchのbash構文、Python AST、git whitespace検査も成功。rawは後掲archive内の `pytest-ci-final.log`、`pytest-collection.log`、`ruff-final.log`。前回の未完走logを成功の証拠として再利用していない。

## 次の投入と見込み

Claudeの投入commitはこのbranchの修正からfresh buildを要求し、先頭に `[require-build] [cluster:firmus] [nodes:2] [prefill-pp4]` を付ける。その投入commitには `[skip ci]` を付けない。decode_greedy_verify_gateは既にtrueなので変更不要。image pinは今回変更せず既存のfresh-build/provision経路に任せる。gate採用後にDecode processの使用logでvocab=151936を確認し、通常servingと公式parser・artifact回収まで確認する。fallbackがあれば理由を確認し、融合が動いたと扱わない。

条件付きの見込みは「Decode の1 stepで0.12〜0.16 ms、Prefillが律速なのでscoreへの効果は小さい見込み」。これは旧152064合成gateからの条件付きの見込みであり、実語彙幅のkernel実測でもservingの利得でもない。

## 証拠の保存と片付け

指定scratchの `round25-cpu-proof-20261103.tar.gz` に、固定依存archive、両版のsource・生成TU・compile command/log・object/SASS、実際に数値検証で生成したC++ source/library、XML、全体suite/収集一覧/ruffのlog、環境、再現手順を保存した。**52570902 bytes、457 entry**を読み戻し、manifestの全entryのsize/SHA256を照合済み。

SHA256: `5b9c51889e3be5ddb3082446e8e1f3cc894d958be5181bedc07e7e2c712fe9a1`

展開した依存source、loose compiler生成物、検証用helper/overlay、今回のpytest一時directory、worktree内の検証venv/cacheを削除した。提供済みの過去logと参照rawには触れていない。新しいscheduler/Actions/GPU jobは作成せず、ローカルcompilerもterminal状態を確認した。未commitだった16ケースは0989d5aで引き継ぎ済み。trackerは手で1段落だけ追記し、kernel-theory-model.jsonは変更していない。

## GPU結果（run 41573354672、commit 14c3e86、Slurm 28478）

fresh buildのimageは `a001f90142b571380894e9532e467c57b831397c9e2e9d613d9e04004c87def9`（12298305536 bytes、SGLang fc27567）。Slurm 28478はCOMPLETED、13:22。

- **Decode greedy gate：** 実モデルconfigから読んだvocab 151936で4 GPUすべてpass、adopted（"all four Decode GPUs passed"）。全caseのindexが一致し、状態変更と復帰も確認した。B44/Q4のgate計時では、従来経路が約0.271 ms、融合が約0.134 ms（両順とも）。融合1回あたり約0.137 msの短縮で、見込みの0.12〜0.16 msの範囲に入る。これはgate内のkernel実測で、servingのstep時間を測ったものではない。
- **servingでの適用：** `decode-0.log` に4 rank（DP0〜3）それぞれ1回の `Decode greedy verify using fused kernels greedy_tiles/greedy_finish: rows=4 vocab=151936 stride=151936`。fallbackのlogは0件。round 22と違い、融合はservingで実際に動いた。
- **他のgate：** pair gateはadopted（12 H200、両順で5 µs以上）、prepack gateはP3（pooled中央値5.20 µs）。round 22〜24と同じ選択。
- **公式score：** 10021.60 tok/s、eligible、good_request_fraction 0.9820、error 0、OSL mismatch 0、6320 request。同じ構成で融合なしの3回は10009.52・10025.16・10024.51。この差はrun間の揺れの範囲で、融合によるscoreの改善は見えない。Prefill律速という見込みどおり。EAGLE3の受理長（bonus込み）は約2.21。
- **dispatchの記録の不具合：** sbatchが残した `results/decode-greedy-dispatch.txt` は `Decode greedy verify using fused kernels g` で切れていた。ERE bracketの `[^\r]` は「CRを除く」ではなく、「`\` と `r` を除く」を意味する。そのため `greedy` の `r` の手前で一致が止まった。ローカルのgrepでは全行が出たので、CPUテストでは検出できなかった。rawのlogは無傷なので判定には影響しない。次の投入からは `.*` に改め、testもその形を確かめる。

採否：融合は正しく動き、scoreは同等。kernelの仕事を減らす変更で精度はbit一致なので採用し、image a001f901をmainのpinにする。score上の利得は主張しない。
