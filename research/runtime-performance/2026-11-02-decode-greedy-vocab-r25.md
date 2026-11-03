# Round 25: Decode greedy verify の実モデル語彙幅への接続

2026-11-03 作業。base は main 38b1e6a、作業 branch は perf/pp4-r25-greedy-vocab-20261102。SSH、Actions 操作、Slurm 投入、GPU 実行は行わない。以下は CPU 検証と投入前の実装であり、serving 実行の証拠ではない。

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

CPUのchain helperでもaccept index/length/bonusと復帰を比較したが、公式GPU verifyの実行ではない。実patchのdispatch・各1回のfallback/使用log、model config読込、gate集約/拒否、shell timeout/予算、ordered overlayとLM head/verify不変も514件に含まれる。数値正本は `decode-greedy-vocab-r25-cpu-validation-20261103.json`、rawのXML/logは指定scratchに保存する。

CUDA13.4/sm_90aのcompile/SASSと、CIと同形の全体pytest/ruffの完走結果は次の区切りで追記する。

imageのCUDAは13.0.88。CPUのCUDA13.4 compileは代わりの証拠にすぎず、image build・link/load・実GPUの精度/性能・通常servingの公式結果は未確認。

## 次の投入と見込み

Claudeの投入commitはこのbranchの修正からfresh buildを要求し、Firmus/2nodes/PP4の実行タグを付ける。decode_greedy_verify_gateは既にtrueなので変更不要。image pinは今回変更せず既存のfresh-build/provision経路に任せる。gate採用後にDecode processの使用logでvocab=151936を確認し、通常servingと公式parser・artifact回収まで確認する。fallbackがあれば理由を確認し、融合が動いたと扱わない。

条件付きの見込みは「Decode の1 stepで0.12〜0.16 ms、Prefillが律速なのでscoreへの効果は小さい見込み」。これは旧152064合成gateからの条件付きの見込みであり、実語彙幅のkernel実測でもservingの利得でもない。
