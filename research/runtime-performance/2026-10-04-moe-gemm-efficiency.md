# 現行Prefill MoE GEMMの達成演算性能と改善設計

2026-10-04。profile run **37135273516**、計測commit `1ec209d`、container `20c3a38359ccc61967343ec23d600f783df8d487c6339d12b9ca53c14b55d3d3`。固定SGLang `fc275674a97d78553b22597ee72c8fd48f5420f4`＋既存bundle、P2TP4EP1DP1 / D2TP4EP1、c48、EAGLE3、FA3、P native CUTLASS。対象はPの**1GPUあたり**で、4rank・2workerを足し合わせたFLOP/sではない。

結論は、**up約902–922 TFLOP/s、down約559–563 TFLOP/s**。公称dense FP8 Tensor Core peakの約46%・28%に相当する。ただしdownはK384と巨大BF16出力による低いarithmetic intensityを持つ。低いpeak比をそのまま高速化可能量としない。まず既存Hopper expert-specialization (ES)をup/down別に成立・単体性能検証し、通常E2Eで改善を確認する。down scratch outputとcombineの融合は別の中規模候補である。

## ハードウェアと公称値

raw `logs/logs/node-0-environment.log:5–12` はP node `a2ap-dgx026` の8GPUを **NVIDIA H100 80GB HBM3** と記録し、Torch trace devicePropertiesはSM90、132SM、85,019,590,656bytesを記録する。D nodeはdgx030。`profile/profile/gpu-node-0.csv` のactive時間帯00:13:55–00:14:00.5の1秒samplingは全8GPUともSM clock **1980MHz**（各GPU5sample）。kernelごとのclock変動やHBM clock、power throttleはこのsamplingだけでは判定できない。

[NVIDIA H100製品仕様](https://www.nvidia.com/en-us/data-center/h100/)のSXM表はFP8 3958 TFLOP/sに **with sparsity** と注記する。今回は2:4 structured sparsityを使用しないため、比較値は半分の **1979 TFLOP/s dense FP8**、HBM帯域公称 **3.35TB/s**。BF16出力であってもGEMM入力はFP8 E4M3×FP8 E4M3、FP32 accumなのでFP16/BF16 peakを分母にしない。top8 routingの疎性はTensor Coreの2:4 weight sparsityとは異なる。公称値とsampling clockの比例換算は行わない。[NVIDIA Hopper解説](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)もFP8入力とFP32 accumulator対応を明記する。

## token rowsとGEMMの対応

active窓はUTC **16:13:55.383751–16:14:00.388142**、5.004391秒。51秒のCPU trace export尾は分母に含めない。

全8CPU traceの `step[EXTEND ... toks=8192]` を確認した。P0各rankはbs3の注釈34個（target worker外/内の二重span）、P1各rankはbs3が34個とbs4が2個で、いずれも8192tokens。P0のrawログ913–962行、P1の対応時刻のPrefill batchも8192tokens。CUDA runtime launchのglobalPid/correlationIdをCPU stepへ結び、そこからNsight graph nodeを含むGPU kernelへ結ぶ。GPU grid数だけからTを決めていない。

model configはhidden4096、expert128、top8、global intermediate1536。TP4なのでlocal intermediate384。従って一回のgrouped GEMMではexpert割当行の合計 **ΣM_e = 8192×8 = 65536**、expert平均M512（個々のM_eは未回収）。

- up gated: `Σ_e 2 M_e×768×4096` = **412,316,860,416 FLOPs/call**。
- down: `Σ_e 2 M_e×4096×384` = **206,158,430,208 FLOPs/call**。

FMAを2FLOPsと数え、routing/quant/SiLU/scale epilogueはこの値に含めない。GEMM本体のkernel時間だけを使う。同一CUTLASS kernel名のup/downはsourceの `shuffle(float)→get_group_gemm_starts→GEMM(up)→fused SiLU/group quant→get_group_gemm_starts→GEMM(down)` 順で判別。全対象GEMMを分類できた。窓境界を跨ぐkernelを除外したためP0 up1569/down1568、P1各1572callという端数になる。これを「完全17forward×94layer」へ丸めない。

## useful FLOP/sとtile padding

| worker/rank | up call / ms | up TFLOP/s | dense peak比 | down call / ms | down TFLOP/s | dense peak比 |
|---|---:|---:|---:|---:|---:|---:|
|P0/TP0|1569 / 714.658|905.22|45.74%|1568 / 577.083|560.16|28.30%|
|P0/TP1|1569 / 701.727|921.90|46.58%|1568 / 575.876|561.33|28.36%|
|P0/TP2|1569 / 704.112|918.78|46.43%|1568 / 575.111|562.08|28.40%|
|P0/TP3|1569 / 705.504|916.97|46.33%|1568 / 575.422|561.77|28.39%|
|P1/TP0|1572 / 714.412|907.27|45.84%|1572 / 578.770|559.95|28.29%|
|P1/TP1|1572 / 718.460|902.15|45.59%|1572 / 579.456|559.28|28.26%|
|P1/TP2|1572 / 707.083|916.67|46.32%|1572 / 575.932|562.71|28.43%|
|P1/TP3|1572 / 716.748|904.31|45.70%|1572 / 579.324|559.41|28.27%|

これらは**useful** FLOP/s。native `prepare_moe_input.cu:35–40` は実occurrenceをM_eへ書き、EP1は実rowを並べる。global graph token paddingは対象batch8192がcapture shape8192で一致する。現在kernel template/sourceのM/N/K tile128に対しN768/4096とK4096/384は128の倍数なのでN/Kの末端paddingはない。しかしexpertごとのMは不明で、Tensor Core tile処理量を実測したわけではない。

M128tileでの計算枠は `Σ 128 ceil(M_e/128)`。合計65536とexpert128だけから **65536–81792 rows (1.000–1.248047倍)** に限定できる。従ってtile-work相当TFLOP/sはuseful値から最大1.248倍の範囲であり、実値は記入しない。平均M512を全expert M512と置くのは不可。grouped schedulerの負荷偏り、idle SM、個別expert分布も未測定。traceのpersistent grid(66,2,1)はこれらを復元できない。

## モデルrooflineとmemory traffic

以下は**全128expertのweightを1回読み、materialized Aを1回読み、BF16 Cを1回書くoperand-volumeモデル**。HBM counterの実測ではない。scale、descriptor、workspace、再読出し、L2 reuse、routing偏りを除く。active expert数が128未満ならB量は小さく、実際のHBM量はcache挙動でも変わる。Tensor Core tile paddingとHBM traffic paddingも同一ではない。

| stage | A FP8 | B FP8(128experts) | C BF16 | 合計 | AI | 3.35TB/sでのモデルroof |
|---|---:|---:|---:|---:|---:|---:|
|up|268.435MB|402.653MB|100.663MB|771.752MB|534.26 FLOP/B|1789.77 TFLOP/s|
|down|25.166MB|201.327MB|536.871MB|763.363MB|270.07 FLOP/B|904.72 TFLOP/s|

dense FP8 peak / HBM帯域のridgeは約590.75FLOP/B。モデルでは両方その下で、特にdownはBF16 Cだけで512MiB/callある。P0TP0平均up455.49us/down368.04usと対比し、operand-volume/model durationは約1.69/2.07TB/sに相当する。ただしこれはHBM実帯域の計測値ではない。

upはK4096による長いmainloop、downはK384の3個K128 blockとN4096の広い出力を持つ。downの28%というdense peak比にはdata movement/epilogue/短Kの比重がある。ESがTensor Core schedulingだけ改善しても出力trafficは残る。既存combineはdown outputをさらに読み、active窓で約324ms。将来down GEMMからtop8加重和を直接作る/中間layoutを変える設計ならscratch outputとcombineの費用へ作用し得るが、並行expertのreduction、数値誤差、write orderingを守る中規模以上の改修になる。

## 実装担当への設計指針

固定 `fp8.py:2444` は `enable_es=(use_mxfp8,use_mxfp8)` なので今回Hopper通常FP8ではfalse。`cutlass_moe.py:219–231,285–297` に既存SM90 ESのup/down分岐がある。**upのみ、downのみ、必要なら両方**を別commit条件で接続し、native APIのcan_implement、workspaceサイズ、current stream/event依存、breakable CUDA graph capture/replay、shapeとerror toleranceをまず検証する。ES専用streamがtarget/draft/TP通信に対して安全にjoinされることが前提。FP8 quantization規則と既存fused SiLU/group quantを変更せず、GEMM主要因だけを比較する。

shape fixtureの中心はE128 / ΣM65536 / N768,K4096とN4096,K384、block128、BF16 output。現route分布は無いので全M512だけでfixtureを終えず、合計固定のskew/非128整列/zero expertを含める。小T、他graph shape、empty/failure shapeも既存serving経路で発生する範囲を確認する。ES単体で勝ってもE2E改善は保証されず、同じイメージ条件・公式workloadの通常非profile c48で反復比較する。

P0TP0のactive費用配分はup714.7ms、down577.1ms、AR1043.3ms、shuffle+scale+combine655.4ms。NVLSとESを同時変更せず主要因を分ける。ES microでspeedup sを得た時に元profileのGPU kernel減少期待は `t_before(1−1/s)`。例えば**仮に**up/downが各1.10倍なら約65.0/52.5ms、計117.4ms（5.004秒窓の2.35%）が減少する計算になる。これはtask score/TTFT gain予測ではない。profile overhead、kernel重複、queue、D受理率、P/D balanceが違うため通常runの測定と混ぜない。NVLSが同様に1.10倍という仮定ならtarget AR減少は約94.8msで、kernel費用規模を比較する条件付き計算に留める。

## 来歴と限界

raw正本 `/srv/qwen-artifacts/37135273516/1`。解析 `/tmp/qwen-current-prefill-profile-analysis-20261004/` の `gemm_efficiency.py` / `gemm-efficiency.json` / stage-window / substages JSONが再計算の根拠。native source対応は読み取り専用source snapshotを使用した。全8rankでkernel/runtime/graph tableが非空で、Nsight diagnosticにdrop/lost/error報告なし。一方CPU Kineto multiple-subscriber warning、driver13.2対CUPTI13.1 warningはある。絶対的完全性や通常run性能の証明ではない。本稿は解析・改善設計であり、実装/E2E成否はaudit担当の後続履歴へ記録する。

実装/E2E追跡は[Hopper ES実験履歴](2026-10-04-moe-es-campaign.md)へ記録する。
