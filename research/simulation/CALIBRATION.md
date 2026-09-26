# Issue #15: ASPIRE予測の校正研究

追跡: https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/issues/15

## 研究の問い

既存のAIPerf結果、実配備の構成、ASPIRE固有のGPU・通信測定を取得すると、汎用H100データによる予測誤差を減らせるか。

判定には同一モデルrevision・コンテナ・SGLang source・トポロジー・workload・profile有無が必要。校正に用いていない独立runに対するTTFT、ITL、throughputの相対誤差とrequest単位のSLA判定を比較する。校正用runへの一致のみで精度改善を宣言しない。

## 履歴から引き継ぐ点

- Qwenの9/16会話とresearch: scheduler状態、worker/profile境界、raw artifactの退避先を確認。実行時間全体から演算・通信ボトルネックを推測しない。
- 前段のAISimulate調査: `bc3fda9` の候補は未校正。TP8 prefillのsimulator拒否は実配備SGLangの失敗原因の証明ではない。

## 収集経路

`collect.py` はarchiveを読み取り専用で走査し、source/provenance、model、container、worker ServerArgs、AIPerf metrics/workload、環境確認、IBカウンタを収集する。exportの同内容コピーは同じrun内でSHA-256により重複除外する。warmupは除外。曖昧な複数export、別cluster、profile、欠けた来歴、単位不一致、終了/ローカル検証の証跡欠落を比較不可として記録する。元ファイルのパスとSHA-256を保存し、rawをGitへ複製しない。

```bash
scp research/simulation/collect.py archive-host:/tmp/qwen-collector.py
ssh archive-host 'python3 /tmp/qwen-collector.py /srv/qwen-artifacts /tmp/qwen-inventory.json'
scp archive-host:/tmp/qwen-inventory.json /tmp/qwen-inventory.json
```

出力先は未作成のファイルを指定する。GPU不要。`simulation.yml` のinventory jobでも同じ収集を実行しartifactへ保存する。`condition_key` は比較候補の構成をまとめるためのキーで、同一GPU個体・実行時クロック・温度・配置の一致までは保証しない。

IBの `port_xmit_data` / `port_rcv_data` は4 octets単位として差分を変換する。reset・非正時間間隔を捨てる。算出値は観測期間のport trafficであり、リンク容量、NIXL単体帯域、benchmark区間だけの通信量ではない。

## 固有性能の短時間測定

`probe.py` はPBS割当内だけで動作し、2ノード各8 GPUに対して同一条件を取得する。

- GPU名、実メモリ量、SM数、PyTorch/CUDA/NCCL version。
- 256 MiBのdevice copy（読み＋書きのbyte量を明示）。
- 4096³ BF16 GEMM（FLOP数を明示）。
- 1 / 16 / 256 MiBのノード内8-rank all-reduce。
- 同じサイズのノード間2-rank all-reduceを8組同時実行。
- 各点warmup 3回、10反復×5サンプル。CUDA eventでGPU完了まで測定し、raw latencyを保存。
- host側でnvidia-smi XML/topology、IB state/rate/link_layer、共有・node-local filesystemを保存。rateは公称値であり実効帯域ではない。

`summarize_probe.py` は16 ranks・2 hosts・同一job・測定点の完全性を検証する。コピー帯域、BF16 TFLOP/s、NCCL algorithm bandwidthを別の値として集計し、NIXL帯域やQwen whole-forward calibrationはnullのままにする。

投入は既存Qwen workflowから `[cluster:aspire2ap] [nodes:2] [profile] [calibration]` を付けたcommitで行う。既存SquashFSを再利用し、測定スクリプトを共有ディレクトリからコンテナへmountする。serving source/依存を変えないため新規コンテナbuildは不要。通常のASPIRE PDベンチは同じpushでは投入しない。ジョブwalltimeは10分。最大割当は16 GPU×10分=2.67 GPU hours（待機課金やサイト補正は含まない）。model load、Shakespeare生成、BigQuery認証情報の配布は行わない。

結果は既存の `qwen-artifacts/<Actions run>/<attempt>` へ退避する。`run-success`とPBS terminal exitだけでなく、16 ranksのJSONと測定値を検査する。GPU固有の成功はこのE2E後にのみ記録する。

## AISimulateへ反映できる範囲

- hardware inventory: H100 SXM、実メモリ、ライブラリ、GPU/NIC配置の仮定を検査できる。
- operator diagnostics: HBM/GEMM/NCCLの汎用profileとの大きな乖離を検出できる。測定したshape/precision/collectiveの範囲に限定する。
- AIPerf E2E: 同条件で予測誤差を検証できるが、TTFTはqueue/prefill/通信/routerを含むためkernel latencyへ直接変換しない。
- 真の校正には実配備SGLangのFP8 GEMM/MoE/attention、role別whole-forward時間、NIXLの実payload・転送時間が必要。BF16 GEMMやNCCLを代用品にした補正係数を生成しない。
- AISimulateのFPMにはexact model/backend/version/parallelism/precisionとParquet+metadataが必要。[upstream FPM設計](https://github.com/ai-dynamo/aisimulate/blob/992dddc7ee2bad8fd523e7db99eea4ac64254766/python/aisimulate/docs/fpm/aic-fpm-modeling-plan.md)では既存collectorはvLLM中心。SGLang収集の契約と対応を先に検証し、現行SGLangへ未確認のFPM collectorを差し込まない。

最終的な正しさ・公式性能は公式AIPerf条件と主催者の再検証で判定する。この研究のmicrobenchmarkは公式taskの代替ではない。
