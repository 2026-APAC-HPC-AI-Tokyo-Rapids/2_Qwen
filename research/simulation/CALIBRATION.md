# Issue #15: ASPIRE予測の校正研究

追跡: https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/issues/15

後続のforward/NIXL校正と反復AIPerf評価は完了し、結果を [OPTIMIZATION.md](OPTIMIZATION.md) にまとめた。forward単体の誤差改善と、serving全体に残る過大予測を区別して報告する。

## 研究の問い

既存のAIPerf結果、実配備の構成、ASPIRE固有のGPU・通信測定を取得すると、汎用H100データによる予測誤差を減らせるか。

判定には同一モデルrevision・コンテナ・SGLang source・トポロジー・workload・profile有無が必要。校正に用いていない独立runに対するTTFT、ITL、throughputの相対誤差とrequest単位のSLA判定を比較する。校正用runへの一致のみで精度改善を宣言しない。

## 履歴から引き継ぐ点

- Qwenの9/16会話とresearch: scheduler状態、worker/profile境界、raw artifactの退避先を確認。実行時間全体から演算・通信ボトルネックを推測しない。
- 前段のAISimulate調査: `bc3fda9` の候補は未校正。TP8 prefillのsimulator拒否は実配備SGLangの失敗原因の証明ではない。

## 収集経路

`collect.py` はarchiveを読み取り専用で走査し、source/provenance、model、container、worker ServerArgs、AIPerf metrics/workload、環境確認、IBカウンタを収集する。exportの同内容コピーは同じrun内でSHA-256により重複除外する。warmupは除外。曖昧な複数export、別cluster、profile、欠けた来歴、単位不一致、終了/ローカル検証の証跡欠落を比較不可として記録する。元ファイルのパスとSHA-256を保存し、rawをGitへ複製しない。

`simulation.yml` のinventory jobは `astral-sh/setup-uv` でuvを導入し、共有archiveを次のコマンドで走査する。自宅サーバーへの接続は `ssh archive-host`。ホストのPython導入状態には依存しない。

```bash
uv run --no-project --python 3.12.12 python research/simulation/collect.py /srv/qwen-artifacts /tmp/qwen-inventory.json
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
- AISimulateのFPMにはexact model/backend/version/parallelism/precisionとParquet+metadataが必要。[upstream FPM workflow](https://github.com/ai-dynamo/aisimulate/blob/992dddc7ee2bad8fd523e7db99eea4ac64254766/python/aisimulate/docs/fpm/end-to-end-workflow.md)では既存collectorはvLLM中心。SGLang収集の契約と対応を先に検証し、現行SGLangへ未確認のFPM collectorを差し込まない。

最終的な正しさ・公式性能は公式AIPerf条件と主催者の再検証で判定する。この研究のmicrobenchmarkは公式taskの代替ではない。

## 既存archiveの調査結果（2026-09-26）

Actions [36221902702](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/36221902702) はuvによるinventory収集とCPU予測の両方に成功した。artifact `calibration-inventory-ae6013e72030b9606ba97f44ec1a3a64c2e9c09c` の集計は次のとおり。

|項目|件数|
|---|---:|
|provenanceを持つ記録|25|
|ASPIRE記録|7|
|ASPIRE AIPerf export|0|
|同条件の比較候補|0|
|JSON読取エラー|0|

ASPIREの予測誤差を算出できない直接の理由は、保存済み7記録にAIPerf exportがないこと。最新の旧ASPIRE run `35108498091` のartifactにはmodel/provenance/container-sidecarの3ファイルだけがあり、停止原因を特定するログがない。一方、環境確認run `34700473422` は2ノード各1 GPUでH100 80GB HBM3、driver 595.71.05、torch 2.11+cu130、NIXL 1.4.1を記録している。今回の2ノード各8 GPU測定で割当時の構成を新たに取得する。

### 収集経路で観測した失敗と修正

- `36221806212`: runnerに `python3` がなくinventoryがexit 127。`ae6013e` でsetup-uvと `uv run --python 3.12.12` を導入し、上記runで収集成功を確認した。
- `36221806385`: PBS `212963.pbs111` がRの時、待機チェックのexit 75がrunner側で1になり監視が終了。短時間のremote-shell診断でも再現した。`f17df11` でpendingを成功した問い合わせの出力として扱ったが、再実行 `36222189137` でも明示的exit 0が1に変わった。runner経由の短時間診断では、明示的exit 0は1、自然終了は0と再現した。`bc96649` でcheckerを自然終了させるよう修正。待機・正常終了・異常終了はfixtureで検証した。回収済みrawは自宅archiveの同run/attempt 1にある。
- 同runのFirmus archive確定はSSH接続切断（exit 255）。ASPIREの既存コンテナ選択・archive確定は成功した。

### 精度改善の評価手順

1. ASPIREの公式workloadを通常benchとして完走し、公式parserの結果、AIPerf export、実配備構成を回収する。同条件を反復して変動幅を得る。
2. 別profile runでSGLangのprefill/decode別whole-forward時間とNIXLのpayload・転送時間を取得する。collectorとAISimulateのFPM入力契約を対応付ける。
3. 実測で校正した予測と汎用H100の予測を、校正に使用していない同条件runへ適用する。TTFT/ITL/throughputの相対誤差とSLA判定を比較する。

GPU copy、BF16 GEMM、NCCLの短時間測定は、汎用profileのどの仮定を優先的に調べるか決める材料になる。精度改善の判定は上記の独立したQwen測定で行う。

継続研究: [Issue #16](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/issues/16) に反復baseline、SGLang/NIXL計測、独立runによる誤差比較を整理した。

`bc96649` のrun `36222428390` では監視がpendingのまま継続できた。`bootstrap-a2ap-dgx019.log` が `PBS_NODEFILE: PBS node list required` を記録し、片側のpbsdsh childにPBS_NODEFILEがないと特定できた。`c656a83` で既存 `check-qwen-container.pbs` と同様、親PBSで割当ノード一覧を共有ファイルへ出力し、各workerがそこからrankを決める方式に修正した。

## ASPIRE固有性能の取得結果

`c656a83` の [Actions 36222671527](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/36222671527) は全体がsuccess。PBS `212968.pbs111` は `F / Exit_status=0`、実行36秒。`a2ap-dgx019` / `a2ap-dgx031` 各8 GPUの16 rankを検証し、集計・upload・自宅archive保管・計算側staging削除まで完了した。

正本は `ssh archive-host` で参照できる `/srv/qwen-artifacts/36222671527/1`。`SHA256SUMS` を現地で再検証し成功した。`summary.json` が集計、`results/results/rank-*.json` が5サンプル×各測定点のraw、`logs/logs/pbs-accounting.txt` がscheduler証跡を持つ。

|測定条件|今回の観測値|
|---|---:|
|256 MiB device copy、read+write合計、rank中央値の中央値|2,979.54 GB/s|
|BF16 GEMM 4096³、rank中央値の中央値|782.84 TFLOP/s|
|ノード内8-rank all-reduce、256 MiB、最も遅いrankの中央値|241.71 GB/s（algorithm bandwidth）|
|ノード間2-rank all-reduce×8組同時、256 MiB、最も遅いrankの中央値|45.29 GB/s（algorithm bandwidth）|

rank 0のGPUはNVIDIA H100 80GB HBM3、CUDAから見えるmemory 85,019,590,656 bytes、132 SM。torch 2.11.0+cu130、CUDA 13.0、NCCL 2.28.9。GPU/NIC topologyとdriver XMLも同じarchiveへ保存した。

通信はpayload依存が大きい。ノード間ペアall-reduceのalgorithm bandwidthは1 MiBで14.67 GB/s、16 MiBで38.18 GB/s、256 MiBで45.29 GB/sだった。したがって校正候補はpayload別に扱う必要がある。先のAISimulateで置いたKV転送50 GB/sはaggregate NIXLの仮定であり、今回のペアごとのNCCL値と対応する測定はIssue #16で取得する。

今回得られたのは1割当内の反復測定。ASPIRE固有のoperator診断値を再取得できる経路が完成した。予測精度改善の判定に必要なQwenのwhole-forward/NIXL実測と、校正に使用しないAIPerf評価runはIssue #16の完了条件にした。

検証: 新規13 unittest、既存145 pytest、Ruff、actionlint、シェル構文検査に成功。CPU run `36222671404` もsuccess。失敗した先行run `36222428390` はterminal failureとなりraw保管が成功。測定後のqstatで残留ジョブがないことを確認した。
