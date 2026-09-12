# 公式Qwen3 PD実装記録

## 正本

- 公式Task Rules: <https://github.com/hpcac/2026-APAC-HPC-AI/blob/main/task/1.%20Qwen3%20AI%20Task%20Rules.md>
- Firmus Application Notes: <https://github.com/hpcac/2026-APAC-HPC-AI/blob/main/task/3.1%20Qwen3%20Application%20Notes%20-%20Firmus%20AI%20Cloud.md>
- AIPerf v0.12.0: <https://github.com/ai-dynamo/aiperf/releases/tag/v0.12.0>
- 実装追跡: <https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/issues/13>

2026-09-12にTask Rules、Application Notes、公式Slackを再照合した。公式runはQwen3-235B-A22B-Instruct-2507のFP8 weights、2ノード16 GPU、Prefill–Decode分離、単一OpenAI互換gatewayを必須とする。AIPerfはclosed-loopで、固定長4000/200 token、warmup 60秒、benchmark 120秒、grace 30秒、seed 2026を使う。TTFT 3000 msと平均ITL 20 msを同時に満たすrequestの割合が0.90以上のrunだけが採点対象で、scoreは`goodput(request/s) × 200`である。

## 固定した来歴

| 対象 | 固定値 |
|---|---|
| Model repository | `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` |
| Model revision | `e156cb4efae43fbee1a1ab073f946a1377e6b969` |
| Tokenizer repository | `Qwen/Qwen3-235B-A22B-Instruct-2507` |
| Tokenizer revision | `ac9c66cc9b46af7306746a9250f23d47083d689e` |
| AIPerf | `v0.12.0`, commit `be53bf2953d30e46c500e6a80fc1f8b6f84bc718` |
| sglang-router | `0.3.2` |

Firmusの`/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8`には24個のsafetensors shardがあり、Hugging Face download metadataのrevisionも表のmodel revisionと一致した。玄界にも221 GiBのweightsがある。ASPIRE 2A+は確認時点で77 GiBのdownload cacheだけで、model rootにsafetensorsがないため実行対象にしていない。

## 実行と判定

Firmusの公式検証済み開始構成を採用し、node 0へ4 Prefill workerをTP2で、node 1へ2 Decode workerをTP4/EP4で置く。Decodeの`max-running-requests`は256である。全workerでNIXLを選び、計算ノードごとにactiveなInfiniBand deviceを`UCX_NET_DEVICES`へ渡す。各workerのAPI portと各Prefill workerのbootstrap portは重複させない。

`scripts/run-qwen-pd-job.sh`はworker、router、AIPerfの順に起動する。AIPerfから見えるendpointはrouterの`/v1/chat/completions`だけである。`scripts/run-aiperf.sh`は公式modeと探索modeを明示的に分け、公式modeのworkload引数を固定する。探索modeだけは`configs/firmus-pd.json`の短い時間を使える。どちらにも`--request-rate`を追加しない。

`scripts/parse-aiperf-result.py`はAIPerf 0.12.0の`profile_export_aiperf.json`からgoodput、good request fraction、request/error count、観測ISL/OSLを読み、公式の0.90 gateとscoreをJSONへ保存する。公式modeでgate未達ならjobを失敗させる。探索modeでは未達点も分析可能なartifactとして残す。

通常runの観測はservice log、AIPerf artifacts、workerのPrometheus snapshotである。`[profile]` runだけは1秒間隔のGPU telemetryも取得する。profileの計測負荷を含む値を通常runのscoreとして扱わない。
