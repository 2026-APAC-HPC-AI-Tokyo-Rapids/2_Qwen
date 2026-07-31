# 取得・送信lifecycle

## provision

Actions runnerはruntime bundleへ次を追加する。

```text
qwen-experiment-metrics/
qwen-experiment-metrics/.venv/
```

OpenFOAMと同様、checkoutからprojectをcopyし、`uv sync --frozen --no-dev`する。
collector変更のためにSIFを再buildしない。

runnerはrepository secret
`EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_JSON`をBase64化し、次とともにPJMへ渡す。

```text
GITHUB_RUN_ID
GITHUB_RUN_ATTEMPT
GITHUB_REPOSITORY
GITHUB_SHA
EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_BASE64
```

## PJM開始

PJM scriptは認証Base64をshell変数へ移して元の環境変数をunsetする。
metrics CLIは呼び出しごとにcommand-scoped environmentから認証を受け取り、
process開始直後にメモリへcaptureしてunsetする。

順序は次に固定する。

```text
run-start送信
  → host platform / env送信
  → source / artifact identity送信
  → static NVIDIA GPU送信
  → benchmark stage開始
```

## benchmark stage

host側stage runnerは次を一つのsupervisor processで行う。

```text
Unix socket listen
  → NVIDIA sampler開始
  → Singularity benchmark起動
  → SGLang configuration event受信・即送信
  → SGLang result event受信・即送信
  → benchmark wait
  → sampler停止・GPU集約送信
  → stage wall time / exit code送信
```

event受信とGPU samplingはthreadまたはasync taskで並行させる。senderは同一batchの
順序を保ち、複数threadからBigQuery clientを同時操作しない。

socket pathはrun固有directory内に作り、終了時にunlinkする。socketにはmetric dataを
永続化しない。SGLang processが接続しなくてもstageはbenchmarkを実行し続ける。

## PJM終了

PJM scriptのEXIT trapは可能な範囲で次を行う。

```text
run-end送信
  → 従来のJIT cache成功時merge
  → node-local temporary directory cleanup
```

metrics送信失敗を元のbenchmark成功へ上書きしない。逆にJIT cache merge失敗等の
既存pipeline errorをmetrics collectorが隠さない。

## runner側終了処理

PJM終了後、runtime directoryを削除する前にrunner側CLIで`pjstat --history`を読み、
scheduler batchを送る。credential JSONはrunner環境から直接captureする。

成功・失敗・cancelのいずれでも、既存のlog/result artifact uploadとscheduler収集を
試みる。実行中ActionsやPJM jobをmetrics導入の都合で停止しない。

## batchと重複

同時に取得できた値を次のbatch単位で送る。

```text
run start
host platform + env
source + artifact identity
static GPU
SGLang configuration
benchmark result
GPU runtime aggregate
stage result
run end
scheduler accounting
```

各batchは送信成功までprocess memoryにだけ保持する。HTTP応答が不明な再試行では
重複し得るため、OpenFOAMと同じdeterministic `insertId`でbest-effortに抑制する。
raw table以外のviewは作らない。
