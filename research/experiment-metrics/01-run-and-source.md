# 実行識別・source・入力artifact

## run名

Actionsから投入する通常経路は次に固定する。

```text
qwen-<GITHUB_RUN_ID>-<GITHUB_RUN_ATTEMPT>-<PJM_JOBID>
```


```text
qwen-manual-<PJM_JOBID>-<YYYYMMDDTHHMMSSffffffZ>
```

現行workflowは`GITHUB_RUN_ATTEMPT`と`GITHUB_REPOSITORY`をPJMへ渡していないため、
実装時に追加する。

## run lifecycle

| キー | 論理型 | 取得元 |
|---|---|---|
| `run.started_at` | timestamp | collector UTC clock |
| `run.finished_at` | timestamp | collector UTC clock |
| `run.elapsed_sec` | float | PJM scriptのmonotonic clock |
| `run.outcome` | enum | supervisor |
| `run.exit_code` | int | benchmark pipeline |
| `run.signal` | int | signal終了時 |
| `run.command` | string | 実argvを`shlex.join()`したshell表現 |

`run.outcome`は`succeeded`, `failed`, `cancelled`だけを使う。PJM scriptが構築した
arrayをcollectorへ渡し、空白や引用符を含む引数も復元できるよう`shlex.join()`で
単一のshell表現へ変換する。

## software source

repository側commitは`env.GITHUB_SHA`を正本とし、別名へ複製しない。
SIF内SGLangはrepository commitと独立しているため、次を個別に保存する。

| キー | 取得元 |
|---|---|
| `software.sglang.commit` | SIF内`/opt/sglang/sglang-commit.txt` |
| `software.sglang.version` | SGLang `__version__` |
| `software.python.version` | benchmark process |
| `software.torch.version` | benchmark process |
| `software.cuda.runtime_version` | PyTorch runtime |
| `software.cuda.driver_version` | NVIDIA driver query |
| `software.singularity.version` | host `singularity --version` |

packageを問答無用で全件展開するとrun当たりのrow数とversion差分noiseが大きくなるため、
性能に直接関係する主要packageだけを個別キーにする。SIFは生成元workflow run番号、
SGLang commit、主要package versionで識別する。

## 入力とartifact

| キー | 取得元 |
|---|---|
| `artifact.sif.source_workflow_run_number` | `sif-source-run.txt` |
| `artifact.sif.size_bytes` | `stat` |
| `artifact.dataset.size_bytes` | `stat` |

artifact hashを実験条件の代理にしない。runtime scriptは`env.GITHUB_SHA`と
`run.command`、SIFは生成元workflow run番号、modelとdatasetはSGLang実引数および
workflow環境から識別する。sizeは成果物の存在と回収状態を診断する補助値である。
