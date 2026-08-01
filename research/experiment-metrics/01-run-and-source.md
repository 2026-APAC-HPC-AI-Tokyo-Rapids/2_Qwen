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
`requirements-frozen.txt`全体はhashで同一性を表し、性能に直接関係する主要packageだけを
個別キーにする。

## 入力とartifact

| キー | 取得元 |
|---|---|
| `artifact.sif.source_workflow_run_number` | `sif-source-run.txt` |
| `artifact.sif.size_bytes` | `stat` |
| `artifact.sif.sha256` | SIF file |
| `artifact.sif.requirements_frozen.sha256` | SIF内file |
| `artifact.runtime_script.run_qwen35.sha256` | runtime bundle |
| `artifact.runtime_script.exec_profiler.sha256` | runtime bundle |
| `artifact.model.config.sha256` | model `config.json` |
| `artifact.dataset.sha256` | ShareGPT JSON |
| `artifact.dataset.size_bytes` | `stat` |

巨大SIFとdatasetのSHA-256をrunごとに計算すると共有filesystem I/Oが増える。
実装時はprovision段階で一度計算しruntime bundleへscalarとして渡すか、同じ
hardlink/source runに対するcacheを共有filesystem上に置く。ただしcacheを使う場合も
`size`, `mtime`, inodeだけで内容一致を断定せず、hashの正本を明確にする。

modelは複数fileの集合なので、`config.json`だけではcheckpoint全体を識別できない。
Hugging Face local metadataからcommitを機械的に得られる場合は
`model.huggingface.commit`を追加する。得られない場合は公称repository revisionを
代入せず空文字にし、将来canonical manifest hashを追加する。
