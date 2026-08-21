# 旧Qwen 3.5 runtime構成

この文書はOCI / Enroot移行前の正本を履歴として残したものである。現行構成は [OCI / Enroot移行記録](../container-runtime-migration.md) を参照する。

ここでいう「現在」は `main` の [Genkai workflow](../../.github/workflows/genkai.yml) と [PJM script](../../scripts/run-qwen35.pjm) を指す。

## provision

workflow 全体で model と dataset の共有 filesystem 上の path を定義する。

```text
MODEL_PATH=/fast/PROJECT/Qwen/Qwen3.5-397B-A17B-FP8
DATASET_PATH=/fast/PROJECT/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json
```

`provision` job は GPU を使わず、host の `jq` で model config が Qwen 3.5 であることを確認する。さらにsafetensors indexを読み、列挙された全shardが存在し、空でないことを確認する。巨大なcheckpoint全体を読み直すchecksum検証は行わない。

ShareGPT は `hf download` により上記の path へ取得する。model と dataset は SIF に封入しない。

検証後、`/fast/PROJECT/sglang-runs` に実行ごとの一時 runtime directory を作る。その中に次を揃える。

```text
sglang.sif
sif-source-run.txt
scripts/
logs/
results/
```

SIF は別ファイルシステム上のarchiveを指すsymbolic linkであり、scriptはそのworkflowのcheckoutからcopyする。PJM scriptの内容と出力先はrepository checkoutやbuild directoryに依存しない。ただし、後述のとおり、現行の投入cwdにはcheckoutへの依存が残っている。

## PJM job の lifecycle

workflow は `pjsub -x` で `RUNTIME_DIR`、`MODEL_PATH`、`DATASET_PATH`、workflow metadata を渡し、Node Group B の2ノード（各ノードH100 4 GPU、合計8 GPU）、10 分の PJM job を投入する。標準の `pjwait` で開始と終了を待ち、実行中の log を Actions に stream する。

benchmark の終了 code は profiler wrapper、PJM script、workflow へ伝播する。成功時だけ `results.jsonl` の存在を確認して `run-success` marker を作る。workflow は job 終了後にこの marker を検証する。

成功、失敗、cancel のいずれでも logs と results の回収を試みる。その後、残っている PJM job を停止してから、一時 runtime directory を削除する。

ただし、現行 workflow は `pjsub` 前に共有 filesystem 上の `RUNTIME_DIR` へ移動していない。ephemeral runner の checkout がqueue待ち中に消えると、PJM jobが `NO CURRENT DIR` で開始前に終了することを過去に確認している。経緯と必要な修正は [trial-record.md](trial-record.md#7-ephemeral-runner-と-no-current-dir) を参照する。

## 計算ノード上の実行環境

PJM script は NVIDIA HPC SDK 25.11、HPC-X 2.25.1、CUDA 13.2.0、Singularity CE 4.1.3をloadする。Open MPIはCPU bindingを無効化して各ノードに1プロセスを配置し、それぞれのcontainerが4 GPUを担当する。SGLangは`tp_size=8`、`nnodes=2`としてノード間tensor parallelismを構成する。Genkai計算ノードでは通常のcontainer process生成が許可されないため、containerは`--userns --nv --cleanenv --containall`で起動する。FUSE mountも許可されない場合はSingularityの一時sandbox展開へfallbackする。次だけを明示的にbind mountする。

- model: `/model:ro`
- dataset の単一 JSON: `/dataset/ShareGPT_V3_unfiltered_cleaned_split.json:ro`
- node-local disk上のwritable homeとJIT cache: `/runtime-home` と `/jit-cache`
- 今回の result directory: `/output`

Hugging Face、Transformers、Datasets は offline mode にする。GPU architecture や device を必要とする Triton、DeepGEMM などの JIT は計算ノード側だけで実行する。推論引数は環境変数ではなく、PJM script 内の command として版管理する。

## JIT cache

永続 cache は共有 filesystem の `${HOME}/.cache/sglang-jit` に置く。ただし benchmark 中に共有 filesystem を直接読まず、各計算ノードの `/tmp/sglang-jit.*` へ snapshot を staging して bind mount する。

staging は共有 lock、成功後の `rsync` による書き戻しは排他 lock で保護する。benchmark が失敗または cancel された場合は local cache を書き戻さない。node-local JIT cache と Singularity の一時 directory は終了時に削除する。

GPU構成と1秒間隔のtelemetryは各ノードで取得し、raw sampleをdiskへ保存せずmemory上で
GPUごとのscalarへ集約する。rank 0のcollectorが2ノード分をBigQueryへ送り、
`hardware.gpu.count`は各ノードの実測GPU数の合計とする。

## 固定 benchmark command

現在は SGLang の `sglang.benchmark.offline_throughput` を次の主要条件で実行する。

| 項目 | 値 |
|---|---|
| load format | `auto` |
| tensor parallel size | 8 |
| dataset | ShareGPT |
| prompt 数 | 32 |
| output length | 256 |
| context length | 262144 |
| seed | 42 |
| static memory fraction | 0.85 |

結果は `/output/results.jsonl`、command と console output は `/output/throughput.log` に保存する。
