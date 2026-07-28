# Qwen 3.5 runtime の試行記録

この文書は、2026-07-12 から 2026-07-28 までの Codex 会話、GitHub Actions、PJM job の診断結果を整理したものである。現在の構成で継続的な性能比較を始める前に、実行環境と lifecycle の問題を切り分けた記録であり、throughput の性能結論ではない。

## 1. runtime bundle を作る理由

初期案では repository checkout から直接 PJM script を起動する構成も検討した。しかし Actions runner の checkout は一時的であり、長い queue待ちの間に runner が再登録される可能性がある。また、どの SIF、script、結果が一つの実験に属するか分かりにくい。

そのため、共有 filesystem 上の run 固有 directory に次をまとめる。

- 選択済み SIF の hardlink
- その workflow revision の PJM script と profiler wrapper
- SIF の再利用元 run number
- logs
- results

model と dataset は大きく、複数実行で共通なので bundle へ copyせず、固定 pathから read-only bindする。

## 2. model と dataset の扱い

入力 dataset は SGLang の標準的な ShareGPT workload とし、SIFには入れない。`provision` で Hugging Face Hub の最新 datasetを共有 pathへ取得し、実行時は単一 JSONだけを bindする。

modelは単に `config.json` の存在を見るだけでなく、Qwen 3.5の model type / architectureであることを hostの `jq` で検査する。さらに `hf cache verify` により local checkpointをHubの最新 revisionと照合する。

この設計ではHub側の repositoryが更新されると、以前は完全だった local checkpointも不足扱いになることがある。これは「固定済みlocal copyを永久に使う」設計ではなく、「実行時点の最新Hub revisionに対して完全であることを要求する」設計上の選択である。性能比較では model revisionが変わっていないことも確認する必要がある。

## 3. JIT cache を共有 filesystemから直接読まない理由

初期実行では、共有 filesystem 上の SGLang cacheをbenchmarkから直接参照すると読み取りが不安定になることがあった。現在は実行開始時に共有 lockを取り、node-local directoryへ一貫したsnapshotを stagingする。成功したrunだけを排他 lock下で共有 cacheへmergeする。

失敗runの不完全なJIT生成物は書き戻さない。複数runはsnapshotの読み取りを並行できるが、mergeは直列化する。

## 4. `/dev/shm` ではなく `/tmp` に JIT cache を置く理由

計算ノードの `/dev/shm` は実効的に `noexec` だった。Tritonが生成した `cuda_utils.so` をここからloadしようとすると、次のような失敗になった。

```text
cuda_utils.so: failed to map segment from shared object
```

同じ計算ノードの `/tmp` は node-local な XFSで、`exec` が許可されていた。診断したノードでは2 driveのmd RAID上にあり、buffered fioはread/writeとも約1.6 GiB/sだった。物理diskの持続帯域そのものではないが、Triton、DeepGEMM、TorchInductorのJIT cacheには十分である。

このため、実行可能な `.so` を生成する JIT cacheだけを `/tmp/sglang-jit.*` に置く。Singularityの一時領域は `/dev/shm/sglang-run.*` のままである。両者は用途が異なるため、一括して同じscratchへ移さない。

## 5. FlashInfer cache の二層

名前が似ているが、次は別物である。

| 対象 | 役割 |
|---|---|
| `flashinfer-jit-cache` Python package | 事前配布された多数のkernel artifact。SIF内でread-only |
| `/root/.cache/flashinfer` | 未収録条件のJIT結果、生成source、log。実行時にwriteが必要 |

事前配布packageを導入しても、runtime workspaceは必要になる。そこでnode-local JIT cacheを `/root/.cache` にも bindし、SIF内のartifactを利用しながら不足分だけを書き込む。SIF testでは一時的なwritable tmpfsを使うが、そこで生成したcacheは永続化しない。

## 6. PJM の待機とログ表示

`pjwait <job-id>` はqueue待ち、実行、終了まで全体を待つ。実行開始だけを通知するコマンドとしては使えない。

Actions上でqueue待ちと計算ログを別stepに見せるため、現在は次の順にしている。

1. `pjsub` でjob IDを得る。
2. `pjwait -w waitnone` とlog fileの存在を使い、job開始または開始前終了を判定する。
3. log fileができた後、別stepで `tail --follow=name` と `pjwait -z` を並行する。
4. 最後に `pjwait` の完了状態と `run-success` markerを確認する。

独自の大きなscheduler polling scriptは一度導入したが、`pjstat` 自体の一時失敗とjob終了の区別が複雑になったため撤去した。標準の `pjwait` を中心にし、開始待ちに必要な最小限だけを残す。

## 7. ephemeral runner と `NO CURRENT DIR`

2026-07-26 のActionsでは、PJM jobが長時間queueにいる間にself-hosted runnerがGitHubとの通信を失った。buildとprovisionは成功していたが、artifact回収とcleanupは実行されず、PJM jobだけが残った。

診断では次を確認した。

- host自体は再起動していない。
- runnerを管理するuser systemdが途中で再起動し、runnerが別IDで再登録された。
- `Linger=no` で、runnerはuser service配下にあった。
- `nohup` ではsystemd cgroupの停止を回避できない。

孤立したPJM jobは後に開始したが、GPU処理へ入る前に終了した。

```text
PJM code: 16
reason: NO CURRENT DIR
GPU execution time: 0
```

`pjsub` を実行したcwdがephemeral runnerのcheckoutであり、job開始時にはそのdirectoryが削除されていたことが原因である。runtime bundle自体は共有 filesystemに残っていても、schedulerが最初にcwdへ移動できなければPJM scriptは開始されない。

### 現在の未解決点

現行 [workflow](../../.github/workflows/genkai.yml) の `Submit Qwen job` は、`RUNTIME_DIR` 内のscriptを絶対pathで指定しているが、`pjsub` の前に `RUNTIME_DIR` へ `cd` していない。このため、同じrunner再登録がqueue待ち中に起きれば `NO CURRENT DIR` が再発する可能性がある。

修正する場合は、少なくとも `pjsub` 前に共有 filesystem上の `${RUNTIME_DIR}` へ移動し、投入cwdもruntime lifecycleの一部にする必要がある。runnerそのものについては、`loginctl enable-linger` またはsystem service化が本筋であり、workflowだけでは通信断を完全には防げない。

## 8. 現時点の検証範囲

過去のActions #80では、当時のSIFとruntime構成でend-to-end実行に成功した。その後の #91/#94 はFlashInfer version不整合、さらに新しいrunはrunner消失と `NO CURRENT DIR` で失敗している。

したがって、現在のrecipe、現在のJIT cache配置、現在のruntime bundleを組み合わせたend-to-end成功は、会話履歴の範囲ではまだ確立していない。throughputの比較へ進む前に、次を一度のActions runで確認する必要がある。

1. SIF buildまたは再利用
2. modelとdatasetのprovision
3. queue待ちを越えたPJM script開始
4. GPU上のFlashInfer、Triton、SGLang初期化
5. `results.jsonl` の生成
6. JIT cacheの成功時merge
7. artifact回収とruntime cleanup
