# NVIDIA GPU構成・実行中telemetry

## 対象

Genkai node group BはNVIDIA H100を搭載する。実際に割り当てられたfull GPUまたは
MIG instanceだけを計算ノード上で問い合わせ、公称構成を代入しない。

GPU keyのindexはjobから見える論理indexとし、UUIDとPCI bus IDを必ず併記して
物理deviceへ対応付ける。

## 静的構成

benchmark開始前に`nvidia-smi --query-gpu=... --format=csv,noheader,nounits`を
shellを介さず実行する。

```text
hardware.gpu.count
hardware.gpu.<index>.logical_index
hardware.gpu.<index>.uuid
hardware.gpu.<index>.name
hardware.gpu.<index>.pci_bus_id
hardware.gpu.<index>.compute_capability
hardware.gpu.<index>.memory.total_bytes
hardware.gpu.<index>.power.limit_watt
hardware.gpu.<index>.mig.mode.current
hardware.gpu.<index>.driver_version
```

query fieldがdriver versionで未対応なら、そのfieldだけ空文字にする。単位付きCSVを
直接保存せず、`nounits`で取得してbyte/Watt等へ変換する。

MIG利用時はGPU instance/compute instance UUIDとprofileを別階層にする。full GPUと
MIG deviceを同じUUID namespaceへ押し込まない。

## sampling範囲

host側stage wrapperがSingularity processを起動する直前にsamplerを開始し、
process終了直後に停止する。範囲には次を含む。

```text
model load → JIT/warmup → timed generation → Engine shutdown
```

sampling intervalは1秒を初期値とする。NVIDIAのutilization値自体が直近sample periodの
割合なので、過剰に短いpollingは行わない。

## raw sample

第1段階で各GPUから次を読む。

```text
timestamp
index
uuid
utilization.gpu
utilization.memory
memory.used
power.draw
temperature.gpu
clocks.sm
clocks.mem
pstate
```

raw sampleはcollector processのmemoryにだけ保持し、BigQueryにもartifactにも
保存しない。

## run-level集約

各GPUについて次を送る。

```text
resource.stage.benchmark.gpu.<index>.sample_count
resource.stage.benchmark.gpu.<index>.sample_interval_sec
resource.stage.benchmark.gpu.<index>.utilization.gpu_percent.mean
resource.stage.benchmark.gpu.<index>.utilization.gpu_percent.max
resource.stage.benchmark.gpu.<index>.utilization.gpu_percent.p50
resource.stage.benchmark.gpu.<index>.utilization.gpu_percent.p95
resource.stage.benchmark.gpu.<index>.utilization.memory_percent.mean
resource.stage.benchmark.gpu.<index>.utilization.memory_percent.max
resource.stage.benchmark.gpu.<index>.memory.used_bytes.mean
resource.stage.benchmark.gpu.<index>.memory.used_bytes.max
resource.stage.benchmark.gpu.<index>.power.draw_watt.mean
resource.stage.benchmark.gpu.<index>.power.draw_watt.max
resource.stage.benchmark.gpu.<index>.energy_joule
resource.stage.benchmark.gpu.<index>.temperature_celsius.max
resource.stage.benchmark.gpu.<index>.clock.sm_mhz.mean
resource.stage.benchmark.gpu.<index>.clock.sm_mhz.min
resource.stage.benchmark.gpu.<index>.clock.memory_mhz.mean
resource.stage.benchmark.gpu.<index>.pstate.<normalized-state>.sample_count
```

`energy_joule`は隣接するpower sampleをtimestamp差で台形積分する。sample欠損区間を
補間で埋めず、観測できた区間だけを積算する。観測時間も
`resource.stage.benchmark.gpu.<index>.sampled_duration_sec`として保存する。

percentileのalgorithmと補間方法は実装で固定し、unit testに具体例を置く。

## failure

- `nvidia-smi`が開始できなくてもbenchmarkは実行する。
- 特定fieldの`N/A`は空文字sampleとして扱い、0にしない。
- 有効sampleがない集約値は空文字にする。
- sampler異常終了は標準errorへ警告し、benchmark終了codeを変えない。
- benchmark processが通常失敗した場合も、それまでのsampleを集約して送る。
- node crashやSIGKILLで終了処理自体が動かなければruntime集約batchが欠ける。
  既送信のrun/platform/static GPU batchは残るため、欠損がrun未完了の印になる。

## DCGMは第2段階

DCGMが計算ノードで利用可能で、権限と同時metric groupをActions経由で確認できた場合は
次を追加候補とする。

```text
resource.stage.benchmark.gpu.<index>.engine_active_ratio.mean
resource.stage.benchmark.gpu.<index>.tensor_active_ratio.mean
resource.stage.benchmark.gpu.<index>.dram_active_ratio.mean
resource.stage.benchmark.gpu.<index>.pcie.rx_bytes_per_sec.mean
resource.stage.benchmark.gpu.<index>.pcie.tx_bytes_per_sec.mean
resource.stage.benchmark.gpu.<index>.nvlink.bytes_per_sec.mean
resource.stage.benchmark.gpu.<index>.xid_error_count
```

DCGM profiling fieldはhardware、driver、metric groupの競合により同時取得できない場合が
ある。利用可能性を推測したfallback値は入れず、unsupportedは空文字とする。
Nsight Systems/Computeは詳細profiling artifactの責務であり、通常runのcollectorには
組み込まない。
