# 実験メトリクス再設計計画

## 目的

メトリクスは、ログやGitを複製する保管庫ではない。次の四つの問いに、BigQueryと
スプレッドシートから短く答えるための索引である。

1. どのGit管理下の条件で実行したか。
2. Gitの外で、実行時にどの条件が選ばれたか。
3. benchmarkは妥当で、どれだけ速かったか。
4. 比較不能または要調査と判断すべき欠損がないか。

変更内容と変更理由の正本はGit、実行の詳細の正本はlog、run間比較の正本は
BigQuery viewとする。

## 正本

| 情報 | 正本 | メトリクスの役割 |
|---|---|---|
| script、設定、patch | `2_Qwen`のcommit | repositoryとcommitを指す |
| 仮説、変更理由、期待する効果 | commit messageのbody | 保存しない |
| SGLangの変更 | `5_SGLang_Src`のcommit | SIF provenanceからcommitを指す |
| stdout、stderr、SGLang結果 | run artifactのlog/JSONL | 比較に必要な観測値だけを索引化する |
| host/container環境 | 実processから取得した値 | Gitから復元できないため全件保存する |
| SIF、model、dataset | immutable artifact | revision、manifest、必要ならcontent hashを指す |
| throughput、speedup、平均、順位 | BigQuery view | raw tableへ保存しない |

Git管理されたrecipeやpatchごとにSHA-256 metricを作らない。`2_Qwen`のcommitとpathで
内容が一意になるためである。hashは、Git外のbinaryや大きなartifactの同一性、転送後の
冪等性を保証する場合に限定する。

## Gitによる実験条件の管理

- 性能、workload、parallelism、cache、build、collectorの挙動を変える変更はjob投入前に
  commitする。
- benchmarkはcleanなtreeからだけ投入する。dirty tree用のdiff metricは作らない。
- 一つのcommitは一つの説明可能な変更単位にする。複数fileが同じ仮説を実現する場合は
  同じcommitでよい。
- subjectは変更の要約、空行後のbodyは変更理由、仮説、予想する影響、確認方法を書く。
- 同じ条件の反復測定ではcommitを増やさず、同じcommitから複数runを作る。
- 性能条件をworkflow dispatch inputにしない。model、dataset、SGLang引数、配置は
  workflowまたはscriptへcommitする。
- SIFを再利用した場合、現在の実行commitとは別に、そのSIFを作った`2_Qwen` commitと
  `5_SGLang_Src` commitを記録する。

commit messageの形は次を基準とする。

```text
Pin one SGLang process to each NUMA domain

The current process spans both sockets while each GPU pair is local to one NUMA
domain. Bind one process per domain because the hypothesis is that local CPU and
memory placement shortens model initialization without reducing generation
throughput. Compare repeated load and generation times against the parent commit.
```

この文章を`experiment.hypothesis`や`run.note`へ複製しない。runからcommitをたどる。

## 保存形式

raw tableは現在の三列を維持する。

```text
run    STRING NOT NULL
key    STRING NOT NULL
value  STRING NOT NULL
```

値が自然に一つのscalarならそのscalarを文字列化する。順序付きの複数値は、添字付きkeyへ
分解せずcompact canonical JSONの一つのvalueにする。Pythonの`str(list)`は使わない。

collectorはBigQueryへ送る同じrowを、先にrun artifact内の`metrics.jsonl`へappendする。
BigQuery送信失敗時はこのfileから再送する。別のevent envelope、別schemaのspool、
型別value列は作らない。

```text
collect
  -> metrics.jsonlへappend
  -> 同じrun/key/valueをBigQueryへ送信
  -> logとmetrics.jsonlをrun artifactとして保存
```

nodeごとのcollectorは一時JSONLを作り、rank 0またはrunner側で一つへmergeする。
keyにnode番号を含め、同じ`(run, key)`を複数producerが所有しない。

## 重複を避ける規則

候補メトリクスAが次のいずれかなら保存しない。

1. Git commitとpathだけから復元できる。
2. 既存メトリクスBだけから決定的に計算できる。
3. 同じsource、scope、取得時点で、Aの集合がBの完全な部分集合である。
4. raw logの診断にしか使わず、runの選別、比較、group化には使わない。

完全な集合と、その一部を検索しやすくするための抜粋をraw tableへ同時に置かない。
抜粋はBigQuery viewで作る。

ただし、見た目が似ていても意味が異なる値は残す。

- command lineでrequestedだった値と、SGLangがdefault/auto解決したeffective値
- PJMへrequestedしたnode数と、schedulerまたは各node collectorがobservedしたnode数
- benchmark generation timeとprocess lifecycle全体のstage time
- 現在の実行commitと、再利用SIFを作ったbuild commit

全host環境を保存するなら、`MODEL_PATH`から導出した`model.name`や、`GITHUB_SHA`の別名を
作らない。resolved `ServerArgs`を全field保存するなら、その一部だけを別namespaceへ
複製しない。

## Qwen/SGLangで保存する最小の事実

### 実行と外部条件

- `run.started_at`
- `run.elapsed_sec`
- `run.exit_code`
- `run.command`: 実際のSingularity/SGLang benchmark commandを一つのshell表現で保存
- `env.*`: host job processの全環境変数
- `sglang.env.*`: benchmark container processの全環境変数
- `scheduler.*`: PJM終了後に得るactual accounting
- `hardware.node.<rank>.*`: CPU、NUMA、memory、hostname、cpuset、各GPUのUUIDと静的構成
- `parallel.node.<rank>.*`: process affinity、NUMA placement、GPU割当、実際に選ばれた
  network/collective transport
- `artifact.sif.*`: SIF build commit、SGLang commit、SIF ID、size、content hash
- `artifact.model.*`: immutable revisionまたはmanifest identity
- `artifact.dataset.*`: immutable revisionまたはmanifest identity

modelやdataset全体のhash計算が高価な場合は、artifact作成時に一度だけmanifestを作り、
runではそのimmutable manifest IDを記録する。path、size、mtimeの組だけをcontent identityと
呼ばない。

`run.finished_at`、`run.outcome`、GPU総数は、他の保存値から一意ならviewで導出する。

### requestedとeffective configuration

- `run.command`はユーザーが指定したactual argvを表す。
- `sglang.benchmark_arg.*`は解決済み`BenchArgs`の全fieldを表す。
- `sglang.server_arg.*`は初期化直前の解決済み`ServerArgs`の全fieldを表す。

commandとeffective argsは意味が異なるため両方残す。一方、同じresolved objectの全文JSONと
leaf展開を同時には保存しない。現在のleaf展開を維持し、list/tupleだけを一つのJSON valueに
する。

SGLang configuration eventが来なければ、成功runであってもcomparison viewでは
`incomplete`とする。空の設定値を送って成功扱いにはしない。required key集合は
使用中のSGLang commitに対応するtest fixtureから決め、別の手書きschemaへ複製しない。

### benchmark結果の原始観測値

- `result.benchmark.successful_requests`
- `result.token.input.total`
- `result.token.output.total`
- `perf.benchmark.generation.wall_time_sec`
- `perf.phase.model_load.wall_time_sec`
- `perf.phase.jit_and_warmup.wall_time_sec`
- `perf.stage.benchmark.wall_time_sec`
- `run.stage.benchmark.exit_code`
- `run.stage.benchmark.result_present`

次はraw tableへ保存せずviewで計算する。

- request、input token、output token、total tokenの各throughput
- generation以外のlifecycle overhead
- succeeded/failed/incompleteの表示
- baseline差、speedup、反復runのmedian、MAD、CV

SGLangが出力するthroughput値はraw `results.jsonl`に残る。raw tableに同じthroughputを
複製せず、total token数とgeneration timeからviewで計算し、必要ならlog値との一致を
testする。

### telemetryとcache

GPU raw sampleはnodeごとの圧縮logへ保存する。BigQueryには比較に使う次の集約だけを
保存する。

- sample countとobserved duration
- generation phaseのGPU utilization mean/p95
- memory max
- power meanとenergy
- temperature maxとSM clock min

model loadからteardownまでの全stage aggregateとgeneration aggregateを同じ名前で
混ぜない。generation境界が取得できないrunはgeneration resource値を空にし、全stage値で
代用しない。

JIT/cacheはGit外の可変条件である。run開始前のcache状態をcold/warm、file count、sizeで
保存する。sizeからfile countを導出できないため両方を残せるが、同じtreeの総sizeと
subdirectory sizeを重複して保存する必要はない。

## BigQuery viewとスプレッドシート

raw tableは事実だけを持つ。workload別viewが次を行う。

1. `env.GITHUB_REPOSITORY`と`env.GITHUB_SHA`からcommit URLを作る。
2. effective argsからmodel、parallelism、dtype、backend等の比較列を選ぶ。
3. total token数とgeneration timeからthroughputを求める。
4. exit code、result、configuration event、node/GPU観測数からcompletenessを求める。
5. 実行commit、SIF/model/dataset identity、node/GPU数が同じrunを反復groupにする。
6. bestとは別にcount、median、MAD、CVを表示する。
7. 二run間ではenvとeffective argsの差だけを表示する。

commit messageはmetricへコピーしない。人間はcommit URL、エージェントは`git show`から
変更と仮説を読む。

## 実装順序

1. benchmark jobがclean treeから投入されることを確認するgateを追加する。
2. SIF provenanceへ`2_Qwen` build commit、`5_SGLang_Src` commit、SIF content hashを
   記録し、symlinkではなく参照先binaryを測る。
3. modelとdatasetにimmutable artifact manifestを用意する。
4. senderを`metrics.jsonl`への先行appendと再送に対応させる。
5. collector間のkey集合が交差しないことをtestする。
6. 397B runで欠落しているconfiguration eventを修正し、欠落runをincompleteにする。
7. outer MPI wrapperではなく、実際のbenchmark commandを`run.command`へ保存する。
8. GPU raw sampleを圧縮artifactへ残し、generation phaseの集約を作る。
9. raw tableに保存中のthroughput、model名、GPU総数などの派生・部分集合metricを廃止する。
10. Qwen比較viewを作り、SGLang `results.jsonl`と同じthroughputになることをfixtureで
    確認する。
11. 同一commitを複数回実行し、一つの比較groupになることを確認する。
12. BigQuery送信を意図的に失敗させ、保存したJSONLから同じrowを再送できることを
    確認する。

## 完了条件

- 性能に影響する変更がすべてcommitされ、benchmark runがcommitを指している。
- commit bodyだけを読めば、変更理由と仮説が分かる。
- SIF、model、datasetがimmutable IDまたは必要なcontent hashで一意になる。
- full log、GPU sample、`metrics.jsonl`が成功・失敗runの両方で保存される。
- collectorが所有するkey集合に重複がない。
- raw metricから別のraw metricだけで導出できる値が残っていない。
- configuration/result/node/GPUの必須観測が欠けたrunをcomparison viewが除外する。
- viewのthroughputがSGLang resultに一致する。
- 同じcommitの反復と異なるcommitの変更がSheets上で明確に対照できる。
