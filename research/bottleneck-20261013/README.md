# 会話履歴・raw計測の再監査（2026-10-13）

2026-10-14更新: 通常比較2本は完走したが改善を確認できず、採用しない。短縮profileは失敗し、完全な再収録は未達。

## 投入した3本の結果（2026-10-14確認）

| run | Slurm / elapsed | good tokens/s | 基準4732.87との差 | 判定 |
|---|---|---:|---:|---|
| 38345930105 / Decode SYMM_MEM | 23084 COMPLETED / 12:47 | 4734.0191 | +0.0243% | 優位性を確認できず |
| 38346456470 / Decode MoE tile32 | 23086 COMPLETED / 12:49 | 4728.9631 | −0.0825% | 改善を確認できず |
| 38350383403 / profile-only | 23091 FAILED / 13:53、exit134 | 対象外 | 対象外 | 完全なprofileを取得できず |

通常2本はSLA適格で、保存済みscoreを同一parserで再計算して一致。SYMM_MEM runのDecodeログにdispatch `['SYMM_MEM', 'PYNCCL']`、tile32 runには専用 `source/vllm-decode-moe/` の設定読込を確認した。単一runの±0.1%未満を改善とは扱わず、追加反復やmainへの性能設定採用は行っていない。全3 runsのarchive SHA256SUMSを全件照合し一致、Actionsの回収・保存・runtime削除は成功。

短縮profileはREADY604秒、capture_active21:56:33Z、stop_requested21:56:38Z。AIPerf終了処理で `Assertion failed: check () (src/msg.cpp:387)` → exit134となり、launcherのerrexitによるcleanupがDecodeのNsight report書出し中にworkerを停止した。これは14分のTIMEOUTではない。Decode reportは存在するが、同じNsight2026.3.2によるCPU exportが `Section Table Reference magic number mismatch` で失敗し、完全なreportとして使えない。Prefill6 reportsはCPU exportでき各2 GPUのkernelを含むが、CUDA event欠損等の診断警告が残っており、厳格なcapture validatorの合格とは扱わない。Actionsのprofile validation stepは失敗後にskipされていた。再投入せずCPUで検証した結果であり、計測終了時のclient failureとreport保存の寿命管理、警告の対象processの調査が残る。

## 調査範囲と正本

ローカルCodexの281履歴ファイルを検索し167候補を抽出、自宅サーバーの69履歴から13候補を抽出した。候補には本作業や無関係な言及も含む。既存 `research/optimization-history/` の横断索引と関連会話の結論を突き合わせ、最新判断に関係する会話本文を精読した。「全履歴を一行ずつ読んだ」という意味ではない。

自宅サーバーのdiscrawl SQLiteには9,432 messages、123 channels、98 threadsがあり、最終同期は2026-10-13T03:50:32Z。全messageをローカル一時領域へ読み出して検索し、252 keyword hitsと関連thread文脈を確認した。特に最新ジョブのnsys解析thread `1558884373988905121` を参照した。OpenCodeの該当directoryにある3 sessionsも確認。古い会話の「decodeはHBM非飽和／CPU待ち」という解釈を、現行vLLMへそのまま移さない。私的な会話全文はリポジトリへ複製しない。

raw正本は `ssh -o BatchMode=yes archive-host` の `/srv/qwen-artifacts/<run>/1/`。5 runsの431 files、約2.69 GBをSHA256SUMSと全バイト照合し、不一致なし（[artifact-audit.json](artifact-audit.json)）。基準4 runsは各commitのparserを取り出して再実行し、保存済みscoreと一致（[baseline-replay.json](baseline-replay.json)）。このparserは公開採点条件を実装したリポジトリのparserであり、主催者提供の完全accuracy評価ではない。

## 定常性能とSLA

| Actions run | 構成 | good tokens/s | good request fraction | 解釈 |
|---|---|---:|---:|---|
| [38194609674](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38194609674) | vLLM c104 / proposals 3 | 4710.02 | 98.44% | 非profile基準 |
| [38331343958](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38331343958) | vLLM c104 / proposals 2 | 4732.87 | 98.72% | +0.485%の単一比較。優位性未確定 |
| [38034080121](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38034080121) | SGLang c104 | 4429.95 | 93.33% | 別engine構成の参考値 |
| [38164365415](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38164365415) | SGLang c192 / MegaMoE | 0 | 36.88% | raw throughputが高くてもSLA不適格 |

vLLM基準はFirmus g1/g4、16 H200、P6×TP2 + D1×TP4、EP1、FA3、FP8 KV、公式FP8 target、BF16 EAGLE3 draft。同一の固定image、target/tokenizer/draft revisionsを使う。公式ISL4000 / OSL200、warmup60 / measurement120 / grace30、TTFT≤3000ms、平均ITL≤20ms、good fraction≥0.90で比較する。2 proposalsではITLが16.51→15.23msに改善する一方、TTFTは1000→1245msに増加した。[request-analysis.json](request-analysis.json) の全リクエスト再集計では、ITL違反8→0件、TTFT違反38→38件。decode duration平均3286→3032msの改善がTTFT増加でほぼ相殺され、平均request latencyは4286→4276msに留まる。少ない投機が全体のgoodputを必ず増やすとはいえない。

調査中に完了した別実験 [38335858973](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38335858973) はTensorRT-LLM、c128、P6TP2/D1TP4EP4、EAGLE3 steps3。READY685秒、計測2,732件・request error0だが、平均ITL26.025ms、TTFT550.598ms、good fraction1.4275%でscore0。Actions failureは採点条件不適合であり、起動失敗ではない。raw output throughput4398.78 tokens/sを有効scoreと扱わない。保存済み全artifactのSHA256SUMS照合とcommit `4fdca5b1` のparser再実行で保存scoreとの一致を確認した。vLLM基準と同時実行数・EP・engine等が異なるため、個別要因の効果は分離できないが、この構成を現行候補より有利とは判定しない。別実験のジョブ・branchには介入していない。

TTFT違反38件は両基準runとも計測phase開始直後の最初の104件に限られた。request start順の0始まりindexは3 proposalsで65〜103、2 proposalsで66〜103、最初の開始からそれぞれ31.8〜43.2ms／40.4〜51.7msに送信されたrequestである。その後のTTFT違反は0件。warmup終了時にinflightをdrainしてから測定phaseの104並列を立ち上げるAIPerfログとも一致する。Prefillの10秒間隔ログの非ゼロWaitingも主にphase開始付近で、定常的な待ちの偏りと断定しない。30〜120秒に開始したrequestのTTFT平均は925→1194ms、ITLは16.63→15.28msで、定常域でもDecode短縮とTTFT増加が併存する。phase立ち上がりのrequestも公式scoreに含めたままとし、この区間別分析で採点条件を変更しない。

## GPU内の時間配分

[38268462101](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38268462101) の7 native Nsight reportsをCPUで再exportした。[analyze_nsys.py](analyze_nsys.py) は各workerの全rankに共通する時間窓を選び、GPU別の区間unionとkernel duration sumを分ける。結果は `vllm-decode-profile.json` と `vllm-prefill-*.json`。

| recorded kernel durationの内訳 | Decode | Prefill |
|---|---:|---:|
| Triton fused MoE | 43.35% | 47.7–48.1% |
| FA3 attention（device_kernel、demangled名で照合） | 16.89% | 約10.4% |
| TP all-reduce | 12.66%（custom 2-stage） | 9.3–9.9%（symm two-shot） |
| dense FP8 GEMM | 5.42% | 約8.4% |
| per-token group quantization | 4.81% | 7.4–7.6% |

Decodeは共通5.297秒窓で各GPUのrecorded kernel unionが96.98–97.06%、kernel recordsの約92.5%がCUDA Graph nodeに属する。単にPython呼び出しを減らせば大幅改善する、という根拠にはならない。MoEはP/DともTRITON backend。DeepGEMMのログだけを見てMoEもDeepGEMMと判断すると誤る。

**制限:** 旧収録toolにはdriver13.3非対応とCUDA event欠損の警告がある。未記録の空白をCPU待ちと断定できず、duration shareはcritical path上の削減率でもない。GPU utilization、電力、memory utilizationだけではHBM帯域飽和／演算飽和を区別できない。これを解消する再収録を実施中。

優先候補はMoEのtoken tileとDecodeのcollective。TP4のMoE中間幅は384。image内のH200向け設定はbatch256/512でM tile64を使うが、現行の小さなexpert batchではpaddingの影響が候補になる。実際のexpert分布・改善率はまだ未計測。activation+quantization融合も残るが、現行E8M0 scale条件ではnative融合が使われない。scale方式をまとめて変更するとdense GEMMも変わるため、まず限定した要因で比較する。

## 実験全体の時間

基準run 38331343958は配備115秒、Slurm割当12分51秒、READY473秒。scheduler accountingではSubmit01:19:38Z、Start03:02:26Z、End03:15:17Zで、queue待ちは1時間42分48秒。成果物回収60秒、Actions artifact保存52秒、実行環境削除25秒、自宅archive job25秒。queue待ちとGPU起動を混同しない。

Decodeのtarget重みload21.92秒、model load全体28.57秒に対し、engine initialization275.06秒（compilation74.59秒を内包）。Graph収録68秒と30秒もこの初期化の内数。Prefillの重みload30–32秒、engine initialization132–168秒。これらは並行workerの値なので足し合わせて総時間にはしない。前runでもDecode initialization267.31秒／compilation74.32秒で、起動最適化には繰返し根拠がある。現在のvLLM配備でコピーしているcacheは主に `.tensorrt_llm` で、torch.compile cacheがrunを跨いで再利用されることは確認できていない。

具体的には `provision-firmus-framework.sh` はworkerごとの `.tensorrt_llm` をコピーし、vLLMのhomeはrun専用directoryへmountする。Decodeログのtorch.compile cache keyは2 proposalsで `727492dbb4`、3 proposalsで `c131bfe8f6`。既存保存cacheの別keyを強制流用してはならない。`TRITON_CACHE_DIR=/tmp/vllm-$name/triton` と `TORCHINDUCTOR_CACHE_DIR=/tmp/vllm-$name/inductor` も回収対象になっていない。nativeの互換性判定を保った保存・復元は次の起動改善候補だが、現時点で改善を実測したとは扱わない。

DecodeのGraph設定は2 proposalsで53 request bucketsを3倍して最大768 tokensまで収録する。うち19 bucketsは現在のconcurrency104を超える112〜256用。旧3-proposal profileのNVTXにはgeneration68〜97 requests（272〜388 tokens）等が実際に記録されているが、短い定常窓だけでは起動・mixed batchを含む必要shape全体は保証できない。request数とtoken数、FULLとPIECEWISE、targetとdraftの必要shapeを区別して削減すべき候補であり、単純にtoken上限104へ切り詰めない。追加GPUジョブは投入していない。

旧profile runは有効なbenchとnative reports生成後、CPU exportをGPU割当内で続けて17分枠を超えた。再収録branchでexportをterminal確認後のlogin runnerへ移し、GPUを使用しない処理で割当を消費しないようにした。既存rawは回収済みで再ビルド不要。

## 今回の実装・検証

すべて `f85c30b51958d4aaa8c093227d28a9329751e8bb` を親とする別commit。serving image／モデルは固定。mainへの採用は反復結果で判定する。

| commit / branch | 単独で変える要因 | Actions |
|---|---|---|
| `3841c582` / `perf/vllm-decode-symm-20261013` | Decodeのcustom/FlashInfer dispatchを外しSYMM_MEMへ | [38345930105](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38345930105) |
| `5b96e630` / `perf/vllm-decode-tile32-20261013` | Decodeだけnative tuned configのbatch256/512のM tile64→32 | [38346456470](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38346456470) |
| `a54355ae` / `perf/vllm-profile-compatible-20261013` | 同一baselineのprofile収録をNsight2026.3.2に更新、CPU exportを割当外へ | [38346936664](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38346936664) |

各branchでCPU単体テスト一式、静的検査、shell構文を通し、Actionsのvalidateも成功。profile validatorにはCUDA欠損・非対応driverの実警告を拒否するfixtureを追加した。今回の成果物は未回収、性能採否は未判定。既存の他者ジョブには介入しない。

## ジョブ数・占有時間の制約

ジョブ数と時間がscheduler優先順位へ影響するため、追加投入は採否に必要な最小限に限定する。投入済み3本の結果を見る前に総当たりや反復を追加しない。通常benchの実測12分39秒／12分51秒を根拠に、今後の同等runは15分枠を目安にし、変更による起動コストも確認する。旧profileは開始20:48:26Z、capture_stopped21:04:35Zで収録停止まで16分09秒かかっていたため17分枠を維持する。CPU exportは割当外へ移した。その後、ユーザーが「待機中なら短縮版へ差し替える」を承認したため、profile 23087だけをPENDING状態フィルタで停止した。実行0秒・割当0ノード・Start=Noneを確認し、他の2実験は変更していない。

ユーザー指摘を受け、短縮profile版 `f256abb` をCPU検証済み・`[skip ci]` で保存した。通常benchを省略し、診断用20/30/30・2,048入力候補・5秒収録のみとする。profile-only来歴と終了判定を分け、公式scoreを生成しない。承認後、commit `9815b6103b8068bd5f0a68d71754fdba1305d1eb` から [短縮profile 38350383403](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38350383403) を1本だけ差し替え投入した。短縮版の上限は14分。旧run 38346936664はCANCELLEDを検知してfailureで終了したが、ログ・sourceのarchive、SHA256SUMS生成、実行環境削除は成功した。正本の `logs/logs/slurm-accounting.txt` は `23087|CANCELLED by 3014|0:0|00:00:00|0`。意図した未実行停止であり、計測失敗として扱わない。現在の優先度設定はFairShare寄与377、Age寄与0、使用量減衰半減期2日だったため、今後の追加比較は特に限定する。
