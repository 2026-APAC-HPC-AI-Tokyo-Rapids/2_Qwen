# SGLang MegaMoE の vLLM 再現チェックリスト

基準は SGLang の Actions `38664373784/1`、実装 `e6e295b`。
引継ぎ元は `codex/firmus-vllm-structural-optimization` の `841e80b`。
対象は MegaMoE カーネルに加え、router、DP 制御、投機デコード、KV 転送、起動から公式結果回収まで。
2026-10-16 に見直しを開始した。

**GPU 投入は保留。**
CPU で解決できる不一致、検証コード自体の欠陥、実装が未確認の項目を先に潰す。
同じ設定名だけで同等と判定しない。
CPU 合格、実イメージ CPU 合格、GPU 合格、公式実行合格を区別する。
チェック完了には、対象の実装と実行した検証の根拠を併記する。
チェックが未完了でも進める項目は、GPU が必要な理由と一度の割当での確認方法を記す。

## 基準と判定

| ID | 確認対象と合格条件 | 根拠と方法 | 状態 |
| --- | --- | --- | --- |
| B01 | 基準 run の raw throughput、SLA goodput、good 率、エラー数を混同しない | 正本の score、raw request、公式 parser を照合 | 確認済み。6344.964606904409 output tok/s、good率0.9597478176527643、4124 requests、error0。正本 score.json を照合 |
| B02 | 実際に起動した argv と全 DP rank の有効設定を基準にする | 正本 source、server-info、起動ログを照合 | 正本の4 server-info を確認。P192/DP4→48、D128/DP4→32。実起動は未実施 |
| B03 | target、draft、tokenizer、chat template、入力データの revision と内容が一致 | モデル manifest、tokenizer hash、draft config 差分 | 未確認 |
| B04 | SGLang に適用した全 patch の移植要否を一つずつ判定 | image manifest と patch 一覧を対応付け。未対応を残さない | 下表で全14 patchを棚卸し。KV／draft／handoffの同等性は未完 |
| B05 | 元の重みと FP32 scale を保持し、E8M0 再量子化をしない | load、transform、dense、draft の実ソースと値の検査 | 未確認 |

## 外側の PD router

対象は `platforms/firmus/framework-comparison/vllm-proxy.py`。
SGLang の native router と同じく、Prefill グループを選ぶ処理を検証する。
内部 DP rank の選択は次節で別に検証する。

| ID | 確認対象と合格条件 | 根拠と方法 | 状態 |
| --- | --- | --- | --- |
| R01 | 最少の未完了数を持つ Prefill を選び、同数時は順番に回す | 非同期同時投入、偏った応答時間、回復後の選択を試す | 既存同時投入テスト合格。予約[2,1,1]、同数時の巡回とbusy回避。複数Pの実HTTP検証は残る |
| R02 | 最初の await より前に予約し、応答 body 完了まで保持する | ヘッダー先行、body 遅延で予約数を観測 | 実HTTP合格。Pがヘッダーを返した後もbody完了まで予約保持 |
| R03 | 成功、接続失敗、HTTP エラー、JSON 不正、timeout、cancel で一度だけ予約を解放 | 各出口で負荷計数、接続の close、次の request を検査 | P503、JSON不正、P待ちcancel、D429、D stream cancelを実HTTPで確認。connect/timeoutは未完 |
| R04 | 未起動、異常、復旧した worker の選択が基準と対応する | SGLang health policy と vLLM の起動確認、実行中の扱いを照合 | 未確認 |
| R05 | Prefill 完了後に不要な待機や同期を挟まず Decode に渡す | body 完了時刻と Decode POST 開始を CPU HTTP で検査 | 実HTTPでP body以前にDへ送らないことを確認。handoffの実測遅延はGPU待ち |
| R06 | prompt、sampling、stop、ignore_eos、max/min tokens を保存する | P の一時的な max_tokens=1 が D へ漏れないことを検査 | 実HTTP合格。Pだけmax_tokens=1、Dは元のsampling/min/max/stop/ignore_eosを保持 |
| R07 | request ID と返却された KV metadata を P から D まで保存する | engine ID、rank、host、port、block IDs の転送を検査 | 実HTTP合格。request IDとKV metadataを変更せず引渡し。実NIXL値はGPU待ち |
| R08 | 最初の token、終了 token、usage、SSE 終端の欠落や二重送信がない | completions と chat の実 HTTP fixture、分割 chunk を検査 | SSE bytesとDONE、JSON content type/usageのfixture合格。実モデルtokenはGPU待ち |
| R09 | Decode が HTTP エラーを返した場合に 200 の SSE として隠さない | 上流 response header を受ける前後の失敗を試す | 修正済み。元実装は実HTTPで失敗、修正後はP503/D429をそのまま返す |
| R10 | クライアント切断で上流 stream を閉じ、後続 request を汚さない | D body の途中で cancel。接続と予約の残留を検査 | 修正済み。P待ち切断の接続／予約残留を再現後に解消。D切断も実HTTP合格 |
| R11 | 接続 pool、retry、timeout、keepalive の差が重複生成や待ちを作らない | 実装と遅延 HTTP。P は fresh connection、D は pool の理由を確認 | 未確認 |
| R12 | 公式 client の streaming と router の API が一致する | 実際の AIPerf command とリクエスト fixture を照合 | 未確認 |

## 内部 DP と scheduler

| ID | 確認対象と合格条件 | 根拠と方法 | 状態 |
| --- | --- | --- | --- |
| D01 | P3×DP4 と D1×DP4、attention TP1、MoE EP4 が全 rank で成立 | native group の構築、rank mapping、worker argv を実行検査 | 未確認 |
| D02 | P と D の内部 rank 選択が待機中と実行中の request を考慮する | SGLang total_requests と vLLM load balancer の値、更新周期を照合 | 差あり。SGLang total_requests/毎iteration、vLLMはinflight床値＋KV圧力加算／100ms。未再現 |
| D03 | KV 受信待ちの request と実行 batch の上限を混同しない | admission、waiting、running、transfer pending の遷移を追う | vLLMはwaiting＋skipped_waitingを集計しKV受信待ちも含む。running上限とは分離。SGLangとの全遷移照合は未完 |
| D04 | P48、D32 の request 上限と P4096、D2048 の token budget の意味が一致 | 投機 token と padding を含む native scheduler を照合 | 未確認 |
| D05 | prefill chunk と前回結果の引渡しが追加の iteration 待ちを作らない | SGLang early handoff patch と vLLM output processing を追う | 未確認 |
| D06 | 空 rank も他 rank と同じ回数、同じ順序で MoE collective に入る | idle dummy、prefill/decode 混在、EAGLE proposal の実コード検査 | 未確認 |
| D07 | DP 同期時の padded rows が workspace と graph の上限内に収まる | 0、1、各 graph 境界、4096 と最大 dummy batch を検査 | 未確認 |
| D08 | prefix cache、chunked prefill、preemption の有効条件が対応する | 両 framework の有効設定と cache lifecycle を照合 | 未確認 |

## モデルと MegaMoE

| ID | 確認対象と合格条件 | 根拠と方法 | 状態 |
| --- | --- | --- | --- |
| M01 | 128 experts、top8、H4096、I1536、rank ごとの32 experts が一致 | 実 checkpoint loader と native expert map を検査 | 未確認 |
| M02 | topk、softmax、renormalize、global expert ID、同点時の扱いを対応付ける | SGLang と vLLM router の実装、固定 logits の比較 | 未確認 |
| M03 | gate/up の連結順、interleave、scale 配置、down 配置が一致 | 識別可能な FP8 byte と scale で transform の往復検査 | 未確認 |
| M04 | load 時の shard と packed weight が全層で正しく登録される | native loader を meta/CPU で通す。94層の重複保持を確認 | 新規CPU診断はnative factoryと全128 expertの所有／weight・scale loaderを通す。Actions実行待ち |
| M05 | kernel 本体と precision patch が基準と一致 | provider、header、patch の hash と演算順の照合 | 未確認 |
| M06 | activation FP8、SiLU、routing weight、FP32 combine の演算順が一致 | independent reference。許容値を変更せず比較 | CPU と実 GPU の対応を再確認待ち |
| M07 | native runner が dispatch、combine、all-reduce を二重に実行しない | runner 全体から実 method まで呼出しを検証 | installed runnerの部分CPUテスト合格済み。native factoryからの構築をActionsで追加確認予定 |
| M08 | workspace を層間で共有しても同時使用せず、graph replay が参照先を保持する | stream、DP group、lifetime、再入、DBO 無効を追跡 | 未確認 |
| M09 | 空 rank、hot expert、境界 expert、全 inactive、変更した入力を処理できる | eager と graph、実寸、4096 rows/rank。旧検証との対象差を確認 | 新 adapter の GPU 検証未到達 |
| M10 | target の backend 選択が dense EAGLE と通常 DeepGEMM を変えない | native dispatch と module namespace、draft config を検査 | 未確認 |
| M11 | 不対応設定をロード前に拒否し、CPU の stub が本番処理に混ざらない | TP、EP、placement、dtype、shared expert、LoRA の拒否検査 | 既存 guard あり。全拒否項目の再確認待ち |

## EAGLE と CUDA Graph

| ID | 確認対象と合格条件 | 根拠と方法 | 状態 |
| --- | --- | --- | --- |
| E01 | SGLang steps=3 / draft_tokens=4 と vLLM proposals=3 の処理内容を対応付ける | bonus token、verify shape、採択と reject 後の処理を追う | 未確認 |
| E02 | draft weight、config 変換、hidden-state 入力、KV dtype が対応する | original/runtime config と prepare_eagle を照合 | 未確認 |
| E03 | P eager と D graph の対象範囲、capture sizes、padding が対応する | request bucket×4 の妥当性を native graph dispatcher で検査 | 22 bucket×4＝最大128 tokenはnative uniform_decode_query_len=1+3と対応。capture範囲の同等性は未完 |
| E04 | proposal 回数や生きている request が rank 間で違っても同期が崩れない | target/draft の DP coordination と dummy を検査 | 未確認 |
| E05 | graph が古い input、route、mask、buffer length を使わない | 同じ pointer の入力を変更し複数 replay。eager と比較 | GPU 必須。CPU では capture 経路を先に検査 |
| E06 | full model の出力が基準に対して妥当で、変更した算術の影響を評価できる | 固定 prompt と同じ sampling、logits/出力の比較。bitwise と精度を区別 | GPU 必須 |

## KV cache と NIXL

| ID | 確認対象と合格条件 | 根拠と方法 | 状態 |
| --- | --- | --- | --- |
| K01 | P131072、D393216 tokens は DP rank ごとの容量として確保する | 実ログ、block64、target と draft の実 allocation を照合 | 基準server-infoはP131072/D393216。vLLM block64×2048/6144もrankごと同値。実allocationはGPU待ち |
| K02 | FP8 KV の scale、head 配置、RoPE、layer ordering が一致する | 量子化と書込、attention と転送の各処理を追跡 | 未確認 |
| K03 | target と draft 両方の KV を正しい rank から受け取る | TP1→TP1 と DP suffix、group mapping を実ソースで検査 | 既存 CPU mapping 検証あり。内容は GPU 未確認 |
| K04 | port、engine ID、IPC path が同じ node の P2組とDで衝突しない | 実 argv と native offset の全 rank 展開 | 静的照合済み。side-channelのDP offsetを含め5600–5603/5610–5613/5620–5623/5700–5703。実bindはGPU待ち |
| K05 | P 完了時に KV が可視であり、D 使用完了まで送信元 buffer を保持する | ready、transfer、ack、release のイベント順を追う | 未確認 |
| K06 | cancel、timeout、転送失敗で block を漏らしたり早期再利用したりしない | native lifecycle、異常 HTTP fixture、既存ログを照合 | 未確認 |
| K07 | model、workspace、target/draft KV、graph、通信 buffer が同居する | rank ごとの byte 見積りと実イメージ metadata。最終確認は GPU | 未確認 |

## 起動、イメージ、実験の制御

| ID | 確認対象と合格条件 | 根拠と方法 | 状態 |
| --- | --- | --- | --- |
| O01 | PyTorch ABI、TVM FFI、CUDA、private provider が実イメージで整合する | 既存 image の CPU tensor bridge と native compile の根拠 | 過去 build 合格。修正後の実イメージは未検証 |
| O02 | 検証プログラムが本番と同じ分散初期化を通る | native init、TP1/DP4/EP4 group、graph context を検査 | 841e80bがnative worldの初期化漏れを修正。ただしgateはまだ合成TP4 group。本番TP1/DP4/EP4のNCCL経路は未完 |
| O03 | 本番 native layer 作成、loader、method 選択、runner 初期化を CPU で通す | 部分 AST や mock だけで合格にしない | 新規check_native_cpu.pyをActionsで実行予定。Gloo/CPUとNCCL/CUDAを区別 |
| O04 | GPU 非依存の設定エラーを scheduler 割当前に検出する | CLI config、モデル metadata、patch replay、静的検査、CPU suite | Linux既存395件合格。変更後suiteとCPUイメージ検証はActionsで追跡 |
| O05 | full serving を未合格 adapter で投入できない | 同じ image/source の成功 gate を pre-submit で照合 | 実装済み。megamoe_gate_run未設定は配備前に停止。同一image SHA/size、診断source、数値、rank/replay、Slurm完了を照合 |
| O06 | 起動した4グループと16 rank の readiness を確認する | argv と実 config、worker 生存、endpoint、GPU UUID | 未確認 |
| O07 | CPU assertion の失敗で GPU 測定結果の回収を失わない | GPU 前の検査と回収後の設定判定を分離 | 未確認 |
| O08 | 公式 workload、warmup60/measure120/grace30、parser と入力を固定する | official-aiperf、実 command、raw 集計を照合 | 未確認 |
| O09 | profile、追加 probe、監視通信を公式測定に混ぜない | launcher と時間区間、metrics 前後 snapshot を検査 | 未確認 |
| O10 | 失敗しても最初の traceback と各段階の結果を保存する | init 前の状態保存、finally、Actions の always 回収を検査 | 初期化前に失敗状態を保存し各rankのtracebackを残す。失敗再現fixture合格。Actions回収はalways |
| O11 | 同じ実験の重複投入や無制限 retry をしない | commit/run/attempt/Slurm ID を一意に追跡。再投入前に terminal を確認 | この見直しからの投入0件 |
| O12 | terminal、公式結果、原データ、checksum、残留 job を確認して完了とする | Actions、Slurm accounting、保存正本を照合 | GPU 実行後 |

## GPU 投入前の完了条件

- [ ] 上表の CPU で確認できる項目に未解決の不一致がない。
- [ ] 新しいテストが修正前の問題を検出し、修正後に合格する。
- [ ] 固定された実ソースの patch replay と既存テスト一式が合格する。
- [ ] 実イメージ上で native layer と分散初期化の CPU 検査が合格する。
- [ ] router の実 HTTP テストで遅延、失敗、cancel と metadata の受渡しを確認する。
- [ ] 同じ機能でも SGLang と vLLM で意味が異なる設定をすべて列挙し、扱いを決める。
- [ ] GPU だけに残る確認項目と、有限の一回で取る証拠を確定する。
- [ ] full serving を投入する前に、修正後 adapter の実 GPU 数値と graph 検証が合格する。

## 現在の失敗根拠

[Actions 38751723386](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38751723386) / Slurm 23815 は37秒で失敗。
正本 `archive-host:/srv/qwen-artifacts/38751723386/1/logs/logs/gpu-gate.log` では、`adapter_gpu_gate.py:42` の `GroupCoordinator(..., group_name="tp")` から FlashInfer の初期化へ入り、`get_node_count()` が `distributed environment is not initialized` で停止している。
MegaMoE の数値検証には到達していない。
ログ SHA256 は `3d481b8f76f4f949ff635932d93d1a715270fbf8d012660e1553b9532b5892fa`。

## 2026-10-16 の再確認

最新pushは `841e80ba8a912a6c1a2da32d6fd82040e5faaa5d`。
[Actions 38763496666](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38763496666) は追加テストの Ruff S102 で停止しており、新しいGPU結果はない。
この見直しでは該当execに検査対象とstubの範囲を明記して修正した。
修正前のrouterに実HTTPテストを当てると、JSON応答とP/Dエラー伝播の6例が失敗した。
さらにP待ち中の切断2例で上流と予約の残留を再現した。
修正後の14例はMacのloopback HTTPで合格した。
GPUジョブの投入は0件。

## 基準イメージの全patch

正本は `38664373784/1/source/source/image-manifest.json`。
以下は移植の判断表であり、未確認の同等性を合格扱いしない。

| SGLang patch | vLLMでの対象と判断 |
| --- | --- |
| prefill-cutlass-silu-quant | targetはMegaMoEでCUTLASS MoEを通らない。dense/draftの経路は別途確認 |
| draft-extend-selected-head-rows | EAGLEのlogits対象行選択。native speculatorとの比較が必要 |
| prefill-fp8-scale-row-shuffle | CUTLASS MoE用。MegaMoE weight/scale transformの一致をM03で確認 |
| prefill-tiled-routing | CUTLASS MoE用。MegaMoE pre-dispatchは別のnative providerを使用 |
| firmus-staged-eagle-kv | target/draft KV staging。NixlConnectorのgroup mappingと寿命をK03/K05で確認 |
| qwen3-sm90-megamoe | private provider＋Sm90MegaMoEMethod。kernel math以外のloader/runnerもM01–M11で検査 |
| firmus-staging-completion | staging完了とdecode通知の順序。K05未完 |
| chunk-cache-page-release | chunked prefill cacheの解放。D08/K06未完 |
| decode-silu-quant | native dense/draftのactivation quantizationとの差を確認する必要あり |
| nixl-expired-chunks | timeout/cancel後のtransfer cleanup。K06未完 |
| firmus-dp-rank-query | rank選択とserver-infoのDP対応。内部DPのD02/D03に差がある |
| prefill-ready-handoff | 前iterationのcopy完了を次の長いP計算より先に通知。vLLMとのイベント順照合が必要 |
| deepgemm-activation-layout | private providerのactivation layoutと通常dense DeepGEMMを区別して検査 |
| prefill-load-lifetime | 外側P groupのleast-live選択とbody完了までの負荷保持。実HTTPまで確認 |

## 残っている投入阻害要因

- native factory/loaderの新しい実イメージCPU診断が未実行。
- 実GPU gateは本番groupとnative runner全体をまだ通していない。
- 内部DPの負荷評価と通知周期に実装差がある。
- EAGLE、early handoff、target/draft KVの寿命はソース照合が未完。
- full modelのKV/workspace/graph同居、数値、SLA goodputはGPUでしか確定できない。

Actionsの `[megamoe-cpu]` は既存イメージを使うCPU診断専用。
cluster/build/gateタグとの混在はvalidateで拒否する。
`megamoe_gate_run` は成功した有限GPU gateの実行ディレクトリが確認できるまで `null` とする。
