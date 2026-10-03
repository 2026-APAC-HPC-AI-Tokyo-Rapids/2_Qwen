# Prefill DP2 の NIXL 初発障害を native 層まで追跡する

更新: 2026-10-04 JST。2026-10-03 campaign の継続調査。**初発は UCX backend の転送失敗まで絞れたが、元の UCX status と endpoint の失敗理由は既存ログに出ておらず、root cause は未確定。metadata 再登録による stale handle、Decode プロセス死亡、Python clear race を初発原因と断定しない。** GPU追加投入、build、source mirror変更、runtime変更は行っていない。

対象は Actions `37129209750`、runtime commit `a953ec8`、image SHA256 `20c3a38359ccc61967343ec23d600f783df8d487c6339d12b9ca53c14b55d3d3`、SGLang `fc275674a97d78553b22597ee72c8fd48f5420f4`。Qwen3-235B-A22B-Instruct-2507-FP8、EAGLE3 dense BF16 draft、target/draft FP8 KV。Pのみ DP attention2、P attention TP2→D TP4、MoE TP4 EP1、2P/2D independent replicas、c48。元の結果・terminal監査は [Prefill topology記録](2026-10-03-prefill-topology.md)。公式profiling成功数0のため性能比較は成立しない。

## 実際の native version を合わせる

freezeは `nixl==1.5.0` / `nixl-cu13==1.5.0`。しかし公式 `v1.5.0` tag は `1683cf3b`、imageの `libnixl_common.so` 埋込 git hash は **`075b2397`**。対応する公式commitは `075b2397029fc67a0b69a34df566c1ae3d00f5ab`。このrevisionの `nixl_agent.cpp:1198` は raw の同名ファイル・行番号・メッセージと一致する。tagだけを読むと行番号や接続実装がずれる。

imageの UCX plugin ELF は `libucp-b39b3ed9.so.0.0.0` / `libucs-98a42727.so.0.0.0` を NEEDED とし、RPATHはwheelの `nixl_cu13.libs` を指す。同梱libucpの `ucp_get_version` を逆アセンブルすると **1.23.0**。system `ucx_info` の版をそのまま採用しない。NIXLの対応wheel Dockerfileは UCX `65e342a82dd2609a28c8c2b8c45ed7fe2cb33fa0` にpinしている。このpinはwheel buildのsource根拠で、実行時 `/proc/PID/maps` やUCX build revision文字列は既存artifactにない。ELFからの通常のlink先は確認できたが、runtime loaderの例外まで実測したとはしない。[公式wheel build設定](https://github.com/ai-dynamo/nixl/blob/075b2397029fc67a0b69a34df566c1ae3d00f5ab/contrib/Dockerfile.manylinux#L373)

## `REMOTE_DISCONNECT` の正確な意味

raw の `nixl_agent.cpp:1198` は `engine->postXfer(...)` の返却値が `NIXL_ERR_REMOTE_DISCONNECT` の分岐。remote metadata generation のweak reference失効は1135付近で別の `NIXL_ERR_NOT_FOUND` を返す。metadata refreshは同じremote sectionにmergeされ、明示invalidateまたはload失敗でentryを消す。今回の行は「登録し直したので古いhandleを拒否した」経路と一致しない。[対応native core](https://github.com/ai-dynamo/nixl/blob/075b2397029fc67a0b69a34df566c1ae3d00f5ab/src/core/nixl_agent.cpp#L1187)

UCX statusの `NOT_CONNECTED` / `CONNECTION_RESET` / `ENDPOINT_TIMEOUT` / `CANCELED` は同じNIXLエラーに変換される。[status変換](https://github.com/ai-dynamo/nixl/blob/075b2397029fc67a0b69a34df566c1ae3d00f5ab/src/plugins/ucx/ucx_enums.cpp#L43) endpoint error callbackはCONNECTED→FAILEDへ変更し、その元statusを **DEBUG** で記録する。以後writeはFAILED状態を検出しただけでも同じエラーを返す。`ucp_ep_create`成功をCONNECTEDとするので、登録成功・handle作成成功はremote handshakeと最初のデータ転送成功を保証しない。[endpoint処理](https://github.com/ai-dynamo/nixl/blob/075b2397029fc67a0b69a34df566c1ae3d00f5ab/src/plugins/ucx/ucx_utils.cpp#L80)

## UUID・room・lifetime の再照合

初発P1は22:31:52 SGT、P0は22:31:53。room `3468568658199182252` / `89826285245956172`、P0の `3177069301943947541` 等は既存監査参照。native1198のfailed remote UUIDは両Pで同じ4つ。**旧独立noteの4 UUIDをすべてD1とした記述は訂正する。**

| failed UUID | 初期化log（物理改行） | P0 native1198回数 | P1回数 |
|---|---|---:|---:|
| b1c1008c-40dd-4e21-adbb-cbedb03c1999 | decode-1.log:469、22:30:15 | 8 | 5 |
| 37fcfbb4-3d25-4603-bfb7-6e349857bca6 | decode-1.log:472、22:30:15 | 9 | 5 |
| afa240f1-334a-4bc4-9ca9-3cd6dd7932eb | decode-0.log:469、22:30:18 | 5 | 9 |
| 0c471e49-759f-4780-98d0-01cae8de32cd | decode-0.log:471、22:30:18 | 9 | 3 |

残り各Dの2 UUIDにはnative1198行がない。ただし別endpointの転送完了は確認できず、これを正常なrankとみなさない。INFO初期化行にはTP rank/PIDがなく、各UUIDのD内rankは断定不可。Dログには初発より前のPython fatal/OOM/restartがなく、22:36:30の300秒timeoutまで処理が残る。明示agent invalidationも観測されない。endpoint failureはプロセス死亡と同義ではない。

固定SGLangの `_add_remote_peer` はUUID登録済なら再登録をskip。DP bootstrap/cacheにはprefill DP rankを含み、TP2→4 head分割算術とtarget/draft双方FP8のitem sizeは静読で整合した。実wire metadata bytes/hashは出力されていないので、正しい算術だけで送信先address/rkeyの実体まで保証しない。

Python側は初発失敗後にsender clearがroomのstatus/transfer_infosを削除し、queued chunkがdefault WaitingForInput→assertへ進む二次race。missing-room skipや300秒deadline延長はUCX初発を修復しない。

## 原因候補と判別に欠ける証拠

| 優先度 | 候補 | 根拠と限界 |
|---|---|---|
| 1（同順位） | UCX wireup/transportがpeerをreachableと維持できない | native1198の直接条件。両D replicaの複数endpointへ広がる。選択transport/device・元UCS status・endpoint callbackが欠落し、接続拒否、IB retry、peer timeoutを分離できない |
| 1（同順位） | DのUCX progressが不足/停止しhandshake・ACKが進まない | PのNIXL thread8に対しDは0。Dはreceiver.poll→update_transfer_status→get_new_notifs→native getNotifs→ucp_worker_progressで進む。Dscheduler/Gloo待機、queueのpoll gapはログなし。設定は成功対照でも同じため「D0threadsだけ」が原因とは言えない |
| 2 | heterogeneous transferのdescriptor/address/rkeyかsession実体の不整合 | TP2→4が新条件で、FAILEDになる前の最初のwrite/protocol statusが未記録。geometryとFP8 dtypeは整合、metadata登録成功後も正しいremote実体かは未確認 |
| 低い | 単一D restart、metadata replacement、draftだけmixed KV dtype、bootstrap DP key collision | restart先行fatalなし、failed UUIDはD0/D1双方、stale-handle native返却値不一致、raw双方FP8、cache keyはDPを区別。完全な排除ではなく、現証拠が支持しない |

PのUCX threadpoolは既定split threshold1024、8 dedicated workers＋shared workerを用い、remote worker addressから各local workerのendpointを作る。heterogeneous pathでconnection/descriptor分割数が変わる余地はあるが、実descriptor数・worker/endpoint対応は未記録。CPU算術だけでnative障害を再現したとはしない。

UCX pinのUD実装にはACK未受信peer timeout経路があり、既定TIMEOUT30秒。ただし今回UDが選択された証拠もinitial endpoint生成時刻もないため、初発時刻を30秒timeoutと決めつけない。[UCX UD timeout経路](https://github.com/openucx/ucx/blob/65e342a82dd2609a28c8c2b8c45ed7fe2cb33fa0/src/uct/ib/ud/base/ud_ep.c#L268)

## 次に必要なら実施する最小診断（未投入）

同image・同DP2・P/D配置を維持し、通常benchmarkと分けた短いdiagnosticをActionsで行う案。最初は **NIXL_LOG_LEVEL=DEBUG、UCX_LOG_LEVEL=info** を全rankに設定し、1短promptの転送だけを確認する。node/role/PID/TP/DP/UUID、実link先、UCX device/transport、metadata hash/bytes長、最初のpostとendpoint callbackのUCS statusを回収する。native DEBUGに元statusが既にあるため、その観測前にnative buildやretry機構を足さない。

1prompt成功時のみc48 burstで再現性を確認し、Dのreceiver.poll/get_new_notifsの時刻・最大gapとGloo前後を有限ログで測る。callbackがENDPOINT_TIMEOUTかつDprogress gapが先行ならprogress仮説が強まり、頻回progressでもIB retry/RESETならtransport/session側を優先する。metadata/agent lifetimeのhash変化またはNOT_FOUNDなら別系統を調べる。instrumentation結果を通常性能値に混ぜない。まだGPU投入は承認されておらず、原因に基づく具体修正も決まっていない。

Decode中間状態をDで取得する設計は [別調査](2026-10-03-decode-side-eagle-initialization.md)。draft処理をDへ移す案でもtarget KV転送が必要であり、今回NIXL障害の修復を自動的に代替しない。

証跡は既存derived正本 `research-issue-17/20261003-existing-data-analysis/prefill-dp2-audit/nixl-native-diagnosis/`。rawは `qwen-artifacts/37129209750/1` を参照し重複保存しない。imageからの対象ELFのみ読取・version抽出、公式exact revisionの読取、UUID集計scriptと出力を保存する。
