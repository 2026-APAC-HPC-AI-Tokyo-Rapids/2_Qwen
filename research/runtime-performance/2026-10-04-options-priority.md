# 次の性能改善候補と改修量（2026-10-04）

現時点の優先は「PだけのNVLS・要求数制限を個別に試す準備」と「Hopper既存ES GEMMの適合診断」。ユーザー案のD通常1step→短context draftはPの仕事を移す設計候補として優先CPU設計に値するが、既存CLIでは完成しない。DP2は実測性能が悪かったのでなく、NIXL warmup transfer失敗で採点不能。現在CLI不可という理由だけでEP/TP/PPを理論上棄却しない。改善率未測定の候補を実測勝者として扱わない。

| 優先 | 候補 | なぜ検討するか | 作業量 | 現在の証拠と制約 | 最小次検証 |
|---|---|---|---|---|---|
| A | PだけNVLS | 大prefillのNCCL all-reduceを改善できる可能性 | 既存設定 | P T3992/8192はAR fusion/custom ARの小shape制限外。NVLS現在off。fabric対応・実選択・時間未測定 | effective NCCL env/fabric確認、Pのみflag、D固定。採点前短診断でNVLS選択確認→公式E2E |
| A | P `prefill-max-requests 2`、chunk8192維持 | 2×3992の後の短い断片を抑える候補 | 既存設定 | c48 Pの8192batchが約93%、queue増。充填率7984/8192低下とのtradeoff。速度未測定 | CPU scheduler fixtureで継続chunk含むshape確認、P cap2単独condition |
| A/B | CUTLASS H100 expert-specialization up/down接続 | 既存SM90 grouped GEMMを対象shapeで使える | 小 | 現normal FP8はenable_es false。upN768/K4096、downN4096/K384はalignment適合。graph/workspace実適合不明 | can_implement、workspace/capture/replay、数値・短shape timing。新image後E2E |
| B（CPU設計優先） | D最初通常target1step→coldstart短context draft | 全prompt draftprefill/全prompt hidden転送を省きPを早く解放できる設計 | 中〜大 | y1はP生成済。D y1通常stepでy2とaux_y1取得→draft1row(seed_y2)。draftprefix未初期化のまま既存API不可 | seedless wire、draftstart/validlen独立index、acceptedprefix lifecycleのCPUfixture。品質/acceptanceとTTFT第2token/ITLを分離 |
| B | EP1 DeepGEMM contiguous route配置 | masked容量を避けGEMM以外のlayout問題を直す | 中〜大 | 単純DG切替はEP1masked入力4.125GiB/down8.25GiB容量。GPU件数/scan/scatterとstaticgraph契約が必要 | device-only route layout/無効行、top8復元、FP32合成、graph静的容量fixture |
| B | CUTLASS route prepare＋FP8 input/scale shuffle統合 | count/assignmentと重複copyを減らす | 中 | logical2ER参照はcacheableでHBM律速証明なし。明示top8 inputcopyあり | 現P境界profile→契約を保持したhistogram/scan/scatter、2shuffle統合の局所数値/timing |
| B/C | dense FP8 GEMM backend Pのみ比較 | 既存DeepGEMM dense射影をshape別に評価 | 既存設定 | 現autoで対象denseはDG。MoE backend比較とは別。BF16draftへのFP8話の適用不可 | fixedsource flagが変える対象dense/fallback条件を特定、sameimageのshape測定後P-only1因子 |
| C | chunk12288/reqcap3、graph3992/7984 buckets | バッチ形成を改善する余地 | 既存設定 | 大chunkは旧CPUscreenで不利、現graphpadding削減proxy c48約0.092%。単独主targetとして弱い | cap2結果とactualshapeを先に確認。chunk/reqcap/graphを一度に変えない |
| C | fusedQKnorm/RoPE、largeprefill norm＋quant接続 | 小融合・large-T量子化境界の削減 | 設定／中 | fusedflagはFP8KVstore融合ではなくint32position cast追加。large-T AR/norm制限解除だけ不可 | 対象形状・castコスト/consumer再quant確認→局所数値・graph検証 |
| C（互換解決先） | DP2 NIXL修復、後にDP4 | attentionTP小分割とMoE TP4維持の候補 | 不明（原因調査後） | DP2起動/メモリ/graph成立、warmup REMOTE_DISCONNECT、0/48採点。room cleanupassertは二次、timeout延長で初発は直らない | exact native照合済、元UCS status欠落。必要なら同imageの1prompt DEBUG/info診断を別承認で行う。小修正で直る保証なし |
| C | P TP4EP4＋Triton | 幅384→1536、localexperts128→32、gatherGEMM形状を変える | 設定（比較設計は中） | nativeCUTLASS EP1assert回避。localtop平均2でも全top8bufferを確保、up384MiB/act192/down512。旧両P/D EP4score2202はP単独棄却不可 | 同backend EP1/EP4か現在CUTLASS対総合candidateを明記。exactshape config/graph/memory診断先 |
| D | TP8EP1 padding/scale/loader対応 | expert tensor幅192を有効幅256等へpadできる理論余地 | 大 | CLI非整列は現実装制約。weight/scales、downinput、zero padding、checkpoint shardingを一貫変更必要。worker2→1も変わる | correctness設計とweight/scale CPUfixture、同TP候補の局所shape timing。paddingだけで勝ちとはしない |
| D | P PP2＋D EAGLE | P-onlyspecを外せれば既存PPassertは回避可能 | 大 | disableoverlap必要。3aux層captureはlastPPだけ、earlystage auxを渡さない。Pworker2→1+pipelinebubbleも変化 | coldstartならPaux不要だがPD PP transfer/overlap無しを別検証。exactaux移設ならstage間aux配線も必要 |
| D | down-output/combine fusion・gather対応GEMM | T8192 downBF16512MiB中間を省く狙い | 大 | top8所有権・FP32sum・scale/roundingが必要、BF16atomic代用不可。現P時間分離なし | 現Pdown/combine割合→tile ownership設計→数値/graph/局所timing |
| D | exact Ptargethidden→Ddraftprefill | exact draftprefixをDへ移す比較案 | 中〜大 | rawaux93.75MiB/ISL4000、Pfc projected31.25MiB。現draftKV3.906MiBより転送増、Dprefill interference | GPUhiddenwire/chunk/fc境界のCPU設計。coldstart案と品質/転送/性能tradeoffを比較 |

SiLU/group128融合・BF16 draft head等はすでに測定済みの統合imageに含まれる候補で、未実装の新規利得へ数えない。c44→c48はgood tok/s2667.013→2673.508の1比較、TTFT1297.590→1581.031msで、さらに並列度を増やす自動根拠ではない。旧target-onlyシミュレーションを現在EAGLE/CUTLASS/FP8KVへ速度校正せず、4P/12D非対称には大prefill service改善が必要という必要条件のみに使う。

source/toolchain/image内容を変更するES/layout/fusion/TPpadding/draftcoldstartは新build、runtimeだけのNVLS/cap/backend/topologyは既存image再利用。実機候補は条件ごとの追跡commit、安価検証先、D固定、公式parserとterminal/artifact確認で評価する。このノートでは実装・GPU投入・外部送信なし。

詳細：[NIXL native調査](2026-10-03-nixl-dp2-root-cause.md)、[Decode側EAGLE初期化](2026-10-03-decode-side-eagle-initialization.md)、[Prefill構成](2026-10-03-prefill-topology.md)。現在CLIで対象外としたTP/EP/PPは、その比較範囲での制約であり、source改修による理論上の対応まで否定したものではない。
