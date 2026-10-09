# AISimulate、ベンチの妥当性、並列化の再評価

2026-10-09。目標は公式4000/200・joint SLA条件でH100 6000、H200 10000 good output tok/s。これは達成報告ではなく、追加GPU実験の選別に使う調査である。現在のTP/EP設定が最適だとは確認できていない。既存Actionsは継続し、新たなGPU投入はこの調査中は保留した。

## 判断

最優先の候補は、8 GPUの役割グループ内で **Attention TP4 × Attention DP2、MoE DP1、EP2/4/8** とし、現在の独立TP4×2にあるMoE重複を解消する構成。PとDで同一のEP、backend、graph設定を使う必要はない。Prefillでは処理能力とTTFT、DecodeではHBM帯域・実効バッチ・ITLを別に最適化してから、16 GPU全体の配分を決める。

ただし、共有による重み容量削減だけでPrefillが倍速になるわけではない。直近Firmusはpoolと同時接続数を増やしても約64k input tok/sのままPの待ち行列が増えた。H200 10k good output tok/sには少なくとも200k input tok/sが必要であり、Pの供給能力の改善も不可欠。容量問題だけと判断してDのEPを変える計画では不足する。

AISimulateは候補の除外・代表shape選定・感度分析には使える。現時点の校正で、新しい並列構成の公式goodput順位やSLA合格を保証することはできない。以下の推定値と資源計算を、実機の成功スコアとして扱わない。

## AISimulateで何を評価できるか

評価に使用した環境は`/tmp/qwen17-simulation/venv`のAISimulate **0.13.0**。既存校正の来歴はupstream `992dddc7ee2bad8fd523e7db99eea4ac64254766`、最新sourceの機能確認は`d088d2efd36a654e611d44903a58e7dd6012230d`。古いresearch READMEのdev version表記とインストール済みwheelを混同しない。

| 段階 | 確認できたこと | この結果だけでは確認できないこと |
|---|---|---|
| schema/API | TP、Attention DP、MoE TP/EP、PPを指定できる | エンジンが同じ通信・kernel・scheduleを実行すること |
| generic op-level | H100/H200、FP8 Attention/KVの今回の40 queryは全件値を返した | 各operatorが実測補間内か、現行CUTLASS/DeepEPに一致するか |
| calibrated forward | 既存TP4EP1/TP8EP8の測定範囲内で高精度 | H200、FP8 KV、Attention DP、PP、EAGLE3への外挿 |
| serving simulation | PD worker・queue・handoff等のモデルを持つ | 実NIXL、予約、CUDA graph、EAGLE3のSLA分布の忠実な再現 |
| official E2E | 実機・gateway・固定client・per-request判定で評価 | simulatorだけで代用できない |

AISimulateの`tp`はAttention TPであり、SGLangのworld TPと同じではない。CP1では`tp × attention_dp = moe_tp_size × moe_ep_size`、lower SDKのCPを含める場合は左辺に`× cp_size`が入る。world8/EP8/AttentionDP2は`tp=4, attention_dp=2, moe_tp_size=1, moe_ep_size=8`となる。scheduled batch/token数はAttention DP-localで、MoE側で一度DP数を掛ける。DP2のB32はglobal64、DP4のB32はglobal128である。同じB指定の表を性能順位に使うと不公平になる。

CPは確認したpublic schema/native FPE SDKの`cp`・`cp_size`・`context_parallel`指定では表現できなかった。一方、lower Python `ModelConfig`には`cp_size`・`cp_style`と一部モデル実装があるため、CP実装が皆無という意味ではない。後述のlower SDK CPU実行は確認できたが、実機・servingは未検証。PPの指定が通ることと、有限microbatch・pipeline bubble・実装制約を含むserving予測の妥当性は別である。最新の投機APIはNgram、vLLM、op-levelの制約があり、現在のSGLang EAGLE3 + calibrated FPMを直接再現するものではない。[AISimulate source](https://github.com/ai-dynamo/aisimulate/blob/d088d2efd36a654e611d44903a58e7dd6012230d/python/aisimulate/src/aisimulate/config/engine.py)

### 既存校正の強さと限界

FPMは413行。TP4EP1がP156/D47、TP8EP8がP163/D47。全てPP1/DP1/CP1、BF16 KV、target-onlyで、D batchは1–32。SHA256は`d185ab5430d90031a5b8f6e92a5234e989d278e86a79e3ee3970cf586eaf3215`。既知controlのTP4EP1・P1×4000は154.796252msを再現した。旧FPMに対するH200/FP8 KV/PP2指定はidentity不足で拒否される。

別allocationの110点では校正済みFPMのAPE中央値0.234%、最大3.57%だが、既知shapeを別allocationで測定した結果であり、未知topologyのholdoutではない。同じ比較のgenericモデルは中央値7.48%、最大32.87%であり、generic値と校正済み値の精度を混同しない。3回repeat自体のrange/median中央値は0.961%、最大13.07%なので、0.234%を全体の誤差保証にしない。4000長のshape holdoutは中央値0.672%、最大6.291%、範囲外2点は拒否された。

servingの独立c40・別node pairの2 runでは、無補正goodput誤差31.98–32.80%、c16で決めた係数を掛けても10.49–11.18%。forwardの精度がそのままgoodput精度になるわけではない。係数はTTFT/ITL分布やqueue、SLAを校正せず、H200・EAGLEへ持ち越せない。来歴は既存`research/simulation/`およびraw `/srv/qwen-artifacts/research-issue-16/20260926/measured-search/`、`/tmp/qwen17-simulation/goodput-prospective.json`にある。

最新SGLang schedulerモデルにも実機の512-token予約とone-request completion guardについてTODOが残る。PD eventsがあっても実NIXL bootstrap/progress/preallocationの忠実度とは別。graph用追加予約量は明示できるが、それだけで全予約挙動は校正されない。[固定source](https://github.com/ai-dynamo/aisimulate/blob/d088d2efd36a654e611d44903a58e7dd6012230d/crates/core/src/engine/scheduler/sglang/core.rs#L432)

### 精度とbatchのidentityを修正してから比較する

最初のgeneric診断ではBF16 Attention + FP8 KVのPrefill tableが欠け、20 Prefill queryが失敗した。しかし固定SGLang `fc275674`のFA3は、明示FP8 KVかつhead_dim128の今回の条件でQをKV dtypeへcastする（`flashattention_backend.py:1215–1227,1801–1812`）。**このmixed-dtype診断の失敗は、実構成のcoverage不足の証拠ではない。** FP8 Attention + FP8 KVへ修正すると40/40 queryが値を返した。

これでも`fallback_policy=deny`は別estimatorへの切り替えを禁止するだけで、operator内部の外挿を禁止しない。MoE tableは`sglang_fused_moe_triton`、legacy provenanceを含み、現行CUTLASS/DeepEP/DeepGEMMそのものではない。Qwenのshapeがtableに存在しても、queryごとの全operatorが実測内だとは証明できない。数値はgeneric target-onlyのscreenとして扱い、EAGLE3性能やkernel選択の根拠にしない。

さらに8 GPU/role、Pはglobal4/8 request×4000 token、Dはglobal64/128 request・context4100へ揃えた48 queryも全件値を返した。1 request×16384 token等の不適切なqueryを修正し、DP-localのrequest数とtoken数を同時に揃えた。raw・script・環境来歴の正本は`/home/USER/qwen-artifacts/research-simulation/20261009-op-coverage/`（`sha256.json`付き）。以下はその一部、単位ms。Pはglobal4×4000、Dはglobal128、target-onlyである。

| 8 GPU構成 | H100 P | H100 D | H200 P | H200 D |
|---|---:|---:|---:|---:|
| 独立TP4×2 | 283.162 | 33.167 | 253.455 | 27.301 |
| A-DP2/TP4 + EP2/MoETP4 | 389.651 | 29.295 | 347.676 | 27.027 |
| A-DP2/TP4 + EP4/MoETP2 | 423.112 | 28.882 | 362.742 | 26.307 |
| A-DP2/TP4 + EP8/MoETP1 | 485.431 | 29.102 | 411.807 | 26.430 |
| A-DP4/TP2 + EP8/MoETP1 | 457.265 | 29.431 | 389.092 | 26.365 |

このgenericモデルは共有MoEをPでは遅く、Dではわずかに速く見積もる。**共有すればPも必ず速くなるという仮説は、このscreenでも支持されない。** H200 DのEP候補間の数%差は未校正モデルで順位確定するには不十分。P/Dでbackendや並列化を変える根拠として扱う。

不利な推定を単に精度不足として捨てず、operator分解も確認した。H100の独立TP4×2→EP8ではMoE150.886→307.558ms、dispatch60.297→105.894ms。差202.269msの77.5%がMoE、22.5%がdispatchで、全op合計がFPEと一致する。DP token数は一度だけ掛けられ、TP/EP shardもDBへ正しく渡る。MoEの該当点は測定Triton tableの内挿であり、この部分を単なる外挿失敗とは説明できない。

このqueryの`moe_comm_backend`は未指定なのでnon-DeepEP reduce-scatter/all-gather、MoEはlegacy Triton、routingは既定`power_law_alpha=1.2`。balanced測定行を使った感度計算でもH100 MoEは148.988→207.086msとなる。理想MoE math時間は双方28.703ms、memory時間は33.728→14.220msへ改善するのに、kernel効率・負荷偏り・通信が利益を打ち消している。現行P CUTLASSや候補DeepGEMMのcompute、DeepEP normalの通信を実測して、この不利を覆せるかを判断する。理想帯域だけでEP8を採用する根拠はない。

PP2の返却値は全94層とP2Pを含むmicrobatch通過時間で、stage時間でも定常pipeline間隔でもない。H200の上記P/D条件で484.788/33.394msだが、この値をそのままTP構成のthroughputと比べない。microbatch overlapとbubbleを別にモデル化する必要がある。

CPはlower SDKもCPUで実行した。`ModelConfig → get_model → SILICON DB → estimate_static_latency_breakdown_with_rust`ではSGLangのTP4/CP2をmodel gateが拒否する一方、TP1/CP8/AttentionDP1、MoETP1/EP8、global4×4000は成功した。H100/H200の全forwardは439.414/371.045ms、内訳のCP all-gatherは6.206/6.157ms。したがってCPを全く評価できないわけではない。ただしこれはstatic operator modelの到達確認であり、VRAM admission・serving SLAを証明しない。固定Qwen3/FA3にはCPとzigzag/interleaveの実装経路があるが、今回のEAGLE・PDとの組合せは実機未検証。TP1/CP2のH100例も774.192msを返したが、2 GPUへの重量収容は成立しない概算であり、**値が返ることと実行可能性は別**だと確認できた。

## ベンチの質と測定の使い分け

公式ベンチはfixed-length、cache-bust、closed-loopという特定ワークロードの評価として有効。汎用chat workload、長文、open-loop burstへの一般化はできない。最適化対象をこの公式条件に合わせる一方、モデルの計算を省略したりclientを変更したりしない。最終合否には実gatewayと固定60/120/30秒、seed2026、4000/200、TTFT≤3秒かつ平均ITL≤20ms、good fraction≥90%を使う。[公式規則](https://github.com/hpcac/2026-APAC-HPC-AI/blob/main/1.%20Qwen3%20AI%20Task%20Rules.md)

- closed-loopのclient concurrencyはDecode batchではない。c96にした今回のFirmusは平均約27.13件だけがDecode中で、残りの多くがPrefill待ち。Littleの法則による概算でも15.96 req/s × 5.88秒 ≈93.8件となる。単にcを上げるとqueueとTTFTだけが増え得る。
- 全requestのjoint SLAを評価する。平均TTFTが3秒以下でも合格とは限らず、raw throughputの増加もgoodput増加とは限らない。warmup・計測・drainの境界を揃え、errorsを分母から落とさない。
- EAGLEはaccepted lengthだけで比べない。draft時間、target verify行数、graph padding、受理token数、commit・通信までを一つのcycleとして測る。受理長2.2は2.2倍速を意味しない。
- 短い探索runはshapeや壊れた候補の除外用。公式規則はローカル探索で10–20秒を許可しており、同一allocation内で代表条件をまとめる余地がある。ただしqueueが定常化していない短時間runを最終SLA判定にしない。
- 校正と検証をallocation/node pair、shape、並列構成、concurrency単位で分離する。同じrunのrandom splitや同じ測定値を補間して自己評価する方法では、未知構成の予測精度を測れない。
- profiler/通信計測は通常benchから分離する。全forward→必要なkernel/collectiveの順に調べ、build・model load・graph capture時間をgeneration性能に混ぜない。
- repositoryの公式条件検証parserによる再計算と、主催者の最終判定を区別する。full-set accuracyは別gateであり、基準FP8モデルの99%以上が必要。今回の容量・負荷runでは未実施。

## 8 GPUの役割グループでの資源比較

計算の正本は[parallelism-bounds.json](2026-10-09-parallelism-bounds.json)。以下は1 byte/parameterで概算したtarget重みのみであり、実VRAM使用量の厳密な下限ではない。BF16部分、scale、draft、activation、graph、workspace、通信bufferが追加される。embedding/LM headの配置も簡略化している。

| 構成（PまたはDに8 GPU） | Attention | MoE | 重み概算 GiB/GPU | 意味 |
|---|---|---|---:|---|
| 現在の独立TP4×2 | TP4、独立2組 | 各TP4/EP1 | 54.74 | 大きな通信groupを避けるがモデル2コピー |
| world8 / A-DP2 | TP4×DP2 | TP4/EP2 | 28.30 | 現行のFFN分割幅384を保って共有 |
| world8 / A-DP2 | TP4×DP2 | TP2/EP4 | 28.30 | 幅768、local expert32 |
| world8 / A-DP2 | TP4×DP2 | TP1/EP8 | 28.30 | 幅1536、local expert16 |
| world8 / A-DP4 | TP2×DP4 | TP1/EP8 | 30.16 | Attention通信減、Attention重み複製増 |
| world8 / A-DP8 | TP1×DP8 | TP1/EP8 | 33.89 | Attention完全独立、local batch縮小 |
| PP2×TP4 | TP4、47層/段 | 各段TP4/EP1 | 27.37 | 容量節約、pipeline待ちとmicrobatchが課題 |
| 独立TP2×4 | TP2、独立4組 | 各TP2/EP1 | 109.47 | H100には収まらず、H200も余裕を消費 |
| pure TP8 | TP8 | TP8/EP1 | 27.37 | KV head重複、FFN幅192のblock128非整合 |

global D batch128、context4100、FP8 target KVの例ではCPなしの多くの構成が約5.88 GiB/GPU。pure TP8ではKV head数4を超えて複製され約11.76 GiBになる。仮に理想的にcontextを分散保存するならworld8/EP8/A-DP2/CP2/AttentionTP2でもKV約5.88 GiBとなるが、**固定FA3のMHA/GQA Prefill CPはfull K/Vを各CP rankへall-gatherして保存する**（`cp_utils.py:395`、`flashattention_backend.py:1153–1175`）。この実装の同容量例は約11.76 GiBであり、理想分散保存の値を実VRAM計画に使わない。Prefill CP、Decode CP、decode時のAttention weight slicingを区別する。

ここでいう独立DP、Attention DP、MoE DPは別物。独立DPはモデル全体を複製し、Attention DPはrequest群を分けつつMoEを共有できる。MoE DPを増やすとexpertコピーも増える。上表ではMoE DP1に固定した。

## EPの通信をallreduce一本で見積もらない

固定sourceのnative StandardDispatcherでは、Attention DPのtokenをgatherし、各GPUのMoE入力はglobal M行になる。Mはrequest数ではなくalignment・graph padding込みのtoken行数。EAGLE verifyならdraft/verify token数を数える。

world8、Attention DP2/TP4、native dispatcherでMAX_LEN経路に入る場合、1層当たりAttentionTP4 reduce-scatter → world8 all-gather → MoE → world8 reduce-scatter → AttentionTP4 all-gather。Qwen内のEP/MoE TP allreduceはskipされる。EP2/4/8でこの通信構造は共通であり、EP8だから通信量が少ないとは言えない。

`A=M×4096×2 byte`と置くring送信量近似は1層2.5A。M16384、94層なら約29.375 GiB/rank/target forward。これは物理実測ではなく、4 collectiveの近似合計である。breakable prefill graphもMAX_LENを強制する場合があるため、Pを常にSUM_LENと仮定しない。

一方SUM_LENではworld8 allreduceでgatherし、さらにEPとMoE TPのallreduceを行う。この近似はEP2/4が4.25A、EP8が3.5Aとなる。backend、graph eligibility、padding modeで順位が変わり得る。根拠は固定`fc275674`の`qwen3_moe.py:750–845`、`communicator.py:380–438,1105–1187,1348–1380`、`dp_attention.py:446–528,827–845`、`forward_batch_info.py:1215–1290`。

DeepEPはtoken dispatch/combineを使う別経路で、上記式を流用できない。top8/EP8の一様routingならtokenが触る異なるrank数の期待値は`8×(1−(7/8)^8)≈5.25`。top8だから必ず8 rankへ送るわけではない。pack、metadata、dtype、重複排除、overlapと、最も忙しいrankのexpert負荷を測る必要がある。

EP8はlocal expertを減らしGEMM幅を広くできる一方、小batch・偏ったroutingでは不均衡が増える。平均expert負荷だけでなく各層各batchの`max_rank_assignments / (topk×M/EP)`を記録する。EP2/4/8を容量だけで順位づけしない。

## P/D配分、PP、CPを含む理論上の可能性

6000/10000 good tok/sは30/50 good req/s、最低120k/200k input tok/sを要求する。good fractionが90%なら実際の処理要求は約133k/222k input tok/s。P、D、KV転送の各stageの持続能力の最小値が全体を制限する。

モデル次元から、input token当たり主要linear計算は約41.9 GFLOP、4000長のcausal Attentionは約6.16 GFLOP。FA3の今回の経路はFP8入力である。GPUあたりdense FP8 peak1979 TFLOP/sを仮定した粗い計算量screenでは、8 P GPU・90%goodのH200 10k目標は理想peakの約67%、H100 6kは約40%を必要とする。6 P GPUでH200 10kなら約90%になり、draft・通信・quantize・launch等をまだ含めない段階で厳しい。peakとの比率は達成予測ではなく、不利な配分を除外するための必要条件である。[H200仕様](https://www.nvidia.com/en-us/data-center/h200/)

H100とH200はHBM帯域・容量が異なる一方、同じ世代の演算peakの増加だけで1.67倍の目標差を説明できない。H200の余剰VRAMは、Dのactive batchと共有MoEを成立させる余地として活用する。Pを12 GPUに増やしてDを4 GPUに減らすような配分も、Dの重み読出しとITLを同時に満たさなければ全体改善にならない。

Decodeの主要関係は`output rate ≈ active requests × accepted tokens / cycle time`。cycleにはdraft、verify、通信、schedulerが入る。10k good tok/s・90%goodでglobal active B128なら、1 token換算のcycleは約11.52ms以内が必要。batch増加に対してcycleも増えるため、空きVRAM量だけで最適batchは決まらない。

FP8 target KVのPD転送量は4000 inputにつき概算385 MB/request（94層、4 KV head、128 dim、K/V）。50 good req/sなら約19.25 GB/s、90%goodで約21.39 GB/sが最低のpayload水準となる。実際はdraft、layout、複製、metadata、待ち、経路帯域を加味する。公称NIC帯域だけで転送待ちを無視しない。

**PP**は重みを分けながらAttention TP4を維持できる。単純なPP2 fill/drainではm microbatchの効率は`m/(m+1)`、m1で50%、m4で80%。ただしmicrobatch化はDecodeの重み読出しを増やし得る。continuous/asynchronous scheduleでは式が変わり、autoregressiveの循環待ちも残る。現行SGLangのPPとEAGLE/overlapの制約は実装上の制約で、PPが理論的に常に遅いことの証明ではない。

**CP**は長いcontextを分け、P AttentionやKV容量に効く可能性がある。ただし4000長では独立requestも多いため、まずAttention DPで並列化した方がcontext通信を避けやすい。causal Attentionを単純に前後半へ分けると仕事量は1:3で、均等分割したつもりでも不均衡になる。balanced partition、KV ring、MoE再配置まで含めて評価する。CP2のKV ring payload近似は`(CP−1)/CP × B_local × S × 2 × local_KV_heads × head_dim × communication_bytes`。KV保存dtypeと通信dtypeを混同しない。CPはpublic serving APIでは直接指定できず、lower SDKではstatic prefill比較まで確認した。Attentionの実測比率が高い場合に実機候補の優先度を引き上げる。

## 次の実験の選別

1. 現在走っているP chunk16384とH100容量・負荷runを回収し、P supply、D active batch、SLA分布を確認する。現在のP2TP4/D2TP4を対照として残す。
2. 同じ8 GPU・同じglobal batch/token量で、まずEP2とEP8 + AttentionDP2/TP4の端点を比較し、必要ならEP4を追加する。Pは実chunkとrouting、DはEAGLE verifyの実行行数・graph境界で測る。同じ構成commitの割当内で代表shape測定と短いE2Eをまとめ、起動費用を償却する。構成を変える実験は条件ごとのcommitからActionsを起動し、独立した候補を予算内で並行投入する。workflow matrixへ条件を隠さない。
3. 既存imageのDeepEP/DeepGEMM EP8候補と、native CUTLASS EP候補は別backend実験として扱う。EPだけの効果を主張する場合はbackendを固定した対照が必要。native CUTLASS patchはCPU検証済みでも、GPU数値・empty expert/all-local-zero・graph replay未確認なので採用不可。既存の小probeは最大128展開行だが、kernelは2048行で分岐し、本番Pは65536行以上になる。非一様scale、実loader/dispatcher/collective、大shape側のall-local-zeroまで検証する。またEP8 patchはglobal M×topkのc1をzero化しfusionも外すため、重み節約と別にactivation容量・帯域が悪化し得る。制約解除だけを速度改善としない。
4. AttentionDP4/TP2は次順位。Attention通信削減と小local batch、Attention重み複製を比較する。P/Dで違う並列度を使う場合はKV handoff geometryも検証する。PP/CPは上記の必要条件とstage profileが根拠を与える場合に進める。
5. 有望候補だけを別allocationで再測定し、公式条件・artifact hash・parser replay・accuracyのgateを通す。既存FPMの0.234%を未知候補の信頼区間として使わず、校正から独立した観測で予測誤差を更新する。

新しいsource候補はbranchに留め、優位性未確認のままmainへmergeしない。ASPIREのポイントは`myprojects`で再確認し、最終検証用6144 SUを保護する。16 GPU・128 SU/GPU-hourでは10分の割当が約341.3 SU、30分wall上限は1024 SU。並行投入は経過時間を短縮するが、SUはGPU時間に比例する。候補の事前除外、同じ構成内での起動費用の共有、不要な長い探索runの削減がポイント節約に効く。
