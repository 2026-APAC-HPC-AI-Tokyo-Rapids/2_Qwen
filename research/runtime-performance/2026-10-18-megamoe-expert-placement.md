# 2026-10-18: Prefill MegaMoEのEP rank負荷不均衡と静的expert配置

## 1. 目的と結論

Prefill MegaMoE kernelはserving中に平均2.17 ms/layerかかるが、配備前gate（`scripts/check-firmus-megamoe.py`、4096 tokens/rankの均等random routing）では1.66 msである。Prefill chunkはほぼ満杯（平均約4028 tokens/rank）なので、約0.5 ms/layerが実routingの条件で失われている。fused kernelは最も遅いEP rankを待つため、rank間のtoken偏りが候補の1つだった。

結論:

- 公式workloadでの実routingは強く偏っている。現行のtrivial配置（rank rがexpert 32r..32r+31を持つ）では、最も重いrankの負荷が平均の1.338倍（94 layerの平均。token加重したpass単位のmax/mean）で、最悪はlayer 86の2.21倍。
- 精度に影響しない静的置換（冗長expertなし、128 expertを4 rankへ32個ずつ配置）でこれが1.034倍まで下がる。ほかのworkerや後半passで評価しても1.034〜1.035倍で、偏りはモデルとworkloadの性質でありノイズではない。
- 置換を入れたbenchでは、Prefill形状のMegaMoE kernel平均が2.163 ms → 1.987 ms（−0.176 ms、−8.1%）になった。予測した約0.50 msのうち実現したのは約35%。Prefill queueが空かない割合は40% → 23.5%、TTFT SLA違反は140 → 110件に減った。
- 公式scoreは6418.50 good output tok/sで、参照6521.98より1.6%低い。ITL SLA違反が82 → 129件に増えたことが主因で、Decodeの設定とコードは不変である。同じ参照imageとtrivial配置で採ったstatistics run（recorderあり）は6699.30なので、run間のばらつきは約±3%あり、単一runでは優劣を判定できない。**未採用・未merge。** 判定には同条件の反復runが要る。

## 2. Run一覧

| 用途 | commit | Actions run | image | 結果 |
|---|---|---|---|---|
| 参照（main構成） | a6acf5f系 | [38945855851](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38945855851) | e29dfde4… | 6521.98、TTFT>3 s 140件、ITL>20 ms 82件 |
| expert統計（recorder有効、scoreは比較に使わない） | aef6cf6（`perf/firmus-expert-stats-20261018`） | [39051152456](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39051152456) | e29dfde4…（再利用） | 6699.30、TTFT 125件、ITL 37件 |
| 静的配置bench | a61d08d（`perf/firmus-expert-placement-20261018`） | [39053527142](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39053527142) | cb4d23d7…（新規build） | 6418.50、TTFT 110件、ITL 129件 |

3つとも c192、P3/D1、各worker TP4/DP4/EP4、EAGLE3 steps 3 / draft 4、公式60/120/30 clientで、official validationはvalid、errorは0件。

## 3. 統計の採り方（run 39051152456）

- Prefill workerだけに`--expert-distribution-recorder-mode stat --expert-distribution-recorder-buffer-size 2048`を付けた。MegaMoE a2a backendでは、SGLangのselect-experts gathererがlayerとforward passごとのtopk idを数える。
- 公式clientの直前に3つのPrefill workerへ`/start_expert_distribution_record`を送り、終了直後に`/stop_…`と`/dump_…`を送った。したがって記録窓はclient全体（warmup、計測、grace）である。clientのphase境界はserver側からは時刻推定なしに切り分けられないため、窓を絞らなかった。
- dumpはworker内の4 EP rankについてpassごとの和をとった`(2048, 94, 128)`の割当数で、rank 0が`results/expert-distribution/<worker>/`に書く。raw dumpは成果物（`/srv/qwen-artifacts/39051152456/1/results/results/expert-distribution/`、各98.6 MB）にある。
- MoEを含むpassは547 / 551 / 542（残りは中身のないdraft passと空き領域）、tokenは計2567万。どのpassでも全layerの割当合計がtoken数×8で一致することを確認した。

解析は`2026-10-18-megamoe-expert-placement.py`（numpy、`.pt`の読み込みにだけtorch CPU）で行い、結果全体（layer別を含む）は`2026-10-18-megamoe-expert-placement.json`にある。再現は次のとおり。

```
uv run --no-project --python 3.12 --with numpy --with torch --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match python -I research/runtime-performance/2026-10-18-megamoe-expert-placement.py /srv/qwen-artifacts/39051152456/1/results/results/expert-distribution/*/*.pt --placement-out placement.json --report-out report.json
```

## 4. 不均衡の測定結果

rank負荷 = 各passでそのrankのexpertへ割り当てられた(token, expert)数、max/mean = 最も重いrankの負荷 / 4 rankの平均。passのtoken数で加重平均した。

| 指標（94 layer） | trivial | 置換（in-sample） | random置換 |
|---|---|---|---|
| pass単位max/meanの平均 | 1.338 | 1.034 | 1.413 |
| 中央値 | 1.314 | 1.034 | 1.347 |
| p90 | 1.502 | 1.044 | 1.716 |
| 最悪layer | 2.208（L86） | 1.054（L48） | 2.282 |
| 全pass合計のmax/mean | 1.335 | 1.000003 | — |

- trivialでは80 layerが1.2倍を超え、10 layerが1.5倍を超える。1.1倍未満は4 layerだけ。上位はL86 2.21、L74 1.77、L65 1.69、L84 1.67、L89 1.64と深い層に多い（最初の10 layerの平均1.41、最後の10 layerの平均1.51）。
- pass単位の値は全pass合計の値とほぼ同じ（1.338と1.335）。つまり偏りはpassごとの揺らぎではなく系統的で、静的配置で除去できる種類のものである。
- 置換後に残る1.034は、1 expertの集中による下限とpass単位のノイズである。最も熱いexpertはlayer平均で全割当の約11%を占め、rank平均（25%）の半分近くを1つのexpertが持つ。
- out-of-sample: 2 workerから作った配置を残り1 workerで評価すると1.0340〜1.0349、前半passから作った配置を後半passで評価すると1.0347。
- expertの偏り（layerごとの全pass合計）: 最も熱いexpertは平均の7.7〜16.0倍（平均13.8倍）、上位8 expertが割当の52%（26〜71%）を占め、割当0のexpertがあるlayerも多い。熱いexpertの番号はlayerごとに異なり、0番などに固まる計測上の不具合ではない。
- M tile（64 / 128行）単位に丸めても結論は同じ: trivial 1.327 / 1.317、置換 1.033 / 1.033。
- 予測した節約: 1.66 ms × (1.338 − 1.034) ≈ 0.50 ms/layer。servingの2.17 msに比例させると0.47 ms/layer。

## 5. 実装（`perf/firmus-expert-placement-20261018`、a61d08d）

1. `patches/sglang/qwen3-sm90-megamoe.patch`: MegaMoE分岐の`self.topk(...)`に`expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)`を追加した（DeepEP分岐と同じ）。従来は論理idのままkernelへ渡していた。FusedMoEのweight loaderはweightを物理id順に置くので、配置ファイルを与えると無言で誤る状態だった。trivial配置では`ep_dispatch_algorithm`がNoneでdispatch infoもNoneになるため、挙動は変わらない。配置ファイルがあり、a2a backendが`none`でなければ、SGLangは`ep_dispatch_algorithm=static`を自動で選ぶ。これは既にcompile済みのtopk後処理に、層ごと128 entryのgatherが1つ加わるだけである。kernelは物理id 32r..32r+31をrank rのものとして扱い、SGLangの物理layoutと一致する。adapterのguard（EPLB再配置と冗長expertの禁止）はそのまま残した。patch chain全体はpinned source（fc27567）に適用でき、`scripts/check-firmus-megamoe-source.py`でdispatch infoが渡ることを検査する。
2. 配置ファイル: `configs/firmus-prefill-expert-location.json`（`{"physical_to_logical_map": 94×128}`、SGLangの`init_by_mapping`形式）。全3 workerの合計countからlayerごとにSGLang EPLBの`balanced_packing`（greedy LPT、各rankちょうど32 expert）で作った。sha256 `c31515f7…`。
3. 起動: configの`prefill.expert_location`に`trivial`またはrepository pathを書く。provisioningがそのファイルを`source/prefill-expert-location.json`へ置き、Prefillにだけ`--init-expert-location`を渡す。Decodeはtrivialのままである。`scripts/check-firmus-server-info.py`は、全Prefill DP stateについてそのpathと`ep_dispatch_algorithm=static`を、Decodeについて`trivial`/`None`を要求する。
4. 公式parserは出力内容を判定しないので、bench前に1件のgreedy応答を`results/sanity-chat.json`へ保存する。

## 6. 配置benchの検証（run 39053527142）

- build、配備、serving、成果物回収のすべてのjobが成功した。MegaMoE数値gateはPASS。Prefill logには全rankで`init_expert_location from init_by_mapping`が出ており、server-infoでは3つのPrefill worker全stateが`…/source/prefill-expert-location.json` / `static`、Decodeが`trivial` / `None`だった。
- 精度: sanity応答は「The capital of France is Paris. It is historically important because …」で、文として正しい。Decodeの平均EAGLE3 accept lenは2.206で、参照2.208、統計run 2.202と同等。Prefillの出力が壊れていればKVもdraft入力も壊れるので、置換は正しく効いている。
- score: 6418.50（eligible、good fraction 0.944、request 4238件）。TTFT>3 sは110件（参照140）、ITL>20 msは129件（参照82）。TTFT p99は4625 ms（参照5120）、ITL平均は17.7 ms（参照17.3）。
- Prefill: queueが空かない割合は40.1〜40.3%から23.5〜24.4%に、平均queueは0.87〜0.93から0.63〜0.67に下がった。
- kernel（profile、3 worker合計）: Prefill形状（BLOCK_M 128、112 registers）のMegaMoEは、参照の2.144〜2.179 ms（平均2.163）に対して1.940〜2.015 ms（平均1.987）。Prefill workerで時々Decode形状（BLOCK_M 64、swap_ab）のMegaMoEが3.45〜3.49 msかかる呼び出しがあり、今回は6〜8 pass分だった。これは置換と無関係に過去のrunでも出ている（39018872189、39035989837、39040135834で1〜6 pass分、いずれも約3.4〜3.7 ms）。全呼び出しの加重平均は2.142 ms（参照2.163）。

## 7. 解釈と残課題

- 予測の35%しか実現しなかった。均等routingのgateに置換後の偏り1.034を掛けると約1.72 msになるが、実測は1.99 msで、まだ約0.27 ms残る。kernel時間は最も重いrankの割当数だけに比例するわけではなく、ほかの要因が残っている。候補は、送信元rankごとのcombine通信の偏り、熱いexpertに集中するtileの直列化、持続負荷でのclockである。
- Decode形状のMegaMoEがPrefill workerで3.5 msかかる件は、置換と別の改善候補である。選択条件（局所token数が小さいrankだけDecode形状になり、他rankを待たせている可能性）を確認する価値がある。
- 配置は公式benchのsynthetic promptで測った分布に合わせている。精度には影響しないが、分布の異なる入力では効果が変わりうる。
- scoreはDecode側ITLのばらつきに支配され、単一runの差（−1.6%）は、trivialの統計run（+2.7%）とのばらつきの範囲内にある。採否を決めるには、同じimage（cb4d23d7…）で配置あり / trivialを交互に複数回走らせる比較が必要である。そのため`perf/firmus-expert-placement-20261018`は残しておく。統計branchは`archive/perf/firmus-expert-stats-20261018`タグへ退避して削除した。
