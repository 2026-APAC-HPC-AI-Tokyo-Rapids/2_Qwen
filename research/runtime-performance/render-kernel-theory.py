#!/usr/bin/env python3
"""Recalculate the kernel ledger from archived profile data; never launch jobs."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = {
    "moe": "MegaMoE", "attention": "FA3 target", "qkv": "QKV GEMM",
    "o": "O GEMM", "qknorm": "QKnorm RoPE", "add_norm": "add RMSNorm",
    "norm_quant": "norm quant", "dispatch": "pre dispatch", "gate": "router gate",
    "add_quant": "add RMSNorm quant", "merge": "verify split merge",
    "draft_attention": "draft FA3", "draft_gemm": "draft等 BF16 GEMM",
    "routing": "routing topk", "other": "その他 未モデル化",
}


def classify(name, role):
    p = role[0]
    if "sm90_fp8_mega_moe_impl" in name:
        return p + "_moe"
    if "sm90_mega_moe_pre_dispatch_kernel" in name:
        return p + "_dispatch"
    if "sm90_fp8_gemm_1d2d_impl" in name:
        if role == "prefill":
            qkv = "(unsigned int)9216, (unsigned int)4096" in name
        else:
            qkv = "(unsigned int)128, (unsigned int)144, (unsigned int)128" in name
        return p + ("_qkv" if qkv else "_o")
    if "FlashAttnFwdSm90" in name:
        return p + ("_attention" if "apac_fa3_private" in name else "_draft_attention")
    if "qknorm_rope" in name:
        return p + "_qknorm"
    if "add_rmsnorm_group_quant_kernel" in name:
        return p + "_add_quant"
    if "fused_add_rmsnorm" in name:
        return p + "_add_norm"
    if name == "norm_quant":
        return p + "_norm_quant"
    if "apac_fa3_verify::combine_fixed" in name:
        return p + "_merge"
    if "nvjet_sm90_tst_64x80" in name:
        return p + "_gate"
    if "nvjet" in name:
        return p + "_draft_gemm" if role == "prefill" else p + "_other"
    if "_router_triton_kernel" in name:
        return p + "_routing"
    return p + "_other"


def formula(key, config):
    """Work per call. Logical global I/O is NOT a measured HBM lower bound."""
    m, hw = config["model"], config["hardware"]
    h, e, k = m["hidden"], m["expert_intermediate"], m["topk"]
    role = "prefill" if key.startswith("p_") else "decode"
    t = m[role + "_representative_rows"]
    q, v, d = m["q_heads"], m["kv_heads"], m["head_dim"]
    kind = key[2:]
    out = dict(useful_flops=None, tile_flops=None, logical_bytes=None,
               tensor_dtype="fp8", formula="未モデル化", assumption="")
    if kind == "moe":
        out.update(useful_flops=6*t*k*h*e,
                   fc1_flops=4*t*k*h*e, fc2_flops=2*t*k*h*e,
                   fc2_partial_bytes=t*k*h*2,
                   combine_read_write_bytes=t*h*(k+1)*2,
                   physical_weight_bytes=m[role+"_physical_experts_per_rank"]*3*h*e,
                   formula="FC1=4*T*topk*H*I, FC2=2*T*topk*H*I",
                   assumption="全EP rank均衡の平均仕事量。rank別負荷/padding/通信は未確定。")
        if role == "decode":
            out.update(logical_bytes=m["decode_physical_experts_per_rank"]*3*h*e,
                       assumption="32 expertのFP8重みcold readだけ。activation/scale/通信を含まない。")
    elif kind in ("qkv", "o"):
        n, inner = ((q+2*v)*d, h) if kind == "qkv" else (h, q*d)
        bm, bn = (256,128) if role == "prefill" else ((128,144) if kind == "qkv" else (64,96))
        out.update(useful_flops=2*t*n*inner,
                   tile_flops=2*math.ceil(t/bm)*bm*math.ceil(n/bn)*bn*inner,
                   logical_bytes=t*inner+n*inner+2*t*n,
                   formula=f"2*M*N*K; M={t}, N={n}, K={inner}; tile={bm}x{bn}",
                   assumption="FP8 A/B + BF16 Cの各1回I/O。scaleと再読出しは別。")
    elif kind == "attention" and role == "prefill":
        visits = v*sum(math.ceil(min((i+1)*8,t)/192) for i in range(math.ceil(t/8)))
        out.update(useful_flops=2*t*(t+1)*q*d, tile_flops=visits*4*128*192*d,
                   logical_bytes=t*(2*q*d+2*v*d+q*8),
                   formula="causal QK+PV=2*L*(L+1)*Qheads*D; M128/N192 PackGQA",
                   assumption="1 sequence/rank。論理I/Oはunique Q/K/V/O+scale/LSEのみ。KV tile再読出しはL2/DRAM配分不明。",
                   kv_tile_visits=visits,
                   kv_tile_global_bytes=visits*192*d*2,
                   softmax_tile_elements=visits*128*192)
    elif kind == "attention" and role == "decode":
        b, length = m["decode_representative_requests"], m["decode_representative_kv_length"]
        out.update(useful_flops=4*t*length*q*d,
                   logical_bytes=b*length*v*d*2,
                   formula="QK+PV=4*M*KVlen*Qheads*D; KV bytes=B*KVlen*KVheads*D*2",
                   assumption="44 requests, verify4, KV4100の条件模型。KVを各1回読む場合。実profileは可変batch/長さ。")
    elif kind == "qknorm" and role == "prefill":
        out.update(logical_bytes=t*((q+2*v)*d*2+(q+4*v)*d),
                   formula="BF16 packed QKV read + FP8 Q write + contiguous KV write + paged KV write",
                   assumption="既に融合済み。trig/weight/metadata/reduction/変換を除く。")
    elif kind == "add_norm" and role == "prefill":
        out.update(logical_bytes=t*h*8+h*2,
                   formula="2 BF16 reads + 2 BF16 writes + weight once",
                   assumption="target代表shape。100 callsにはdraft等も含むため全体床へ掛けない。")
    elif kind in ("norm_quant", "add_quant"):
        out.update(logical_bytes=t*h*7+t*(h//128)*4+h*2,
                   formula="2 BF16 reads + BF16 residual write + FP8 Q write + FP32 scales",
                   assumption="既に融合。入力/出力がL2に残る場合、この量はHBM量と一致しない。")
    elif kind == "dispatch" and role == "prefill":
        out.update(logical_bytes=t*h*3+t*(h//128)*4+t*k*(4+4+8+4),
                   formula="BF16 X read + FP8 X write + FP32 scale + topk ids/weights read/write",
                   assumption="T3994代表shape。padding/metadata初期化を除く。")
    elif kind == "gate" and role == "prefill":
        out.update(useful_flops=2*t*h*m["logical_experts"], tensor_dtype="bf16",
                   formula="2*T*H*128", assumption="小N GEMM。logitsのI/O dtype/scaleは未確定。")
    elif kind == "merge" and role == "decode":
        out.update(logical_bytes=t*q*d*2*4+t*q*2*4+t*q*d+t*q*4,
                   formula="split2 FP32 partial O/LSE read + FP8 O/scale write",
                   assumption="verify176行。実call shape混在、metadata/reductionを除く。")
    if out["useful_flops"] is not None:
        per_sm = hw["dense_"+out["tensor_dtype"]+"_flops_per_sm_cycle"]
        clocks = hw[role+"_clock_scenario_mhz"]
        rates = [hw["sm_count"]*per_sm*f*1e6 for f in clocks]
        out["tensor_only_us"] = [out["useful_flops"]/rates[1]*1e6, out["useful_flops"]/rates[0]*1e6]
        out["nominal_tensor_only_us"] = out["useful_flops"]/(hw["nominal_dense_"+out["tensor_dtype"]+"_tflops"]*1e12)*1e6
        if out["tile_flops"] is not None:
            out["tile_tensor_only_us"] = [out["tile_flops"]/rates[1]*1e6,out["tile_flops"]/rates[0]*1e6]
    if out["logical_bytes"] is not None:
        out["logical_io_at_hbm_peak_us"] = out["logical_bytes"]/hw["hbm_bytes_per_second"]*1e6
    if "softmax_tile_elements" in out:
        clocks = hw[role+"_clock_scenario_mhz"]
        per_cycle = hw["sm_count"]*hw["scalar_exp2_results_per_sm_cycle"]
        out["tile_exp2_only_us"] = [out["softmax_tile_elements"]/per_cycle/f for f in reversed(clocks)]
    return out


def analyze(config):
    output = {}
    for role, source in config["profiles"].items():
        path = ROOT/source["file"]
        profile = json.loads(path.read_text())
        assert profile["run"] == config["profile_run"] and profile["role"] == role
        assert profile["alignment"]["selected_interval_target_counts_ok"]
        assert profile["alignment"]["prefix_first_layer_counts_ok"]
        grouped = collections.defaultdict(list)
        for row in profile["kernels"]:
            assert row["per_cycle_overlap_ms"] >= 0
            grouped[classify(row["name"],role)].append(row)
        result = []
        for key, rows in grouped.items():
            calls = sum(r["calls"] for r in rows)
            duration_ms = sum(r["total_ms"] for r in rows)
            result.append(dict(id=key, label=LABELS[key[2:]],
                               ms=sum(r["per_cycle_overlap_ms"] for r in rows),
                               mean_us=duration_ms*1000/calls if calls else None,
                               calls_per_cycle=calls/profile["selected_cycles"],
                               theory=formula(key,config),
                               next=config["approach"].get(key,config["approach"]["other"])))
        accounted = sum(r["ms"] for r in result)*profile["selected_cycles"]
        assert math.isclose(accounted, profile["kernel_window_overlap_sum_ms"], abs_tol=1e-6)
        output[role] = dict(profile=profile, rows=sorted(result,key=lambda r:-r["ms"]),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return output


def megamoe_phase_lines(config):
    evidence = config.get("megamoe_phase_evidence")
    if evidence is None:
        return []
    d = json.loads((ROOT/evidence["file"]).read_text())
    phases = [("first_stage_wait", "最初のstage待ち"), ("fc1_loop", "FC1 mainloop"),
              ("fc1_stage_wait_subset", "うちFC1 stage待ち"), ("fc1_epilogue", "FC1 SwiGLU/FP8/store"),
              ("fc2_loop", "FC2 mainloop"), ("fc2_stage_wait_subset", "うちFC2 stage待ち"),
              ("fc2_epilogue_scatter", "FC2 BF16/address/scatter"),
              ("pre_combine_grid_nvlink_wait", "combine前grid/NVLink待ち"), ("combine_cleanup", "combine/cleanup")]
    lines = ["## MegaMoEの既存内部計装と今回の変更", "",
             f"run{d['run']}、image{d['image_run']}、{d['case']}。代表callのWG0、CTA中央値（µs）。rawとparserのhashは`{evidence['file']}`に保存。両WGのmedian/maxも同JSONに保持。現行servingとは別条件。", "",
             "| 区間 | rank0 | rank1 | rank2 | rank3 |", "|---|---:|---:|---:|---:|"]
    for key, name in phases:
        lines.append("| "+name+" | "+" | ".join(f"{r['warpgroups'][0]['phases_us'][key]['cta_median']:.1f}" for r in d["ranks"])+" |")
    lines += ["", f"通常bench中央値{d['bench_median_us']:.1f}µs、計装bench中央値{d['instrumented_bench_median_us']:.1f}µs、別の8-call dumpのcalls2–8中央値{d['timeline_calls_2_end_median_us']:.1f}µs。これらを同じ測定値にしない。CTA中央値の和もcritical pathではない。stage待ちはmainloopの内数。", "",
              "FC2出力の変換・アドレス・store発行はepilogueに入り、完了待ちはgrid/NVLink同期に混ざる。転送の単独所要時間は未確定。localケースとの差もcache/仕事分布が変わるため純NVLink時間にはならない。", "",
              "| rank | 同call SM GHz | FC1 compute床 µs | FC2 compute床 µs | FC1 useful/peak÷loop | FC2 useful/peak÷loop |", "|---|---:|---:|---:|---:|---:|"]
    for r in d["ranks"]:
        w, ratios = r["work"], r["warpgroups"][0]["useful_to_peak_over_median_cta_loop"]
        lines.append(f"| {r['rank']} | {r['sm_clock_ghz_cta_median']:.4f} | {w['fc1_compute_only_us']:.1f} | {w['fc2_compute_only_us']:.1f} | {ratios['fc1']:.1%} | {ratios['fc2']:.1%} |")
    lines += ["", "この比率はpadding/setup/stage待ち/FP32 promotionを含むCTA中央値との診断比で、Tensor pipe実測稼働率ではない。旧73–75%を訂正するが、残差全部を短縮可能と主張しない。", "",
              f"このfixtureのremote率は{d['remote_fraction_all_ranks']:.2%}。FC2 partialは平均268.44MB/rank、remote平均は{268435456*d['remote_fraction_all_ranks']/1e6:.2f}MB/rank。combineの論理read+writeは301.99MB/rankで、4.8TB/s換算62.9µs。combineのFP32 FMAは0.268GFLOP/rankで、memory以外の変換・同期も残る。実転送量のcounterではない。source別expert countsがない旧runのrank別remote量とpadding量は未確定として残す。", "",
              "今回の候補：6f0164dの5つのFC2出力改善が別branchのまま未統合だった。アドレスを行単位で先に計算し、行の有効性分岐・shared address計算を減らし、metadata loadをBF16 packingに先行させる。BF16 payload、combine加算順、同期は保持。転送量削減とは呼ばない。", "",
              "旧run40116925551はscore8854.80→8922.77だったが、単回・別run比較で因果効果は未確立。現行D5aとの組合せをcontrol/candidate/control-repeatで一度検証する。無関係なQKV候補は再実行しない。現行controlの代表routingとlocalだけに別passのtimelineを付け、同時にsource-rank×expert countsを保存する。このcountsは集約済みlayer分布から生成したfixtureの値であり、本番source別routingではないため本番配置改善の根拠に流用しない。候補timelineはCPU compileでspill増加（store32B/load80–84B）が出たため実行しない。通常版は双方168 registers、stack/spill 0、Prefill命令slotは7512→7200 / 8072→7768、Decode3形状のencodingは完全一致。", ""]
    communication = config.get("megamoe_communication_evidence")
    if communication:
        lines += ["### 通信は律速ではないのか", "",
                  "既存run40071892077/40313004834のSQLiteには、NVLink request/response別のUser/Protocol Data Throughput %が残っていた。403130の全4 CUDA GPUがMegaMoE実行中の共通区間で、TX Requests User Data平均18.11–18.35%、p95 31%、最大37%。同窓のTensor Active平均50.96–52.99%、DRAM read15.08–15.31%、write6.77–6.82%。詳細とraw hashは`megamoe-communication-metrics.json`。", "",
                  "これは約1msごとのサンプルで、約200µsのFC2出力区間や約80µsのcombineを分解できない。sampleの集計期間はkernel外を跨ぎ得る。typeIdとCUDA deviceIdの直接対応も未取得なので、下位bitから推測せず全GPU共通区間で比較した。", "",
                  "[NVIDIA定義](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#available-metrics)のNVLink bytesはsample期間の最大byteに対する割合だが、このrequest/response細分metricを足して総利用率やGB/sへ換算しない。低い全体平均は短い転送区間の飽和を否定しない。raw link bytes、L2 traffic、PC stallは未取得。今回のaddress変更は減らせる仕事が具体的な候補であり、『通信ではなく命令が主因』という結論ではない。", ""]
    return lines


def render(config, data):
    lines = ["# カーネルの理論値と改善台帳", "",
             f"更新日 {config['updated_at']}。時間と仕事量の正本は `kernel-profile-40310899466-*.json` と `kernel-theory-model.json`。この文書は `python3 research/runtime-performance/render-kernel-theory.py --write` で再生成する。計算・解析のみでGPU jobは起動しない。", "",
             "実測費用、理論上必要な仕事量、削る対象、過去の結果を同じ単位で追う。最大の未解明箇所はPrefill MegaMoE内部の計算と出力転送、および現在のN192 FA3の依存待ち。小kernelの融合を重複計上しない。", "",
             "## 比較条件", "",
             "対象はrun40310899466、image40197369083、H200、3P1D、EP4、94層。通常scoreと別のprofile passからprefill-0/decode-0 GPU0を再抽出した。両端2周期を外し、同じ94 MegaMoE周期内の全kernelを保持する。PDLで周期境界を跨ぐ仕事は外側の窓だけでclipし、kernel時間の和と重複区間のunionを別々に記録する。元nsys-repのSHA256はarchive manifestと一致。", "",
             "このprofileはBF16 draft時代のもの。後続のFP8 draft採用後run40480817670、router run40528222753のkernel時間として扱わない。新しい通常runのscoreと古いprofileのkernel差を因果関係で結ばない。", "",
             "Prefill代表shapeは3994行（norm_quantの実gridは3994/3995）。旧4032行仮定を訂正。Decodeは144〜176行の混在で、以下の理論欄は176行・44要求・KV4100の条件模型。shape別の新しい実測とは異なる。", "",
             "## 理論値の定義", "",
             "GEMMはFMAを2 FLOPとし、`2MNK`。MoEはFC1とFC2を分け、有効行数とtile丸め後の行数も分ける。Tensor Coreのdense FP8は8192 FLOP/SM/cycle、BF16は4096。132 SMとの積から計算し、旧 `1979×clock/1.98` は使わない。公称FP8 1979 TFLOPSとHBM4.8 TB/sは[NVIDIA仕様](https://www.nvidia.com/en-us/data-center/h200/)、SM単位の算出は[Hopper解説](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)と[NVIDIA技術回答](https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727)に基づく。", "",
             "同窓GPC rawのworker内平均をHzと解釈した場合、Prefill約1.35〜1.38GHz、Decode約1.65〜1.73GHz。表の演算欄はP1.35〜1.40GHz/D1.65〜1.74GHzの明示的な条件計算。NsightのMHzラベルとraw値の桁が不一致で、kernel別clockとdevice対応は未確定。公称値の床も機械計算に保持する。", "",
             "独立資源の必要時間は `max(F/P, B_HBM/W_HBM, B_NVLink/W_NVLink, 命令/発行率)`。因果関係のある直列区間だけを足す。表の『論理I/O換算』は各bufferの論理byteを4.8TB/sで割った参考値であり、L2 hit・再読出し・同期を含むHBM実測値でも、厳密なkernel下限でもない。理論との差をそのまま削れる時間としない。", ""]
    current = config.get("current_megamoe_candidate", {})
    if current:
        lines += ["## 現在の実行状態", "",
                  "2026-10-28、ユーザー指示で旧Firmus kernel benchを停止。Actions40563900060を取消し、停止専用workflow40567082710でSlurm26800のCANCELLED・経過0秒・割当0台・残存runtime削除を確認した。この停止済みbenchのGPU測定は0回。旧benchとは別に通常serving試験を完了し、今回FC2候補の改善は認められなかった。", "",
                  "kernel benchはASPIRE H100 EP4を優先する。従来のFirmus移行条件の目安はASPIRE通常serving公式6500 output tok/sで、kernel単体の改善率では代替しない。最新ASPIREはc112の6081.54（run40274854588、BF16 draft、FA3 control）。6500には約6.88%必要。c104の5925.07はFA3 n192だったため、差をconcurrencyだけの効果としない。今回の通常Firmus結果は以下に記録する。", ""]
    if any(attempt.get("run") == "40569440787" for attempt in current.get("attempts", [])):
        lines += ["ASPIRE EP4経路はcommit396f1fbで実装済み。run40569440787のLinux全検証は成功したが、接続中継WIN-5IM4VJ0SFOSへのSSH timeoutで準備前に停止した。PBS投入・GPU測定・runtime作成は0。中継は10/27 23:36 JST以降offlineで、直接ASPIRE接続もtimeout。このASPIRE benchは再投入していない。", ""]
    for attempt in current.get("attempts", []):
        if attempt.get("mode") == "normal_serving" and attempt.get("stage") == "Linux unit tests before Firmus preparation":
            fixed = attempt["fix_validation"]
            lines += [f"旧通常run{attempt['run']}はGPU前CIで{attempt['failed_tests']}件の設定期待値testに失敗。GPU測定0・runtime作成なし。ASPIRE固定7本とFirmusのFC2追加、D5a後のFC2順序、凍結submissionとの差分を3test fileで明示した。修正commit{attempt['fix_commit'][:7]}のローカル検証は{fixed['passed']} passed・{fixed['skipped']} skipped（既存source mirror未配置）、Ruffとdiff検査はPASS。実行設定・FC2候補・submission実体は変更していない。", ""]
    normal = next((attempt for attempt in reversed(current.get("attempts", [])) if attempt.get("mode") == "normal_serving"), None)
    if normal:
        lines += [f"ユーザーの追加指示『いやfirmusで、benchじゃなくて、普通に投入して欲しいなって』により、今回の通常Firmus FC2試験はASPIRE6500条件より優先して明示許可された。ASPIRE6500は未達のまま。通常serving run{normal['run']}（commit{normal['commit'][:7]}）は投入済み、記録時点の状態は{normal['status']}。kernel benchは実行せず、9235.28のc215・FP8 draft・Prefill開始整列を保持しFC2 patchだけを追加した。FC1候補は非active。", ""]
    if current.get("serving_result"):
        lines += ["通常run40653531712は公式9234.34、reference9235.28比−0.0102%で実質横ばい。平均TTFT1345.50→1329.99ms、ITL15.890→15.939ms、全応答4507.64→4501.79ms。公式validation/eligibility合格・error0、Slurm26960 COMPLETED0:0・14分15秒、artifact回収とruntime削除を確認。設定はFC2追加だけで、実際のPrefill/Decode選択envも一致した。単回の別run比較なので微小差の因果は断定しない。採用済み改善として数えず、同候補の自動再投入を止める。詳細は `megamoe-fc2-serving-40653531712.json`。", "", "ユーザー目標は公式good output 10000超。9234.34から約8.29%必要。raw outputも9403.98なので、SLA違反の解消だけでは10000に届かない。追加GPU計測・subagent再開は行わず既存routingを解析する。", ""]
    if current.get("latest_serving_reference"):
        lines += ["直近の完了Firmus reference run40630160781（840b40f、c215、FP8 draft、Prefill開始整列）は公式9235.28 good output tok/s。imageと7本のDeepGEMM overlayのSHAは本候補の既存controlと全一致した。作業baseをこのcommitへ進め、c215等を保持するFC2追加設定の比較基準とする。FC2 bundleはGitに存在するが9235のserving設定では未有効。旧6f0164dの5本を再適用したheaderともbyte一致するため、新しいFC2手法とは呼ばない。同bundleの旧servingは8922.77（旧8854.80比+0.77%の単回比較）、個別/結合kernel benchの記録は見つからない。bulk/fragment/transpose等の過去の棄却案とは異なり、それらの再測定は不要。今回の9235への追加試験は9234.34（−0.0102%）で上積みなし。詳細は `megamoe-fc2-history-and-9235-compatibility.json`。", ""]
    if config.get("prefill_moe_input_cpu_candidate"):
        lines += ["PP4 Prefill（7988行/microbatch）のMoE入力候補。post-attention add-RMSNormがBF16 y（gate用、in place）とMegaMoE用FP8・FP32 scale・pad slotを1 passで書き、routerが同bufferへtop-kを書くことでpre_dispatch（32.6µs、BF16再読出し65.4MB、計66.5MB/層）を消す。FP8とscaleはpre_dispatchと同じIEEE除算・cvt.rn.satfinite（同じyからbit一致）、residualはflashinferとbit一致、yはinput norm_quantと同じ演算（1 BF16丸め以内）。gate GEMM＋routerを1 kernelにする版（BF16 logitの丸めが変わりうる）も選択肢。見込みは13〜24µs/層（仕事削減、κ0.7〜1.0）、gate融合は追加4〜9µs/層（大半が起動とlatency）。PP4の通常job内でDecode用4 GPUを使い、数値gateと24層CUDA graphの速度比較に通った場合だけ採用する。詳細は `prefill-moe-input-cpu-validation-20261029.json`。", ""]
    if config.get("megamoe_fc1_cpu_candidate"):
        lines += ["FC1にもscale読取アドレスの共通化候補を作成した。約509万組のpointer一致、6 geometryのfresh CPU compileを確認。Prefill128/140のFC1静的spanは290→277/280→267命令（整数address-family各−10）、168 registers・spill0を維持。その他4 geometryは命令encoding一致。動的命令数や速度改善率ではなく、GPUでの数値一致・速度は未確認。候補patchはactive bench inventoryに含めない。詳細は `megamoe-fc1-cpu-validation-20261028.json`。", ""]
    if config.get("megamoe_prefill_4stage_cpu_candidate"):
        lines += ["Prefill 2WGの4 stage候補（host `DG_SM90_PREFILL_4STAGE` + header overlay）。64KiBのFC2 BF16 CDを、FC1 epilogueが既に使うwarp別16行×144byte（計18KiB）へ置き換え、FC2を64列ずつ4回で書き出す。共有メモリは225,472→228,560B（3→4 stage）。FC2 epilogueの直線spanはPP4 chainのEP1で765→223命令/thread/tile（swizzle XOR・行分岐・行毎SHFLが消え、STS/LDS/STG/F2FPは同数）、EP4 main chainで625→241。168 registers・spill0、3 stageとDecodeの全instantiationはSASS一致。見込みはPP4で23〜129µs/層（epilogueは仕事削減、stageは待ち削減でκ0.35〜0.6）。GPUでは未測定で、PP4の通常job内で全12 Prefill GPUのbit一致・FP32 reference・速度を測り、全GPUで速い場合だけ採用する。同じ選択に3つ目の候補として、64行以下の末尾tile（EP1 tileの13.4%、D5aでWG1が遊ぶ半tile）を両WGのm64n128半分で処理するtail splitを追加した（`DG_SM90_PREFILL_TAIL_SPLIT`、末尾template引数のみ追加、FC1 amaxは相方warpとpadで交換）。D5a半tileの実測費用0.64〜0.70 tileを0.52〜0.55へ下げる見込みで、待ち削減なのでκ0.35〜0.6を掛け11〜36µs/層、4 stageと合わせ34〜165µs/層。split版も168 registers・spill0、split無しの全instantiationはSASS一致。詳細は `megamoe-prefill-4stage-cpu-validation-20261029.json`。", ""]
    if config.get("megamoe_prefill_direct_fc2_cpu_candidate"):
        lines += ["Prefill MegaMoE（PP4/EP1、8064 token）の時間模型を作り直した。K loopは共有メモリ量で決まる（2WG 128KiB/K block＝1215clk、4WG 144KiB＝1330clk、4WG→2WGはTMA量不変でoperand読出し減だけで−8%、交互発行はMMA待ち−52µsでもK loop+5%、≤64行tailのskip 88KiBとsplit 96KiBが実測同等）。168 registersで1 MACあたりの共有メモリ量を減らせるtileはなく、multicastも共有メモリ書込みを減らさない。層あたり予算（2.73ms）：有効行のmainloop約1.9〜2.0ms、padding・半tile約190µs、combine 140〜160µs、FC2 epilogueのstaging約55µs、FC1 epilogueの露出25〜40µs、dispatchのpool複写25〜50µs。候補 `DG_SM90_PREFILL_DIRECT_FC2`（host＋header overlay、4 stageのsplit無しtileのみ）は、FC2 partialを256列block内で並べ替えた列順（byte 64j+16cにlane四つ組位置cのn8 chunk 4j..4j+3）でaccumulatorから直接STG.128し、combineが自然順に戻してyを書く。FC2 tileあたり共有メモリ128KiBと直列のSTS→LDS→STG鎖が消え、EP1の直線spanは223→126命令/thread（STS 64→0、LDS 16→0）。bit一致の設計、168 registers・spill0、他の21 instantiation（3 stage、4 stage、split、Decode全形）はSASS一致。見込みは仕事削減κ≈1で40〜55µs/層（1.5〜2.0%）、24層stageで1.0〜1.3ms。GPU未測定、PP4 job内選択に4つ目の変種として追加。詳細は `megamoe-prefill-direct-fc2-cpu-validation-20261030.json`。", ""]
    if config.get("fa3_prefill_direct_o_cpu_candidate"):
        lines += ["Prefill FA3 N192のepilogueを、accumulatorから直接group-128 FP8へ量子化する候補 `n192-direct-o`（variant 32）を作成した。BF16 smem staging、quad SHFLの列並べ替え、named barrier2回、staged layoutのBF16展開とgroup-8 shuffleを除き、consumer warpあたりM-tileごとの実行命令は808→343（CUDA 13.4、3994行で1回7.44M warp命令・smem131MB減）。定常KV loopは597命令/tileのまま、registers・spillは不変、他のgeometryのSASSは一致。出力codeとscaleはn192とbyte一致する設計で、in-job選択のGPU gateがn192とのbyte一致と速度を判定する。見込みは単体−4〜−6%、電力上限下−2〜−5%でFA3目標10%に届かない。GPU未測定・未有効化。詳細は `fa3-prefill-direct-o-cpu-validation-20261029.json`。", ""]
    if config.get("fa3_prefill_varlen_cpu_candidate"):
        lines += ["PP4（run40825417205）では2要求のmicrobatchが固定経路に入らず、prepared N224（899.7µs/回＋prepare 4.1µs）で動いていた。固定schedulerを1〜4要求の連結順へ広げたvariant 64の `n192-varlen` と `n192-direct-o-varlen` を追加し、PP4ではこの2行だけを3長と3986+4027の組でin-job選択する（controlは組ではprepared N224）。数値gateは要求ごとのn192とのbyte一致を要求する。定常KV loopは不変、spillはproducer側で72/132→96/124B、他kernelのSASS本体は一致。見込みは124ms周期あたり電力上限下で−0.7〜−1.6ms（direct-o併用）。draft extend（stock paged N160、BF16出力）は同じ64/4/128だが出力・KV形式が異なり、stage3は律速でないため未配線。GPU未測定。詳細は `fa3-prefill-varlen-cpu-validation-20261029.json`。", ""]
    for role in ("prefill", "decode"):
        p = data[role]["profile"]
        busy = p["kernel_union_busy_ms"]/p["selected_cycles"]
        lines += [f"## {role.capitalize()} の収支", "",
                  f"{p['selected_cycles']}周期、周期平均 **{p['cycle_mean_ms']:.3f} ms**、kernel union **{busy:.3f} ms**、kernelのない区間 **{p['cycle_mean_ms']-busy:.3f} ms**。{len(p['kernels'])}個のkernel名とgeometryの組を全て保持。重なったkernelの時間和を周期時間へ足し直さない。", "",
                  "| kernel | 実測 ms/周期 | 実測 µs/回 | 演算だけの床 µs/回 | 論理I/O換算 µs/回 |",
                  "|---|---:|---:|---:|---:|"]
        for row in data[role]["rows"]:
            th = row["theory"]
            tensor = th.get("tensor_only_us")
            io = th.get("logical_io_at_hbm_peak_us")
            lines.append(f"| {row['label']} | {row['ms']:.3f} | {row['mean_us']:.2f} | "+(f"{tensor[0]:.1f}–{tensor[1]:.1f}" if tensor else "未算定")+" | "+(f"{io:.1f}" if io is not None else "未算定")+" |")
        lines += ["", "µs/回はその分類の実測平均。shape混在・draft混在の行に単一shapeの床を掛けて周期全体の『余地』を作らない。", ""]
    lines += megamoe_phase_lines(config)
    lines += ["## 仕事量と改善の対応", ""]
    for role in ("prefill", "decode"):
        for row in data[role]["rows"]:
            th, action = row["theory"], row["next"]
            if th["useful_flops"] is None and th["logical_bytes"] is None:
                continue
            lines += [f"### {role.capitalize()} {row['label']}", "", th["formula"]+"。", ""]
            parts = []
            if th["useful_flops"] is not None:
                parts.append(f"有効演算 {th['useful_flops']/1e9:.3f} GFLOP/回")
            if th["tile_flops"] is not None:
                parts.append(f"tile演算 {th['tile_flops']/1e9:.3f} GFLOP/回（{th['tile_flops']/th['useful_flops']:.3f}倍）")
            if th["logical_bytes"] is not None:
                parts.append(f"論理I/O {th['logical_bytes']/1e6:.3f} MB/回")
            lines += ["、".join(parts)+"。"+th["assumption"], "",
                      "追う差："+action["gap_to_explain"], "", "次の作業："+action["next"], "", "測定へ進む条件："+action["gate"], ""]
            if row["id"].endswith("_moe"):
                peak = config["hardware"]["hbm_bytes_per_second"]
                link = config["hardware"]["nvlink_one_direction_bytes_per_second"]
                partial = th["fc2_partial_bytes"]
                lines += ["内部の仕事量（時間の分解とは別）：", "",
                          "| 区間 | 仕事量/層/rank | 追跡する分母 |", "|---|---:|---|",
                          f"| FC1 | {th['fc1_flops']/1e9:.3f} GFLOP | 同runのFC1時間、実rowとtile row |",
                          f"| FC2 | {th['fc2_flops']/1e9:.3f} GFLOP | 同runのFC2時間、実rowとtile row |",
                          f"| 全expert重み | {th['physical_weight_bytes']/1e6:.3f} MB | HBM cold readなら{th['physical_weight_bytes']/peak*1e6:.1f}µs、実cache traffic未確定 |",
                          f"| FC2 partial出力 | {partial/1e6:.3f} MB | 全てlocal HBMなら{partial/peak*1e6:.1f}µs相当 |",
                          f"| combine read＋output write | {th['combine_read_write_bytes']/1e6:.3f} MB | HBM換算{th['combine_read_write_bytes']/peak*1e6:.1f}µs、cache/演算は別 |",
                          f"| FC2 remote出力 | {partial/1e6:.3f} MB × remote率r | 片方向450GB/s換算 {partial/link*1e6:.1f}×r µs |", "",
                          "rはこのrunでは未取得。旧routing fixtureの48.6%を現在の実測値へ代入しない。通信が計算に重なる部分は足し算せず、追加bufferのread/writeも必ず勘定する。", ""]
            if row["id"] == "p_attention":
                lines += [f"N192の条件模型ではKV tile訪問{th['kv_tile_visits']:,}回、softmax対象{th['softmax_tile_elements']/1e6:.3f}M要素。tile単位のKV読出しは{th['kv_tile_global_bytes']/1e9:.3f}GB相当で、unique I/Oの71.6MBとは別物。L2 hit率を無視してどちらかをHBM実測量と呼ばない。tile Tensorだけの床は{th['tile_tensor_only_us'][0]:.1f}〜{th['tile_tensor_only_us'][1]:.1f}µs。", ""]
                lines += [f"各tile要素にscalar exp2を1回実行する模型では、[CUDAの16結果/SM/cycle](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html#arithmetic-instructions)からexpだけで{th['tile_exp2_only_us'][0]:.1f}〜{th['tile_exp2_only_us'][1]:.1f}µs。同じ条件のTensor時間とほぼ等しい。これらは重なり得るので足し算しない。追加rescale exp、変換、ALU、転置、依存latencyを加えた現在の命令モデルは未完成であり、436µsとの差を単に『expが遅い』とは判定しない。", ""]
    lines += ["## 優先順位と不足する証拠", "",
              "1. Prefill MegaMoE：142ms/周期が最大。旧40057117293のFC1/FC2をrawから分解し、同callのclockで演算比を訂正した。上記表をこのserving profileへ差し引かない。未統合だったFC2 address bundleを現行D5aと組み合わせて検証する。source rank別remote行数が無ければ配置の得は未判定。hot4複製は既に有効。", "",
              "2. Prefill FA3：41ms/周期。N192のtile仕事量と有効演算の差を分離した。Tensorだけの床との差にはsoftmax/FP8変換/転置/同期がある。旧N224 NCUのpipe比率を現在の内訳へ転用しない。", "",
              "3. post-attention norm→MegaMoE入力：gate用BF16を残しつつFP8/scaleを直接書くと、BF16再読出し1回を削れる可能性がある。3994×4096×2=32.72MB/層、94層で3.08GB。4.8TB/s換算0.64ms/周期はbyteだけの参考値で、予測短縮ではない。12.3ms丸ごとの削減案ではない。", "",
              "4. Decodeも同時に追う：MoEの重み604MB/層、verifyのKV読出しを別の上限として記録。Prefill供給だけを増やすとITLが悪化し得る。CPU起動間隙の除去だけで大幅改善とは見積もらない。", "",
              "不足：現行FP8 draftのkernel profile、同runのMoE subphase、実HBM/NVLink bytes、rank別routing、N192のopcode/依存内訳、Decodeの各callのKV長。これらは0ではなく未確定。既存成果物で解ける項目から埋め、採否が変わらない測定は増やさない。", "",
              "## 過去の結果", "", "| 候補 | run | 状態と結果 |", "|---|---|---|"]
    for trial in config["previous_trials"]:
        lines.append(f"| {trial['candidate']} | {trial['run'] or 'CPU compileのみ'} | {trial['status']}: {trial['result']} |")
    lines += ["", "## 更新手順", "",
              "- 候補を変える前にkernel名、shape/dtype、source/image、FLOP/bytes/命令の何を減らすか、過去の棄却理由を更新する。", "- 未検証の改善幅は成功条件として記載し、実測欄には入れない。", "- 実験後は同shapeのcontrol/candidate、clock/電力、数値品質、normal goodput・完了時間・ITL・受理長を追加する。失敗も残す。", "- profileは通常benchと別に扱う。更新は既存のジョブに必要な観測を同乗させる判断を優先する。", "",
              "9500は旧通常score9161.6688から約3.69%増が必要。仮にPrefillだけが律速し他条件が固定なら、251msから約242msへ約9ms削る規模になる。これは必要条件の逆算で、達成予測ではない。routerで先頭だけ約201ms早めても完了時間は約1msしか変わらなかったため、今後の採用は完了時間とgoodputで判定する。", "",
              "## 再現", "", "```sh",
              "nsys export --type sqlite --output /tmp/prefill-0.sqlite /srv/qwen-artifacts/40310899466/1/profile/profile/nsys-prefill-0.nsys-rep",
              "python3 research/runtime-performance/extract-kernel-profile.py /tmp/prefill-0.sqlite --role prefill --run 40310899466 > research/runtime-performance/kernel-profile-40310899466-prefill.json",
              "nsys export --type sqlite --output /tmp/decode-0.sqlite /srv/qwen-artifacts/40310899466/1/profile/profile/nsys-decode-0.nsys-rep",
              "python3 research/runtime-performance/extract-kernel-profile.py /tmp/decode-0.sqlite --role decode --run 40310899466 > research/runtime-performance/kernel-profile-40310899466-decode.json",
              "python3 research/runtime-performance/render-kernel-theory.py --write", "```", "",
              "GPU計算なし。元nsys-repのhashはmodel JSON、export SQLiteと抽出scriptのhashはprofile JSONに保存。窓・呼出し数・重複・shape仮定を変更した場合は旧値へ上書きせず、差と理由をGitに残す。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT/"kernel-theory-model.json").read_text())
    data = analyze(config)
    report = render(config,data)
    target = ROOT/"kernel-theory-tracker.md"
    if args.check:
        assert target.read_text() == report, "Generated ledger is stale"
        print("Profile accounting and generated ledger: PASS")
    elif args.write:
        target.write_text(report)
        print(target)
    else:
        print(report)


if __name__ == "__main__":
    main()
