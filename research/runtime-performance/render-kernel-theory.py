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
              f"このfixtureのremote率は{d['remote_fraction_all_ranks']:.2%}。FC2 partialは平均268.44MB/rank、remote平均は{268435456*d['remote_fraction_all_ranks']/1e6:.2f}MB/rank。combineの論理read+writeは301.99MB/rankで、4.8TB/s換算62.9µs。実転送量のcounterではない。source別expert countsがない旧runのrank別remote量とpadding量は未確定として残す。", "",
              "今回の候補：6f0164dの5つのFC2出力改善が別branchのまま未統合だった。アドレスを行単位で先に計算し、行の有効性分岐・shared address計算を減らし、metadata loadをBF16 packingに先行させる。BF16 payload、combine加算順、同期は保持。転送量削減とは呼ばない。", "",
              "旧run40116925551はscore8854.80→8922.77だったが、単回・別run比較で因果効果は未確立。現行D5aとの組合せをcontrol/candidate/control-repeatで一度検証する。無関係なQKV候補は再実行しない。現行controlの代表routingとlocalだけに別passのtimelineを付け、同時にsource-rank×expert countsを保存する。候補timelineはCPU compileでspill増加（store32B/load80–84B）が出たため実行しない。通常版は双方168 registers、stack/spill 0、Prefill命令slotは7512→7200 / 8072→7768、Decode3形状のencodingは完全一致。", ""]
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
