"""Default-off composed native-V/sum gate, sequenced after the round-12 gate."""
import argparse
import json
import math
import os
import runpy
import statistics
import time
from pathlib import Path

KEY = "SGLANG_PREFILL_FA3_VERIFY"
CONTROL = "n192-direct-o-varlen-pipe+m192n96-pair"
DISPATCH = "n192-direct-o-varlen-pipe+m192n96-pair-vsum"
SCHEMA = "firmus-prefill-fa3-vsum/v2"
SHAPES = ((3994,3994),(3986,4027),(3986,3986),(4027,4027),(4096,4096),(3900,4096),(4096,3901))
SINGLE_SHAPES = ((3994,),(4096,),(3584,))
ROUNDS, REPS, ENDURANCE, SECONDS = 10, 12, 1200, 100


def row_name(lengths, candidate):
    prefix = "m192n96" if len(lengths)==2 else "n192"
    return prefix+"-direct-o-varlen-pipe"+("-vsum" if candidate and len(lengths)==2 else "")


def timing_summary(samples):
    medians = {}
    for order in ("forward", "reverse"):
        medians[order] = {}
        for mode in ("control", "candidate"):
            values = samples[order][mode]
            assert len(values)==ROUNDS and all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in values), "invalid timing"
            medians[order][mode] = statistics.median(values)
    savings = {order: medians[order]["control"]-medians[order]["candidate"] for order in medians}
    savings["pooled"] = statistics.median(samples["forward"]["control"]+samples["reverse"]["control"])-statistics.median(samples["forward"]["candidate"]+samples["reverse"]["candidate"])
    return {"medians_us":medians, "savings_us":savings}


def valid_metrics(metrics):
    assert len(metrics)==3, "missing replays"
    for metric in metrics:
        for key,limit in (("relative_l2",.05),("max_scaled_error",.25)):
            value=metric[key]
            assert type(value) in (int,float) and math.isfinite(value) and 0<=value<=limit, "numerical tolerance exceeded"


def reject_constant(value):
    raise ValueError("nonfinite JSON: "+value)


def finite_float(value):
    number=float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number: "+value)
    return number


def aggregate(directory, env_output, pair_selection, run_id, deadline, workers_ok):
    result={"schema":SCHEMA,"adopted":False,"gpus":[],"run_id":run_id,"rule":{"shapes":SHAPES,"single_checks":SINGLE_SHAPES,"minimum_saving_us":5,"seconds":SECONDS,"endurance_calls":ENDURANCE,"timing":"QK post + FA3 + O"}}
    try:
        pair=json.loads(pair_selection.read_text(),parse_constant=reject_constant,parse_float=finite_float)
        assert pair["schema"]=="firmus-prefill-fa3-pair/v1" and pair["adopted"] is True, "round-12 gate did not adopt"
        pair_gpus=pair["gpus"]
        assert len(pair_gpus)==12 and len({g["uuid"] for g in pair_gpus})==12, "incomplete round-12 UUIDs"
        # Independent srun steps may swap worker labels on the same node.
        # Every physical GPU must still have passed the pair gate in this allocation.
        pair_identity={g["uuid"]:g for g in pair_gpus}
        assert workers_ok and time.time()<=deadline, "workers failed, timed out or late"
        uuids=set()
        for worker in range(3):
            for rank in range(4):
                record=json.loads((directory/f"prefill-{worker}"/f"rank-{rank}.json").read_text(),parse_constant=reject_constant,parse_float=finite_float)
                result["gpus"].append(record)
                assert record["schema"]==SCHEMA and record["run_id"]==run_id, "stale result"
                assert record["worker"]==f"prefill-{worker}" and record["rank"]==rank, "wrong worker"
                assert isinstance(record["uuid"],str) and record["uuid"] in pair_identity and record["uuid"] not in uuids and "H200" in record["device"], "wrong GPU"
                # Same Slurm allocation; the pair gate's timestamp suffix is independent.
                assert record["run_id"].split("-",1)[0]==pair_identity[record["uuid"]]["run_id"].split("-",1)[0], "wrong round-12 allocation"
                uuids.add(record["uuid"])
                assert record["passed"] is True and 0<record["elapsed_seconds"]<SECONDS and record["finished_at"]<=deadline, "failed or late GPU"
                assert set(record["shapes"])=={"+".join(map(str,s)) for s in SHAPES}, "incomplete shapes"
                for shape in record["shapes"].values():
                    assert shape["byte_image_first"] is True and shape["producer_bytes_equal"] is True and shape["replays"]==2, "writer/dump/replay failure"
                    valid_metrics(shape["candidate_fp32"])
                    valid_metrics(shape["control_fp32"])
                    assert all(v>=5 for v in timing_summary(shape["samples_us"])["savings_us"].values()), "below 5 us margin"
                assert set(record["singles"])=={str(s[0]) for s in SINGLE_SHAPES}, "incomplete single checks"
                for single in record["singles"].values():
                    assert single["row"]==row_name((3994,),False) and single["writer"]=="fp32", "single route changed"
                    assert single["producer_bytes_equal"] is True and single["attention_bytes_equal"] is True and single["replays"]==2, "single byte/replay failure"
                    assert "samples_us" not in single, "single speed gate forbidden"
                    valid_metrics(single["candidate_fp32"])
                    valid_metrics(single["control_fp32"])
                endurance=record["endurance"]
                assert endurance["calls"]==ENDURANCE and endurance["bytes_equal"] is True and endurance["dispatch_equal"] is True, "failed endurance"
                valid_metrics([endurance["initial_fp32"],endurance["final_fp32"],endurance["final_fp32"]])
        lines=env_output.read_text().splitlines(keepends=True)
        assert [l.rstrip("\n") for l in lines if l.startswith(KEY+"=")]==[KEY+"="+CONTROL], "round-12 composite env required"
        assert time.time()<=deadline, "aggregate deadline expired"
        temporary=env_output.with_name(env_output.name+".vsum.tmp")
        try:
            temporary.write_text("".join(KEY+"="+DISPATCH+"\n" if l.startswith(KEY+"=") else l for l in lines))
            temporary.replace(env_output)
        finally:
            temporary.unlink(missing_ok=True)
        result.update(adopted=True,reason="all twelve H200s passed pair margins, single fallback bytes, reference and endurance")
    except Exception as error:  # noqa: BLE001 - gate refusal preserves serving
        result["reason"]=f"{type(error).__name__}: {error}"
        result["gpus"]=[]
    return result


def native_producer(common, image, n):
    from sglang.kernels.ops.attention.rope import apply_qknorm_rope_fp8_prefill_native
    apply_qknorm_rope_fp8_prefill_native(*common["rope_args"],image,common["cu"],n,max(common["lengths"]))


def steps(torch, checks, common):
    import deep_gemm
    lengths=common["lengths"]
    assert len(lengths)==2 and all(3900<=n<=4096 for n in lengths)
    n=96
    image=torch.empty((len(lengths),4,(max(lengths)+n-1)//n,n*128),device="cuda",dtype=torch.uint8)
    control=checks["prefill_selection_step"](common,row_name(lengths,False),1)
    candidate=checks["prefill_specs"]().index(next(s for s in checks["prefill_specs"]() if s["name"]==row_name(lengths,True)))
    out=torch.empty_like(control.output[0])
    scales=torch.empty((64,(common["rows"]+3)//4*4),device="cuda",dtype=torch.float32).T[:common["rows"]]
    metadata=torch.empty(17,device="cuda",dtype=torch.int32)
    lse=torch.empty((64,common["rows"]),device="cuda",dtype=torch.float32)
    # Same actual O projection and compiled nk contract as prefill_fp8.linear.
    data=checks["dense_inputs"](common["rows"],4096,8192,49183)
    weight=data[1]
    o_outputs={mode:torch.empty_like(data[2]) for mode in ("control","candidate")}

    def native():
        native_producer(common,image,n)
        torch.ops.apac_fa3_verify.prefill_variant_fwd_out(common["q"],common["k"],common["v"],image,common["cu"],out,scales,lse,metadata,max(lengths),128**-.5,candidate)
        return out,scales

    attention={"control":control,"candidate":native}
    outputs={"control":control.output,"candidate":(out,scales)}
    closures={}
    for mode,call in attention.items():
        def step(mode=mode,call=call):
            for _ in range(REPS):
                a,sa=call()
                deep_gemm.fp8_gemm_nt((a.view(common["rows"],8192),sa),weight,o_outputs[mode],compiled_dims="nk")
        closures[mode]=step
    # Retain every buffer and closure for the lifetime of captured graphs.
    return {"image":image,"n":n,"attention":attention,"output":outputs,"timed":closures,"weight":weight,"o_outputs":o_outputs,"storage":(out,scales,metadata,lse,data)}


def validate(torch, checks, common, state):
    metrics={mode:[] for mode in state["attention"]}
    for replay in range(3):
        if replay:
            common["packed"].copy_(torch.randn_like(common["packed"])*(replay+.5))
        # Run no attention before independently dumping/checking the deployed V image.
        native_producer(common,state["image"],state["n"])
        produced=[x.clone() for x in (common["q"],common["k"],common["kc"],common["vc"])]
        from sglang.kernels.ops.attention.rope import apply_qknorm_rope_fp8_prefill_fp32
        apply_qknorm_rope_fp8_prefill_fp32(*common["rope_args"])
        assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(produced,(common["q"],common["k"],common["kc"],common["vc"]),strict=True)), "Q/K or paged KV byte mismatch"
        dump=torch.empty_like(state["image"])
        torch.ops.apac_fa3_verify.prefill_dump_native(common["v"],dump,common["cu"],state["n"])
        assert torch.equal(dump,state["image"]), "native V dump mismatch including request padding"
        indices,reference=checks["prefill_fp32_reference"](common)
        for mode,call in state["attention"].items():
            out,scales=call()
            metrics[mode].append(checks["prefill_assert_errors"](out.float()[indices]*scales[indices,:,None],reference,mode+"_fp32"))
    return {"byte_image_first":True,"producer_bytes_equal":True,"replays":2,"candidate_fp32":metrics["candidate"],"control_fp32":metrics["control"]}


def bytes_equal(torch, left, right):
    return all(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)) for a,b in zip(left,right,strict=True))


def single_check(torch, checks, common):
    """No timing: the new env must preserve round-12 single producer/output bytes."""
    from sglang.kernels.ops.attention.rope import apply_qknorm_rope_fp8_prefill_fp32
    from sglang.srt.layers.attention.prefill_fp8 import (
        attention,
        pair_candidate,
        prefill_candidate,
    )
    os.environ[KEY]=DISPATCH
    prefill_candidate.cache_clear()
    selected=pair_candidate(common["lengths"])
    assert selected["name"]==row_name(common["lengths"],False) and not selected.get("native_v_n"), "single route changed"
    control=checks["prefill_selection_step"](common,row_name(common["lengths"],False),1)
    metrics={"candidate":[],"control":[]}
    common["kc"].zero_(); common["vc"].zero_()
    for replay in range(3):
        if replay:
            common["packed"].copy_(torch.randn_like(common["packed"])*(replay+.5))
        # Same host-selected old-writer branch as the model; no native V buffer.
        apply_qknorm_rope_fp8_prefill_fp32(*common["rope_args"])
        produced=tuple(x.clone() for x in (common["q"],common["k"],common["v"],common["kc"],common["vc"]))
        out,scales=attention(common["q"],common["k"],common["v"],common["cu"],max(common["lengths"]),128**-.5,common["rows"],fixed=True,v_image=None,lengths=common["lengths"])
        candidate=(out.view(common["rows"],64,128).clone(),scales.clone())
        baseline=control()
        assert bytes_equal(torch,produced,(common["q"],common["k"],common["v"],common["kc"],common["vc"])), "single Q/K/V or paged KV changed"
        assert bytes_equal(torch,candidate,baseline), "single attention differs from round-12 served row"
        indices,reference=checks["prefill_fp32_reference"](common)
        for mode,output in (("candidate",candidate),("control",baseline)):
            metrics[mode].append(checks["prefill_assert_errors"](output[0].float()[indices]*output[1][indices,:,None],reference,"single_"+mode))
    return {"row":selected["name"],"writer":"fp32","producer_bytes_equal":True,"attention_bytes_equal":True,"replays":2,"candidate_fp32":metrics["candidate"],"control_fp32":metrics["control"]}


def measure(torch, checks, state):
    graphs={mode:checks["prefill_selection_capture"](step) for mode,step in state["timed"].items()}
    samples={order:{mode:[] for mode in graphs} for order in ("forward","reverse")}
    for _ in range(ROUNDS):
        for order,modes in (("forward",("control","candidate")),("reverse",("candidate","control"))):
            for mode in modes:
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record(); graphs[mode].replay(); b.record(); b.synchronize()
                samples[order][mode].append(a.elapsed_time(b)*1000/REPS)
    return samples


def component_timings(torch, checks, common, state):
    """First-shape diagnostics only; isolated times are not additive or adoption gates."""
    import deep_gemm
    from sglang.kernels.ops.attention.rope import apply_qknorm_rope_fp8_prefill_fp32

    def repeated(call):
        def step():
            for _ in range(REPS):
                call()
        return step

    producers={"control":lambda:apply_qknorm_rope_fp8_prefill_fp32(*common["rope_args"]),"candidate":lambda:native_producer(common,state["image"],state["n"])}
    producer_samples=measure(torch,checks,{"timed":{mode:repeated(call) for mode,call in producers.items()}})
    specs=checks["prefill_specs"]()
    metadata=torch.empty(17,device="cuda",dtype=torch.int32)
    lse=torch.empty((64,common["rows"]),device="cuda",dtype=torch.float32)
    empty=torch.empty(0,device="cuda",dtype=torch.uint8)
    calls={}
    for mode in ("control","candidate"):
        candidate=mode=="candidate"
        index=next(i for i,spec in enumerate(specs) if spec["name"]==row_name(common["lengths"],candidate))
        image=state["image"] if candidate else empty
        def attention_o(mode=mode,index=index,image=image):
            out,scales=state["output"][mode]
            torch.ops.apac_fa3_verify.prefill_variant_fwd_out(common["q"],common["k"],common["v"],image,common["cu"],out,scales,lse,metadata,max(common["lengths"]),128**-.5,index)
            deep_gemm.fp8_gemm_nt((out.view(common["rows"],8192),scales),state["weight"],state["o_outputs"][mode],compiled_dims="nk")
        calls[mode]=repeated(attention_o)
    attention_samples=measure(torch,checks,{"timed":calls})
    return {"scope":"isolated producer and FA3+O; not additive, not an adoption criterion","producer_samples_us":producer_samples,"producer_summary":timing_summary(producer_samples),"attention_o_samples_us":attention_samples,"attention_o_summary":timing_summary(attention_samples)}


def endurance(torch, checks, common, state):
    import deep_gemm
    from sglang.srt.layers.attention.prefill_fp8 import (
        attention,
        pair_candidate,
        prefill_candidate,
    )
    os.environ[KEY]=DISPATCH
    prefill_candidate.cache_clear()
    assert pair_candidate(common["lengths"])["name"]==row_name(common["lengths"],True)
    state["attention"]["candidate"]()
    direct=tuple(x.clone() for x in state["output"]["candidate"])

    def call():
        native_producer(common,state["image"],state["n"])
        out,scales=attention(common["q"],common["k"],common["v"],common["cu"],max(common["lengths"]),128**-.5,common["rows"],fixed=True,v_image=state["image"],lengths=common["lengths"])
        deep_gemm.fp8_gemm_nt((out,scales),state["weight"],state["o_outputs"]["candidate"],compiled_dims="nk")
        return out,scales

    def equal(a,b):
        return all(torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)) for x,y in zip(a,b,strict=True))

    first=tuple(x.clone() for x in call())
    first_o=state["o_outputs"]["candidate"].clone()
    assert equal(first,(direct[0].view(common["rows"],8192),direct[1])), "serving dispatch mismatch"
    indices,reference=checks["prefill_fp32_reference"](common)

    def numeric(output):
        return checks["prefill_assert_errors"](output[0].view(common["rows"],64,128).float()[indices]*output[1][indices,:,None],reference,"endurance")

    initial=numeric(first)
    for _ in range(ENDURANCE):
        last=call()
    torch.cuda.synchronize()
    final=numeric(last)
    assert equal(first,last) and torch.equal(first_o.view(torch.uint8),state["o_outputs"]["candidate"].view(torch.uint8)), "endurance FA3/scale/O bytes changed"
    return {"calls":ENDURANCE,"bytes_equal":True,"dispatch_equal":True,"initial_fp32":initial,"final_fp32":final}


def run(output, run_id, deadline):
    begin=time.monotonic()
    import torch
    from sglang.srt.layers.attention.decode_fa3_verify import load
    rank=int(os.environ["LOCAL_RANK"])
    result={"schema":SCHEMA,"worker":output.name,"rank":rank,"run_id":run_id,"passed":False,"shapes":{},"singles":{}}
    output.mkdir(parents=True,exist_ok=True)
    try:
        assert int(os.environ["WORLD_SIZE"])==4
        torch.set_num_threads(1); torch.cuda.set_device(rank)
        result.update(device=torch.cuda.get_device_name(rank),uuid=str(torch.cuda.get_device_properties(rank).uuid))
        assert "H200" in result["device"] and torch.cuda.get_device_capability()==(9,0)
        load()
        checks=runpy.run_path(str(Path(__file__).with_name("check-firmus-decode-kernels.py")))
        manifest=json.loads(Path("/opt/apac/fa3-verify/manifest.json").read_text())
        assert [{k:v for k,v in row.items() if k!="shared_bytes"} for row in manifest["prefill_candidates"]]==checks["prefill_specs"](), "native table mismatch"
        result["binary_sha256"]=manifest["binary_sha256"]
        for lengths in SHAPES:
            assert time.time()<=deadline
            common=checks["prefill_selection_inputs"](lengths,padding=0)
            common["fp32_producer"]=True
            state=steps(torch,checks,common)
            record=validate(torch,checks,common,state)
            record["samples_us"]=measure(torch,checks,state)
            record["timing_summary"]=timing_summary(record["samples_us"])
            result["shapes"]["+".join(map(str,lengths))]=record
            if lengths==SHAPES[0]:
                record["components"]=component_timings(torch,checks,common,state)
            assert all(v>=5 for v in record["timing_summary"]["savings_us"].values()), "below 5 us margin"
            del common,state
        for lengths in SINGLE_SHAPES:
            assert time.time()<=deadline
            common=checks["prefill_selection_inputs"](lengths,padding=0)
            common["fp32_producer"]=True
            result["singles"][str(lengths[0])]=single_check(torch,checks,common)
            del common
        common=checks["prefill_selection_inputs"](SHAPES[0],padding=0)
        common["fp32_producer"]=True
        state=steps(torch,checks,common)
        validate(torch,checks,common,state)
        result["endurance"]=endurance(torch,checks,common,state)
        assert time.monotonic()-begin<SECONDS and time.time()<=deadline
        result["passed"]=True
    except Exception as error:  # noqa: BLE001 - persist rejection, continue serving
        result["error"]=f"{type(error).__name__}: {error}"
    result.update(elapsed_seconds=time.monotonic()-begin,finished_at=time.time())
    temporary=output/f"rank-{rank}.tmp"
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    temporary.replace(output/f"rank-{rank}.json")
    print(json.dumps(result,allow_nan=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aggregate",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--env-output",type=Path)
    p.add_argument("--pair-selection",type=Path)
    p.add_argument("--run-id",required=True)
    p.add_argument("--deadline",type=float,required=True)
    p.add_argument("--workers-ok",type=int,choices=(0,1),default=0)
    a=p.parse_args()
    if a.aggregate:
        result=aggregate(a.aggregate,a.env_output,a.pair_selection,a.run_id,a.deadline,bool(a.workers_ok))
        a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k!="gpus"}),flush=True)
    else:
        run(a.output,a.run_id,a.deadline)


if __name__=="__main__":
    main()
