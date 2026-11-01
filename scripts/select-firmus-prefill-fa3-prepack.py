"""Default-off round-14 P/S selection and factorial diagnostics in one process group."""
import argparse
import hashlib
import json
import math
import os
import runpy
import statistics
import time
from pathlib import Path

KEY = "SGLANG_PREFILL_FA3_VERIFY"
CONTROL = "n192-direct-o-varlen-pipe+m192n96-pair"
DISPATCH = {"P": CONTROL+"-prepack", "S": CONTROL+"-sum"}
SCHEMA = "firmus-prefill-fa3-prepack/v1"
SHAPES = ((3994,3994),(3986,4027),(3986,3986),(4027,4027),(4096,4096),(3900,4096),(4096,3901))
SINGLE_SHAPES = ((3994,),(4096,),(3584,))
ROUNDS, REPS, ENDURANCE, SECONDS = 10, 12, 1200, 175
CANDIDATE_SECONDS = 45  # Full validation reservation, never shorten its shapes/endurance.
VARIANTS = {"C":8416,"S":24800,"V":41184,"VS":57568,"P":41184}
ROWS = {"C":"m192n96-direct-o-varlen-pipe", "S":"m192n96-direct-o-varlen-pipe-sum", "V":"m192n96-direct-o-varlen-pipe-native", "VS":"m192n96-direct-o-varlen-pipe-vsum", "P":"m192n96-direct-o-varlen-pipe-native"}


def row_name(lengths, candidate):
    return ROWS[candidate] if len(lengths)==2 else "n192-direct-o-varlen-pipe"


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



def check_candidate(record, candidate):
    assert record["passed"] is True, record.get("error","candidate incomplete")
    assert set(record["shapes"])=={"+".join(map(str,s)) for s in SHAPES}, "incomplete shapes"
    for shape in record["shapes"].values():
        assert shape["row"]==ROWS[candidate] and shape["variant"]==VARIANTS[candidate], "wrong candidate row/variant"
        assert shape["byte_image_first"] is True and shape["producer_bytes_equal"] is True and shape["replays"]==2, "byte/replay failure"
        valid_metrics(shape["candidate_fp32"])
        valid_metrics(shape["control_fp32"])
        assert all(v>=5 for v in timing_summary(shape["samples_us"])["savings_us"].values()), "below 5 us margin"
    assert set(record["singles"])=={str(s[0]) for s in SINGLE_SHAPES}, "incomplete singles"
    for single in record["singles"].values():
        assert single["row"]=="n192-direct-o-varlen-pipe" and single["writer"]=="fp32", "single route changed"
        assert single["producer_bytes_equal"] is True and single["attention_bytes_equal"] is True and single["replays"]==2, "single bytes/replay"
        assert "samples_us" not in single, "single speed gate forbidden"
        valid_metrics(single["candidate_fp32"]); valid_metrics(single["control_fp32"])
    e=record["endurance"]
    assert e["calls"]==ENDURANCE and e["bytes_equal"] is True and e["dispatch_equal"] is True, "endurance failed"
    valid_metrics([e["initial_fp32"],e["final_fp32"],e["final_fp32"]])


def aggregate(directory, env_output, pair_selection, run_id, deadline, workers_ok, image_sha256):
    result={"schema":SCHEMA,"adopted":False,"selected":None,"gpus":[],"run_id":run_id,"candidate_rejections":{},"rule":{"shapes":SHAPES,"minimum_saving_us":5,"wall_seconds":180,"endurance_calls":ENDURANCE,"priority":["P","S"]}}
    try:
        def read(path):
            return json.loads(path.read_text(),parse_constant=reject_constant,parse_float=finite_float)
        pair=read(pair_selection)
        assert pair["schema"]=="firmus-prefill-fa3-pair/v1" and pair["adopted"] is True, "round-12 gate did not adopt"
        identities={(g["worker"],g["rank"]):g for g in pair["gpus"]}
        assert len(identities)==12 and len({g["uuid"] for g in identities.values()})==12, "incomplete pair GPUs"
        assert workers_ok and time.time()<=deadline, "workers failed, timed out or late"
        assert isinstance(image_sha256,str) and len(image_sha256)==64 and all(c in "0123456789abcdef" for c in image_sha256), "image hash required"
        uuids=set(); binaries=set()
        for worker in range(3):
            for rank in range(4):
                r=read(directory/f"prefill-{worker}"/f"rank-{rank}.json")
                result["gpus"].append(r)
                prior=identities[(f"prefill-{worker}",rank)]
                assert r["schema"]==SCHEMA and r["run_id"]==run_id, "stale result"
                assert r["worker"]==f"prefill-{worker}" and r["rank"]==rank, "wrong worker"
                assert r["uuid"]==prior["uuid"] and r["uuid"] not in uuids and "H200" in r["device"], "wrong GPU"
                assert prior["run_id"].split("-",1)[0]==run_id.split("-",1)[0], "wrong allocation"
                assert r["healthy"] is True and 0<r["elapsed_seconds"]<SECONDS and r["finished_at"]<=deadline, "fatal or late GPU"
                assert r["image_sha256"]==image_sha256 and len(r["binary_sha256"])==64, "hash mismatch"
                uuids.add(r["uuid"]); binaries.add(r["binary_sha256"])
        assert len(binaries)==1, "different binaries"
        for candidate in ("P","S"):
            try:
                for record in result["gpus"]:
                    check_candidate(record["candidates"][candidate],candidate)
                result["selected"]=candidate
                break
            except (AssertionError,KeyError,TypeError,ValueError) as error:
                result["candidate_rejections"][candidate]=str(error)
        assert result["selected"] is not None, "no candidate passed all twelve GPUs"
        lines=env_output.read_text().splitlines(keepends=True)
        assert [l.rstrip("\n") for l in lines if l.startswith(KEY+"=")]==[KEY+"="+CONTROL], "round-12 env required"
        assert time.time()<=deadline, "aggregate deadline expired"
        temporary=env_output.with_name(env_output.name+".prepack.tmp")
        try:
            temporary.write_text("".join(KEY+"="+DISPATCH[result["selected"]]+"\n" if l.startswith(KEY+"=") else l for l in lines))
            assert time.time()<=deadline, "aggregate deadline expired"
            temporary.replace(env_output)
        finally:
            temporary.unlink(missing_ok=True)
        result.update(adopted=True,reason="all twelve H200s passed; P preferred to S")
    except Exception as error:  # noqa: BLE001 - refusal preserves round-12 selection
        result.update(reason=f"{type(error).__name__}: {error}",selected=None,gpus=[])
    return result


def native_producer(common, image):
    from sglang.kernels.ops.attention.rope import apply_qknorm_rope_fp8_prefill_native
    apply_qknorm_rope_fp8_prefill_native(*common["rope_args"],image,common["cu"],96,max(common["lengths"]))


def steps(torch, checks, common):
    import deep_gemm
    from sglang.kernels.ops.attention.rope import apply_qknorm_rope_fp8_prefill_fp32
    lengths=common["lengths"]; rows=common["rows"]
    assert len(lengths)==2 and all(3900<=n<=4096 for n in lengths)
    image=torch.empty((2,4,(max(lengths)+95)//96,96*128),device="cuda",dtype=torch.uint8)
    empty=torch.empty(0,device="cuda",dtype=torch.uint8)
    data=checks["dense_inputs"](rows,4096,8192,49183)
    outputs={}; attention={}; storage=[]
    for mode in ("C","S","V","VS"):
        out=torch.empty((rows,64,128),device="cuda",dtype=common["q"].dtype)
        scale=torch.empty((64,(rows+3)//4*4),device="cuda",dtype=torch.float32).T[:rows]
        lse=torch.empty((64,rows),device="cuda",dtype=torch.float32)
        metadata=torch.empty(17,device="cuda",dtype=torch.int32)
        spec=next(s for s in checks["prefill_specs"]() if s["name"]==ROWS[mode])
        index=checks["prefill_specs"]().index(spec)
        native=spec.get("native_v_n")
        def fa3(out=out,scale=scale,lse=lse,metadata=metadata,index=index,native=native):
            torch.ops.apac_fa3_verify.prefill_variant_fwd_out(common["q"],common["k"],common["v"],image if native else empty,common["cu"],out,scale,lse,metadata,max(lengths),128**-.5,index)
            return out,scale
        attention[mode]=fa3; outputs[mode]=(out,scale); storage.extend((out,scale,lse,metadata))
    attention["P"]=attention["V"]; outputs["P"]=outputs["V"]
    def old():
        apply_qknorm_rope_fp8_prefill_fp32(*common["rope_args"])
    def pack():
        torch.ops.apac_fa3_verify.prefill_prepack(common["v"],image,common["cu"])
    def packed():
        old(); pack()
    writers={"C":old,"S":old,"V":lambda:native_producer(common,image),"VS":lambda:native_producer(common,image),"P":packed}
    o_outputs={mode:torch.empty_like(data[2]) for mode in attention}
    composite={}
    for mode,fa3 in attention.items():
        def call(mode=mode,fa3=fa3):
            writers[mode]()
            out,scale=fa3()
            deep_gemm.fp8_gemm_nt((out.view(rows,8192),scale),data[1],o_outputs[mode],compiled_dims="nk")
            return out,scale
        composite[mode]=call
    def only_o():
        out,scale=outputs["C"]
        deep_gemm.fp8_gemm_nt((out.view(rows,8192),scale),data[1],o_outputs["C"],compiled_dims="nk")
    return {"image":image,"attention":attention,"output":outputs,"writers":writers,"pack":pack,"composite":composite,"weight":data[1],"o_outputs":o_outputs,"only_o":only_o,"storage":(storage,data,empty)}


def host_image(torch, common, image):
    # Independent forward map, generated from the old producer CuTe byte map.
    x=torch.arange(96*128,dtype=torch.int64,device="cpu")
    mapping=torch.zeros_like(x)
    for bit,mask in enumerate((256,32,64,144,512,1024,2048,1,4,8,2,16,4096,8192)):
        mapping^=((x>>bit)&1)*mask
    logical=torch.zeros_like(image).view(2,4,-1,128)
    v=common["v"].view(torch.uint8)
    offset=0
    for request,length in enumerate(common["lengths"]):
        logical[request,:,:length]=v[offset:offset+length].permute(1,0,2)
        offset+=length
    expected=torch.empty_like(image)
    expected[...,mapping.to(device="cuda")]=logical.reshape_as(image)
    return expected


def validate(torch, checks, common, state, modes):
    metrics={mode:[] for mode in ("C",*modes)}; errors={}; graphs=state.setdefault("validation_graphs",{})
    for replay in range(3):
        if replay:
            common["packed"].copy_(torch.randn_like(common["packed"])*(replay+.5))
        state["writers"]["C"]()
        old=tuple(x.clone() for x in (common["q"],common["k"],common["v"],common["kc"],common["vc"]))
        # Pack itself is never the oracle. Compare old contiguous V -> host map,
        # independently executed LDSM/PRMT/STSM dump, and the native writer image.
        if any(mode in modes for mode in ("P","V","VS")):
            expected=host_image(torch,common,state["image"])
            dump=torch.empty_like(state["image"])
            torch.ops.apac_fa3_verify.prefill_dump_native(common["v"],dump,common["cu"],96)
            try:
                assert torch.equal(dump,expected), "old GPU transpose vs host byte map"
            except AssertionError as error:
                errors.update({mode:str(error) for mode in modes if mode in ("P","V","VS")})
            state["pack"]()
            try:
                assert torch.equal(state["image"],expected), "pack vs independent maps"
                assert bytes_equal(torch,old,(common["q"],common["k"],common["v"],common["kc"],common["vc"])), "pack touched Q/K/V or paged KV"
            except AssertionError as error:
                errors["P"]=str(error)
            native_producer(common,state["image"])
            try:
                assert bytes_equal(torch,(old[i] for i in (0,1,3,4)),(common["q"],common["k"],common["kc"],common["vc"])), "native Q/K or paged KV byte mismatch"
                assert torch.equal(state["image"],expected), "native writer vs independent maps"
            except AssertionError as error:
                errors.update({mode:str(error) for mode in modes if mode in ("V","VS")})
        state["writers"]["C"]()
        indices,reference=checks["prefill_fp32_reference"](common)
        for mode,values in metrics.items():
            if mode in errors:
                continue
            try:
                # Changed-input replay exercises the captured serving composition.
                if mode not in graphs:
                    graphs[mode]=checks["prefill_selection_capture"](state["composite"][mode])
                graphs[mode].replay(); torch.cuda.synchronize()
                assert bytes_equal(torch,old,(common["q"],common["k"],common["v"],common["kc"],common["vc"])), mode+" Q/K/V or paged KV byte mismatch"
                if mode in ("P","V","VS"):
                    assert torch.equal(state["image"],expected), mode+" changed-input V image mismatch"
                out,scale=state["output"][mode]
                values.append(checks["prefill_assert_errors"](out.float()[indices]*scale[indices,:,None],reference,mode+"_fp32"))
            except AssertionError as error:
                errors[mode]=str(error)
    result={}
    for mode in modes:
        record={"byte_image_first":True,"producer_bytes_equal":True,"replays":2,"candidate_fp32":metrics[mode],"control_fp32":metrics["C"],"row":ROWS[mode],"variant":VARIANTS[mode]}
        if mode in errors or "C" in errors:
            record["error"]=errors.get(mode,errors.get("C"))
        else:
            record["error_delta"]=[{key:a[key]-b[key] for key in ("relative_l2","max_scaled_error")} for a,b in zip(metrics[mode],metrics["C"],strict=True)]
        result[mode]=record
    return result


def bytes_equal(torch, left, right):
    return all(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)) for a,b in zip(left,right,strict=True))


def single_check(torch, checks, common, candidate="P"):
    """No timing: the new env must preserve round-12 single producer/output bytes."""
    from sglang.kernels.ops.attention.rope import apply_qknorm_rope_fp8_prefill_fp32
    from sglang.srt.layers.attention.prefill_fp8 import (
        attention,
        pair_candidate,
        prefill_candidate,
    )
    os.environ[KEY]=DISPATCH[candidate]
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



def measure(torch, checks, calls, graphs=None):
    def repeat(call):
        def step():
            for _ in range(REPS):
                call()
        return step
    graphs={} if graphs is None else graphs
    for mode,call in calls.items():
        if mode not in graphs:
            graphs[mode]=checks["prefill_selection_capture"](repeat(call))
    samples={order:{mode:[] for mode in calls} for order in ("forward","reverse")}
    for _ in range(ROUNDS):
        for order,modes in (("forward",list(calls)),("reverse",list(calls)[::-1])):
            for mode in modes:
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record(); graphs[mode].replay(); b.record(); b.synchronize()
                samples[order][mode].append(a.elapsed_time(b)*1000/REPS)
    return samples


def paired(samples, candidate):
    return {order:{"control":values["C"],"candidate":values[candidate]} for order,values in samples.items()}


def diagnostics(torch, checks, common, state):
    state["writers"]["P"]()
    for call in state["attention"].values():
        call()
    sections={
        "writer_only":measure(torch,checks,{"old":state["writers"]["C"],"native":state["writers"]["V"],"old_pack":state["writers"]["P"],"pack":state["pack"]}),
        "fa3_only":measure(torch,checks,{mode:state["attention"][mode] for mode in ("C","S","V","VS")}),
        "composite":measure(torch,checks,{mode:state["composite"][mode] for mode in ("C","S","V","VS","P")},state.setdefault("timing_graphs",{})),
        "o_only":measure(torch,checks,{"O":state["only_o"]}),
    }
    sections["interaction_us"]={section:{order:statistics.median(samples["VS"])-statistics.median(samples["V"])-statistics.median(samples["S"])+statistics.median(samples["C"]) for order,samples in sections[section].items()} for section in ("fa3_only","composite")}
    for section in ("fa3_only","composite"):
        samples=sections[section]
        medians={mode:statistics.median(samples["forward"][mode]+samples["reverse"][mode]) for mode in ("C","S","V","VS")}
        sections["interaction_us"][section]["pooled"]=medians["VS"]-medians["V"]-medians["S"]+medians["C"]
    sections.update(shape=SHAPES[0],variants={mode:{"row":ROWS[mode],"variant":VARIANTS[mode]} for mode in ROWS},clock_mhz=None,clock_reason="No independent clock sampler in the gate; no clock from another run is substituted")
    return sections


def endurance(torch, checks, common, state, candidate):
    import deep_gemm
    from sglang.srt.layers.attention.prefill_fp8 import (
        attention,
        pair_candidate,
        prefill_candidate,
    )
    os.environ[KEY]=DISPATCH[candidate]; prefill_candidate.cache_clear()
    assert pair_candidate(common["lengths"])["name"]==ROWS[candidate]
    state["composite"][candidate]()
    direct=tuple(x.clone() for x in state["output"][candidate])
    def call():
        state["writers"][candidate]()
        out,scales=attention(common["q"],common["k"],common["v"],common["cu"],max(common["lengths"]),128**-.5,common["rows"],fixed=True,v_image=state["image"] if candidate=="P" else None,lengths=common["lengths"])
        deep_gemm.fp8_gemm_nt((out,scales),state["weight"],state["o_outputs"][candidate],compiled_dims="nk")
        return out,scales
    first=tuple(x.clone() for x in call()); first_o=state["o_outputs"][candidate].clone()
    assert bytes_equal(torch,first,(direct[0].view(common["rows"],8192),direct[1])), "serving dispatch mismatch"
    indices,reference=checks["prefill_fp32_reference"](common)
    def numeric(output):
        return checks["prefill_assert_errors"](output[0].view(common["rows"],64,128).float()[indices]*output[1][indices,:,None],reference,"endurance")
    initial=numeric(first)
    for _ in range(ENDURANCE):
        last=call()
    torch.cuda.synchronize()
    final=numeric(last)
    assert bytes_equal(torch,first,last) and torch.equal(first_o.view(torch.uint8),state["o_outputs"][candidate].view(torch.uint8)), "endurance bytes changed"
    return {"calls":ENDURANCE,"bytes_equal":True,"dispatch_equal":True,"initial_fp32":initial,"final_fp32":final}


def require_time(deadline, seconds=0):
    assert time.time()+seconds<deadline, "insufficient time for complete candidate"


def run(output, run_id, deadline, image_sha256):
    begin=time.monotonic()
    rank=int(os.environ["LOCAL_RANK"])
    result={"schema":SCHEMA,"worker":output.name,"rank":rank,"run_id":run_id,"healthy":False,"candidates":{mode:{"passed":False} for mode in ("P","S")},"image_sha256":image_sha256}
    output.mkdir(parents=True,exist_ok=True)
    fatal=None
    try:
        import torch
        from sglang.srt.layers.attention.decode_fa3_verify import load
        assert int(os.environ["WORLD_SIZE"])==4
        torch.set_num_threads(1); torch.cuda.set_device(rank)
        result.update(device=torch.cuda.get_device_name(rank),uuid=str(torch.cuda.get_device_properties(rank).uuid))
        assert "H200" in result["device"] and torch.cuda.get_device_capability()==(9,0)
        load()
        checks=runpy.run_path(str(Path(__file__).with_name("check-firmus-decode-kernels.py")))
        manifest=json.loads(Path("/opt/apac/fa3-verify/manifest.json").read_text())
        assert [{k:v for k,v in row.items() if k!="shared_bytes"} for row in manifest["prefill_candidates"]]==checks["prefill_specs"](), "table mismatch"
        with Path("/opt/apac/fa3-verify/libapac_fa3_verify.so").open("rb") as binary:
            result["binary_sha256"]=hashlib.file_digest(binary,"sha256").hexdigest()
        assert result["binary_sha256"]==manifest["binary_sha256"], "binary manifest mismatch"
        result["source_sha256"]=manifest["source_sha256"]
        for mode,row in ROWS.items():
            assert next(s["variant"] for s in checks["prefill_specs"]() if s["name"]==row)==VARIANTS[mode], "variant mismatch"
        cache={}; single_inputs={}
        def inputs(lengths):
            if lengths not in cache:
                common=checks["prefill_selection_inputs"](lengths,padding=0); common["fp32_producer"]=True
                cache[lengths]=(common,steps(torch,checks,common))
            return cache[lengths]
        common,state=inputs(SHAPES[0])
        initial=validate(torch,checks,common,state,("S","V","VS","P"))
        result["diagnostic_numerics"]=initial
        result["diagnostics"]=diagnostics(torch,checks,common,state)
        first_samples=result["diagnostics"]["composite"]
        for candidate in ("P","S"):
            record={"passed":False,"shapes":{},"singles":{}}
            result["candidates"][candidate]=record
            try:
                first={**initial[candidate],"samples_us":paired(first_samples,candidate)}
                record["shapes"]["3994+3994"]=first
                assert "error" not in first, first.get("error")
                assert all(v>=5 for v in timing_summary(first["samples_us"])["savings_us"].values()), "below 5 us margin on first shape"
                require_time(deadline,CANDIDATE_SECONDS)
                for lengths in SHAPES[1:]:
                    require_time(deadline)
                    common,state=inputs(lengths)
                    cell=validate(torch,checks,common,state,(candidate,))[candidate]
                    record["shapes"]["+".join(map(str,lengths))]=cell
                    assert "error" not in cell, cell.get("error")
                    cell["samples_us"]=paired(measure(torch,checks,{mode:state["composite"][mode] for mode in ("C",candidate)},state.setdefault("timing_graphs",{})),candidate)
                    assert all(v>=5 for v in timing_summary(cell["samples_us"])["savings_us"].values()), "below 5 us margin"
                for lengths in SINGLE_SHAPES:
                    require_time(deadline)
                    if lengths not in single_inputs:
                        single_inputs[lengths]=checks["prefill_selection_inputs"](lengths,padding=0)
                        single_inputs[lengths]["fp32_producer"]=True
                    record["singles"][str(lengths[0])]=single_check(torch,checks,single_inputs[lengths],candidate)
                require_time(deadline)
                common,state=inputs(SHAPES[0])
                record["endurance"]=endurance(torch,checks,common,state,candidate)
                require_time(deadline)
                record["passed"]=True
            except AssertionError as error:
                record["error"]=str(error)
        require_time(deadline)
        result["healthy"]=True
    except Exception as error:  # noqa: BLE001 - discard whole selection on a damaged context
        result["fatal"]=f"{type(error).__name__}: {error}"
        fatal=error
    result.update(elapsed_seconds=time.monotonic()-begin,finished_at=time.time())
    temporary=output/f"rank-{rank}.tmp"
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    temporary.replace(output/f"rank-{rank}.json")
    print(json.dumps(result,allow_nan=False),flush=True)
    if fatal is not None:
        raise fatal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aggregate",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--env-output",type=Path)
    p.add_argument("--pair-selection",type=Path)
    p.add_argument("--run-id",required=True)
    p.add_argument("--deadline",type=float,required=True)
    p.add_argument("--image-sha256",required=True)
    p.add_argument("--workers-ok",type=int,choices=(0,1),default=0)
    a=p.parse_args()
    if a.aggregate:
        result=aggregate(a.aggregate,a.env_output,a.pair_selection,a.run_id,a.deadline,bool(a.workers_ok),a.image_sha256)
        a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k!="gpus"}),flush=True)
    else:
        run(a.output,a.run_id,a.deadline,a.image_sha256)


if __name__=="__main__":
    main()
