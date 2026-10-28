"""CPU coverage of selection, graph preparation ownership and opt-in loading."""

import ast
import json
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = json.loads((ROOT / "kernels/fa3-verify/candidates.json").read_text())


def selector():
    return runpy.run_path(str(ROOT / "scripts/select-firmus-decode-fa3.py"))


def records(ns):
    return [{"rank": rank, "batches": {str(batch): {name: {"numeric_passed": True, "blocks": [{"stock": [50., 50.1, 49.9], "candidate": [45., 45.1, 44.9]} for _ in range(2)]} for name in ns["NAMES"]} for batch in (36, 38, 40)}} for rank in range(4)]


def test_only_cap_batch_selects_and_control_is_never_selected():
    ns = selector()
    ranks = records(ns)
    for rank in ranks:
        rank["batches"]["40"]["control96"]["blocks"] = [{"stock": [50.], "candidate": [1.]}] * 2
        rank["batches"]["36"]["p128s2"] = {"numeric_passed": False, "error": "diagnostic only"}
    result = ns["choose_candidate"](ranks, 40)
    assert result["winner"] == "p128s2"
    assert result["env"] == {"SGLANG_DECODE_FA3_VERIFY": "p128s2"}


@pytest.mark.parametrize("failure", ["numeric", "exception", "missing-block", "too-small", "second-block-slower"])
def test_failed_or_inconsistent_candidate_is_excluded_without_losing_others(failure):
    ns = selector()
    ranks = records(ns)
    result = ranks[2]["batches"]["40"]["p128s2"]
    if failure == "numeric":
        result["numeric_passed"] = False
    elif failure == "exception":
        result["error"] = "RuntimeError: compile failed"
    elif failure == "missing-block":
        result["blocks"].pop()
    elif failure == "too-small":
        result["blocks"][0]["candidate"] = [46.9]
    else:
        result["blocks"][1]["candidate"] = [51.]
    assert ns["choose_candidate"](ranks, 40)["winner"] == "p128split2-fp8"


def test_no_eligible_candidate_clears_the_environment():
    ns = selector()
    ranks = records(ns)
    for rank in ranks:
        for entry in rank["batches"]["40"].values():
            entry["numeric_passed"] = False
    result = ns["choose_candidate"](ranks, 40)
    assert result["winner"] is None and result["env"] == {"SGLANG_DECODE_FA3_VERIFY": ""}


@pytest.mark.parametrize("error_type", [RuntimeError, AssertionError, ValueError])
@pytest.mark.parametrize("phase", ["numeric", "capture", "replay"])
def test_candidate_exception_is_recorded_and_next_candidate_prepares(error_type, phase):
    ns = selector()
    calls = []

    def validate(batch, name):
        calls.append((batch, name))
        if name == "bad" and phase == "numeric":
            raise error_type("candidate failure")
        return {"numeric_passed": True}

    def capture(name):
        if name == "bad" and phase == "capture":
            raise error_type("candidate failure")

        def replay():
            if name == "bad" and phase == "replay":
                raise error_type("candidate failure")
        return types.SimpleNamespace(replay=replay)

    checks = {"verify_validate_candidate": validate, "verify_step": lambda commons, name: name, "verify_capture": capture}
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None))
    bad, _, _ = ns["prepare_candidate"](torch, checks, 32, "bad", [])
    good, step, graph = ns["prepare_candidate"](torch, checks, 32, "good", [])
    assert not bad["numeric_passed"] and error_type.__name__ in bad["error"]
    assert good["numeric_passed"] and step == "good" and graph is not None
    assert calls == [(32, "bad"), (32, "good")]


@pytest.fixture
def helper(tmp_path, monkeypatch):
    subprocess.run(["git", "apply", "--include=python/sglang/srt/layers/attention/decode_fa3_verify.py", str(ROOT / "patches/sglang/decode-fa3-verify.patch")], cwd=tmp_path, check=True)
    loads, calls, shapes = [], [], []

    def empty(shape, **kwargs):
        shapes.append((shape, kwargs))
        return object()

    ops = types.SimpleNamespace(prepare=lambda *args: calls.append(("prepare", args)), fwd_out=lambda *args: calls.append(("fwd", args)), fwd_fp8_out=lambda *args: calls.append(("fp8", args)))
    torch = types.SimpleNamespace(ops=types.SimpleNamespace(load_library=loads.append, apac_fa3_verify=ops), library=types.SimpleNamespace(register_fake=lambda _: lambda f: f), empty=empty, float32="fp32", int32="int32", bfloat16="bf16")
    monkeypatch.setitem(sys.modules, "torch", torch)
    module = tmp_path / "python/sglang/srt/layers/attention/decode_fa3_verify.py"
    module.write_text(module.read_text().replace("/opt/apac/fa3-verify", str(ROOT / "kernels/fa3-verify")))
    ns = runpy.run_path(str(module))
    return ns, loads, calls, shapes


@pytest.mark.parametrize("batch", range(32, 49))
@pytest.mark.parametrize("name,splits", [(c["name"], c["splits"]) for c in CANDIDATES if c["output"] == "bf16"])
def test_workspace_is_owned_by_forward_and_split_buffers_stay_fp32(helper, batch, name, splits):
    ns, loads, calls, shapes = helper
    assert not loads, "import alone must not load the companion"
    lengths = types.SimpleNamespace(numel=lambda: batch, device="cuda:2")
    state = ns["allocate"](lengths, name)
    other = ns["allocate"](lengths, name)
    assert len(loads) == 1 and state["metadata"] is not other["metadata"]
    assert shapes[0][0] == 4 * ((batch + 3) // 4 * 4) + 1 + (batch * 4 if state["spec"].get("merge") else 0)
    assert shapes[2][0] == ((splits, 64, 4 * batch, 128) if splits > 1 else (0,))
    assert shapes[2][1]["dtype"] == "fp32"
    q, k, v, pages, cu, out = [object() for _ in range(6)]
    ns["prepare"](state, lengths, cu)
    for _ in range(94):
        assert ns["forward"](state, q, k, v, pages, lengths, cu, .1, out) is out
    assert calls[0] == ("prepare", (lengths, cu, state["metadata"], ns["NAMES"].index(name)))
    assert len(calls) == 95 and all(call[1][10] is state["metadata"] for call in calls[1:])


def test_build_and_serving_are_wired_without_rejected_probes():
    config = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    assert config["decode_max_running_requests"] == 176
    assert "patches/sglang/decode-fa3-verify.patch" in config["sglang_overlay_patches"]
    assert config["sglang_overlay_patches"][-7:-3] == [
        "patches/sglang/prefill-fp8-integration.patch",
        "patches/sglang/prefill-qkv-fast-fp8.patch",
        "patches/sglang/decode-qknorm-fast-fp8.patch",
        "patches/sglang/decode-add-rmsnorm-group-quant.patch",
    ]
    assert config["sglang_overlay_patches"][-3:] == ["patches/sglang/prefill-async-admission.patch", "patches/sglang/prefill-start-align.patch", "patches/sglang/decode-router-cluster.patch"]
    assert not any("decode-gate" in p or "l2-prefetch" in p for p in config["sglang_overlay_patches"] + config["deep_gemm_overlay_patches"])
    assert "sm90-decode-l2-prefetch" not in (ROOT / "patches/deep-gemm-host/series").read_text()
    shell = (ROOT / "scripts/select-firmus-decode-kernels.sh").read_text()
    assert 'timeout --kill-after=5 175 "${torchrun}"' in shell and '== fa3' in shell
    assert "--select-decode-gate" not in shell and "--tune-l2-prefetch" not in shell
    job = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    assert '"${ROOT}" fa3' in job and 'decode-selection/fa3.env' in job
    assert "select-firmus-decode-fa3.py" in (ROOT / "scripts/provision-firmus-framework.sh").read_text()
    assert "containers/build-fa3-verify.sh" in (ROOT / "containers/Dockerfile").read_text()
    assert 'max_workers=2' in (ROOT / "kernels/fa3-verify/build.py").read_text()


def test_serving_gate_only_checks_the_selected_fa3(monkeypatch):
    path = ROOT / "scripts/check-firmus-decode-kernels.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check_fa3_verify")
    import os
    calls = []
    ns = {"os": os, "FA3_VERIFY_NAMES": selector()["NAMES"], "json": json, "verify_validate_candidate": lambda b, n: calls.append((b, n)) or {"numeric_passed": True}}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)  # noqa: S102 - tracked GPU-free entrypoint with mocks
    monkeypatch.delenv("SGLANG_DECODE_FA3_VERIFY", raising=False)
    ns["check_fa3_verify"](selected_only=True)
    assert calls == []
    monkeypatch.setenv("SGLANG_DECODE_FA3_VERIFY", "p128split2-fp8")
    ns["check_fa3_verify"](selected_only=True)
    assert calls == [(b, "p128split2-fp8") for b in range(32, 49)]


class VerifyTensor:
    dtype = "fp8"

    def __init__(self, elements):
        self.elements = elements

    def numel(self):
        return self.elements

    def contiguous(self):
        return self

    def view(self, *_):
        return self


@pytest.fixture
def backend_method(monkeypatch):
    text = (ROOT / "patches/sglang/decode-fa3-verify.patch").read_text()
    body = text.split('+    def _run_decode_fa3_verify(', 1)[1].split('     def _mxfp8_sf_kwargs', 1)[0]
    source = 'class Backend:\n    def _run_decode_fa3_verify(' + body.splitlines()[0] + '\n' + '\n'.join(line[1:] for line in body.splitlines()[1:] if line.startswith('+'))
    tree = ast.parse(source)
    method = tree.body[0].body[0]
    ns = {"torch": types.SimpleNamespace(float8_e4m3fn="fp8")}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'companion-backend', 'exec'), ns)  # noqa: S102 - complete tracked production method
    calls = []
    native = types.SimpleNamespace(output_fp8=lambda name: False, allocate=lambda *args: calls.append(("allocate", args)) or object(), prepare=lambda *args: calls.append(("prepare", args)), forward=lambda *args: calls.append(("forward", args)) or args[-1])
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.attention", types.SimpleNamespace(decode_fa3_verify=native))
    metadata = types.SimpleNamespace(cache_seqlens_int32=VerifyTensor(32), max_seq_len_q=4, verify_fa3_state=None)
    backend = types.SimpleNamespace(forward_metadata=metadata, fa_impl_ver=3, use_mla=False, attn_cp_size=1, topk=1, page_size=64, _decode_fa3_verify_name="p128split2-fp8")
    layer = types.SimpleNamespace(tp_q_head_num=64, tp_k_head_num=4, tp_v_head_num=4, head_dim=128, v_head_dim=128, logit_cap=0, is_cross_attention=False, scaling=.1)
    fb = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_target_verify=lambda: True), spec_info=None)
    q, kv = VerifyTensor(32 * 4 * 64 * 128), VerifyTensor(1)
    args = [backend, q, kv, kv, layer, fb, True, (-1, -1), False, False, object(), metadata.cache_seqlens_int32, object(), object(), {}]
    return ns["_run_decode_fa3_verify"], args, calls


def test_backend_shares_workspace_and_preparation_across_94_layers(backend_method):
    method, args, calls = backend_method
    for _ in range(94):
        assert method(*args) is args[-2]
    assert [kind for kind, _ in calls].count("allocate") == 1
    assert [kind for kind, _ in calls].count("prepare") == 1
    assert [kind for kind, _ in calls].count("forward") == 94
    old = args[0].forward_metadata.verify_fa3_state
    # The capture-reset statement is part of the production hook's patch.
    patch = (ROOT / "patches/sglang/decode-fa3-verify.patch").read_text()
    reset = next(line[1:].strip() for line in patch.splitlines() if line.startswith('+            self.forward_metadata.verify_fa3_state = None'))
    exec(compile(reset, 'capture-reset', 'exec'), {"self": args[0]})  # noqa: S102 - exact production reset statement
    method(*args)
    assert args[0].forward_metadata.verify_fa3_state is not old
    assert [kind for kind, _ in calls].count("prepare") == 2


@pytest.mark.parametrize("condition", ["descale", "small-batch", "tp", "dtype", "ragged", "local", "fa4"])
def test_unsupported_geometry_returns_before_companion_import_or_allocation(backend_method, monkeypatch, condition):
    method, args, calls = backend_method
    if condition == "descale":
        args[-1] = {"k_descale": object()}
    elif condition == "small-batch":
        args[0].forward_metadata.cache_seqlens_int32 = VerifyTensor(31)
    elif condition == "tp":
        args[4].tp_q_head_num = 16
    elif condition == "dtype":
        args[1].dtype = "bf16"
    elif condition == "ragged":
        args[5].spec_info = types.SimpleNamespace(ragged_verify_layout=object())
    elif condition == "local":
        args[7] = (128, 0)
    else:
        args[0].fa_impl_ver = 4
    monkeypatch.delitem(sys.modules, "sglang.srt.layers.attention")
    assert method(*args) is None and calls == []


@pytest.mark.parametrize("cap,expected", [(176, (40, 42, 44)), (160, (36, 38, 40)), (192, (44, 46, 48))])
def test_measurement_and_selection_batches_follow_the_config(cap, expected):
    tree = ast.parse((ROOT / "scripts/select-firmus-decode-fa3.py").read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_selection")
    nodes = [n for n in run.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"cap", "selection_batch", "batches"} for t in n.targets)]
    ns = {"config": {"decode_max_running_requests": cap, "decode_dp_size": 4}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "selection-geometry", "exec"), ns)  # noqa: S102 - actual production assignments
    assert ns["batches"] == expected and ns["selection_batch"] == expected[-1]


@pytest.mark.parametrize("serving_enabled", [False, True])
def test_image_compile_control_preserves_configured_overlay_order(tmp_path, serving_enabled):
    config = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    candidate = "sm90-decode-independent-wg.patch"
    config["deep_gemm_overlay_patches"] = [p for p in config["deep_gemm_overlay_patches"] if Path(p).name != candidate]
    if serving_enabled:
        config["deep_gemm_overlay_patches"].insert(-1, "patches/deep-gemm-overlay/" + candidate)
    expected = [ROOT / p for p in config["deep_gemm_overlay_patches"]]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    tree = ast.parse((ROOT / "containers/check-sm90-independent-wg.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    node = next(n for n in main.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "patches" for t in n.targets))
    ns = {"args": types.SimpleNamespace(config=path, patch_dir=ROOT / "patches/deep-gemm-overlay"), "Path": Path, "json": json, "CANDIDATE": candidate}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "image-control-patches", "exec"), ns)  # noqa: S102 - actual build-helper selection
    assert ns["patches"] == expected
    assert all(p.read_bytes() for p in ns["patches"])


@pytest.mark.parametrize("spec", [c for c in CANDIDATES if c["output"] == "fp8"])
def test_fp8_forward_preserves_preallocated_gemm_input_and_never_calls_bf16_op(helper, spec):
    ns, _, calls, shapes = helper
    lengths = types.SimpleNamespace(numel=lambda: 40, device="cuda:0")
    state = ns["allocate"](lengths, spec["name"])
    assert shapes[2][0][0] == spec["splits"]
    assert ns["output_fp8"](spec["name"])

    class Tensor:
        shape = (160,64,128)

        def view(self, *shape):
            return self

    output, scales = Tensor(), object()
    before = len(shapes)
    q = Tensor()
    pair = ns["forward"](state,q,object(),object(),object(),lengths,object(),.1,(output,scales))
    assert pair == (output,scales) and len(shapes) == before
    assert calls[-1][0] == "fp8" and calls[-1][1][-1] is scales
