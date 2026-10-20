"""CPU checks against the patched Qwen adapter, including an idle DP rank."""
import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS


def check_rope_wiring(tree, path):
    attention = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3MoeAttention")
    init = next(n for n in attention.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    guard_index = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "use_fp8_q_rope" for t in n.targets))
    guard = compile(ast.Module(body=init.body[guard_index:guard_index + 2], type_ignores=[]), str(path), "exec")
    calls = []

    class Tensor:
        def __init__(self, shape, dtype="bf16", device="cuda"):
            self.shape, self.dtype, self.device = shape, dtype, device

        def view(self, *shape):
            return self

        def split(self, sizes, dim):
            return tuple(Tensor((self.shape[0], size), self.dtype) for size in sizes)

    class Rotary:
        use_fallback_kernel, is_neox_style, rotary_dim = False, True, 128
        cos_sin_cache = object()

        def __call__(self, positions, q, k, **kwargs):
            calls.append(("old_rope", kwargs["fused_set_kv_buffer_arg"]))
            return q, k

    args = NS(attention_backend="fa3", kv_cache_dtype="fp8_e4m3", cuda_graph_config=NS(prefill=NS(tc_compiler="piecewise")))
    deterministic = NS(enabled=False)
    attention_layer = NS(use_fused_qk_norm_rope=False, head_dim=128, num_heads=64, num_kv_heads=4, q_size=8192, kv_size=512, rotary_emb=Rotary(), q_norm=NS(weight=Tensor((128,)), variance_epsilon=1e-6), k_norm=NS(weight=Tensor((128,)), variance_epsilon=1e-6), alt_stream=None)
    ns = {"self": attention_layer, "_is_cuda": True, "get_moe_a2a_backend": lambda: NS(is_megamoe=lambda: True), "get_server_args": lambda: args, "MRotaryEmbedding": type("MRotaryEmbedding", (), {}), "torch": NS(bfloat16="bf16", float8_e4m3fn="fp8", empty=Tensor), "envs": NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda: deterministic.enabled)), "apply_qk_norm": lambda **kw: (calls.append(("norm",)), (kw["q"], kw["k"]))[1], "apply_qknorm_rope_fp8_q": lambda *xs: calls.append(("qknorm_fp8_rope", xs)), "apply_rope_fp8_q": lambda *xs: calls.append(("fp8_rope", xs)), "enable_fused_set_kv_buffer": lambda fb: True, "create_fused_set_kv_buffer_arg": lambda **kw: "fused_kv", "ForwardMode": NS(EXTEND="EXTEND")}
    nodes = [n for n in attention.body if isinstance(n, ast.FunctionDef) and n.name in ("apply_qk_norm_rope", "forward_core")]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)  # noqa: S102 - real pinned methods, CPU-only collaborators
    attention_layer.attn = lambda q, k, v, fb, **kw: (calls.append(("attention", kw["save_kv_cache"])), q)[1]
    attention_layer.o_proj = lambda output: (output, None)
    # A mode without the fused FP8 KV write (check-firmus-latency-source.py covers its selection).
    other_mode = NS(forward_mode=NS(is_target_verify=lambda: False))
    for backend, kv_dtype, input_dtype, fused in [("fa3", "fp8_e4m3", "bf16", True), ("fa4", "fp8_e4m3", "bf16", False), ("fa3", "auto", "bf16", False), ("fa3", "fp8_e4m3", "fp16", False)]:
        calls.clear()
        args.attention_backend, args.kv_cache_dtype = backend, kv_dtype
        attention_layer.compatible_with_fused_kv_buffer = True
        exec(guard, ns)  # noqa: S102
        q, k, v = ns["apply_qk_norm_rope"](attention_layer, Tensor((7, 9216), input_dtype), "positions", other_mode)
        assert [c[0] for c in calls] == (["qknorm_fp8_rope"] if fused else ["norm", "old_rope"])
        assert q.dtype == ("fp8" if fused else input_dtype) and k.dtype == v.dtype == input_dtype
        ns["forward_core"](attention_layer, (None, None, (q, k, v, None)))
        assert calls[-1] == ("attention", attention_layer.use_fp8_q_rope), "FP8 RoPE must still save KV exactly once"
    args.attention_backend, args.kv_cache_dtype = "fa3", "fp8_e4m3"
    exec(guard, ns)  # noqa: S102
    # Preserve the existing unfused-normalization fallbacks, then cast in RoPE.
    for condition in ("deterministic", "inductor", "epsilon", "weight_dtype"):
        calls.clear()
        deterministic.enabled = condition == "deterministic"
        args.cuda_graph_config.prefill.tc_compiler = "inductor" if condition == "inductor" else "piecewise"
        attention_layer.k_norm.variance_epsilon = 1e-5 if condition == "epsilon" else 1e-6
        attention_layer.k_norm.weight.dtype = "fp32" if condition == "weight_dtype" else "bf16"
        ns["apply_qk_norm_rope"](attention_layer, Tensor((7, 9216)), "positions", other_mode)
        assert [c[0] for c in calls] == ["norm", "fp8_rope"], condition
    print("PASS: fused QK norm feeds FP8 Q to FA3 without a second norm; guarded fallbacks; KV stored once")


def check(source, replicas=False):
    path = source / "python/sglang/srt/models/qwen3_moe.py"
    tree = ast.parse(path.read_text())
    check_rope_wiring(tree, path)
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "forward_megamoe")
    method.body = [n for n in method.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    # Evaluate the real adapter's constructor guard without loading GPU modules.
    supported = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "supported" for t in n.targets))
    guard = compile(ast.Expression(body=supported), str(path), "eval")
    args = NS(enable_dp_attention=True, dp_size=4, disaggregation_mode="prefill", enable_two_batch_overlap=False, enable_eplb=False, ep_num_redundant_experts=0, init_expert_location="/runtime/source/prefill-expert-location.json", enable_dp_lm_head=True, moe_dense_tp_size=1, speculative_moe_a2a_backend=None, speculative_algorithm="EAGLE3", speculative_eagle_topk=1, speculative_num_draft_tokens=4)
    guard_ns = {"torch": NS(cuda=NS(get_device_capability=lambda: (9, 0))), "config": NS(hidden_size=4096, moe_intermediate_size=1536, num_experts=128, num_experts_per_tok=8), "self": NS(tp_size=1, ep_size=4), "args": args, "quant_config": NS(get_name=lambda: "fp8", weight_block_size=[128, 128])}
    assert eval(guard, guard_ns)
    for mode in ("prefill", "decode"):
        args.disaggregation_mode = mode
        for redundant in (0, 12, 24, 36, 48):
            args.ep_num_redundant_experts = redundant
            assert bool(eval(guard, guard_ns)) == (redundant == 0 or replicas and mode == "prefill" and redundant == 12), (mode, redundant)
    if replicas:
        args.disaggregation_mode, args.ep_num_redundant_experts, args.init_expert_location = "prefill", 12, "trivial"
        assert not eval(guard, guard_ns)
        args.init_expert_location = "/runtime/source/prefill-expert-location.json"
    args.ep_num_redundant_experts = 0
    args.disaggregation_mode = "decode"
    assert eval(guard, guard_ns)
    for key, bad in [("enable_dp_lm_head", False), ("moe_dense_tp_size", None), ("speculative_moe_a2a_backend", "none"), ("speculative_algorithm", None), ("speculative_eagle_topk", 2), ("speculative_num_draft_tokens", 8), ("dp_size", 2), ("enable_two_batch_overlap", True)]:
        original = getattr(args, key)
        setattr(args, key, bad)
        assert not eval(guard, guard_ns), key
        setattr(args, key, original)
    # The dense EAGLE draft inherits MegaMoE's DP bookkeeping, even though it
    # has no MoE layer. Overriding only its A2A backend to "none" makes the
    # specialized draft graph runners incorrectly require global MLP buffers.
    common = source / "python/sglang/srt/utils/common.py"
    names = {"require_mlp_tp_gather", "require_attn_tp_gather", "require_gathered_buffer"}
    functions = [n for n in ast.parse(common.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    for function in functions:
        function.body = [n for n in function.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    backend = NS(is_none=lambda: False, is_flashinfer=lambda: False)
    dp_ns = {"ServerArgs": NS, "get_moe_a2a_backend": lambda: backend}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(common), "exec"), dp_ns)  # noqa: S102 - pinned graph predicates
    args.tp_size, args.elastic_ep_backend, args.disable_attn_tp_gather = 4, None, False
    assert not dp_ns["require_gathered_buffer"](args)
    backend.is_none = lambda: True
    assert dp_ns["require_gathered_buffer"](args)
    calls = []
    tensor = NS(to=lambda dtype: tensor)
    topk = NS(topk_ids=tensor, topk_weights=tensor)
    class Router:
        def __call__(self, hidden, logits, **kwargs):
            calls.append(("router", kwargs))
            return topk

        def empty_topk_output(self, device):
            calls.append(("empty", device))
            return topk

    namespace = {"torch": NS(int32="int32", float32="float32"), "get_moe_ep_group": lambda: NS(device_group="EP4"), "envs": NS(SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=NS(get=lambda: 4096)), "is_sm90_fp8_mega_moe_available": lambda _: True, "_get_mega_moe_symm_buffer": lambda *args, **kwargs: (args, kwargs), "run_sm90_mega_routed": lambda *args: calls.append(("native", args)), "ExpertLocationDispatchInfo": NS(init_new=lambda layer_id: ("dispatch", layer_id))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)  # noqa: S102 - named method from pinned build input
    block = NS(layer_id=7, experts=NS(num_experts=140 if replicas else 128), gate=lambda h: (tensor, None), topk=Router(), config=NS(num_experts=128, num_experts_per_tok=8, hidden_size=4096, moe_intermediate_size=1536))
    for count in [0, 4, 7, 120, 128, 4096]:
        calls.clear()
        namespace["forward_megamoe"](block, NS(shape=(count, 4096), device="cuda"), NS(num_token_non_padded="non-padded"))
        assert [c[0] for c in calls] == (["empty", "native"] if count == 0 else ["router", "native"])
        # SGLang's static placement remaps logical ids to the physical ids FusedMoE loaded; trivial placement passes None.
        assert count == 0 or calls[0][1] == {"num_token_non_padded": "non-padded", "expert_location_dispatch_info": ("dispatch", 7)}
        args = calls[-1][1]
        assert args[2] is tensor and args[3] is tensor and args[-1] == count
        assert args[-2][0] == ("EP4",) and args[-2][1]["num_max_tokens_per_rank"] == 4096
        assert args[-2][1]["num_experts"] == (140 if replicas else 128)
    try:
        namespace["forward_megamoe"](block, NS(shape=(4097, 4096)), None)
    except ValueError:
        pass
    else:
        raise AssertionError("token-cap overflow accepted")
    # Native FC1 reads gate/up scales separately despite the 8-row interleave.
    for packed_row in range(3072):
        up = packed_row % 16 >= 8
        original_row = packed_row // 16 * 8 + packed_row % 8 + (1536 if up else 0)
        native_scale_row = packed_row // 128 // 2 + (12 if up else 0)
        assert native_scale_row == original_row // 128
    print("PASS: Prefill and DP-local Decode guards; empty/nonempty/cap DP paths; placement-mapped expert IDs; original block128 scale mapping")


def check_replica_geometry(source):
    path = source / "python/sglang/srt/layers/moe/mega_moe_sm90.py"
    wanted = {"idle_output_rows", "_use_idle_prefill_geometry", "run_sm90_mega_routed"}
    functions = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions], type_ignores=[])
    calls = []

    class Tensor:
        def __init__(self, shape, **kwargs):
            self.shape = shape

        def __getitem__(self, item):
            return Tensor((item.stop, self.shape[1]))

    server = NS(disaggregation_mode="prefill", ep_num_redundant_experts=12)
    fakes = {"deep_gemm": NS(mega_moe_pre_dispatch_sm90=lambda *a, **k: calls.append(("pre", k["num_tokens"], k["group_size"])), fp8_mega_moe=lambda y, *a, **k: calls.append(("moe", y.shape))), "sglang.srt.runtime_context": NS(get_server_args=lambda: server), "sglang.srt.distributed": NS(get_moe_ep_group=lambda: NS(world_size=4))}
    saved = {name: sys.modules.get(name) for name in fakes}
    sys.modules.update(fakes)
    try:
        for mode, redundant, enabled in (("prefill", 12, True), ("prefill", 12, False), ("prefill", 0, True), ("prefill", 0, False), ("decode", 0, True)):
            server.disaggregation_mode, server.ep_num_redundant_experts = mode, redundant
            namespace = {"torch": NS(bfloat16="bf16", empty=Tensor), "_IDLE_PREFILL_GEOMETRY": enabled}
            exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)  # noqa: S102 - real overlay functions with CPU-only collaborators
            moe = NS(experts=NS(should_fuse_routed_scaling_factor_in_topk=True, mega_l1_weights=None, mega_l2_weights=None), config=NS(hidden_size=4096, num_experts=128, num_experts_per_tok=8))
            buf = NS(num_experts=128 + redundant, x=None, x_sf=None, topk_idx=NS(shape=(4224, 8)), topk_weights=None)
            for count in (0, 1, 7, 65, 128, 129, 256, 279, 280, 281, 4096):
                calls.clear()
                result = namespace["run_sm90_mega_routed"](moe, NS(device="cuda"), None, None, buf, count)
                expected_rows = max(count, 280) if redundant else (256 if mode == "prefill" and enabled and count == 0 else max(count, 1))
                assert calls == [("pre", count, 128), ("moe", (expected_rows, 4096))], (mode, redundant, enabled, count, calls)
                assert result.shape == (count, 4096)
            assert namespace["idle_output_rows"](140, 8, 4) == 280
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    print("PASS: physical-140 Prefill output rows >=280; real token count retained; Decode and physical-128 behavior preserved")


def check_replica_placement(source, placement):
    path = source / "python/sglang/srt/eplb/expert_location.py"
    names = {"_find_nearest_expert", "_compute_gpu_id_of_physical_expert", "_compute_node_id_of_physical_expert"}
    functions = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)  # noqa: S102 - real pinned static dispatch selector
    mapping = json.loads(placement.read_text())["physical_to_logical_map"]
    assert len(mapping) == 94
    for row in mapping:
        counts = Counter(row)
        assert len(row) == 140 and set(counts) == set(range(128))
        hot = {e for e, count in counts.items() if count == 4}
        assert len(hot) == 4 and all(counts[e] == (4 if e in hot else 1) for e in counts)
        for rank in range(4):
            local = row[35 * rank:35 * (rank + 1)]
            assert len(set(local)) == 35 and hot <= set(local)
            for expert in range(128):
                candidates = [p for p, logical in enumerate(row) if logical == expert]
                selected = namespace["_find_nearest_expert"](candidates, 35, rank, 4, 140)
                assert selected in candidates
                assert selected // 35 == rank if expert in local else len(candidates) == 1
    print("PASS: 94x140 placement, same-GPU hot selection, one cold owner, all logical experts present")
    # Execute the pinned FP8 loader's real fan-out and rank filtering. Symbolic
    # weight/scale objects prove each local slot receives its logical source,
    # without requiring torch, CUDA or the full checkpoint on the CPU.
    path = source / "python/sglang/srt/layers/moe/fused_moe_triton/layer.py"
    names = {"weight_loader", "_weight_loader_physical", "_map_global_expert_id_to_local_expert_id"}
    functions = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef) and n.name in names]
    metadata = NS()
    namespace = {"get_global_expert_location_metadata": lambda: metadata, "KTEPWrapperMethod": type("KTEPWrapperMethod", (), {})}
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)  # noqa: S102 - pinned FP8 checkpoint loader with symbolic payloads
    for row in (mapping[0], mapping[-1]):
        metadata.logical_to_all_physical = lambda layer, expert, require_global, row=row: [p for p, e in enumerate(row) if e == expert]
        for rank in range(4):
            seen = []
            block = NS(quant_config=NS(get_name=lambda: "fp8"), quant_method=None, _has_fused_shared=False, num_fused_shared_experts=0, num_local_experts=35, _num_local_routed=35, _num_global_routed=140, _expert_storage_rank=rank, layer_id=0, _weight_loader_impl=lambda seen=seen, **kwargs: seen.append(kwargs))
            for name in names:
                setattr(block, name, namespace[name].__get__(block))
            for shard in ("w1", "w3", "w2"):
                for kind in ("weight", "weight_scale_inv"):
                    sources = [object() for _ in range(128)]
                    seen.clear()
                    for expert in range(128):
                        block.weight_loader(NS(), sources[expert], kind, shard, expert)
                    assert len(seen) == 35
                    for call in seen:
                        assert call["loaded_weight"] is sources[row[rank * 35 + call["expert_id"]]]
                        assert call["shard_id"] == shard and call["weight_name"] == kind
    print("PASS: existing FP8 checkpoint loader fans logical w1/w3/w2 weights and scales into all 35 local slots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--replicas", action="store_true", help="Check the serving replica overlay, not the base image")
    parser.add_argument("--placement", type=Path)
    args = parser.parse_args()
    check(args.source, args.replicas)
    if args.replicas:
        assert args.placement is not None, "--replicas requires --placement"
        check_replica_geometry(args.source)
        check_replica_placement(args.source, args.placement)
