"""CPU contracts for the explicit Qwen3 MegaMoE integration."""

import ast
import hashlib
import runpy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "containers/firmus-sm90-megamoe"
INSTALL = runpy.run_path(str(HELPER / "install_vllm_adapter.py"))
SOURCE = (HELPER / "vllm_method.py").read_text()


def config():
    moe = NS(
        tp_size=1,
        dp_size=4,
        ep_size=4,
        pcp_size=1,
        sp_size=1,
        num_experts=128,
        num_local_experts=32,
        experts_per_token=8,
        hidden_dim=4096,
        intermediate_size_per_partition=1536,
        activation="silu",
        has_bias=False,
        is_lora_enabled=False,
        skip_final_all_reduce=False,
        in_dtype="bf16",
    )
    parallel = NS(
        tensor_parallel_size=1,
        data_parallel_size=4,
        enable_expert_parallel=True,
        enable_eplb=False,
        enable_dbo=False,
        expert_placement_strategy="linear",
    )
    cfg = NS(
        parallel_config=parallel,
        model_config=NS(hf_config=NS(model_type="qwen3_moe")),
        scheduler_config=NS(max_num_batched_tokens=4096),
    )
    quant = NS(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    layer = NS(
        moe_config=moe,
        renormalize=True,
        use_grouped_topk=False,
        scoring_func="softmax",
        custom_routing_function=None,
        apply_router_weight_on_input=False,
        routed_scaling_factor=1.0,
    )
    return quant, layer, cfg


@pytest.mark.parametrize(
    "field,value",
    [
        (None, None),
        ("tensor_parallel_size", 4),
        ("data_parallel_size", 1),
        ("enable_eplb", True),
        ("enable_dbo", True),
        ("expert_placement_strategy", "round_robin"),
    ],
)
def test_qwen_parallel_guard(field, value):
    tree = ast.parse(SOURCE)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "check_config"
    )
    namespace = {
        "torch": NS(bfloat16="bf16"),
        "MoEActivation": NS(SILU="silu"),
        "CAPACITY": 4096,
    }
    exec(  # noqa: S102 -- Selected host-only guard, no external code.
        compile(
            ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
            "adapter-guard",
            "exec",
        ),
        namespace,
    )
    q, layer, cfg = config()
    if field:
        setattr(cfg.parallel_config, field, value)
        with pytest.raises(ValueError, match="Qwen3"):
            namespace["check_config"](q, layer, cfg)
    else:
        namespace["check_config"](q, layer, cfg)


@pytest.mark.parametrize("name", INSTALL["PATCHES"])
def test_patch_is_hash_guarded_and_changes_one_anchor(name, monkeypatch):
    table = INSTALL["PATCHES"]
    expected, before, after = table[name]
    assert len(expected) == 64
    # Mechanics fixture: actual image source is independently verified at build.
    source = (
        "if True:\n" + before
        if before.startswith("    ")
        else before + '    "fixture",\n]\n'
    ).encode()
    if before.startswith("    "):
        # Anchors inside classes/functions need their original nesting depth.
        source = (
            "class Fixture:\n    def f(self):\n" + before
            if before.startswith("            ")
            else (
                "class Fixture:\n    def f(self):\n"
                if before.startswith("        ")
                else "class Fixture:\n"
            )
            + before
        ).encode()
        if "def supports_internal_mk" in before:
            source += b"        return False\n"
    monkeypatch.setitem(
        table, name, (hashlib.sha256(source).hexdigest(), before, after)
    )
    result = INSTALL["patch"](source, name)
    assert result.count(after.encode()) == 1
    with pytest.raises(ValueError, match="Unsupported pinned"):
        INSTALL["patch"](source + b"\n# changed\n", name)


def test_explicit_backend_and_image_cpu_gate_are_wired():
    recipe = (ROOT / "containers/vllm.Dockerfile").read_text()
    assert recipe.index("install_vllm_adapter.py") < recipe.index(
        "bash /opt/apac/firmus-sm90-megamoe/check_cpu_image.sh"
    )
    check = (HELPER / "check_adapter_cpu.py").read_text()
    assert 'torch.device("meta")' in check
    assert all(f'"{name}"' in check for name in ("_maybe_dispatch", "_maybe_combine", "_maybe_reduce_final_output"))
    assert '"apac_sm90_megamoe"' in check
    assert "vllm_adapter_cpu" in (ROOT / "containers/check-vllm-image.py").read_text()
    assert "@cache\ndef workspace(group):" in SOURCE
    assert "rows > CAPACITY" in SOURCE
    assert SOURCE.index("mega.mega_moe_pre_dispatch_sm90(") < SOURCE.index(
        "mega.fp8_mega_moe("
    )
    assert "all_reduce(" not in SOURCE and "all_gather(" not in SOURCE
