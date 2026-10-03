from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).parents[2]


def flags_function():
    patch = (REPOSITORY / "patches/sglang/prefill-cutlass-es.patch").read_text()
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    function = added.split("def _prefill_cutlass_es_flags", 1)[1].split("\n\n", 1)[0]
    scope = {}
    exec(compile(ast.parse("def _prefill_cutlass_es_flags" + function), "patched-flags", "exec"), scope)  # noqa: S102 - Execute only this repository's patched pure function.
    return scope["_prefill_cutlass_es_flags"]


@pytest.mark.parametrize("mode,expected", [("off", (False, False)), ("up", (True, False)), ("down", (False, True)), ("both", (True, True))])
def test_es_modes_are_prefill_hopper_only_and_preserve_mxfp8(mode, expected):
    flags = flags_function()
    assert flags(mode, True, True, False) == expected
    assert flags(mode, False, True, False) == (False, False)
    assert flags(mode, True, False, False) == (False, False)
    assert flags(mode, True, False, True) == (True, True)


def test_unknown_es_mode_fails_instead_of_silent_fallback():
    with pytest.raises(ValueError, match="Invalid SGLANG_PREFILL_CUTLASS_ES"):
        flags_function()("typo", True, True, False)


def test_microbenchmark_cpu_inventory_has_real_large_shapes_and_empty_experts():
    spec = importlib.util.spec_from_file_location("moe_es_micro", REPOSITORY / "scripts/benchmark-prefill-moe-es.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.cases() == [(8192, "balanced"), (8192, "skewed"), (4096, "skewed"), (32, "few_experts")]
