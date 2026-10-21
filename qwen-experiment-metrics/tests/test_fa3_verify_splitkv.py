"""CPU checks of the pinned private metadata ABI used by the bench overlay."""
from __future__ import annotations

import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


class Tensor:
    def __init__(self, shape=(), dtype="fp8", data=None):
        self.shape, self.dtype, self.data = shape, dtype, data

    def numel(self):
        return len(self.data) if self.data is not None else self.shape[0]

    def __getitem__(self, part):
        parent = self

        class View:
            def fill_(self, value):
                parent.data[part] = [value] * len(parent.data[part])

        return View()


def load_helper(tmp_path, monkeypatch, metadata):
    subprocess.run(["git", "apply", str(ROOT / "patches/sglang-bench/03-fa3-verify-splitkv.patch")], cwd=tmp_path, check=True)
    calls = []

    def prepare(**kwargs):
        calls.append(kwargs)
        return metadata

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float8_e4m3fn="fp8"))
    monkeypatch.setitem(sys.modules, "sgl_kernel", types.ModuleType("sgl_kernel"))
    monkeypatch.setitem(sys.modules, "sgl_kernel.flash_attn", types.SimpleNamespace(get_scheduler_metadata=prepare))
    helper = runpy.run_path(str(tmp_path / "python/sglang/kernels/ops/attention/fa3_verify_splitkv.py"))["get_verify_splitkv_metadata"]
    return helper, calls


@pytest.mark.parametrize("batch", [32, 33, 34, 35, 36])
@pytest.mark.parametrize("splits", [2, 3, 4, 6, 8])
def test_only_live_dynamic_entries_change_and_capacity_matches(tmp_path, monkeypatch, batch, splits):
    rounded = (batch + 3) // 4 * 4
    # Distinct sentinels catch accidental changes to alignment padding, M,
    # sorted-batch map, head swizzle and mutable semaphore.
    before = list(range(4 * rounded + 1))
    metadata = Tensor(data=before.copy())
    helper, calls = load_helper(tmp_path, monkeypatch, metadata)
    actual = helper(Tensor((batch * 4, 64, 128)), Tensor((2377, 64, 4, 128)), Tensor((batch, 128), "int32"), Tensor((batch,), "int32"), Tensor((batch + 1,), "int32"), splits)
    assert actual is metadata
    assert metadata.data == [splits] * batch + before[batch:]
    assert calls[0]["num_splits"] == splits
    assert calls[0]["max_seqlen_q"] == 4
    # Match fwd's page-table bound, not max(cache_seqlens).
    assert calls[0]["max_seqlen_k"] == 8192
    assert (calls[0]["num_heads"], calls[0]["num_heads_k"]) == (64, 4)
    assert calls[0]["pack_gqa"] is True


def test_changed_metadata_abi_is_rejected_before_mutation(tmp_path, monkeypatch):
    metadata = Tensor(data=list(range(144)))  # expected 145 for B=36
    before = metadata.data.copy()
    helper, _ = load_helper(tmp_path, monkeypatch, metadata)
    with pytest.raises(AssertionError):
        helper(Tensor((144, 64, 128)), Tensor((2377, 64, 4, 128)), Tensor((36, 66)), Tensor((36,)), Tensor((37,)), 8)
    assert metadata.data == before
