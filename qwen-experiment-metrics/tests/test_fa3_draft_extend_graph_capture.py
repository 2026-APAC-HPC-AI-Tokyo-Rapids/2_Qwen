"""CPU execution checks for the FA3 EAGLE draft-extend capture fixture."""

import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "patches/sglang/fa3-draft-extend-graph.patch"
RUNNER_DIFF = "diff --git a/python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py"


def added_runner_lines():
    runner_diff = PATCH.read_text().split(RUNNER_DIFF, 1)[1]
    return [
        line[1:]
        for line in runner_diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def fixture_class():
    lines = added_runner_lines()
    start = lines.index("    @contextlib.contextmanager")
    end = lines.index("                allocator.free(cache_locs)") + 1
    source = "class Fixture:\n" + "\n".join(lines[start:end]) + "\n"
    namespace = {"contextlib": contextlib, "torch": torch}
    exec(compile(ast.parse(source), str(PATCH), "exec"), namespace)  # noqa: S102
    return namespace["Fixture"]


class Allocator:
    def __init__(self, page_size, *, fail=False, duplicate=False):
        self.page_size = page_size
        self.fail = fail
        self.duplicate = duplicate
        self.freed = []
        self.extend_args = None

    def alloc(self, count):
        if self.fail:
            return None
        locs = torch.arange(101, 101 + count, dtype=torch.int64)
        return locs[:1].expand(count).clone() if self.duplicate else locs

    def alloc_extend(
        self, prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, count
    ):
        self.extend_args = (
            prefix_lens.clone(),
            prefix_lens_cpu.clone(),
            seq_lens.clone(),
            seq_lens_cpu.clone(),
            last_loc.clone(),
            count,
        )
        if self.fail:
            return None
        offsets = torch.arange(count, dtype=torch.int64)
        width = count // len(seq_lens)
        locs = 6400 + offsets // width * 64 + offsets % width
        return locs[:1].expand(count).clone() if self.duplicate else locs

    def free(self, locs):
        self.freed.append(locs.clone())


class DeviceModule:
    def __init__(self):
        self.sync_count = 0

    def synchronize(self):
        self.sync_count += 1


def make_fixture(page_size, *, fail=False, duplicate=False):
    fixture_type = fixture_class()
    fixture = fixture_type()
    fixture.captured_req_width = 4
    fixture.device_module = DeviceModule()
    fixture.model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            free_slots=[3, 1, 4],
            req_to_token=torch.arange(60, dtype=torch.int32).view(5, 12),
        ),
        token_to_kv_pool_allocator=Allocator(
            page_size, fail=fail, duplicate=duplicate
        ),
    )
    fixture.buffers = SimpleNamespace(
        req_pool_indices=torch.tensor([8, 9, 10], dtype=torch.int64),
        out_cache_loc=torch.arange(20, 32, dtype=torch.int64),
        positions=torch.arange(30, 42, dtype=torch.int64),
        mrope_positions=torch.arange(36, dtype=torch.int64).view(3, 12),
        seq_lens=torch.tensor([7, 8, 9], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([7, 8, 9], dtype=torch.int64),
    )
    return fixture


def state(fixture):
    return {
        "req_to_token": fixture.model_runner.req_to_token_pool.req_to_token.clone(),
        "req_pool_indices": fixture.buffers.req_pool_indices.clone(),
        "out_cache_loc": fixture.buffers.out_cache_loc.clone(),
        "positions": fixture.buffers.positions.clone(),
        "mrope_positions": fixture.buffers.mrope_positions.clone(),
        "seq_lens": fixture.buffers.seq_lens.clone(),
        "seq_lens_cpu": fixture.buffers.seq_lens_cpu.clone(),
    }


@pytest.mark.parametrize("page_size", [1, 64])
def test_fixture_writes_a_coherent_batch_then_restores_pool_and_buffers(page_size):
    fixture = make_fixture(page_size)
    before = state(fixture)
    allocator = fixture.model_runner.token_to_kv_pool_allocator
    width = fixture.captured_req_width
    with fixture._fa3_capture_input_fixture(2):
        locs = fixture.buffers.out_cache_loc[: 2 * width]
        assert fixture.buffers.req_pool_indices[:2].tolist() == [3, 1]
        assert fixture.buffers.seq_lens[:2].tolist() == [width, width]
        assert fixture.buffers.seq_lens_cpu[:2].tolist() == [width, width]
        assert locs.unique().numel() == 2 * width
        assert torch.equal(
            fixture.model_runner.req_to_token_pool.req_to_token[[3, 1], :width],
            locs.view(2, width).to(torch.int32),
        )
        assert fixture.buffers.positions[: 2 * width].tolist() == [0, 1, 2, 3] * 2
        assert torch.equal(
            fixture.buffers.mrope_positions[:, : 2 * width],
            fixture.buffers.positions[: 2 * width].unsqueeze(0).expand(3, -1),
        )
        if page_size > 1:
            prefix, prefix_cpu, seq, seq_cpu, last, count = allocator.extend_args
            assert prefix.tolist() == [0, 0] and prefix_cpu.tolist() == [0, 0]
            assert seq.tolist() == [width, width] and seq_cpu.tolist() == [width, width]
            assert last.tolist() == [-1, -1] and count == 2 * width

    after = state(fixture)
    assert all(torch.equal(after[name], value) for name, value in before.items())
    assert len(allocator.freed) == 1 and allocator.freed[0].numel() == 2 * width
    assert fixture.device_module.sync_count == 2


@pytest.mark.parametrize("page_size", [1, 64])
def test_fixture_restores_and_frees_after_forward_error(page_size):
    fixture = make_fixture(page_size)
    before = state(fixture)
    allocator = fixture.model_runner.token_to_kv_pool_allocator
    with pytest.raises(ValueError, match="forward failed"), fixture._fa3_capture_input_fixture(2):
        raise ValueError("forward failed")
    after = state(fixture)
    assert all(torch.equal(after[name], value) for name, value in before.items())
    assert len(allocator.freed) == 1 and fixture.device_module.sync_count == 2


def test_fixture_cleans_up_duplicate_allocator_output_and_handles_shortfall():
    duplicate = make_fixture(64, duplicate=True)
    before = state(duplicate)
    with pytest.raises(RuntimeError, match="aliased KV slots"), duplicate._fa3_capture_input_fixture(2):
        raise AssertionError("duplicate locations must fail before yield")
    after = state(duplicate)
    assert all(torch.equal(after[name], value) for name, value in before.items())
    assert len(duplicate.model_runner.token_to_kv_pool_allocator.freed) == 1

    shortfall = make_fixture(64, fail=True)
    before = state(shortfall)
    with pytest.raises(RuntimeError, match="could not allocate"), shortfall._fa3_capture_input_fixture(2):
        raise AssertionError("allocation shortfall must fail before yield")
    after = state(shortfall)
    assert all(torch.equal(after[name], value) for name, value in before.items())
    assert not shortfall.model_runner.token_to_kv_pool_allocator.freed


def test_capture_patch_keeps_exact_full_logits_and_hidden_parity():
    added = "\n".join(added_runner_lines())
    assert "reference_logits = eager.next_token_logits.clone()" in added
    assert "reference_hidden = (" in added and "eager.hidden_states.clone()" in added
    assert "replay.next_token_logits," in added and "replay.hidden_states," in added
    assert "rtol=0" in added and "atol=0" in added and "equal_nan=False" in added


def test_padding_fill_one_with_query_width_four_masks_three_rows_under_bottom_right_causality():
    def causal_key_counts(query_len, key_len):
        # FlashAttention's bottom-right alignment: j <= i + key_len - query_len.
        return [max(0, min(key_len, i + key_len - query_len + 1)) for i in range(query_len)]

    assert causal_key_counts(4, 1) == [0, 0, 0, 1]
    assert causal_key_counts(4, 4) == [1, 2, 3, 4]
