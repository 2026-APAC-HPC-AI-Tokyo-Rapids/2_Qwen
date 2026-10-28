"""Check the first-stage transaction split independently of GPU timing."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_first_stage_barrier_cannot_complete_from_weights_alone():
    # The single producer arrival stays outstanding until activation readiness.
    # Exercise either completion order after arrive_and_expect_tx(A+SFA).
    for activation_scale_bytes in (256, 512):
        b_bytes = 16384
        a_bytes = 8192 + activation_scale_bytes
        for b_completes_before_arrival in (False, True):
            pending_arrivals, pending_bytes = 1, b_bytes
            if b_completes_before_arrival:
                pending_bytes -= b_bytes
            assert pending_arrivals or pending_bytes
            pending_bytes += a_bytes
            pending_arrivals -= 1
            assert pending_bytes > 0
            pending_bytes -= a_bytes
            if not b_completes_before_arrival:
                assert pending_bytes == b_bytes
                pending_bytes -= b_bytes
            assert pending_arrivals == pending_bytes == 0


def test_candidate_changes_only_iwg_loader_and_preserves_scale_loads():
    patch = (ROOT / "patches/deep-gemm-archive/decode-m1/decode-m1-b-first/decode-m1-b-first.patch").read_text()
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("-") and not line.startswith("---"))
    assert "empty_barriers[stage_idx]->wait(phase ^ 1)" in added
    assert added.index("expect_transaction(SMEM_B_SIZE_PER_STAGE)") < added.index("tma::copy<")
    assert "if (k != 0)" in added
    assert "(k == 0 ? 0 : SMEM_B_SIZE_PER_STAGE)" in added
    assert "SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE" in removed
    assert "init(" not in added and "__ldg" not in added
    assert "num_k_blocks" not in added
    assert patch.count("@@") == 4  # Two localized loader hunks.
