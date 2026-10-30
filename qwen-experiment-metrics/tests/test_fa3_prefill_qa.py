"""CPU checks for the staged-Q pipelined Prefill FA3 rows (prefill-qa.patch)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FA3 = ROOT / 'kernels/fa3-verify'


def added(name):
    return '\n'.join(line[1:] for line in (FA3 / name).read_text().splitlines() if line.startswith('+') and not line.startswith('+++'))


def removed(name):
    return [line[1:] for line in (FA3 / name).read_text().splitlines() if line.startswith('-') and not line.startswith('---')]


def test_qa_patch_follows_the_pipe_patch_in_the_build_chain():
    source = (FA3 / 'build.py').read_text()
    assert source.index('"prefill-pipe.patch"') < source.index('"prefill-qa.patch"') < source.index('cuda = Path("/usr/local/cuda")')


def test_qa_bits_require_the_pipelined_tiles():
    text = added('prefill-qa.patch')
    assert 'static constexpr bool PrefillPipeQA = (PrefillVariant & 256) != 0;' in text
    assert 'static_assert(!PrefillPipeQA || PrefillPipe);' in text
    assert 'static constexpr bool PrefillPipeRows = (PrefillVariant & 1024) != 0;' in text
    assert 'static_assert(!PrefillPipeRows || PrefillPipe);' in text


def test_qa_patch_changes_other_rows_only_behind_the_flags():
    text = added('prefill-qa.patch')
    # Only the staged-Q row grows smem_q (the staging half) and counts one barrier_Q arrival.
    assert 'cute::cosize_v<SmemLayoutQ> * kHeads * (PrefillPipeQA ? 2 : 1), SmemAlignmentQ> smem_q;' in text
    assert 'barrier_Q.init(Use_TMA_Q || CollectiveMainloop::PrefillPipeQA ? 1 : NumProducerThreads /*numThreads*/);' in text
    # The producer's scheduler path and the pin's per-tile prefetch stay as they were for every other row.
    assert 'if constexpr (!PrefillPipeQA) { scheduler_prefetch(); }' in text
    assert '        } else {\n            scheduler_prefetch();' in text
    assert removed('prefill-qa.patch').count('        scheduler_prefetch();') == 1
    # The store without the flag is the previous store_fp8_direct body.
    assert 'store_fp8_direct_impl<false>(params, tOrO, shared_storage, tiled_mma, thread_idx, block_coord, 0, 0);' in text


def test_staged_q_moves_into_smem_q_after_the_consumers_release_it():
    text = added('prefill-qa.patch')
    handoff = text[text.index('if constexpr (PrefillPipeQA) {\n            // The staged Q'):text.index('} else if constexpr (Use_TMA_Q) {')]
    wait, fence = handoff.index('flash::cp_async_wait<0>();'), handoff.index('fence_view_async_shared();')
    released = handoff.index('FwdNamedBarriers::QueryEmpty')
    expect, copy = handoff.index('arrive_and_expect_tx'), handoff.index('cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes')
    assert wait < fence < released < expect < copy
    # The staging half is reused only after the bulk copy of the current tile has completed.
    assert 'if (work_idx >= 0) { shared_storage.pipelines.barrier_Q.wait(work_idx % 2); }' in text
    # One claim per tile: an atomic before or at the decision step, and the decision once.
    assert text.count('atomicAdd(scheduler_prefetch.semaphore, 1) + int(gridDim.x)') == 2
    assert text.count('scheduler_prefetch.tile_of(pipe_idx)') == 2


def test_qd_reads_q_from_the_half_of_its_tile():
    text = added('prefill-qa.patch')
    assert 'static constexpr bool PrefillPipeQD = (PrefillVariant & 2048) != 0;' in text and 'static_assert(!PrefillPipeQD || PrefillPipeQA);' in text
    # Two textual copies of the tile body: the steady QK reads the tile's half, the boundary QK the other.
    assert text.count('{  // Tile with Q in half 0.') == 1 and text.count('{  // Tile with Q in half 1.') == 1
    assert 'next_tile(cute::Int<1>{})' in text and 'next_tile(cute::Int<0>{})' in text
    # The producers fill half (w + 1) % 2 only after tile w - 1 released it, once per tile.
    assert 'int const half = PrefillPipeQD ? (work_idx + 1) % 2 : 1;' in text
    assert 'if (work_idx >= 1) { cutlass::arch::NamedBarrier::sync(NumMmaThreadsQK + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/); }' in text
    assert 'pipe_tile(shared_storage)[PrefillPipeQD ? work_idx % 2 : 0] = make_int4(0, 0, -1, 0);' in text
