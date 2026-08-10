from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from qwen_experiment_metrics.network import (
    parse_imb_mpi1,
    parse_nccl_tests_json,
    project_mpi_network_results_v2,
)

FIXTURES = Path(__file__).parent / "fixtures"
RUN_ID = "r-" + "a" * 32


def test_imb_handles_zero_scientific_warning_and_multiple_sections() -> None:
    results = parse_imb_mpi1((FIXTURES / "imb-mpi1-complete.txt").read_text())

    assert len(results) == 8
    assert results[0].message_bytes == 0
    assert results[0].time_max_us == Decimal("0.10")
    assert results[0].bandwidth_mbytes_per_sec == Decimal("0.0")
    assert results[-1].process_count == 8
    allreduce = next(row for row in results if row.benchmark == "Allreduce")
    assert (allreduce.time_min_us, allreduce.time_avg_us, allreduce.time_max_us) == (
        Decimal("0.10"),
        Decimal("0.20"),
        Decimal("0.30"),
    )


def test_imb_truncated_or_inconsistent_capture_is_zero_rows() -> None:
    complete = (FIXTURES / "imb-mpi1-complete.txt").read_text()
    assert (
        parse_imb_mpi1(complete.replace("1 1000 2.0e-1 4.0e-1 3.0e-1", "1 1000")) == ()
    )
    assert parse_imb_mpi1(complete.replace(" 3.0e-1 2.0e-1", " 1.0e-1 2.0e-1")) == ()
    assert (
        parse_imb_mpi1(complete.replace("# Benchmarking Sendrecv", "# skipped")) == ()
    )


def test_imb_raw_to_relation_round_trip_and_grain() -> None:
    raw = (FIXTURES / "imb-mpi1-complete.txt").read_text()
    first = project_mpi_network_results_v2(
        RUN_ID,
        parse_imb_mpi1(raw),
        "source/network/imb-mpi1/stdout.txt",
    )
    second = project_mpi_network_results_v2(
        RUN_ID,
        parse_imb_mpi1(raw),
        "source/network/imb-mpi1/stdout.txt",
    )

    assert first == second
    assert len(
        {(r.run_id, r.benchmark, r.process_count, r.message_bytes) for r in first}
    ) == len(first)
    assert set(first[0].to_object()) == {
        "run_id",
        "benchmark",
        "process_count",
        "message_bytes",
        "iterations",
        "time_min_us",
        "time_avg_us",
        "time_max_us",
        "bandwidth_mbytes_per_sec",
        "source_member",
    }


def test_three_official_nccl_json_collectives_project_both_buffer_modes() -> None:
    cases = {
        "all_reduce": ("nccl-all-reduce.json", "sum"),
        "all_gather": ("nccl-all-gather.json", None),
        "reduce_scatter": ("nccl-reduce-scatter.json", "sum"),
    }
    for collective, (fixture, operation) in cases.items():
        data = (FIXTURES / fixture).read_bytes()
        rows = parse_nccl_tests_json(
            data,
            run_id=RUN_ID,
            collective=collective,
            source_member=f"source/network/nccl-tests/{collective}/result.json",
        )
        assert len(rows) == 4
        assert {row.buffer_mode for row in rows} == {"out_of_place", "in_place"}
        assert {row.reduction_op for row in rows} == {operation}
        assert {row.data_type for row in rows} == {"float"}
        assert {row.mpi_process_count for row in rows} == {2}
        assert {row.gpu_count for row in rows} == {2}
        assert len(
            {
                (row.run_id, row.collective, row.message_bytes, row.buffer_mode)
                for row in rows
            }
        ) == len(rows)
        assert rows == parse_nccl_tests_json(
            data,
            run_id=RUN_ID,
            collective=collective,
            source_member=f"source/network/nccl-tests/{collective}/result.json",
        )


def test_nccl_preserves_numeric_fields_losslessly_as_decimals() -> None:
    rows = parse_nccl_tests_json(
        (FIXTURES / "nccl-all-reduce.json").read_bytes(),
        run_id=RUN_ID,
        collective="all_reduce",
        source_member="source/network/nccl-tests/all_reduce/result.json",
    )
    first = rows[0]
    assert first.time_us == Decimal("0.125")
    assert first.algorithm_bandwidth_gbps == Decimal("0.000064")
    assert first.bus_bandwidth_gbps == Decimal("0.000056")
    assert rows[-1].error_count == 1


def test_nccl_partial_truncated_duplicate_or_semantic_mismatch_is_zero_rows() -> None:
    data = (FIXTURES / "nccl-all-reduce.json").read_text()
    kwargs = {
        "run_id": RUN_ID,
        "collective": "all_reduce",
        "source_member": "source/network/nccl-tests/all_reduce/result.json",
    }
    assert parse_nccl_tests_json(data[:-2], **kwargs) == ()
    assert parse_nccl_tests_json(data.replace('"size":16', '"size":8'), **kwargs) == ()
    assert (
        parse_nccl_tests_json(data.replace('"redop":"sum"', '"redop":"none"'), **kwargs)
        == ()
    )
    assert (
        parse_nccl_tests_json(
            data.replace('"version":3', '"version":3,"version":3'), **kwargs
        )
        == ()
    )
    assert (
        parse_nccl_tests_json(data.replace('"type":"float"', '"type":"half"'), **kwargs)
        == ()
    )
