"""Reject incomplete adapter evidence before accepting a gate or serving run."""

import argparse
import json
import math
from pathlib import Path

COUNTS = [[1, 1, 1, 1], [128, 0, 17, 64], [128] * 4, [4096] * 4]


def validate(runtime, serving_config=None):
    result = json.loads((runtime / "results/gate.json").read_text())
    provenance = json.loads((runtime / "source/provenance.json").read_text())
    assert (runtime / "run-success").is_file(), "Missing successful batch marker"
    accounting = [line.split("|") for line in (runtime / "logs/slurm-accounting.txt").read_text().splitlines() if line.strip()]
    assert len(accounting) == 1 and accounting[0][1:3] == ["COMPLETED", "0:0"], "Gate allocation did not complete successfully"
    assert provenance["diagnostic"] == "vllm-sm90-megamoe"
    assert provenance["measurement_mode"] == "kernel-gate" and provenance["profile"] is False
    assert provenance["world_size"] == result["world_size"] == 4
    assert result["schema"] == "vllm-sm90-actual-adapter-gate/v1"
    assert all(result[key] is True for key in ("completed", "correctness_passed", "graph_passed"))
    assert result["adapter_manifest"]["backend"] == "apac_sm90_megamoe"
    assert result["native_factory_runner"] is True
    assert result["parallelism"] == {"tp": 1, "dp": 4, "ep": 4}
    assert result["capacity_requested"] == 4096
    assert sorted(gpu["rank"] for gpu in result["gpus"]) == list(range(4))
    assert all(isinstance(gpu["uuid"], str) and gpu["uuid"] for gpu in result["gpus"])
    assert len({gpu["uuid"] for gpu in result["gpus"]}) == 4, "GPU UUIDs must be distinct"
    assert [case["counts"] for case in result["cases"]] == COUNTS
    for case in result["cases"]:
        records = case["records"]
        assert sorted((record["rank"], record["replay"]) for record in records) == [(rank, replay) for rank in range(4) for replay in range(4)]
        for record in records:
            assert record["eager_compared"] is True
            assert record["pattern"] == ("random", "hot_rank0", "boundary_masked", "all_masked")[record["replay"]]
            assert record["rows"] == case["counts"][record["rank"]]
            error = record["errors"]
            if record["rows"] == 0:
                assert error["empty_rank_participated"] is True
            else:
                for key, limit in (("relative_l2", 0.025), ("max_scaled_error", 0.08)):
                    assert type(error[key]) in (float, int) and math.isfinite(error[key]) and 0 <= error[key] < limit, f"Invalid numerical result: {key}"
                assert error["masked_rows_zero"] is True
    assert result["official_benchmark_score"] is None
    assert not (runtime / "results/score.json").exists(), "Kernel gate is not a serving score"
    if serving_config is not None:
        expected = json.loads(serving_config.read_text())["images"]["vllm"]
        for key in ("sha256", "size_bytes"):
            assert provenance["container"][key] == expected[key], "Gate used a different serving image"
        # Changing the diagnostic invalidates old evidence; rerun only after
        # the new CPU checks pass. Runtime-only proxy changes reuse this gate.
        for name in ("adapter_gpu_gate.py", "gpu_gate.py"):
            assert (runtime / "source" / name).read_bytes() == (Path(__file__).parent / name).read_bytes(), f"Gate used a different {name}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--serving-config", type=Path)
    args = parser.parse_args()
    validate(args.runtime, args.serving_config)
