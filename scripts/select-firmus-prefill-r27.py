"""Post-greedy independent Prefill gates; neither candidate is a prerequisite of the other."""
import argparse
import json
import runpy
import time
from pathlib import Path

import firmus_prefill_norm_fma as norm
import firmus_prefill_poll_backoff as poll


def aggregate_norm(directory, env_output, pair_selection, run_id, deadline, workers_ok, image):
    result = {"schema": norm.SCHEMA, "adopted": False, "run_id": run_id, "gpus": [], "workers_ok": bool(workers_ok)}
    try:
        assert time.time() <= deadline, "expired budget"
        prior = poll.read(pair_selection)
        identities = {g["uuid"] for g in prior["gpus"]}
        assert len(identities) == len(prior["gpus"]) == 12
        assert all(g["run_id"].split("-", 1)[0] == run_id.split("-", 1)[0] for g in prior["gpus"])
        uuids, sources = set(), set()
        for worker in range(3):
            for rank in range(4):
                r = poll.read(directory / f"prefill-{worker}" / f"rank-{rank}.json")
                result["gpus"].append(r)
                assert r["schema"] == norm.SCHEMA and r["run_id"] == run_id and r["rank"] == rank
                assert r["worker"] == f"prefill-{worker}" and r["uuid"] in identities and r["uuid"] not in uuids and "H200" in r["device"]
                assert r["healthy"] is True and r["image_sha256"] == image
                assert 0 < r["elapsed_seconds"] < norm.SECONDS and r["finished_at"] <= deadline
                assert len(r["source_sha256"]) == 64
                norm.validate(r)
                uuids.add(r["uuid"])
                sources.add(r["source_sha256"])
        assert uuids == identities and len(sources) == 1
        lines = env_output.read_text().splitlines()
        assert all(not line.startswith(norm.KEY + "=") or line == norm.KEY + "=0" for line in lines)
        assert time.time() <= deadline
        env_output.write_text("\n".join([line for line in lines if not line.startswith(norm.KEY + "=")] + [norm.KEY + "=1"]) + "\n")
        result.update(adopted=True, reason="all twelve UUIDs: FP64 non-inferiority, 1200 calls and both order medians improve on every shape")
    except (OSError, AssertionError, KeyError, ValueError, TypeError) as error:
        result["reason"] = f"{type(error).__name__}: {error}"
    return result


def aggregate(args):
    decisions = {}
    for name, module, enabled in (("norm-square-fma", norm, args.norm_square_fma), ("poll-backoff", poll, args.poll_backoff)):
        temporary = args.env_output.with_name(args.env_output.name + ".r27.tmp")
        result = {"adopted": False, "reason": "disabled"}
        try:
            if enabled:
                temporary.write_text(args.env_output.read_text())
                function = aggregate_norm if module is norm else poll.aggregate
                result = function(args.aggregate / name, temporary, args.pair_selection, args.run_id, args.deadline, args.workers_ok, args.image_sha256)
                if result["adopted"]:
                    if time.time() <= args.deadline:
                        temporary.replace(args.env_output)
                    else:
                        result.update(adopted=False, reason="aggregate deadline expired")
        finally:
            temporary.unlink(missing_ok=True)
        decisions[name] = result
        directory = args.aggregate / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "selection.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.output.write_text(json.dumps(decisions, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: {p: v for p, v in r.items() if p != "gpus"} for k, r in decisions.items()}), flush=True)


def worker(args):
    # Norm runs first: its short gate can finish independently even when the
    # MegaMoE endurance encounters the earlier observed long-run stalls.
    for name, module, enabled in (("norm-square-fma", norm, args.norm_square_fma), ("poll-backoff", poll, args.poll_backoff)):
        if not enabled:
            continue
        output = args.output.parent / name / args.output.name
        try:
            if module is norm:
                norm.run(output, args.run_id, args.deadline, args.image_sha256)
            else:
                prepare = runpy.run_path(str(Path(__file__).with_name("select-firmus-prefill-megamoe.py")))["prepare"]
                poll.run(prepare, output, args.routing, args.run_id, args.deadline, args.image_sha256)
        except (OSError, AssertionError, RuntimeError, ValueError, TypeError, ImportError) as error:
            import os
            import traceback

            traceback.print_exc()
            output.mkdir(parents=True, exist_ok=True)
            (output / f"rank-{os.environ['LOCAL_RANK']}.json").write_text(json.dumps({"schema": module.SCHEMA, "healthy": False, "reason": f"{type(error).__name__}: {error}"}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-output", type=Path)
    parser.add_argument("--pair-selection", type=Path)
    parser.add_argument("--routing", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--workers-ok", type=int, choices=(0, 1), default=0)
    parser.add_argument("--poll-backoff", choices=("true", "false"), default="false")
    parser.add_argument("--norm-square-fma", choices=("true", "false"), default="false")
    args = parser.parse_args()
    args.poll_backoff = args.poll_backoff == "true"
    args.norm_square_fma = args.norm_square_fma == "true"
    if args.aggregate:
        aggregate(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
