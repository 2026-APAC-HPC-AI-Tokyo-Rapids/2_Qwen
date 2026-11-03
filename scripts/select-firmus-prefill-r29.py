"""Post-greedy QKV tail gate; incomplete evidence keeps control."""
import argparse
import json
import os
import time
from pathlib import Path

import firmus_prefill_r29 as gate


def aggregate_candidate(args, name):
    result = {"schema": gate.SCHEMA, "candidate": name, "run_id": args.run_id, "adopted": False, "gpus": [], "workers_ok": bool(args.workers_ok), "rule": {**gate.SPECS[name], "minimum_fraction": .005, "rounds_per_order": gate.ROUNDS, "endurance_calls_per_shape": gate.CALLS}}
    try:
        assert time.time() <= args.deadline, "expired aggregate budget"
        prior = gate.read(args.pair_selection)
        assert prior["schema"] == "firmus-prefill-fa3-pair/v1"
        identities = {g["uuid"] for g in prior["gpus"]}
        assert len(identities) == len(prior["gpus"]) == 12, "twelve prior UUIDs required"
        assert all(g["run_id"].split("-", 1)[0] == args.run_id.split("-", 1)[0] for g in prior["gpus"]), "wrong allocation"
        sources, uuids = set(), set()
        for worker in range(3):
            for rank in range(4):
                r = gate.read(args.aggregate / name / f"prefill-{worker}" / f"rank-{rank}.json")
                result["gpus"].append(r)
                assert r["schema"] == gate.SCHEMA and r["candidate"] == name and r["run_id"] == args.run_id
                assert r["rank"] == rank and r["worker"] == f"prefill-{worker}"
                assert "H200" in r["device"] and r["uuid"] in identities and r["uuid"] not in uuids
                assert r["image_sha256"] == args.image_sha256 and r["fa3_pin"] == args.fa3_pin, "different serving baseline"
                # The worker reservation ends ten seconds before aggregation.
                assert 0 < r["elapsed_seconds"] < gate.SECONDS and r["finished_at"] <= args.deadline - 10
                assert len(r["source_sha256"]) == 64
                gate.validate(r, name)
                sources.add(r["source_sha256"]); uuids.add(r["uuid"])
        assert uuids == identities and len(sources) == 1
        key = gate.SPECS[name]["key"]
        lines = args.env_output.read_text().splitlines()
        assert [line for line in lines if line.startswith(key + "=")] == [key + "=0"], "exactly one default-off flag required"
        # Publish only complete evidence, within the aggregate deadline.
        temporary = args.env_output.with_name(args.env_output.name + ".r29.tmp")
        try:
            temporary.write_text("\n".join(key + "=1" if line.startswith(key + "=") else line for line in lines) + "\n")
            assert time.time() <= args.deadline, "expired aggregate budget"
            temporary.replace(args.env_output)
        finally:
            temporary.unlink(missing_ok=True)
        result.update(adopted=True, reason="all twelve UUIDs: byte exact, changed-input restoration, 1200 calls per shape and both-order margin")
    except (OSError, AssertionError, KeyError, ValueError, TypeError) as error:
        result["reason"] = f"{type(error).__name__}: {error}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-output", type=Path)
    parser.add_argument("--pair-selection", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--deadline", required=True, type=float)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--fa3-pin", required=True)
    parser.add_argument("--workers-ok", type=int, choices=(0, 1), default=0)
    for name in gate.SPECS:
        parser.add_argument("--" + name, choices=("true", "false"), default="false")
    args = parser.parse_args()
    decisions = {}
    for name in gate.SPECS:
        if getattr(args, name.replace("-", "_")) != "true":
            result = {"adopted": False, "reason": "disabled"}
        elif args.aggregate:
            result = aggregate_candidate(args, name)
        else:
            try:
                gate.run(name, args.output.parent / name / args.output.name, args.run_id, args.deadline, args.image_sha256, args.fa3_pin)
            except Exception as error:  # noqa: BLE001 - preserve worker failure evidence
                import traceback
                traceback.print_exc()
                directory = args.output.parent / name / args.output.name
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"rank-{os.environ['LOCAL_RANK']}.json").write_text(json.dumps({"healthy": False, "reason": f"{type(error).__name__}: {error}"}) + "\n")
            continue
        if args.aggregate:
            decisions[name] = result
            directory = args.aggregate / name
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "selection.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.aggregate:
        args.output.write_text(json.dumps(decisions, indent=2, allow_nan=False) + "\n")
        print(json.dumps({n: {k: v for k, v in r.items() if k != "gpus"} for n, r in decisions.items()}), flush=True)


if __name__ == "__main__":
    main()
