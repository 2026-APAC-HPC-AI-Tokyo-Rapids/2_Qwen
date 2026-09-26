# /// script
# dependencies = ["tokenizers==0.23.2"]
# ///
"""Inspect archived synthetic inputs and observed ASPIRE prefill cache reuse."""
import argparse
import collections
import datetime
import json
import re
import statistics
from pathlib import Path

from collect import evidence
from tokenizers import Tokenizer


def common_prefix(a, b):
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b))


def analyze(root, tokenizer_path):
    inputs_path, = root.rglob("inputs.json")
    records_path, = root.rglob("profile_export.jsonl")
    inputs = json.loads(inputs_path.read_text())["data"]
    texts = {r["session_id"]: r["payloads"][0]["messages"][0]["content"] for r in inputs}
    records = [json.loads(line) for line in records_path.open()]
    profiling = [r for r in records if r["metadata"]["phase_kind"] == "profiling"]
    start = min(r["metadata"]["request_start_ns"] for r in profiling) / 1e9 + 1
    end = max(r["metadata"]["request_end_ns"] for r in profiling) / 1e9 - 1
    workers = []
    pattern = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*#new-token: (\d+), #cached-token: (\d+)")
    for path in sorted(root.rglob("prefill-*.log")):
        batches = []
        for line in path.open(errors="replace"):
            match = pattern.search(line)
            if match:
                stamp = datetime.datetime.fromisoformat(match[1]).replace(tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
                if start <= stamp < end:
                    batches.append((int(match[2]), int(match[3])))
        new, cached = sum(x[0] for x in batches), sum(x[1] for x in batches)
        workers.append({"source": evidence(path), "batches": len(batches), "new_tokens": new, "cached_tokens": cached, "observed_reused_token_fraction": cached / (new + cached) if new + cached else None})
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # Long matches are censored at 64; this is an unlimited, noncausal bound,
    # not an estimate of the actual finite per-worker cache hit rate.
    sequences = sorted(e.ids[:64] for e in tokenizer.encode_batch(["<|im_start|>user\n" + text[:512] for text in texts.values()], add_special_tokens=False))
    neighbors = [max(common_prefix(seq, sequences[i - 1]) if i else 0, common_prefix(seq, sequences[i + 1]) if i + 1 < len(sequences) else 0) for i, seq in enumerate(sequences)]
    counts = collections.Counter(texts.values())
    used = [texts[r["metadata"]["conversation_id"]] for r in records]
    total_new, total_cached = sum(w["new_tokens"] for w in workers), sum(w["cached_tokens"] for w in workers)
    return {"scope": "ASPIRE SGT logs; profiling request window trimmed by 1 second at both ends; batch token counters, not a measured speedup", "sources": {"inputs": evidence(inputs_path), "records": evidence(records_path), "tokenizer": evidence(tokenizer_path)}, "dataset_entries": len(inputs), "distinct_bodies": len(counts), "duplicate_extra_entries": sum(n - 1 for n in counts.values()), "used_requests_including_warmup": len(used), "used_distinct_bodies": len(set(used)), "all_dataset_nearest_other_prefix_capped64": {"scope": "unlimited noncausal body+Qwen chat header bound; before any wire marker; no warmup system message", "median_tokens": statistics.median(neighbors), "p90_tokens": sorted(neighbors)[int(len(neighbors) * .9)], "at_least64_tokens": sum(n == 64 for n in neighbors)}, "workers": workers, "observed_reused_token_fraction": total_cached / (total_new + total_cached) if total_new + total_cached else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("tokenizer_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run, args.tokenizer_json)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("dataset_entries", "duplicate_extra_entries", "observed_reused_token_fraction")}))


if __name__ == "__main__":
    main()
