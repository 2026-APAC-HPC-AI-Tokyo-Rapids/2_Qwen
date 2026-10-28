#!/usr/bin/env python3
"""EAGLE acceptance diagnostics; never an official throughput score."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

from eagle_metrics import ACCEPTANCE_LOG, acceptance_states, summary

PYTHON = "/opt/sglang/venv/bin/python"
AIPERF = "/opt/sglang/venv/bin/aiperf"
URL = "http://127.0.0.1:8000"
TOKENIZER = "Qwen/Qwen3-235B-A22B-Instruct-2507"
BENCHMARK_SECONDS = 20


def server_command(root: Path, draft: str | None = None, port: int = 8000) -> list[str]:
    command = [PYTHON, "-m", "sglang.launch_server", "--speculative-draft-model-path", str(root / "drafts" / draft if draft else root / "draft")]
    command += shlex.split("--model-path /model --served-model-name qwen --trust-remote-code --dtype auto --context-length 8192 --tp-size 4 --mem-fraction-static 0.90 --page-size 64 --kv-cache-dtype fp8_e4m3 --attention-backend fa3 --moe-a2a-backend none --moe-runner-backend auto --max-running-requests 32 --max-total-tokens 131072 --chunked-prefill-size 8192 --speculative-algorithm EAGLE3 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --disable-cuda-graph --host 127.0.0.1 --port 8000")
    if draft:
        command[command.index("--port") + 1] = str(port)
        command += ["--speculative-draft-model-quantization", "fp8", "--dist-init-addr", f"127.0.0.1:{port + 21000}", "--nccl-port", str(port + 22000)]
    return command


def aiperf_command(draft: str | None = None, port: int = 8000) -> list[str]:
    # Official prompt generation; reduced request count and timings for diagnostics.
    command = [AIPERF, "profile", "--url", URL, "--endpoint-type", "chat", "--endpoint", "/v1/chat/completions", "--model", "qwen", "--tokenizer", TOKENIZER, "--streaming", "-H", "Accept: text/event-stream"]
    command += shlex.split(f"--concurrency 32 --workers-max 32 --api-key NOT_USED --benchmark-duration {BENCHMARK_SECONDS} --benchmark-grace-period 30 --synthetic-input-tokens-mean 4000 --synthetic-input-tokens-stddev 0 --output-tokens-mean 200 --output-tokens-stddev 0 --extra-inputs min_tokens:200 --extra-inputs max_tokens:200 --extra-inputs ignore_eos:true --extra-inputs temperature:0.0 --extra-inputs repetition_penalty:1.0 --num-dataset-entries 12800 --random-seed 2026 --cache-bust first_turn_prefix --goodput 'time_to_first_token:3000 inter_token_latency:20' --ui simple")
    if draft:
        command[command.index("--url") + 1] = f"http://127.0.0.1:{port}"
        command[command.index("--benchmark-duration") + 1] = "180"
        command += ["--request-count", "128"]
    return command


def completed_requests(export: dict, draft: str | None = None) -> int:
    if export.get("aiperf_version") != "0.12.0" or export.get("was_cancelled") is not False or export.get("error_summary") != []:
        raise ValueError("AIPerf version, cancellation or error validation failed")
    count = export["request_count"]["avg"]
    if isinstance(count, bool) or not isinstance(count, (int, float)) or not math.isfinite(count) or count <= 0 or int(count) != count:
        raise ValueError("no positive integral completed request count")
    for name in ("error_request_count", "osl_mismatch_count"):
        if export.get(name, {"avg": 0})["avg"] != 0:
            raise ValueError(f"AIPerf {name} is nonzero")
    config = export["input_config"]
    phases = config["phases"]
    if len(phases) != 1 or any(phases[0].get(key) != value for key, value in {"kind": "profiling", "type": "concurrency", "duration": 180 if draft else BENCHMARK_SECONDS, "concurrency": 32, "grace_period": 30, **({"requests": 128} if draft else {})}.items()):
        raise ValueError("expected one un-warmed concurrency32 measurement phase")
    expected_dataset = {"name": "main", "type": "synthetic", "entries": 12800, "random_seed": 2026, "prompts": {"isl": {"mean": 4000.0, "stddev": 0.0}, "osl": {"mean": 200.0, "stddev": 0.0}, "cache_bust": {"target": "first_turn_prefix"}}}
    if config["datasets"] != [expected_dataset] or config["tokenizer"]["name"] != TOKENIZER:
        raise ValueError("exported prompt generation differs from the official workload")
    expected_extra = {"min_tokens": 200, "max_tokens": 200, "ignore_eos": True, "temperature": 0.0, "repetition_penalty": 1.0}
    if config["endpoint"]["extra"] != expected_extra:
        raise ValueError("exported sampling parameters differ from the official workload")
    if draft and count != 128:
        raise ValueError(f"expected exactly 128 completed requests, got {count}")
    return int(count)


def collect(info: dict, server_log: str, export: dict, startup_s: float, draft: str | None = None) -> dict:
    expected = {"speculative_algorithm": "EAGLE3", "speculative_num_steps": 3, "speculative_eagle_topk": 1, "speculative_num_draft_tokens": 4, "tp_size": 4, "dp_size": 1, "enable_dp_attention": False, "disaggregation_mode": "null", "moe_a2a_backend": "none", "dtype": "auto", "context_length": 8192, "kv_cache_dtype": "fp8_e4m3"}
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f"unexpected server setting {key}={info.get(key)!r}, expected {value!r}")
    if draft:
        for key, value in {"speculative_draft_model_quantization": "fp8", "random_seed": 2026, "disable_cuda_graph": True}.items():
            if info.get(key) != value:
                raise ValueError(f"unexpected draft server setting {key}={info.get(key)!r}")
    states = acceptance_states(info, Path("server-info.json"), "huggingface")
    if len(states) != len(info["internal_states"]):
        raise ValueError("every internal state must report a completed acceptance average")
    lengths = [state["accept_length_including_bonus"] for state in states]
    if any(not 1 <= length <= 4 for length in lengths):
        raise ValueError("acceptance length outside the steps3/draft4 range")
    samples = [{"accept_length_including_bonus": float(match["length"]), "accept_rate": float(match["rate"])} for match in ACCEPTANCE_LOG.finditer(server_log.replace("\x1b", ""))]
    stats = summary(lengths)
    return {"schema": "genkai-eagle-accept/v1", "scope": "single TP4 server, no PD; acceptance only, not an official score", "method": "GET /get_server_info internal_states[].avg_spec_accept_length; unweighted state mean including bonus", "states": states, "accept_length": stats["mean"], "accepted_draft": stats["mean"] - 1, "requests": completed_requests(export, draft), "startup_s": startup_s, "benchmark_duration_s": 180 if draft else BENCHMARK_SECONDS, "log_samples": {"accept_length": summary([sample["accept_length_including_bonus"] for sample in samples]), "accept_rate": summary([sample["accept_rate"] for sample in samples])}}


def position_acceptance(log: str) -> dict:
    hist = [0] * 4
    for match in re.finditer(r"EAGLE_POSITION histogram=(\[[0-9, ]+\])", log):
        sample = json.loads(match[1])
        if len(sample) != 4 or any(type(n) is not int or n < 0 for n in sample):
            raise ValueError("invalid positional acceptance histogram")
        hist = [a + b for a, b in zip(hist, sample)]
    batches = len(re.findall(r"EAGLE_POSITION histogram=", log))
    total = sum(hist)
    if not total:
        raise ValueError("no positional acceptance evidence in server log")
    return {"correct_drafts_histogram": hist, "verify_batches": batches, "verification_requests": total, "unconditional_prefix_acceptance": [sum(hist[n:]) / total for n in (1, 2, 3)], "method": "P(correct drafts >= position), including unfinished requests; statistics rank only; CPU settlement before output truncation"}


def request_acceptance(log: str, count: int) -> list[dict]:
    requests = []
    for match in re.finditer(r"EAGLE_REQUEST rid=(\S+) verify_steps=(\d+) accepted_draft_tokens=(\d+)", log):
        rid, steps, accepted = match.groups()
        steps, accepted = int(steps), int(accepted)
        if not steps or not 0 <= accepted <= 3 * steps:
            raise ValueError("invalid completed-request acceptance counters")
        requests.append({"rid": rid, "verify_steps": steps, "accepted_draft_tokens": accepted, "accept_length_including_bonus": 1 + accepted / steps, "accepted_draft_length": accepted / steps})
    if len(requests) != count or len({item["rid"] for item in requests}) != count:
        raise ValueError(f"expected {count} distinct completed request counters, got {len(requests)}")
    return requests


def english_workload(root: Path) -> int:
    prompts = json.loads((root / "source/eagle-english-prompts.json").read_text())["prompts"]
    if len(prompts) != 32 or len(set(prompts)) != 32:
        raise ValueError("expected 32 unique fixed English prompts")

    def request(item):
        index, prompt = item
        body = {"model": "qwen", "messages": [{"role": "user", "content": prompt}], "min_tokens": 200, "max_tokens": 200, "ignore_eos": True, "temperature": 0.0, "repetition_penalty": 1.0}
        req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=150) as response:
            result = json.load(response)
        (root / f"results/english-response-{index:02}.json").write_text(json.dumps(result) + "\n")
        if result.get("error") or result.get("usage", {}).get("completion_tokens") != 200 or not result.get("choices"):
            raise ValueError("English request failed or did not generate 200 tokens")
        return 1

    with ThreadPoolExecutor(max_workers=32) as pool:
        return sum(pool.map(request, enumerate(prompts)))


def collect_english(root: Path, server_path: Path, startup_s: float) -> dict:
    # Pinned flush_cache resets lifetime acceptance counters, without restarting.
    with urllib.request.urlopen(URL + "/flush_cache", timeout=5) as response:
        if response.status != 200:
            raise ValueError("cache/statistics reset failed")
    baseline = get_json("/get_server_info")
    if any(state.get("avg_spec_accept_length") is not None for state in baseline["internal_states"]):
        raise ValueError("acceptance counters were not reset")
    (root / "results/english-baseline-server-info.json").write_text(json.dumps(baseline, indent=2) + "\n")
    offset = server_path.stat().st_size
    count = english_workload(root)
    info = get_json("/get_server_info")
    (root / "results/english-server-info.json").write_text(json.dumps(info, indent=2) + "\n")
    with server_path.open("rb") as stream:
        stream.seek(offset)
        log = stream.read().decode(errors="replace")
    (root / "logs/english-server.log").write_text(log)
    # Reuse setting/state validation; synthetic export validation belongs to (a).
    states = acceptance_states(info, Path("english-server-info.json"), "huggingface")
    lengths = [state["accept_length_including_bonus"] for state in states]
    if len(states) != len(info["internal_states"]) or any(not 1 <= n <= 4 for n in lengths):
        raise ValueError("incomplete English acceptance states")
    return {"workload": "fixed English questions, concurrency 32, 200 output tokens", "requests": count, "states": states, "accept_length": summary(lengths)["mean"], "startup_s": startup_s, "positions": position_acceptance(log), "server_log": "logs/english-server.log", "prompt_note": "Short English questions differ in length from 4000-token synthetic input; this is not an isolated language comparison"}


def stop(child: subprocess.Popen | None, grace: int = 5) -> None:
    if child is None:
        return
    # Also kill descendants after the group leader exits (AIPerf ignores TERM).
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=2)


def interrupted(signum: int, _frame: object) -> None:
    raise RuntimeError(f"probe interrupted by signal {signum}")


def get_json(path: str, port: int = 8000) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}" + path, timeout=3) as response:
        return json.load(response)


def execute(root: Path, started_at: float, firmus: bool = False, draft: str | None = None, port: int = 8000) -> dict:
    origin = time.monotonic() - (time.time() - started_at)

    def elapsed() -> float:
        return time.monotonic() - origin

    startup_limit = 600 if firmus else 450
    deadline = (900 if draft else 945) if firmus else 555
    process_limit = 960 if firmus else 570
    if version("aiperf") != "0.12.0":
        raise ValueError("expected the image's AIPerf 0.12.0")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("compute execution must be offline")
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, interrupted)
    if elapsed() >= startup_limit:
        raise TimeoutError("container setup exhausted the startup budget")
    # Leave 15 seconds for both process groups and the Enroot wrapper to exit.
    signal.setitimer(signal.ITIMER_REAL, deadline - elapsed())
    results = root / "results" / draft if draft else root / "results"
    logs = root / "logs" / draft if draft else root / "logs"
    server_path = logs / "server.log"
    client_path = logs / "aiperf.log"
    url = f"http://127.0.0.1:{port}"
    server = client = None
    try:
        command = server_command(root, draft, port)
        if firmus:
            command += ["--decode-log-interval", "1", "--random-seed", "2026"]
        (results / "server-command.txt").write_text(shlex.join(command) + "\n")
        print(f"server_start elapsed_s={elapsed():.1f} command={shlex.join(command)}", flush=True)
        with server_path.open("w") as output:
            server = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"server exited before health: {server.returncode}")
            if elapsed() >= startup_limit:
                raise TimeoutError(f"health did not pass within the {startup_limit}-second startup budget")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    if response.status == 200 and (not draft or "The server is fired up and ready to roll!" in server_path.read_text(errors="replace")):
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            print(f"waiting_health elapsed_s={elapsed():.1f}", flush=True)
            time.sleep(2)
        startup_s = elapsed()
        print(f"server_ready startup_s={startup_s:.3f} remaining_until_cleanup_s={deadline - elapsed():.1f}", flush=True)
        synthetic_offset = 0
        if firmus:
            with urllib.request.urlopen(url + "/flush_cache", timeout=5) as response:
                if response.status != 200:
                    raise ValueError("initial cache/statistics reset failed")
            synthetic_offset = server_path.stat().st_size
        command = aiperf_command(draft, port)
        (results / "aiperf-command.txt").write_text(shlex.join(command) + "\n")
        client_env = dict(os.environ, AIPERF_SERVICE_REGISTRATION_INTERVAL="1", AIPERF_SERVICE_REGISTRATION_MAX_ATTEMPTS="60", AIPERF_SERVICE_REGISTRATION_TIMEOUT="60", AIPERF_RECORD_PROCESSOR_SCALE_FACTOR="32")
        print(f"aiperf_start elapsed_s={elapsed():.1f} command={shlex.join(command)}", flush=True)
        with client_path.open("w") as output:
            client = subprocess.Popen(command, cwd=results, env=client_env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        while client.poll() is None:
            if server.poll() is not None:
                raise RuntimeError(f"server exited during AIPerf: {server.returncode}")
            time.sleep(1)
        print(f"aiperf_end elapsed_s={elapsed():.1f} status={client.returncode}", flush=True)
        if client.returncode != 0:
            raise RuntimeError(f"AIPerf failed: {client.returncode}")
        info = get_json("/get_server_info", port)
        (results / "server-info.json").write_text(json.dumps(info, indent=2) + "\n")
        # The workflow removes the runtime, including when validation fails.
        print("SERVER_INFO " + json.dumps(info), flush=True)
        exports = list(results.glob("artifacts/*/profile_export_aiperf.json"))
        if len(exports) != 1:
            raise ValueError(f"expected one AIPerf export, got {len(exports)}")
        shutil.copyfile(exports[0], results / "profile_export_aiperf.json")
        shutil.copyfile(exports[0].with_name("profile_export.jsonl"), results / "profile_export.jsonl")
        export = json.loads(exports[0].read_text())
        print("AIPERF_EXPORT " + json.dumps(export), flush=True)
        with server_path.open("rb") as stream:
            stream.seek(synthetic_offset)
            synthetic_log = stream.read().decode(errors="replace")
        result = collect(info, synthetic_log, export, startup_s, draft)
        if firmus:
            (logs / "synthetic-server.log").write_text(synthetic_log)
            result["positions"] = position_acceptance(synthetic_log)
            result["schema"] = "firmus-eagle-accept/v1"
            if draft:
                descriptor = next(item for item in json.loads((root / "source/eagle-drafts.json").read_text()) if item["name"] == draft)
                result.update(draft=descriptor, draft_precision=info["speculative_draft_model_quantization"], revision=descriptor["revision"], per_request=request_acceptance(synthetic_log, result["requests"]))
                result["verify_steps"] = sum(item["verify_steps"] for item in result["per_request"])
                result["verify_steps_note"] = "sum of completed-request verification counts; verify_batches counts scheduler iterations on the statistics rank"
                result["schema"] = "firmus-eagle-draft/v1"
            else:
                (results / "synthetic-eagle-acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
                if deadline - elapsed() < 180:
                    raise TimeoutError("less than 180 seconds remain for required English phase")
                result["english"] = collect_english(root, server_path, startup_s)
                (results / "english-eagle-acceptance.json").write_text(json.dumps(result["english"], indent=2) + "\n")
        (results / "eagle-acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
        # The existing workflow deletes the runtime; retain primary evidence in its log.
        print("EAGLE_ACCEPTANCE " + json.dumps(result), flush=True)
        print(f"collection_end elapsed_s={elapsed():.1f}", flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            stop(client)
        finally:
            stop(server)
        for path in (server_path, client_path):
            if path.exists():
                print(f"BEGIN_LOG {path.name}\n{path.read_text(errors='replace')}\nEND_LOG {path.name}", flush=True)
        print(f"cleanup_end elapsed_s={elapsed():.1f}", flush=True)
    if elapsed() >= process_limit:
        raise TimeoutError("probe exceeded its total process budget")
    return result


def compare_drafts(root: Path, started_at: float) -> dict:
    drafts = json.loads((root / "source/eagle-drafts.json").read_text())
    enabled = [item for item in drafts if item["enabled"]]
    if [item["name"] for item in enabled] != ["nebius", "meituan"]:
        raise ValueError("expected the two reviewed native drafts")
    children = []
    outputs = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, 915 - (time.time() - started_at)))
    try:
        for index, draft in enumerate(enabled):
            name = draft["name"]
            output = (root / "logs" / name / "probe.log").open("w")
            outputs.append(output)
            command = [PYTHON, "-u", str(root / "source/eagle-accept.py"), str(root), str(started_at), "--firmus", "--draft", name, "--port", str(8000 + index * 100)]
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(n) for n in range(index * 4, index * 4 + 4))), start_new_session=True)
            children.append(child)
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError("draft diagnostic failed; see per-draft probe/server/AIPerf logs")
            time.sleep(1)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError("draft diagnostic failed; see per-draft logs")
        results = [json.loads((root / "results" / item["name"] / "eagle-acceptance.json").read_text()) for item in enabled]
        control = results[0]["accept_length"]
        comparison = {"schema": "firmus-eagle-drafts/v1", "scope": "128 official synthetic requests each, concurrency32; single TP4 per draft; acceptance only, not serving throughput", "drafts": [{"name": item["draft"]["name"], "revision": item["revision"], "draft_precision": item["draft_precision"], "accept_length_including_bonus": item["accept_length"], "accepted_draft_length": item["accepted_draft"], "relative_to_nebius": item["accept_length"] / control, "requests": item["requests"], "verify_steps": item["verify_steps"]} for item in results], "excluded": [item for item in drafts if not item["enabled"]]}
        (root / "results/eagle-acceptance.json").write_text(json.dumps(comparison, indent=2) + "\n")
        print("EAGLE_DRAFT_COMPARISON " + json.dumps(comparison), flush=True)
        return comparison
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for child in children:
            # Worker signal handlers clean up their separately grouped server/client.
            stop(child, grace=15)
        for output in outputs:
            output.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("started_at", type=float)
    parser.add_argument("--firmus", action="store_true")
    parser.add_argument("--compare-drafts", action="store_true")
    parser.add_argument("--draft", choices=("nebius", "meituan"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.compare_drafts:
        compare_drafts(args.root, args.started_at)
        print("PASS EAGLE draft comparison; two TP4 diagnostics complete", flush=True)
    else:
        result = execute(args.root, args.started_at, firmus=args.firmus, draft=args.draft, port=args.port)
        if "english" in result:
            print("PASS workload=english accept_length={accept_length:.6f} requests={requests}".format(**result["english"]), flush=True)
        print("PASS accept_length={accept_length:.6f} accepted_draft={accepted_draft:.6f} requests={requests} startup_s={startup_s:.3f}".format(**result), flush=True)
