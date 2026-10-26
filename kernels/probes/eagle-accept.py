#!/usr/bin/env python3
"""Single-server EAGLE acceptance diagnostic; never an official throughput score."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
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


def server_command(root: Path) -> list[str]:
    command = [PYTHON, "-m", "sglang.launch_server", "--speculative-draft-model-path", str(root / "draft")]
    command += shlex.split("--model-path /model --served-model-name qwen --trust-remote-code --dtype auto --context-length 8192 --tp-size 4 --mem-fraction-static 0.90 --page-size 64 --kv-cache-dtype fp8_e4m3 --attention-backend fa3 --moe-a2a-backend none --moe-runner-backend auto --max-running-requests 32 --max-total-tokens 131072 --chunked-prefill-size 8192 --speculative-algorithm EAGLE3 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --disable-cuda-graph --host 127.0.0.1 --port 8000")
    return command


def aiperf_command() -> list[str]:
    # Prompt generation is identical to official-aiperf.sh; only timings and C change.
    command = [AIPERF, "profile", "--url", URL, "--endpoint-type", "chat", "--endpoint", "/v1/chat/completions", "--model", "qwen", "--tokenizer", TOKENIZER, "--streaming", "-H", "Accept: text/event-stream"]
    command += shlex.split(f"--concurrency 32 --workers-max 32 --api-key NOT_USED --benchmark-duration {BENCHMARK_SECONDS} --benchmark-grace-period 30 --synthetic-input-tokens-mean 4000 --synthetic-input-tokens-stddev 0 --output-tokens-mean 200 --output-tokens-stddev 0 --extra-inputs min_tokens:200 --extra-inputs max_tokens:200 --extra-inputs ignore_eos:true --extra-inputs temperature:0.0 --extra-inputs repetition_penalty:1.0 --num-dataset-entries 12800 --random-seed 2026 --cache-bust first_turn_prefix --goodput 'time_to_first_token:3000 inter_token_latency:20' --ui simple")
    return command


def completed_requests(export: dict) -> int:
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
    if len(phases) != 1 or any(phases[0].get(key) != value for key, value in {"kind": "profiling", "type": "concurrency", "duration": BENCHMARK_SECONDS, "concurrency": 32, "grace_period": 30}.items()):
        raise ValueError("expected one un-warmed concurrency32 measurement phase")
    expected_dataset = {"name": "main", "type": "synthetic", "entries": 12800, "random_seed": 2026, "prompts": {"isl": {"mean": 4000.0, "stddev": 0.0}, "osl": {"mean": 200.0, "stddev": 0.0}, "cache_bust": {"target": "first_turn_prefix"}}}
    if config["datasets"] != [expected_dataset] or config["tokenizer"]["name"] != TOKENIZER:
        raise ValueError("exported prompt generation differs from the official workload")
    expected_extra = {"min_tokens": 200, "max_tokens": 200, "ignore_eos": True, "temperature": 0.0, "repetition_penalty": 1.0}
    if config["endpoint"]["extra"] != expected_extra:
        raise ValueError("exported sampling parameters differ from the official workload")
    return int(count)


def collect(info: dict, server_log: str, export: dict, startup_s: float) -> dict:
    expected = {"speculative_algorithm": "EAGLE3", "speculative_num_steps": 3, "speculative_eagle_topk": 1, "speculative_num_draft_tokens": 4, "tp_size": 4, "dp_size": 1, "enable_dp_attention": False, "disaggregation_mode": "null", "moe_a2a_backend": "none", "dtype": "auto", "context_length": 8192, "kv_cache_dtype": "fp8_e4m3"}
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f"unexpected server setting {key}={info.get(key)!r}, expected {value!r}")
    states = acceptance_states(info, Path("server-info.json"), "huggingface")
    if len(states) != len(info["internal_states"]):
        raise ValueError("every internal state must report a completed acceptance average")
    lengths = [state["accept_length_including_bonus"] for state in states]
    if any(not 1 <= length <= 4 for length in lengths):
        raise ValueError("acceptance length outside the steps3/draft4 range")
    samples = [{"accept_length_including_bonus": float(match["length"]), "accept_rate": float(match["rate"])} for match in ACCEPTANCE_LOG.finditer(server_log.replace("\x1b", ""))]
    stats = summary(lengths)
    return {"schema": "genkai-eagle-accept/v1", "scope": "single TP4 server, no PD; acceptance only, not an official score", "method": "GET /get_server_info internal_states[].avg_spec_accept_length; unweighted state mean including bonus", "states": states, "accept_length": stats["mean"], "accepted_draft": stats["mean"] - 1, "requests": completed_requests(export), "startup_s": startup_s, "benchmark_duration_s": BENCHMARK_SECONDS, "log_samples": {"accept_length": summary([sample["accept_length_including_bonus"] for sample in samples]), "accept_rate": summary([sample["accept_rate"] for sample in samples])}}


def position_acceptance(log: str) -> dict:
    hist = [0] * 4
    for match in re.finditer(r"EAGLE_POSITION histogram=(\[[0-9, ]+\])", log):
        sample = json.loads(match[1])
        if len(sample) != 4 or any(type(n) is not int or n < 0 for n in sample):
            raise ValueError("invalid positional acceptance histogram")
        hist = [a + b for a, b in zip(hist, sample)]
    total = sum(hist)
    if not total:
        raise ValueError("no positional acceptance evidence in server log")
    return {"correct_drafts_histogram": hist, "verification_requests": total, "unconditional_prefix_acceptance": [sum(hist[n:]) / total for n in (1, 2, 3)], "method": "P(correct drafts >= position), including unfinished requests; statistics rank only; CPU settlement before output truncation"}


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


def stop(child: subprocess.Popen | None) -> None:
    if child is None:
        return
    # Also kill descendants after the group leader exits (AIPerf ignores TERM).
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=2)


def interrupted(signum: int, _frame: object) -> None:
    raise RuntimeError(f"probe interrupted by signal {signum}")


def get_json(path: str) -> dict:
    with urllib.request.urlopen(URL + path, timeout=3) as response:
        return json.load(response)


def execute(root: Path, started_at: float, firmus: bool = False) -> dict:
    origin = time.monotonic() - (time.time() - started_at)

    def elapsed() -> float:
        return time.monotonic() - origin

    startup_limit = 600 if firmus else 450
    deadline = 945 if firmus else 555
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
    results = root / "results"
    server_path = root / "logs/server.log"
    client_path = root / "logs/aiperf.log"
    server = client = None
    try:
        command = server_command(root)
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
                with urllib.request.urlopen(URL + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            print(f"waiting_health elapsed_s={elapsed():.1f}", flush=True)
            time.sleep(2)
        startup_s = elapsed()
        print(f"server_ready startup_s={startup_s:.3f} remaining_until_cleanup_s={deadline - elapsed():.1f}", flush=True)
        synthetic_offset = 0
        if firmus:
            with urllib.request.urlopen(URL + "/flush_cache", timeout=5) as response:
                if response.status != 200:
                    raise ValueError("initial cache/statistics reset failed")
            synthetic_offset = server_path.stat().st_size
        command = aiperf_command()
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
        info = get_json("/get_server_info")
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
        result = collect(info, synthetic_log, export, startup_s)
        if firmus:
            (root / "logs/synthetic-server.log").write_text(synthetic_log)
            result["positions"] = position_acceptance(synthetic_log)
            result["schema"] = "firmus-eagle-accept/v1"
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


if __name__ == "__main__":
    result = execute(Path(sys.argv[1]), float(sys.argv[2]), firmus="--firmus" in sys.argv[3:])
    if "english" in result:
        print("PASS workload=english accept_length={accept_length:.6f} requests={requests}".format(**result["english"]), flush=True)
    # run.pjm tees this into results/trial-1.log. Never write the same file twice.
    print("PASS accept_length={accept_length:.6f} accepted_draft={accepted_draft:.6f} requests={requests} startup_s={startup_s:.3f}".format(**result), flush=True)
