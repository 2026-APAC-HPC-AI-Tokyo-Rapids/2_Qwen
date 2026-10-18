"""CPU parity and latency gate against the pinned Qwen tokenizer.

Run with gigatoken 0.10.0, transformers 5.12.1 and tokenizers 0.22.2.
Arguments: patched SGLang source directory, local pinned tokenizer directory.
This never loads model weights or modifies the official AIPerf client.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import random
import statistics
import time
from importlib.metadata import version
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument(
        "--adapter-only",
        action="store_true",
        help="Test the new adapter extracted from the patch, without the private SGLang loader source",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        help="Archived AIPerf inputs.json for complete workload parity",
    )
    args = parser.parse_args()
    assert (
        hashlib.sha256((args.tokenizer / "tokenizer.json").read_bytes()).hexdigest()
        == "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    )
    path = args.source / "python/sglang/srt/utils/hf_transformers/gigatoken_encoder.py"
    spec = importlib.util.spec_from_file_location("gigatoken_encoder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hf = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    accelerated = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    accelerated.encode = module.GigatokenEncoder(accelerated)
    rng = random.Random(20261018)
    atoms = [
        "東京",
        "hello",
        " world",
        "\n",
        "\r\n",
        "\t",
        "  ",
        "1234567",
        "é",
        "e\u0301",
        "👩🏽‍💻",
        "عربي",
        "漢字",
        "\x00",
        "[]{}:",
        "can't",
        "ประเทศไทย",
        "한국어",
        "a\u200db",
        "<|im_start|>",
        "<|im_end|>",
    ]
    cases = ["", " ", "\n\n", *atoms, *hf.all_special_tokens]
    cases += ["".join(rng.choices(atoms, k=rng.randrange(1, 150))) for _ in range(1024)]
    # Unique long inputs prevent an identical whole-prompt cache from inflating the result.
    words = [
        "model",
        "compute",
        "cluster",
        "memory",
        "attention",
        "bandwidth",
        "request",
        "latency",
        "decode",
        "prefill",
        "GPU",
        "token",
        "network",
        "performance",
        "optimizer",
        "parallel",
        "sequence",
        "matrix",
        "vector",
    ]
    long_cases = [" ".join(rng.choices(words, k=3950)) for _ in range(48)]
    cases += long_cases
    for text in cases:
        for specials in (False, True):
            assert accelerated.encode(text, add_special_tokens=specials) == hf.encode(
                text, add_special_tokens=specials
            ), repr(text[:80])
    messages = [
        {"role": "system", "content": "Answer precisely."},
        {"role": "user", "content": "Explain 日本語 and 😀."},
    ]
    for tokenize in (True, False):
        assert accelerated.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=True
        ) == hf.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=True
        )
    # The adapter intentionally preserves all non-plain-encode HF APIs.
    assert accelerated.encode("a b c d", truncation=True, max_length=3) == hf.encode(
        "a b c d", truncation=True, max_length=3
    )
    assert accelerated.encode("a", text_pair="b") == hf.encode("a", text_pair="b")
    assert accelerated(["a", "longer b"], padding=True) == hf(
        ["a", "longer b"], padding=True
    )
    assert accelerated.all_special_ids == hf.all_special_ids
    ids = [hf.encode(x) for x in atoms]
    for skip in (True, False):
        assert accelerated.batch_decode(
            ids, skip_special_tokens=skip
        ) == hf.batch_decode(ids, skip_special_tokens=skip)
    # Static loader wiring check; this does not import SGLang's GPU dependencies.
    if not args.adapter_only:
        tree = ast.parse(
            (
                args.source / "python/sglang/srt/utils/hf_transformers/tokenizer.py"
            ).read_text()
        )
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "get_tokenizer"
        )
        assert any(
            isinstance(n, ast.Compare)
            and any(
                isinstance(x, ast.Constant) and x.value == "gigatoken"
                for x in n.comparators
            )
            for n in ast.walk(fn)
        )
    official = []
    if args.inputs:
        for session in json.loads(args.inputs.read_text())["data"]:
            for payload in session["payloads"]:
                text = hf.apply_chat_template(
                    payload["messages"], tokenize=False, add_generation_prompt=True
                )
                assert accelerated.encode(text) == hf.encode(text), session[
                    "session_id"
                ]
                official.append(text)
    result = {
        "parity_cases": len(cases) * 2,
        "archived_chat_prompts_matched": len(official),
        "template_and_fallback_checks": "passed",
        "loader_static_check": "not run (adapter only)" if args.adapter_only else "passed",
        "scope": "Local CPU encode latency; not a Firmus throughput estimate.",
        "versions": {},
    }
    for package in ("gigatoken", "transformers", "tokenizers"):
        result["versions"][package] = version(package)
    if official:
        long_cases = official[:: max(1, len(official) // 128)][:128]
    result["benchmark_prompts"] = len(long_cases)
    result["input_tokens"] = statistics.mean(len(hf.encode(s)) for s in long_cases)
    for name, tokenizer in (("huggingface", hf), ("gigatoken", accelerated)):
        samples = []
        for _ in range(3):
            for text in long_cases:
                start = time.perf_counter_ns()
                tokenizer.encode(text)
                samples.append((time.perf_counter_ns() - start) / 1e6)
        result[name] = {
            "median_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[int(0.95 * len(samples))],
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
