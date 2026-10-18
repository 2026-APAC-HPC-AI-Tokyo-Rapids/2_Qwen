"""CPU parity and latency check for the pinned Qwen tokenizer adapter.

This does not load model weights or claim a GPU throughput result.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("--adapter-only", action="store_true")
    args = parser.parse_args()
    assert hashlib.sha256((args.tokenizer / "tokenizer.json").read_bytes()).hexdigest() == "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    adapter_path = args.source / "python/sglang/srt/utils/hf_transformers/gigatoken_encoder.py"
    spec = importlib.util.spec_from_file_location("gigatoken_encoder", adapter_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hf = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    accelerated = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    accelerated.encode = module.GigatokenEncoder(accelerated)
    rng = random.Random(20261018)
    atoms = ["東京", "hello", " world", "\n", "\r\n", "\t", "  ", "1234567", "é", "e\u0301", "👩🏽‍💻", "عربي", "漢字", "\x00", "[]{}:", "can't", "ประเทศไทย", "한국어", "a\u200db", "<|im_start|>", "<|im_end|>"]
    cases = ["", " ", "\n\n", *atoms, *hf.all_special_tokens]
    cases += ["".join(rng.choices(atoms, k=rng.randrange(1, 150))) for _ in range(1024)]
    words = ["model", "compute", "cluster", "memory", "attention", "bandwidth", "request", "latency", "decode", "prefill", "GPU", "token", "network", "performance", "optimizer", "parallel", "sequence", "matrix", "vector"]
    cases += [" ".join(rng.choices(words, k=3950)) for _ in range(48)]
    for text in cases:
        for specials in (False, True):
            assert accelerated.encode(text, add_special_tokens=specials) == hf.encode(text, add_special_tokens=specials), repr(text[:80])
    messages = [{"role": "system", "content": "Answer precisely."}, {"role": "user", "content": "Explain 日本語 and 😀."}]
    for tokenize in (True, False):
        assert accelerated.apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=True) == hf.apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=True)
    assert accelerated.encode("a b c d", truncation=True, max_length=3) == hf.encode("a b c d", truncation=True, max_length=3)
    assert accelerated.encode("a", text_pair="b") == hf.encode("a", text_pair="b")
    assert accelerated(["a", "longer b"], padding=True) == hf(["a", "longer b"], padding=True)
    assert accelerated.all_special_ids == hf.all_special_ids
    ids = [hf.encode(x) for x in atoms]
    for skip in (True, False):
        assert accelerated.batch_decode(ids, skip_special_tokens=skip) == hf.batch_decode(ids, skip_special_tokens=skip)
    if not args.adapter_only:
        tree = ast.parse((args.source / "python/sglang/srt/utils/hf_transformers/tokenizer.py").read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_tokenizer")
        assert any(isinstance(node, ast.Compare) and any(isinstance(item, ast.Constant) and item.value == "gigatoken" for item in node.comparators) for node in ast.walk(fn))
    long_cases = cases[-48:]
    result = {"parity_cases": len(cases) * 2, "template_and_fallback_checks": "passed", "loader_static_check": "not run (adapter only)" if args.adapter_only else "passed", "versions": {package: version(package) for package in ("gigatoken", "transformers", "tokenizers")}}
    result["benchmark_prompts"] = len(long_cases)
    result["input_tokens"] = statistics.mean(len(hf.encode(text)) for text in long_cases)
    for name, tokenizer in (("huggingface", hf), ("gigatoken", accelerated)):
        samples = []
        for _ in range(3):
            for text in long_cases:
                start = time.perf_counter_ns()
                tokenizer.encode(text)
                samples.append((time.perf_counter_ns() - start) / 1e6)
        result[name] = {"median_ms": statistics.median(samples), "p95_ms": sorted(samples)[int(0.95 * len(samples))]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
