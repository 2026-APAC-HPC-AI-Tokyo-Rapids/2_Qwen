"""Execute selected pinned DP/NIXL source contracts on CPU, without vLLM imports."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np

HASHES = {
    "v1/engine/core.py": "268bf40534a853812867ae8e80040f190983cb364b41182b55969aace20c93f5",
    "config/speculative.py": "ac877ed34977dc5226f52c0dd873963c20e9e37ff1782c7749221b50bc1bb533",
    "config/parallel.py": "f012ace89e3fb730a467f729f602679bc145c49f0d10721811c9d818570e7b2b",
    "distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py": "7a1db8c54e4cd846bdac747344f2aea2c2fcf507817928796063225223b7e169",
    "distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py": "2d43607f98701b2fad6cc0ac2767619a80b7fc2b3a3c2eb6ed0da4fb05dcd8f6",
    "distributed/kv_transfer/kv_connector/v1/nixl/metadata.py": "2b23a07c1ec5ac5490c3b818737c7192e344ea9f52cd5623f75939f741ab4e21",
    "v1/spec_decode/llm_base_proposer.py": "5bd11c41f0a8cf1272b5880d9878043f876bee6c7e04fb781e613f92930b15ab",
}


def execute(nodes, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, "pinned-dp4-contract", "exec"), namespace)  # noqa: S102


def check(sources):
    assert "torch" not in sys.modules and "vllm" not in sys.modules
    trees = {}
    for path, expected in HASHES.items():
        if hashlib.sha256(sources[path].encode()).hexdigest() != expected:
            raise ValueError("Unsupported pinned source: " + path)
        trees[path] = ast.parse(sources[path])
    ids, ports = [], []
    core = trees["v1/engine/core.py"]
    assignment = next(n for n in ast.walk(core) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "engine_id" for t in n.targets) and "dp_rank" in ast.dump(n.value))
    scheduler = trees["distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py"]
    port = next(n for n in ast.walk(scheduler) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "side_channel_port" for t in n.targets))
    for rank in range(4):
        config = SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="decoder"), parallel_config=SimpleNamespace(data_parallel_index=rank))
        state = SimpleNamespace()
        execute([assignment], {"vllm_config": config, "dp_rank": rank})
        execute([port], {"self": state, "vllm_config": config, "envs": SimpleNamespace(VLLM_NIXL_SIDE_CHANNEL_PORT=5700)})
        ids.append(config.kv_transfer_config.engine_id)
        ports.append(state.side_channel_port)
    assert ids == [f"decoder_dp{rank}" for rank in range(4)] and ports == [5700, 5701, 5702, 5703]

    class AttentionSpec:
        pass

    class MambaSpec:
        pass

    namespace = {"np": np, "dataclass": dataclass, "AttentionSpec": AttentionSpec, "MambaSpec": MambaSpec}
    mapping = trees["distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py"]
    execute([n for n in mapping.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))], namespace)
    topology = SimpleNamespace(tp_rank=0, tp_size=1, total_num_kv_heads=4, is_mla=False, dcp_consumer_count=lambda *_: 1)
    plan = namespace["compute_tp_mapping"](topology, 2, (AttentionSpec, AttentionSpec))
    assert plan.all_source_ranks == (0, 1) and plan.source_ranks_per_group == ((0, 1), (0, 1))
    assert plan.rank_to_attention_slot == {0: 0, 1: 1} and plan.rank_offset_factor == 0
    assert plan.local_consumers == 1

    factors_tree = trees["distributed/kv_transfer/kv_connector/v1/nixl/metadata.py"]
    factor = next(n for n in factors_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_get_speculative_compatibility_factors")
    execute([factor], namespace)
    draft = SimpleNamespace(model="/draft", revision=None, code_revision=None, hf_config=SimpleNamespace())
    spec = SimpleNamespace(use_eagle=lambda: True, draft_model_config=draft, kv_cache_dtype=None, method="eagle3", parallel_drafting=False)
    configs = [SimpleNamespace(speculative_config=spec, cache_config=SimpleNamespace(cache_dtype="fp8_e4m3"), parallel_config=SimpleNamespace(tensor_parallel_size=tp, data_parallel_size=dp)) for tp, dp in [(2, 1), (1, 4)]]
    factors = [namespace["_get_speculative_compatibility_factors"](c) for c in configs]
    assert factors[0] == factors[1]

    factory = next(n for n in ast.walk(trees["config/speculative.py"]) if isinstance(n, ast.FunctionDef) and n.name == "create_draft_parallel_config")
    factory.decorator_list = []
    namespace["ParallelConfig"] = lambda **kw: SimpleNamespace(data_parallel_size=1, **kw)
    execute([factory], namespace)
    target = SimpleNamespace(enable_expert_parallel=True, distributed_executor_backend="mp", max_parallel_loading_workers=None, disable_custom_all_reduce=False, ray_workers_use_nsight=False, placement_group=None)
    dense = namespace["create_draft_parallel_config"](target, 1)
    assert dense.tensor_parallel_size == 1
    # The existing pinned call-site repair disables EP for a dense draft.
    assert "if not self.draft_model_config.is_moe:" in sources["config/speculative.py"]
    assert "self.draft_parallel_config.enable_expert_parallel = False" in sources["config/speculative.py"]
    # Target DP coordination is explicitly used by the EAGLE proposer, independently of the draft's factory defaults.
    assert "coordinate_batch_across_dp(" in sources["v1/spec_decode/llm_base_proposer.py"]
    assert "self.vllm_config.parallel_config.data_parallel_size > 1" in sources["v1/spec_decode/llm_base_proposer.py"]
    assert "self.execute_dummy_batch()" in sources["v1/engine/core.py"]

    # Nominal target+draft FP8 attention payload only: excludes padding, grouping, allocator and graph overhead.
    baseline_bytes = 95 * 2 * 64 * 128 * 1 * 8192
    candidate_bytes = 95 * 2 * 64 * 128 * 4 * 2048
    assert baseline_bytes == candidate_bytes and 4 * 2048 * 64 == 524288
    assert 4 * 64 == 256 and 4 * 2048 == 8192
    return {"schema": "vllm-dp4-cpu-contract/v1", "passed": True, "cuda_imported": False, "source_sha256": HASHES, "engine_ids": ids, "side_channel_ports": ports, "producer_tp2_decoder_tp1_mapping": asdict(plan), "speculative_compatibility_factors": factors[0], "aggregate_limits": {"cache_tokens": 524288, "max_sequences": 256, "batch_tokens": 8192}, "nominal_target_draft_attention_bytes_per_gpu": candidate_bytes, "limitations": "AST-selected source contracts and nominal payload only; not distributed execution, memory fit, KV contents, graph safety, accuracy or throughput proof."}


if __name__ == "__main__":
    print(json.dumps(check(json.load(sys.stdin)), indent=2) + "\n")
