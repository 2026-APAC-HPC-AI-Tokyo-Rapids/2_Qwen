"""Execute the pinned metadata, dispatch and loader methods on CPU tensors.

Usage: uv run --with torch python check_placement.py /path/to/patched/sglang
Imports of GPU runtime modules are replaced, not the methods under test.
"""
import __future__

import ast
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS

import torch
import torch.nn.functional as F

torch.set_num_threads(2)
source = Path(sys.argv[1]) / "python/sglang/srt"
root = Path(__file__).resolve().parents[3]
placement = json.loads((root / "configs/firmus-prefill-expert-location.json").read_text())["physical_to_logical_map"]
assert len(placement) == 94 and all(sorted(row) == list(range(128)) for row in placement)
namespace = {"__name__": __name__, "torch": torch, "F": F, "dataclass": dataclass, "random": random, "ModelConfigForExpertLocation": NS(from_model_config=lambda _: NS(num_layers=94, num_logical_experts=128)), "_prefer_same_node_experts": lambda _: True}


def load(path, names):
    tree = ast.parse(path.read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)  # noqa: S102


load(source / "eplb/expert_location.py", {"ExpertLocationMetadata", "_compute_logical_to_all_physical_map", "_pad_nested_array", "compute_logical_to_rank_dispatch_physical_map", "_logical_to_all_physical_raw", "_find_nearest_expert", "_compute_gpu_id_of_physical_expert", "_compute_node_id_of_physical_expert", "_fair_choices"})
load(source / "eplb/expert_location_dispatch.py", {"_topk_ids_logical_to_physical_static"})
load(source / "layers/moe/fused_moe_triton/layer.py", {"weight_loader", "_weight_loader_physical", "_map_global_expert_id_to_local_expert_id"})
namespace["KTEPWrapperMethod"] = type("KTEPWrapperMethod", (), {})
args = NS(device="cpu", ep_size=4, nnodes=1, ep_num_redundant_experts=0, elastic_ep_initial_size=None, moe_a2a_backend="megamoe", ep_dispatch_algorithm="static")
physical_to_logical = torch.tensor(placement)
logical_ids = torch.arange(128, dtype=torch.int32).reshape(16, 8)
checks = 0
for rank in range(4):
    metadata = namespace["ExpertLocationMetadata"].init_by_mapping(args, None, placement, moe_ep_rank=rank)
    namespace["get_global_expert_location_metadata"] = lambda metadata=metadata: metadata
    layer = NS(quant_config=None, quant_method=None, _has_fused_shared=False, num_fused_shared_experts=0, num_local_experts=32, _expert_storage_rank=rank, _num_local_routed=32)
    for name in ("weight_loader", "_weight_loader_physical", "_map_global_expert_id_to_local_expert_id"):
        setattr(layer, name, MethodType(namespace[name], layer))
    for layer_id in range(94):
        layer.layer_id = layer_id
        info = NS(partial_logical_to_rank_dispatch_physical_map=metadata.logical_to_rank_dispatch_physical_map[layer_id])
        ids = namespace["_topk_ids_logical_to_physical_static"](logical_ids, info)
        assert ids.dtype == logical_ids.dtype
        assert torch.equal(physical_to_logical[layer_id, ids], logical_ids)
        # Probe all checkpoint weights and scales through the real logical ->
        # physical -> local loader, retaining distinct values for each expert.
        for name in ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv"):
            loaded = {}
            layer._weight_loader_impl = lambda loaded=loaded, **kw: loaded.__setitem__(kw["expert_id"], kw["loaded_weight"])
            for logical in range(128):
                layer.weight_loader(NS(), logical, name, "w1" if name.startswith("w13") else "w2", logical)
            assert sorted(loaded) == list(range(32))
            assert [loaded[i] for i in range(32)] == placement[layer_id][rank * 32:(rank + 1) * 32]
            checks += 1
print(f"PASS: 94 layer permutations; all EP4 dispatch tables; {checks} weight/scale loading checks; no replicas or missing experts")
