"""Keep expert parallelism off for dense drafts of an EP target in vLLM0.30."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ORIGINAL_SHA256 = "9c642a87264339c0dbc1d0b57330572149df0f9d5e16c6893b9895e50a619e19"
ORIGINAL = """                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config, self.draft_tensor_parallel_size
                    )
                )
"""
REPAIRED = (
    ORIGINAL
    + """                # Dense EAGLE drafts keep attention TP but have no expert group.
                if not self.draft_model_config.is_moe:
                    self.draft_parallel_config.enable_expert_parallel = False
"""
)


def repair(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("unexpected pinned vLLM speculative config source")
    text = source.decode("utf-8")
    if text.count(ORIGINAL) != 1:
        raise ValueError("dense-draft repair anchor is not unique")
    result = text.replace(ORIGINAL, REPAIRED).encode("utf-8")
    ast.parse(result)
    return result


def validate(source: bytes) -> None:
    """Exercise the patched native factory/call block without importing CUDA."""
    tree = ast.parse(source)
    factory = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "create_draft_parallel_config"
    )
    factory.decorator_list = []
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    namespace = {"ParallelConfig": lambda **values: SimpleNamespace(**values)}
    exec(  # noqa: S102 -- Selected native factory after source hash validation.
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[prefix, factory], type_ignores=[])
            ),
            "native-draft-factory",
            "exec",
        ),
        namespace,
    )
    selected = None
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for index, node in enumerate(body):
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Attribute) and t.attr == "draft_parallel_config"
                    for t in node.targets
                )
                and "create_draft_parallel_config" in ast.dump(node.value)
            ):
                selected = body[index : index + 2]
    if selected is None or len(selected) != 2 or not isinstance(selected[1], ast.If):
        raise ValueError("dense-draft EP correction missing at the native call site")
    code = compile(
        ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
        "native-draft-call",
        "exec",
    )
    for target_ep in [False, True]:
        for draft_moe in [False, True]:
            target = SimpleNamespace(
                enable_expert_parallel=target_ep,
                distributed_executor_backend="mp",
                max_parallel_loading_workers=None,
                disable_custom_all_reduce=False,
                ray_workers_use_nsight=False,
                placement_group=None,
            )
            state = SimpleNamespace(
                target_parallel_config=target,
                draft_tensor_parallel_size=4,
                draft_model_config=SimpleNamespace(is_moe=draft_moe),
            )
            scope = {
                "self": state,
                "SpeculativeConfig": SimpleNamespace(
                    create_draft_parallel_config=namespace[
                        "create_draft_parallel_config"
                    ]
                ),
            }
            exec(code, scope)  # noqa: S102
            draft = state.draft_parallel_config
            assert draft is not target and target.enable_expert_parallel == target_ep
            assert draft.tensor_parallel_size == 4 and draft.pipeline_parallel_size == 1
            assert draft.enable_expert_parallel == (target_ep and draft_moe)
            assert draft.disable_custom_all_reduce is False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    patched = repair(args.path.read_bytes())
    validate(patched)
    args.path.write_bytes(patched)
    print(
        json.dumps(
            {
                "repair": "vllm030-dense-draft-ep",
                "before_sha256": ORIGINAL_SHA256,
                "after_sha256": hashlib.sha256(patched).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
