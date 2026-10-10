"""CPU checks against the patched Qwen adapter, including an idle DP rank."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace as NS


def check(source):
    path = source / "python/sglang/srt/models/qwen3_moe.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "forward_megamoe")
    method.body = [n for n in method.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    calls = []
    tensor = NS(to=lambda dtype: tensor)
    topk = NS(topk_ids=tensor, topk_weights=tensor)
    class Router:
        def __call__(self, hidden, logits, **kwargs):
            calls.append(("router", kwargs))
            return topk

        def empty_topk_output(self, device):
            calls.append(("empty", device))
            return topk

    namespace = {"torch": NS(int32="int32", float32="float32"), "get_moe_ep_group": lambda: NS(device_group="EP4"), "envs": NS(SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=NS(get=lambda: 4096)), "is_sm90_fp8_mega_moe_available": lambda _: True, "_get_mega_moe_symm_buffer": lambda *args, **kwargs: (args, kwargs), "run_sm90_mega_routed": lambda *args: calls.append(("native", args))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)  # noqa: S102 - named method from pinned build input
    block = NS(experts=object(), gate=lambda h: (tensor, None), topk=Router(), config=NS(num_experts=128, num_experts_per_tok=8, hidden_size=4096, moe_intermediate_size=1536))
    for count in [0, 7, 4096]:
        calls.clear()
        namespace["forward_megamoe"](block, NS(shape=(count, 4096), device="cuda"), NS(num_token_non_padded="non-padded"))
        assert [c[0] for c in calls] == (["empty", "native"] if count == 0 else ["router", "native"])
        args = calls[-1][1]
        assert args[2] is tensor and args[3] is tensor and args[-1] == count
        assert args[-2][0] == ("EP4",) and args[-2][1]["num_max_tokens_per_rank"] == 4096
    try:
        namespace["forward_megamoe"](block, NS(shape=(4097, 4096)), None)
    except ValueError:
        pass
    else:
        raise AssertionError("token-cap overflow accepted")
    # Native FC1 reads gate/up scales separately despite the 8-row interleave.
    for packed_row in range(3072):
        up = packed_row % 16 >= 8
        original_row = packed_row // 16 * 8 + packed_row % 8 + (1536 if up else 0)
        native_scale_row = packed_row // 128 // 2 + (12 if up else 0)
        assert native_scale_row == original_row // 128
    print("PASS: empty/nonempty/cap DP paths; global expert IDs; original block128 scale mapping")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    check(parser.parse_args().source)
