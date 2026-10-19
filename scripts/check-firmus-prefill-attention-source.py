"""CPU-only routing contract for the Prefill attention patch; no torch import."""
import ast
import copy
import sys
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace as NS

root = Path(sys.argv[1]) / "python/sglang"


class ForwardMode(Enum):
    EXTEND = auto()
    TARGET_VERIFY = auto()
    DECODE = auto()
    IDLE = auto()
    DRAFT_EXTEND = auto()
    MIXED = auto()

    def is_target_verify(self):
        return self == ForwardMode.TARGET_VERIFY


helper = ast.parse((root / "srt/layers/attention/prefill_contiguous.py").read_text())
helper.body = [n for n in helper.body if isinstance(n, ast.FunctionDef)]
ns = {"ForwardMode": ForwardMode}
exec(compile(helper, "prefill_contiguous.py", "exec"), ns)
select = ns["use_contiguous_prefill"]


def batch(mode=ForwardMode.EXTEND, prefixes=(0, 0), lengths=(17, 13)):
    return NS(forward_mode=mode, extend_seq_lens_cpu=lengths, extend_prefix_lens_cpu=prefixes, encoder_lens=None, mha_return_lse=False, global_num_tokens_cpu=[128, 4096, 0, 4096])


for mode in ForwardMode:
    for prefixes in ((0, 0), (0, 2), (3, 0), None):
        for full in (False, True):
            assert select(batch(mode, prefixes), full_graph=full) == (mode == ForwardMode.EXTEND and prefixes == (0, 0) and not full)
assert not select(batch(lengths=(), prefixes=()))
assert not select(batch(lengths=(0, 0)))
assert not select(batch(lengths=None))
b = batch()
b.encoder_lens = [3, 5]
assert not select(b)
b = batch()
b.mha_return_lse = True
assert not select(b)

backend = ast.parse((root / "srt/layers/attention/flashattention_backend.py").read_text())
cls = next(n for n in backend.body if isinstance(n, ast.ClassDef) and n.name == "FlashAttentionBackend")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_extend")
branch = next(n for n in method.body if isinstance(n, ast.If) and "use_contiguous_prefill" in ast.unparse(n.test))
assert "self.flash_attn_varlen_func" in ast.unparse(branch)
assert "page_table=" not in ast.unparse(branch)
assert "metadata.cu_seqlens_q" in ast.unparse(branch)
# Execute the actual production branch with tensor-shaped CPU stubs, including
# the DP padding slice. Any failed condition falls through to the paged marker.
function = ast.parse("def route():\n    return 'paged'\n").body[0]
function.body.insert(0, copy.deepcopy(branch))
module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))


class Tensor:
    def __init__(self, rows, dtype="fp8"):
        self.shape = (rows, 64, 128)
        self.dtype = dtype
        self.device = "cuda"

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return True

    def split(self, sizes, dim):
        result = [Tensor(self.shape[0], self.dtype) for _ in sizes]
        for tensor, size in zip(result, sizes):
            tensor.shape = (self.shape[0], size // 128, 128)
        return result

    def view(self, *shape):
        return self

    def __getitem__(self, key):
        return Tensor(key.stop if key.stop is not None else self.shape[0] - (key.start or 0), self.dtype)

    def new_zeros(self, shape, dtype):
        return Tensor(shape[0], dtype)

    def zero_(self):
        return self

    def copy_(self, value):
        return self


calls = []


def fa(**kw):
    calls.append(kw)
    assert kw["q"].shape[0] == kw["k"].shape[0] == kw["v"].shape[0] == 30
    assert kw["cu_seqlens_q"] is kw["cu_seqlens_k"]
    assert kw["causal"] is True
    return Tensor(30, "bf16")


def environment():
    metadata = NS(cu_seqlens_q=object(), max_seq_len_q=17)
    return dict(self=NS(fa_impl_ver=3, use_mla=False, full_cg_prefill_metadata=None, attn_cp_size=1, num_splits=0, flash_attn_varlen_func=fa), metadata=metadata, use_contiguous_prefill=select, forward_batch=batch(), q=Tensor(37), k=Tensor(37), v=Tensor(37), save_kv_cache=False, is_cp_mode=False, use_local_attn=False, is_swa_layer=False, causal=True, torch=NS(float8_e4m3fn="fp8", bfloat16="bf16", empty=lambda shape, device, dtype: Tensor(shape[0], dtype)), layer=NS(is_cross_attention=False, k_scale=None, v_scale=None, tp_q_head_num=64, tp_k_head_num=4, tp_v_head_num=4, head_dim=128, v_head_dim=128, scaling=128 ** -.5, logit_cap=0.), window_size=(-1, -1), _fa_out=None, kwargs={})


def run(env):
    exec(compile(module, "forward_extend branch", "exec"), env)
    return env["route"]()


assert run(environment()).shape[0] == 37
for key, value in (("save_kv_cache", True), ("is_cp_mode", True), ("use_local_attn", True), ("is_swa_layer", True), ("causal", False)):
    env = environment()
    env[key] = value
    assert run(env) == "paged", key
for mode in ForwardMode:
    if mode != ForwardMode.EXTEND:
        env = environment()
        env["forward_batch"].forward_mode = mode
        assert run(env) == "paged", mode
for prefix in ((0, 1), (1, 0)):
    env = environment()
    env["forward_batch"].extend_prefix_lens_cpu = prefix
    assert run(env) == "paged"
for obj, attr, value in (("self", "fa_impl_ver", 4), ("self", "use_mla", True), ("self", "attn_cp_size", 2), ("layer", "k_scale", 1.), ("k", "dtype", "bf16")):
    env = environment()
    setattr(env[obj], attr, value)
    assert run(env) == "paged", (obj, attr)
env = environment()
env["self"].full_cg_prefill_metadata = env["metadata"]
assert run(env) == "paged"

model = (root / "srt/models/qwen3_moe.py").read_text()
assert "apply_qknorm_rope_fp8_prefill(*rope_args, *kv_args, contiguous_k, contiguous_v)" in model
assert 'full_graph=get_server_args().cuda_graph_config.prefill.backend == "full"' in model
assert "k, v = contiguous_k, contiguous_v" in model
assert "apply_qknorm_rope_fp8_kv(*rope_args, *kv_args)" in model
# Execute the producer too: returning FP8 K/V is the explicit signal consumed
# above, and the paged-write flag must suppress the second cache writer.
node = next(n for n in ast.walk(ast.parse(model)) if isinstance(n, ast.FunctionDef) and n.name == "apply_qk_norm_rope")
node = copy.deepcopy(node)
node.returns = None
for arg in node.args.args:
    arg.annotation = None
class Pool:
    def get_kv_buffer(self, layer):
        return caches
caches = [Tensor(256) for _ in range(2)]
for t in caches:
    t.shape = (256, 4, 128)
args = NS(cuda_graph_config=NS(prefill=NS(tc_compiler="eager", backend="disabled")), speculative_eagle_topk=1, enable_prefill_cp=False)
prologue_calls = []
env = dict(ForwardMode=ForwardMode, use_contiguous_prefill=select, get_server_args=lambda: args, get_token_to_kv_pool=lambda: pool, MHATokenToKVPool=Pool, envs=NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda: False)), torch=NS(bfloat16="bf16", float8_e4m3fn="fp8", empty=lambda shape, dtype, device: Tensor(shape[0], dtype), empty_like=lambda t: Tensor(t.shape[0], t.dtype)), apply_qknorm_rope_fp8_prefill=lambda *a: prologue_calls.append("contiguous"), apply_qknorm_rope_fp8_kv=lambda *a: prologue_calls.append("paged"))
pool = Pool()
exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "Qwen prologue", "exec"), env)
for mode, prefixes, expected in ((ForwardMode.EXTEND, (0, 0), "contiguous"), (ForwardMode.EXTEND, (0, 1), "paged"), (ForwardMode.TARGET_VERIFY, (0, 0), "paged")):
    fb = batch(mode, prefixes)
    fb.out_cache_loc = Tensor(37, "int64")
    norm = NS(weight=Tensor(128, "bf16"), variance_epsilon=1e-6)
    layer = NS(_stored_fp8_kv_last_call=True, use_fused_qk_norm_rope=False, use_fp8_q_rope=True, q_norm=norm, k_norm=norm, q_size=8192, kv_size=512, num_heads=64, num_kv_heads=4, head_dim=128, rotary_emb=NS(cos_sin_cache=None), attn=NS(layer_id=0, k_scale=None, v_scale=None))
    output = env["apply_qk_norm_rope"](layer, Tensor(37, "bf16"), Tensor(37, "int64"), fb)
    assert prologue_calls[-1] == expected
    assert layer._stored_fp8_kv_last_call
    assert output[0].dtype == "fp8"
    assert output[1].dtype == ("fp8" if expected == "contiguous" else "bf16")

kernel = (root / "kernels/jit/csrc/elementwise/rope.cuh").read_text()
assert "if (slot != 0)" in kernel and "slot < p.cache_slots" in kernel
print("PASS: CPU Prefill routing; no-prefix EXTEND, DP padding, prefix/continuation, full graph, CP, scales, dtype and all Decode modes")
