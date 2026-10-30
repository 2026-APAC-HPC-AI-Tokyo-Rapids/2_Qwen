import json
import re
import runpy
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "patches/deep-gemm-bench/prefill-combine-norm/sm90-prefill-combine-norm.patch"
HOST = ROOT / "patches/deep-gemm-host/sm90-prefill-combine-norm.patch"
SGLANG = ROOT / "patches/sglang/prefill-combine-norm.patch"


@pytest.fixture
def selector():
    return runpy.run_path(str(ROOT / "scripts/select-firmus-prefill-megamoe.py"))


def added(patch):
    return "\n".join(line[1:] for line in patch.read_text().splitlines() if line.startswith("+") and not line.startswith("+++"))


def write_results(selector, directory, variant="four-stage-lean-split", control=2560.0, fused=2525.0, workers=3, **changes):
    for worker in range(workers):
        path = directory / f"prefill-{worker}"
        path.mkdir(parents=True, exist_ok=True)
        for rank in range(4):
            routings = {kind: {"residual_bitwise_equal": True, "finite": True, "q_mismatch_fraction": 1e-6, "scale_max_relative": 0.0, "relative_l2": 1e-6, "passed": True, "us": {"control": [control] * selector["ROUNDS"], "fused": [fused] * selector["ROUNDS"]}} for kind in selector["ROUTINGS"] + selector["EXACT_ROUTINGS"]}
            entry = {"rank": rank, "variant": variant, "needed": True, "calls": selector["COMBINE_NORM_CALLS"], "stable": {k: True for k in routings}, "passed": True, "routings": routings, **changes}
            (path / f"rank-{rank}.json").write_text(json.dumps(entry))


def aggregate(selector, tmp_path, winner="four-stage-lean-split"):
    selection, out, env = tmp_path / "selection.json", tmp_path / "combine-norm.json", tmp_path / "prefill-kernels.env"
    selection.write_text(json.dumps({"winner": winner}))
    env.write_text("DG_SM90_PREFILL_4STAGE=1\n")
    selector["aggregate_combine_norm"](tmp_path / "results", selection, out, env, 3)
    return json.loads(out.read_text()), env.read_text()


def test_adopted_when_every_gpu_passed_and_the_sum_saved_the_margin(selector, tmp_path):
    write_results(selector, tmp_path / "results")
    result, env = aggregate(selector, tmp_path)
    assert result["adopted"] and result["reason"].startswith("saved 840.0 us over 24 cases")
    assert env == "DG_SM90_PREFILL_4STAGE=1\nSGLANG_PREFILL_COMBINE_NORM=1\n"
    assert set(selector["COMBINE_NORM_VARIANTS"]) == {"four-stage-lean", "four-stage-lean-pair", "four-stage-lean-split", "four-stage-lean-split-ldsm"}
    assert selector["COMBINE_NORM_CALLS"] >= 600


@pytest.mark.parametrize("change,why", [
    ({"passed": False}, "rank 0 failed"),
    ({"calls": 0}, "rank 0 failed"),
    ({"variant": "four-stage-lean"}, "rank 0 gated four-stage-lean, not four-stage-lean-split"),
])
def test_one_failing_gpu_keeps_the_separate_norm(selector, tmp_path, change, why):
    write_results(selector, tmp_path / "results")
    path = tmp_path / "results/prefill-0/rank-0.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    result, env = aggregate(selector, tmp_path)
    assert not result["adopted"] and result["reason"].startswith(why)
    assert "SGLANG_PREFILL_COMBINE_NORM" not in env


def test_missing_gpu_slow_fused_call_or_non_lean_winner_keep_the_separate_norm(selector, tmp_path):
    write_results(selector, tmp_path / "results")
    (tmp_path / "results/prefill-2/rank-3.json").unlink()
    assert aggregate(selector, tmp_path)[0]["reason"] == "missing combine-norm results (late, hung or failed)"
    write_results(selector, tmp_path / "results", fused=2550.0)
    # 10 us saved per case is below max(5 us, 0.5% of 2560 us) = 12.8 us.
    result, env = aggregate(selector, tmp_path)
    assert not result["adopted"] and result["reason"].startswith("below-margin") and "SGLANG_PREFILL_COMBINE_NORM" not in env
    write_results(selector, tmp_path / "results")
    assert aggregate(selector, tmp_path, winner="four-stage")[0]["reason"] == "four-stage is not on the lean path"


def test_gate_skips_the_gpu_when_the_winner_is_not_on_the_lean_path(selector, tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"winner": "control"}))
    monkeypatch.setenv("LOCAL_RANK", "1")
    selector["run_combine_norm"](selection, tmp_path / "out")
    entry = json.loads((tmp_path / "out/rank-1.json").read_text())
    assert entry["variant"] == "control" and not entry["needed"] and not entry["passed"] and entry["calls"] == 0


def test_pp4_job_gates_the_combine_norm_after_the_megamoe_aggregate_and_before_the_workers(tmp_path):
    shell = (ROOT / "platforms/firmus/framework-comparison/sglang.sbatch").read_text()
    pp4 = shell[shell.index('if [[ -s "${ROOT}/source/pp4-kernel-settings.json" ]]; then'):shell.index('elif [[ -s "${ROOT}/source/reused-kernel-selection.json" ]]; then')]
    gate = next(line for line in pp4.splitlines() if "select-firmus-prefill-megamoe.py\" --combine-norm" in line)
    assert "timeout --kill-after=5 90 " in gate and '--output "${ROOT}/results/prefill-combine-norm/prefill-${worker}"' in gate
    assert '--combine-norm "${ROOT}/results/prefill-megamoe/selection.json"' in gate
    final = next(line for line in pp4.splitlines() if "--aggregate-combine-norm" in line)
    assert '--env-output "${ROOT}/source/prefill-kernels.env"' in final
    assert pp4.index('--aggregate "${ROOT}/results/prefill-megamoe"') < pp4.index(gate) < pp4.index(final)
    assert shell.index("--aggregate-combine-norm") < shell.index('bash "${ROOT}/source/worker.sh" prefill-0')
    script = tmp_path / "pp4.sh"
    script.write_text(pp4 + "fi\n")
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_patches_are_in_the_pp4_chain_after_the_ldmatrix_split():
    config = json.loads((ROOT / "configs/firmus-prefill-pp4.json").read_text())
    dg = config["deep_gemm_overlay_patches"]
    assert dg[-3:] == ["patches/deep-gemm-bench/prefill-split-ldsm/sm90-prefill-split-ldsm.patch", "patches/deep-gemm-bench/prefill-combine-norm/sm90-prefill-combine-norm.patch", "patches/deep-gemm-bench/decode-dense-bm192/sm90-1d2d-decode-bm192.patch"]
    assert config["sglang_overlay_patches"][-4:-2] == ["patches/sglang/decode-moe-input.patch", "patches/sglang/prefill-combine-norm.patch"]
    series = (ROOT / "patches/deep-gemm-host/series").read_text().split()
    assert series[-3:] == ["sm90-prefill-split-ldsm.patch", "sm90-prefill-combine-norm.patch", "sm90-1d2d-decode-bm192.patch"]
    main = json.loads((ROOT / "configs/firmus-framework.json").read_text())
    assert "patches/sglang/prefill-combine-norm.patch" not in main["sglang_overlay_patches"]


def test_host_appends_the_norm_argument_and_flag_only_on_the_lean_path():
    host = added(HOST)
    # All four lean flags are emitted explicitly, then kPrefillCombineNorm=true.
    assert 'args.prefill_combine_norm ? fmt::format(",\\n        true,\\n        {},\\n        {},\\n        {},\\n        true", args.prefill_tail_pair, args.prefill_split_accum, args.prefill_split_ldsm) :' in host
    assert "DG_HOST_ASSERT(not prefill_norm.has_value() or (prefill_lean and hidden == 4096 and num_topk == 8));" in host
    assert "            args.prefill_norm\n" in host and "TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_mega_moe_norm, dg_fp8_mega_moe_norm);" in host
    assert "scale_t.stride(0) == 1 and scale_t.stride(1) >= num_tokens and scale_t.stride(1) % 4 == 0" in host
    # The host and kernel argument structs have the same fields in the same order.
    fields = re.compile(r"struct SM90PrefillCombineNorm \{(.*?)\};", re.DOTALL)
    strip = lambda body: [re.sub(r"\s*//.*", "", line).strip() for line in body.strip().splitlines()]
    assert strip(fields.search(host).group(1)) == strip(fields.search(added(HEADER)).group(1)) == ["const void* weight;", "void* q;", "float* scale;", "int64_t scale_group_stride;", "float eps;"]


def test_kernel_keeps_the_separate_norm_arithmetic():
    kernel = added(HEADER)
    assert "    bool kPrefillCombineNorm = false,\n" in kernel
    assert "                       const SM90PrefillCombineNorm prefill_norm) {" in kernel
    assert "DG_STATIC_ASSERT(not kPrefillCombineNorm or (kPrefillLean and kHidden == 4096 and kNumTopk == 8)," in kernel
    # y rounded to BF16 as before, then FP32 h = y + residual; BF16 h back to the residual; unrounded h normalized.
    for text in ("h[2 * l] = __fadd_rn(y_values.x, r_values.x);", "casted_bf16[l] = __float22bfloat162_rn(make_float2(h[2 * l], h[2 * l + 1]));",
                 'asm("rsqrt.approx.f32 %0, %1;" : "=f"(rstd) : "f"(__fadd_rn(__fmul_rn(sum_squares, 1.0f / kHidden), prefill_norm.eps)));',
                 "__fmul_rn(__fmul_rn(h[2 * l], rstd), w.x)", "amax = fmaxf(amax, 1e-10f);", "const float gain = __fdiv_rn(448.0f, amax);",
                 "fminf(normed[2 * l] * gain, 448.0f)", "amax * (1.0f / 448.0f)"):
        assert text in kernel, text
    # No trap in the math role (the split instantiation needs its setmaxnreg budget).
    assert "TRAP" not in kernel and "DG_DEVICE_ASSERT" not in kernel


def test_lane_layout_covers_each_group_with_sixteen_lanes_and_each_row_slot_once():
    kernel = added(HEADER)
    assert "const uint32_t column = chunk * (kNumChunkBytes / sizeof(nv_bfloat16)) + (j * 32 + lane_idx) * 8;" in kernel
    assert kernel.count("const uint32_t row_idx = ((chunk * kNumUint4PerLane + j) * 2) * 32 + lane_idx;") == 2
    assert "for (uint32_t offset = 8; offset >= 1; offset >>= 1)" in kernel and "for (uint32_t offset = 16; offset >= 1; offset >>= 1)" in kernel
    chunks, per_lane, hidden = 4, 4, 4096
    slots, columns = set(), set()
    for chunk in range(chunks):
        for j in range(per_lane):
            for lane in range(32):
                column = chunk * 1024 + (j * 32 + lane) * 8
                columns.update(range(column, column + 8))
                row_idx = ((chunk * per_lane + j) * 2) * 32 + lane
                slots.update((row_idx, row_idx + 32))
                # The XOR partners 8, 4, 2, 1 stay inside the 16 lanes of the same 128-column group.
                for offset in (8, 4, 2, 1):
                    partner = lane ^ offset
                    assert (chunk * 1024 + (j * 32 + partner) * 8) // 128 == column // 128
    assert columns == set(range(hidden)) and slots == set(range(hidden // 4))


def test_only_the_sum_order_differs_and_stays_inside_the_gate_bounds(selector):
    # CPU model of the two paths on random rows: the separate norm sums h * h in one order,
    # the combine per lane over (chunk, j, element) with FMA and then a 32-lane XOR tree.
    np = pytest.importorskip("numpy")

    def bf16(x):
        bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
        return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)


    def fp8_e4m3(x):
        x = np.minimum(np.asarray(x, dtype=np.float64), 448.0)
        exponent = np.maximum(np.floor(np.log2(np.maximum(np.abs(x), 2.0 ** -20))), -6)
        step = 2.0 ** (exponent - 3)
        return np.clip(np.rint(x / step) * step, -448.0, 448.0)


    def norm_quant(h, weight, sum_squares, eps=1e-6):
        rstd = (1.0 / np.sqrt((sum_squares / np.float32(4096) + np.float32(eps)).astype(np.float32))).astype(np.float32)
        normed = bf16((h * rstd[:, None]).astype(np.float32) * weight)
        groups = normed.reshape(h.shape[0], 32, 128)
        amax = np.maximum(np.abs(groups).max(-1), np.float32(1e-10)).astype(np.float32)
        gain = (np.float32(448.0) / amax).astype(np.float32)
        q = fp8_e4m3((groups * gain[..., None]).astype(np.float32))
        return q, (amax * np.float32(1.0 / 448.0)).astype(np.float32)

    rng = np.random.default_rng(1030)
    rows = 256
    y = bf16(rng.standard_normal((rows, 4096)).astype(np.float32) * 0.5)
    residual = bf16(rng.standard_normal((rows, 4096)).astype(np.float32) * 2.0)
    weight = bf16((1.0 + 0.1 * rng.standard_normal(4096)).astype(np.float32))
    h = (y + residual).astype(np.float32)
    control_sum = np.add.reduce((h * h).astype(np.float32), axis=1, dtype=np.float32)
    lanes = np.zeros((rows, 32), dtype=np.float32)
    for chunk in range(4):
        for j in range(4):
            for e in range(8):
                v = h[:, chunk * 1024 + (np.arange(32) + j * 32) * 8 + e].astype(np.float64)
                lanes = (v * v + lanes.astype(np.float64)).astype(np.float32)
    for offset in (16, 8, 4, 2, 1):
        lanes = (lanes + lanes[:, np.arange(32) ^ offset]).astype(np.float32)
    assert (lanes == lanes[:, :1]).all()
    q_c, s_c = norm_quant(h, weight, control_sum)
    q_f, s_f = norm_quant(h, weight, lanes[:, 0])
    # Residual BF16(h) is the same expression in both paths; (q, scale) differ only through rstd.
    assert (q_c != q_f).mean() <= selector["COMBINE_NORM_MAX_Q_MISMATCH"]
    assert (np.abs(s_f - s_c) / s_c).max() <= selector["COMBINE_NORM_MAX_SCALE_RELATIVE"]
    dequant_c, dequant_f = q_c * np.repeat(s_c, 128, 1).reshape(rows, 32, 128), q_f * np.repeat(s_f, 128, 1).reshape(rows, 32, 128)
    assert np.linalg.norm(dequant_f - dequant_c) / np.linalg.norm(dequant_c) <= selector["COMBINE_NORM_MAX_RELATIVE_L2"]


def test_sglang_layer_hands_the_next_layer_its_qkv_input_only_inside_a_stage():
    patch = added(SGLANG)
    assert '_PREFILL_COMBINE_NORM = os.environ.get("SGLANG_PREFILL_COMBINE_NORM", "0") == "1"' in patch
    # Only the onepass Prefill input, never for an EAGLE capture layer or when the next layer would not norm/quant itself.
    for text in ('or moe_input != "onepass"', 'or getattr(nxt, "_is_layer_to_capture", False)', "is not CommunicateSummableTensorPairFn._trivial",
                 "if not nxt._use_input_norm_quant(residual, residual, forward_batch, None, kwargs):", "return nxt.input_layernorm"):
        assert text in patch, text
    # The next layer skips its own norm for a (q, scale) tuple; the last layer of a stage has no next layer.
    assert "        if isinstance(hidden_states, tuple):\n" in patch
    assert "for i in range(self.start_layer, self.end_layer - 1):" in patch
    assert 'object.__setattr__(self.layers[i], "_next_layer", self.layers[i + 1])' in patch
    # The scales have add_rmsnorm_group_quant_fp8's column-major, TMA-aligned layout.
    assert "scale = torch.empty((residual.shape[1] // 128, (rows + 3) // 4 * 4), dtype=torch.float32, device=residual.device).transpose(0, 1)[:rows]" in patch
    assert "deep_gemm._C.fp8_mega_moe_norm(" in patch
