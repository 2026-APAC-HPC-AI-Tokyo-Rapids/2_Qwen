"""CPU checks of the real host's Prefill opt-in and descriptor/layout contract."""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = Path("deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--fmt-include", type=Path, required=True)
    parser.add_argument("--image-header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("host_contract", ROOT / "scripts/check-sm90-independent-wg-host.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    cases = []
    with tempfile.TemporaryDirectory(prefix="prefill-host-") as directory:
        temp = Path(directory)
        source = temp / "source"
        base.prepare_source(args.source_repo, source)
        cpp = temp / "host.cpp"
        cpp.write_text(base.host_translation_unit(source))
        subprocess.run(["g++", "-std=c++20", "-O0", "-I" + str(args.fmt_include), str(cpp), "-o", str(temp / "host")], check=True)
        control = base.decode_rows(subprocess.check_output([str(temp / "host")], env={**os.environ, "DG_SM90_PREFILL_INDEPENDENT_WG": "0"}, text=True))
        for mode in (None, "0", "a", "b"):
            env = {k: v for k, v in os.environ.items() if k != "DG_SM90_PREFILL_INDEPENDENT_WG"}
            if mode is not None:
                env["DG_SM90_PREFILL_INDEPENDENT_WG"] = mode
            rows = base.decode_rows(subprocess.check_output([str(temp / "host")], env=env, text=True))
            for original, row in zip(control, rows, strict=True):
                inputs, h = row["input"], row["host"]
                active = mode in ("a", "b") and inputs["experts"] in (128, 140) and inputs["prefill_2wg_env"] == 1 and original["host"]["block_m"] == 128
                if not active:
                    assert row == original, (mode, inputs)
                else:
                    assert h["stages"] == (4 if mode == "a" else 3) and h["smem"] == h["launch_smem"] == (208080 if mode == "a" else 199360)
                    assert h["launch_threads"] == 384 and h["math_threads"] == 256
                    assert h["reuse_accum"] == h["arrival_counter"] == 1 and h["swap_ab"] == h["epilogue_full_sync"] == 0
                    for key in ("l1_acts", "l2_acts"):
                        assert row["tma"][key]["box"] == [128, 64]
                    for key in ("l1_sf", "l2_sf"):
                        assert row["tma"][key]["box"] == [64, 1]
                    for key in ("l1_weights", "l2_weights", "l1_output"):
                        assert row["tma"][key] == original["tma"][key]
                    # The existing JIT template arguments, arithmetic and parent layout stay fixed.
                    expected = original["jit_template_args"].split(",")
                    expected[11] = str(h["stages"])
                    assert row["jit_template_args"].split(",") == expected
                cases.append({"mode": mode, "active": active, **row})
        bad = subprocess.run([str(temp / "host")], env={**os.environ, "DG_SM90_PREFILL_INDEPENDENT_WG": "invalid"}, capture_output=True, text=True, cwd=temp, check=False)
        assert bad.returncode != 0
        # Use the actual kernel declarations for the host-generated Prefill specializations.
        overlay = temp / "overlay"
        (overlay / HEADER).parent.mkdir(parents=True)
        (overlay / HEADER).write_bytes(args.image_header.read_bytes())
        for patch in json.loads((ROOT / "configs/firmus-framework.json").read_text())["deep_gemm_overlay_patches"] + ["patches/deep-gemm-overlay/sm90-prefill-independent-wg.patch"]:
            subprocess.run(["git", "apply", str(ROOT / patch)], cwd=overlay, check=True)
        header = (overlay / HEADER).read_text()
        base.KERNEL_NAMES = base.KERNEL_NAMES[:2] + ("kPrefillIWGMode", "kPrefillIWG") + base.KERNEL_NAMES[2:]
        layouts = []
        for mode, number in (("a", 1), ("b", 2)):
            specialized = re.sub(r"#ifdef DG_SM90_PREFILL_INDEPENDENT_WG\n.*?#endif", f"constexpr uint32_t kPrefillIWGMode = {number};", header, flags=re.DOTALL)
            templates = list(dict.fromkeys(c["jit_template_args"] for c in cases if c["active"] and c["mode"] == mode))
            cpp.write_text(base.kernel_translation_unit(source, specialized, templates))
            subprocess.run(["g++", "-std=c++20", "-O0", "-I" + str(args.fmt_include), str(cpp), "-o", str(temp / "kernel")], check=True)
            lines = subprocess.check_output([str(temp / "kernel")], text=True).splitlines()
            for template, line in zip(templates, lines, strict=True):
                values = list(map(int, line.split()))
                assert values[0] == (208080 if mode == "a" else 199360) and values[12:15] == [64, 256, 128]
                layouts.append({"mode": mode, "jit_template_args": template, "kernel_values": values})
        args.output.write_text(json.dumps({"gpu_executed": False, "cases": cases, "layouts": layouts}, indent=2) + "\n")
        print(f"PASS: {len(cases)} actual host cases, {len(layouts)} actual kernel layouts; opt-out/Decode/unsupported geometries unchanged")


if __name__ == "__main__":
    main()
