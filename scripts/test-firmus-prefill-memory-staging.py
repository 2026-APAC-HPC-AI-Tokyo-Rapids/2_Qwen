"""CPU fixture: QKV staging applies configured overlays before preparation."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVISION = (ROOT / "scripts/provision-firmus-kernel-bench.sh").read_text()
RUN = (ROOT / "scripts/run-megamoe-kernel-bench.sh").read_text()


def function(name):
    start = PROVISION.index(name + "() {\n")
    return PROVISION[start:PROVISION.index("\n}\n", start) + 3]


class PrefillStaging(unittest.TestCase):
    def test_control_overlay_precedes_cpu_preparation(self):
        start = PROVISION.index("prefill_memory=null\n")
        end = PROVISION.index('\nfor patch in "${baseline_patches[@]}";', start)
        block = PROVISION[start:end]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace, runtime = root / "repo", root / "runtime"
            probe = workspace / "kernels/probes/prefill-memory.py"
            probe.parent.mkdir(parents=True)
            probe.write_text('''import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--header',type=Path)
p.add_argument('--output',type=Path)
a=p.parse_args()
assert a.header.read_text() == 'configured control\\n'
a.output.mkdir(parents=True)
(a.output/'rope-heads1.cuh').write_bytes(a.header.read_bytes())
(a.output/'prefill-qkv-heads4.patch').write_text('fixture candidate patch\\n')
(a.output/'preparation.json').write_text(json.dumps({'source_sha256':hashlib.sha256(a.header.read_bytes()).hexdigest(),'headers':{'control':'fixture'}}))
''')
            header = runtime / "image/opt/sglang/source/python/sglang/kernels/jit/csrc/elementwise/rope.cuh"
            header.parent.mkdir(parents=True)
            header.write_text("pinned image\n")
            patch = workspace / "control.patch"
            patch.write_text("--- a/python/sglang/kernels/jit/csrc/elementwise/rope.cuh\n+++ b/python/sglang/kernels/jit/csrc/elementwise/rope.cuh\n@@ -1 +1 @@\n-pinned image\n+configured control\n")
            config = workspace / "config.json"
            config.write_text(json.dumps({"sglang_overlay_patches":["control.patch"]}))
            (runtime / "scripts").mkdir()
            (runtime / "source").mkdir()
            marker = workspace / "patches/deep-gemm-bench/prefill-memory"
            marker.parent.mkdir(parents=True)
            marker.touch()
            setup = 'bench_dir="${GITHUB_WORKSPACE}/patches/deep-gemm-bench"; runtime_dir="$1"; config="$2"; sglang_package=opt/sglang/source/python/sglang\n'
            gnu_install = 'install() { command ginstall "$@"; }\n' if shutil.which("ginstall") else ""
            shell = gnu_install + function("apply_patches") + function("patch_records") + setup + block + '\nprintf "%s" "$prefill_memory"\n'
            result = subprocess.run(["bash", "-euo", "pipefail", "-c", shell, "fixture", str(runtime), str(config)], env={**os.environ, "GITHUB_WORKSPACE":str(workspace)}, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            provenance = json.loads(result.stdout)
            self.assertEqual(provenance["overlay_patches"][0]["path"], "control.patch")
            self.assertEqual(len(provenance["script_sha256"]), 64)
            self.assertEqual(len(provenance["candidate_patch_sha256"]), 64)
            self.assertEqual((runtime / "source/control.patch").read_bytes(), patch.read_bytes())

    def test_single_gpu_step_and_terminal_control_contract(self):
        self.assertIn('if [[ "${variant}" == prefill-memory ]]; then gpus=1; fi', RUN)
        self.assertIn('step prefill-memory probe CUDA_VISIBLE_DEVICES=0', RUN)
        self.assertIn('--header "${ROOT}/source/prefill-memory/rope-heads1.cuh" --output "${ROOT}/results/prefill-memory" --run', RUN)
        self.assertLess(RUN.index('step prefill-memory probe'), RUN.index('step control-repeat bench'))
        self.assertIn('--output "${ROOT}/results/control-repeat/bench.jsonl" --reference "${ROOT}/results/image/bench.jsonl"', RUN)
        self.assertIn('if [[ -e "${bench_dir}/prefill-memory" ]]; then steps=$((steps + 1)); fi', PROVISION)


if __name__ == "__main__":
    unittest.main()
