"""CPU-only tests of the provisioner's actual pre-staging host guard."""

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts/provision-firmus-kernel-bench.sh").read_text()
START = SCRIPT.index("check_candidate_host() {\n")
FUNCTION = SCRIPT[START:SCRIPT.index("\n}\n", START) + 3]
PATCH = "sm90-decode-independent-wg.patch"


class HostGuardTests(unittest.TestCase):
    def check_guard(self, env, record_kind, extension_matches=True):
        with tempfile.TemporaryDirectory(prefix="independent-host-guard-") as directory:
            root = Path(directory)
            candidate = root / "independent-wg"
            candidate.mkdir()
            if env is not None:
                (candidate / "env").write_text(env)
            extension = root / "_C.so"
            extension.write_bytes(b"fixture extension bytes; no shared library is loaded")
            record = root / "deep-gemm-host-build.json"
            if record_kind != "missing":
                patch_sha = hashlib.sha256((ROOT / "patches/deep-gemm-host" / PATCH).read_bytes()).hexdigest()
                patches = {} if record_kind == "old" else {PATCH: "0" * 64 if record_kind == "stale" else patch_sha}
                record.write_text(json.dumps({"abi": "tvm-ffi", "patches": patches, "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest() if extension_matches else "0" * 64}))
            return subprocess.run(["bash", "-euo", "pipefail", "-c", FUNCTION + '\ncheck_candidate_host "$1" "$2" "$3"\n', "guard", str(candidate) + "/", str(record), str(extension)], env={**os.environ, "GITHUB_WORKSPACE": str(ROOT)}, capture_output=True, text=True, check=False)

    def test_exact_patch_and_extension_accept(self):
        result = self.check_guard("DG_SM90_DECODE_INDEPENDENT_WG=1\n", "current")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_old_missing_stale_or_mismatched_image_rejects(self):
        for record, extension in (("old", True), ("missing", True), ("stale", True), ("current", False)):
            with self.subTest(record=record, extension=extension):
                result = self.check_guard("DG_SM90_DECODE_INDEPENDENT_WG=1\n", record, extension)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires a rebuilt DeepGEMM host", result.stderr)

    def test_other_candidates_and_opt_out_accept_old_image(self):
        for env in (None, "DG_SM90_PREFILL_2WG=1\n", "DG_SM90_DECODE_INDEPENDENT_WG=0\n"):
            with self.subTest(env=env):
                result = self.check_guard(env, "missing")
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_nonzero_host_switch_also_requires_new_image(self):
        result = self.check_guard("DG_SM90_DECODE_INDEPENDENT_WG=2\n", "old")
        self.assertNotEqual(result.returncode, 0)

    def test_guard_runs_before_baseline_and_candidate_staging(self):
        call = SCRIPT.index('for dir in "${candidate_dirs[@]}"; do check_candidate_host')
        self.assertLess(call, SCRIPT.index('apply_patches "${runtime_dir}/image/'))
        self.assertLess(call, SCRIPT.index('apply_patches "${runtime_dir}/overlay/candidates/'))


if __name__ == "__main__":
    unittest.main()
