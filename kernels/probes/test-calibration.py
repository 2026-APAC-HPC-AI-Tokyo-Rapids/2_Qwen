"""CPU checks for calibration aggregation and the actual PJM/summary protocol."""

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from runpy import run_path

ROOT = Path(__file__).resolve().parents[2]
COMMON = run_path(str(Path(__file__).with_name("calib-common.py")))


class CalibrationTests(unittest.TestCase):
    def test_distribution_rejects_invalid_timings(self):
        for values in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.assertRaises(AssertionError):
                COMMON["distribution"](values)

    def test_report_uses_paired_stock_and_exposes_disagreeing_blocks(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            COMMON["report"]("B32", "candidate", [[8, 8], [12, 12]], [[10, 10], [10, 10]], {"numeric_passed": True})
        record, row = out.getvalue().splitlines()
        self.assertEqual(json.loads(record)["blocks"], [[8, 8], [12, 12]])
        self.assertEqual(row, "CALIB | B32 | candidate | 10.000 | 8.000 | 12.000 | 1.00000 | 0.80000 | 1.20000 |")

    def test_actual_job_check_and_summary(self):
        for calibration, probe_exit in ((True, 0), (True, 7), (False, 0)):
            with self.subTest(calibration=calibration, probe_exit=probe_exit), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for directory in ("source", "results", "logs", "bin"):
                    (root / directory).mkdir()
                (root / "source/provenance.json").write_text(json.dumps({"calibration": calibration}))
                scripts = {
                    "bin/module": "exit 0",
                    "bin/timeout": 'printf "%s\\n" "$1" >> "$RUNTIME_DIR/limits"; shift; exec "$@"',
                    "bin/pjwait": "exit 0",
                    "bin/pjstat": "echo terminal",
                    "source/probe": f'echo called >> calls; echo "CALIB | B32 | stock | 10 | 9 | 11 | 1 | 1 | 1 |"; echo "PASS calibration=fixture"; exit {probe_exit}',
                }
                for name, body in scripts.items():
                    p = root / name
                    p.write_text("#!/bin/bash\n" + body + "\n")
                    p.chmod(0o755)
                env = {**os.environ, "RUNTIME_DIR": tmp, "PJM_JOB_ID": "123", "PATH": f"{root / 'bin'}:{os.environ['PATH']}"}
                job = subprocess.run(["bash", str(ROOT / "platforms/genkai/kernel-probe/run.pjm")], env=env, capture_output=True, text=True, check=False)
                self.assertEqual(job.returncode, probe_exit, job.stderr)
                count = 1 if calibration or probe_exit else 3
                self.assertEqual(len((root / "calls").read_text().splitlines()), count)
                self.assertEqual((root / "limits").read_text().splitlines(), ["165" if calibration else "90"] * count)
                self.assertEqual((root / "results/exit-code").read_text().strip(), str(probe_exit))
                (root / "logs/123.pjm.log").write_text(job.stdout)
                checked = subprocess.run(["bash", str(ROOT / "platforms/genkai/kernel-probe/check-job.sh")], env=env, capture_output=True, text=True, check=False)
                self.assertEqual(checked.returncode == 0, probe_exit == 0, checked.stderr)
                summary = subprocess.run(["bash", str(ROOT / "platforms/genkai/kernel-probe/summary.sh")], env=env, capture_output=True, text=True, check=True).stdout
                self.assertIn(f"Exit code: {probe_exit}", summary)
                self.assertIn("| B32 | stock |" if calibration else "| 3 | PASS", summary)
                if calibration:
                    self.assertNotIn("not completed", summary)


if __name__ == "__main__":
    unittest.main()
