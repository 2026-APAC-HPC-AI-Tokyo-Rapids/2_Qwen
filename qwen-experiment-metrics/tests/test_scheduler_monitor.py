"""Exercise pending, terminal and failed scheduler observations through a shell adapter."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

MONITOR = Path(__file__).resolve().parents[2] / 'scripts/monitor-qwen-job.sh'


class MonitorTest(unittest.TestCase):
    def run_monitor(self, pending_status, terminal_status, observation="scheduler_job_pending=123.pbs111 state=R"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / 'adapter'
            adapter.write_text(f'#!/bin/bash\nif [[ ! -e "{root}/seen" ]]; then touch "{root}/seen"; echo {observation}; exit {pending_status}; fi\necho terminal\nexit {terminal_status}\n')
            adapter.chmod(0o755)
            sleeper = root / 'sleep'
            sleeper.write_text('#!/bin/bash\nexit 0\n')
            sleeper.chmod(0o755)
            return subprocess.run(['bash', str(MONITOR), str(adapter), 'checker'], env={**os.environ, 'PATH': f'{root}:{os.environ["PATH"]}'}, capture_output=True, text=True, timeout=5, check=False)

    def test_zero_status_pending(self):
        result = self.run_monitor(0, 0)
        self.assertEqual(result.returncode, 0)
        self.assertIn('terminal', result.stdout)

    def test_legacy_pending(self):
        self.assertEqual(self.run_monitor(75, 0).returncode, 0)

    def test_terminal_failure(self):
        self.assertEqual(self.run_monitor(0, 1).returncode, 1)

    def test_failed_adapter_is_not_pending(self):
        result = self.run_monitor(1, 0)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn('terminal', result.stdout)

    def test_pending_observation_survives_adapter_tar_failure(self):
        result = self.run_monitor(2, 0)
        self.assertEqual(result.returncode, 0)
        self.assertIn("terminal", result.stdout)

    def test_tar_failure_without_pending_evidence_is_fatal(self):
        result = self.run_monitor(2, 0, "terminal_observation_incomplete")
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("\nterminal\n", result.stdout)
