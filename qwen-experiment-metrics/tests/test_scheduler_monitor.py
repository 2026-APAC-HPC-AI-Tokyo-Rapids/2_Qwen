"""Exercise pending, terminal and failed scheduler observations through a shell adapter."""
import os
import shlex
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

MONITOR = Path(__file__).resolve().parents[2] / 'scripts/monitor-qwen-job.sh'


class MonitorTest(unittest.TestCase):
    def test_provision_waits_for_transport_but_preserves_validation_failure(self):
        workflow = (MONITOR.parents[1] / ".github/workflows/firmus.yml").read_text()
        step = workflow.split("        id: provision\n", 1)[1].split("\n      - name:", 1)[0]
        script = textwrap.dedent(step.split("        run: |\n", 1)[1]).replace("${{ inputs.dynamo }}", "false")
        cases = [(255, "Connection closed by host port 2026", True), (2, "exec request failed on channel 0\r", True), (254, "/home/ACCOUNT/.local/.setenv: fork: Resource temporarily unavailable", True), (1, "image checksum mismatch", False), (2, "invalid workload", False)]
        for status, error, retry in cases:
            with self.subTest(status=status, error=error), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                adapter = root / "firmus-shell"
                adapter.write_text(f'#!/bin/bash\nif [[ ! -e "{root}/seen" ]]; then touch "{root}/seen"; printf "%s\\n" {shlex.quote(error)} >&2; exit {status}; fi\necho provision_complete\n')
                adapter.chmod(0o755)
                for name, body in (("sleep", "exit 0"), ("timeout", 'shift 2; exec "$@"')):
                    tool = root / name
                    tool.write_text(f"#!/bin/bash\n{body}\n")
                    tool.chmod(0o755)
                result = subprocess.run(["bash", "-euc", script], env={**os.environ, "PATH": f"{root}:{os.environ['PATH']}"}, capture_output=True, text=True, timeout=5, check=False)
                self.assertEqual(result.returncode, 0 if retry else status)
                self.assertEqual("provision_complete" in result.stdout, retry)

    def run_monitor(self, pending_status, terminal_status, observation="scheduler_job_pending=123.pbs111 state=R", transport_error=False, transport_message="Connection closed by 150.230.206.3 port 2026"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / 'adapter'
            stderr = f"echo '{transport_message}' >&2; echo 'tar: This does not look like a tar archive' >&2; " if transport_error else ""
            adapter.write_text(f'#!/bin/bash\nif [[ ! -e "{root}/seen" ]]; then touch "{root}/seen"; echo {observation}; {stderr}exit {pending_status}; fi\necho terminal\nexit {terminal_status}\n')
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

    def test_firmus_pending_observation_without_state_survives_tar_failure(self):
        for terminal_status in (0, 1):
            with self.subTest(terminal_status=terminal_status):
                result = self.run_monitor(2, terminal_status, observation="scheduler_job_pending=21732")
                self.assertEqual(result.returncode, terminal_status)
                self.assertIn("terminal", result.stdout)

    def test_tar_failure_without_pending_evidence_is_fatal(self):
        result = self.run_monitor(2, 0, "terminal_observation_incomplete")
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("\nterminal\n", result.stdout)

    def test_disconnected_ssh_stream_wrapped_as_tar_failure_is_retried(self):
        for message in ("Connection closed by 150.230.206.3 port 2026", "exec request failed on channel 0", "exec request failed on channel 0\r"):
            for terminal_status in (0, 1):
                with self.subTest(message=message, terminal_status=terminal_status):
                    result = self.run_monitor(2, terminal_status, observation="", transport_error=True, transport_message=message)
                    self.assertEqual(result.returncode, terminal_status)
                    self.assertIn('terminal', result.stdout)
                    self.assertIn('status=255', result.stderr)
                    self.assertIn(message.rstrip("\r"), result.stderr)

    def test_login_fork_failure_is_retried_but_other_exit_254_is_fatal(self):
        for message in ("/etc/profile.d/which2.sh: fork: retry: Resource temporarily unavailable", "/etc/bashrc: fork: Resource temporarily unavailable\r"):
            result = self.run_monitor(254, 0, observation="", transport_error=True, transport_message=message)
            self.assertEqual(result.returncode, 0)
            self.assertIn("terminal", result.stdout)
            self.assertIn("status=255", result.stderr)
        result = self.run_monitor(254, 0, observation="", transport_error=True, transport_message="checker failed")
        self.assertEqual(result.returncode, 254)
        self.assertNotIn("terminal", result.stdout)

    def test_connection_outage_preserves_last_observed_state(self):
        for state, next_state, limit in [('Q', 'Q', 'none'), ('R', 'R', 'none'), ('Q', 'R', 'none')]:
            with self.subTest(state=state, next_state=next_state), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                adapter = root / 'adapter'
                adapter.write_text(f'#!/bin/bash\nn=$(cat "{root}/count" 2>/dev/null || echo 0)\necho $((n+1)) >"{root}/count"\ncase $n in\n0) echo scheduler_job_pending=123.pbs111 state={state}; exit 0;;\n1) exit 124;;\n2) exit 255;;\n3) echo scheduler_job_pending=123.pbs111 state={next_state}; exit 0;;\n4) exit 124;;\n*) echo terminal; exit 0;;\nesac\n')
                adapter.chmod(0o755)
                sleeper = root / 'sleep'
                sleeper.write_text('#!/bin/bash\nexit 0\n')
                sleeper.chmod(0o755)
                result = subprocess.run(['bash', str(MONITOR), str(adapter), 'checker'], env={**os.environ, 'PATH': f'{root}:{os.environ["PATH"]}'}, capture_output=True, text=True, timeout=5, check=False)
                self.assertEqual(result.returncode, 0)
                self.assertIn(f'last_state={state} limit_seconds={limit}', result.stderr)
                self.assertIn('status=124', result.stderr)
                self.assertIn('status=255', result.stderr)
                self.assertIn(f"last_state={next_state} limit_seconds=none", result.stderr)
                self.assertIn('terminal', result.stdout)

    def test_long_outage_waits_for_definitive_scheduler_result(self):
        for state, expected in [('Q', 0), ('R', 0), ('unknown', 0), ('Q', 1), ('R', 1), ('unknown', 1)]:
            with self.subTest(state=state, terminal_status=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                monitor = root / 'monitor'
                # Inject elapsed time only: connection loss must not become a terminal result.
                monitor.write_text(MONITOR.read_text().replace('connection_failure_seconds=$((SECONDS - connection_failure_started))', 'connection_failure_seconds=1801'))
                adapter = root / 'adapter'
                observation = f'echo scheduler_job_pending=123.pbs111 state={state}; exit 0' if state != 'unknown' else 'exit 124'
                adapter.write_text(f'#!/bin/bash\nn=$(cat "{root}/count" 2>/dev/null || echo 0)\necho $((n+1)) >"{root}/count"\ncase $n in\n0) {observation};;\n1) exit 124;;\n*) echo terminal; exit {expected};;\nesac\n')
                adapter.chmod(0o755)
                sleeper = root / 'sleep'
                sleeper.write_text('#!/bin/bash\nexit 0\n')
                sleeper.chmod(0o755)
                result = subprocess.run(['bash', str(monitor), str(adapter), 'checker'], env={**os.environ, 'PATH': f'{root}:{os.environ["PATH"]}'}, capture_output=True, text=True, timeout=5, check=False)
                self.assertEqual(result.returncode, expected)
                self.assertIn('terminal', result.stdout)
                self.assertIn('limit_seconds=none', result.stderr)



def test_firmus_completed_job_removed_from_squeue_uses_accounting(tmp_path):
    checker = MONITOR.parents[1] / "platforms/firmus/check-qwen-job.sh"
    tools = tmp_path / "bin"
    tools.mkdir()
    runtime = tmp_path / "runtime"
    (runtime / "logs").mkdir(parents=True)
    (runtime / "results").mkdir()
    (runtime / "logs/slurm-123.log").write_text("AIPERF ok\n")
    (runtime / "run-success").touch()
    for name in ("profile_export_aiperf.json", "score.json"):
        (runtime / "results" / name).write_text("{}")
    for name, body in (("squeue", 'if [[ "$*" == *--jobs=* ]]; then echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1; fi\necho "999|RUNNING"'), ("sacct", "echo '123|COMPLETED|0:0|00:12:39|2'")):
        tool = tools / name
        tool.write_text("#!/bin/bash\n" + body + "\n")
        tool.chmod(0o755)
    result = subprocess.run(["bash", str(checker)], env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "SLURM_JOB_ID": "123", "RUNTIME_DIR": str(runtime)}, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "123|COMPLETED|0:0" in result.stdout
    assert (runtime / "logs/slurm-accounting.txt").is_file()
