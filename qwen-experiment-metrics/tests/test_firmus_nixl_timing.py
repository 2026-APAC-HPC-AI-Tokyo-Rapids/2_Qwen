from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

INSTALL = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/profile-firmus-nixl.py"))["install_timers"]


def test_first_peer_timers_preserve_nested_return_and_exceptions(capsys) -> None:
    result = object()

    class Manager:
        kv_args = SimpleNamespace(engine_rank=2)

        def _prepare_payload_xfer(self, peer, *, fail=False):
            if fail:
                raise ValueError("original failure")
            return result

        def _add_remote_peer(self, peer, *, fail=False):
            return self._prepare_payload_xfer(peer, fail=fail)

    INSTALL(Manager)
    manager = Manager()
    peer = SimpleNamespace(agent_name="decoder")
    assert manager._add_remote_peer(peer) is result
    with pytest.raises(ValueError, match="original failure"):
        manager._add_remote_peer(peer, fail=True)
    rows = [json.loads(line.removeprefix("FIRMUS_NIXL_TIMING ")) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 8
    assert all(row["rank"] == 2 and row["peer"] == "decoder" for row in rows)
    assert sum(row["event"] == "end" for row in rows) == 4
    assert all(row["elapsed_seconds"] >= 0 for row in rows if row["event"] == "end")
