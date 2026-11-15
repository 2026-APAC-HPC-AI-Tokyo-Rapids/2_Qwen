"""Temporary first-peer wall timers; copied as sitecustomize for diagnostic workers."""
from __future__ import annotations

import functools
import json
import os
import time


def install_timers(manager_class: type) -> None:
    for method_name in ("_add_remote_peer", "_prepare_payload_xfer"):
        original = getattr(manager_class, method_name)

        def wrap(method, name):
            @functools.wraps(method)
            def measured(self, peer, *args, **kwargs):
                started = time.perf_counter()
                identity = {"method": name, "pid": os.getpid(), "rank": self.kv_args.engine_rank, "peer": peer.agent_name}
                print("FIRMUS_NIXL_TIMING " + json.dumps({**identity, "event": "begin", "unix_ns": time.time_ns()}), flush=True)
                try:
                    return method(self, peer, *args, **kwargs)
                finally:
                    print("FIRMUS_NIXL_TIMING " + json.dumps({**identity, "event": "end", "unix_ns": time.time_ns(), "elapsed_seconds": time.perf_counter() - started}), flush=True)

            return measured

        setattr(manager_class, method_name, wrap(original, method_name))


if __name__ == "sitecustomize":
    from sglang.srt.disaggregation.nixl.conn import NixlKVManager

    install_timers(NixlKVManager)
