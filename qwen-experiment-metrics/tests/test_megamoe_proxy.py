"""Exercise least-live routing and reservation lifetime without network/GPU."""
import ast
import asyncio
import itertools
import logging
import uuid
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("fail", [False, True])
def test_live_prefill_reservations_survive_until_body_read_and_release_on_failure(fail):
    source = (ROOT / "platforms/firmus/framework-comparison/vllm-proxy.py").read_text()
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in ("get_next_client", "_handle_completions")]
    namespace = {"Request": object, "uuid": uuid, "logger": logging.getLogger(__name__), "StreamingResponse": lambda generator, **kw: generator}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), "actual-proxy-reservation", "exec"), namespace)  # noqa: S102

    async def exercise():
        clients = [{"id": i, "in_flight": 0} for i in range(3)]
        state = NS(prefill_clients=clients, prefill_iterator=itertools.cycle(range(3)), decode_clients=[{"id": 0}], decode_iterator=itertools.cycle(range(1)))
        application = NS(state=state)
        release_body = asyncio.Event()
        started = []

        async def send(client, *_args):
            started.append(client["id"])
            await release_body.wait()
            if fail:
                raise RuntimeError("observed prefill failure")
            async def close():
                pass
            return NS(json=lambda: {"kv_transfer_params": {"remote_engine_id": "producer-dp2"}}, aclose=close)

        async def stream(*_args, **_kwargs):
            yield b"payload"

        async def request_json():
            return {"prompt": "keep me", "max_tokens": 200}

        namespace["send_request_to_service"] = send
        namespace["stream_service_response"] = stream
        request = NS(app=application, json=request_json)
        tasks = [asyncio.create_task(namespace["_handle_completions"]("/completions", request)) for _ in range(4)]
        await asyncio.sleep(0)
        assert started == [0, 1, 2, 0]
        assert [c["in_flight"] for c in clients] == [2, 1, 1]
        release_body.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert [c["in_flight"] for c in clients] == [0, 0, 0]
        if fail:
            assert all(isinstance(result, RuntimeError) for result in results)
        else:
            for result in results:
                assert [chunk async for chunk in result] == [b"payload"]
        # After completion ties rotate, and a busy worker is skipped.
        clients[1]["in_flight"] = 9
        selected = namespace["get_next_client"](application, "prefill")
        assert selected["id"] == 2
    asyncio.run(exercise())
