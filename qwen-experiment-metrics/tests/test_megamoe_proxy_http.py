"""Actual proxy and upstreams over loopback HTTP, without models or GPUs."""

import asyncio
import importlib.util
import json
import socket
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).parents[2]


@asynccontextmanager
async def serve(app):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, log_level="critical", access_log=False))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if task.done():
                        await task
                        raise AssertionError("Server exited before readiness")
                    await asyncio.sleep(0.01)
            yield sock.getsockname()[1]
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 5)


@pytest.mark.parametrize("endpoint", ["completions", "chat/completions"])
@pytest.mark.parametrize("mode", ["stream", "json", "prefill_error", "decode_error", "bad_json", "cancel", "prefill_cancel"])
def test_pd_http_contract(endpoint, mode):
    async def exercise():
        spec = importlib.util.spec_from_file_location("actual_pd_proxy", ROOT / "platforms/firmus/framework-comparison/vllm-proxy.py")
        proxy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(proxy)
        prefill_seen, decode_seen = [], []
        release_prefill, prefill_started, prefill_closed, decode_closed = (asyncio.Event() for _ in range(4))
        metadata = {"remote_engine_id": "p0_dp2", "remote_host": "192.0.2.1", "remote_port": 5602, "remote_block_ids": [[2, 7]], "do_remote_prefill": True}
        prefill, decode = FastAPI(), FastAPI()

        @prefill.post("/v1/" + endpoint)
        async def produce(request: Request):
            prefill_seen.append((await request.json(), request.headers["x-request-id"]))

            async def body():
                try:
                    prefill_started.set()
                    await release_prefill.wait()
                    yield b"not json" if mode == "bad_json" else json.dumps({"kv_transfer_params": metadata}).encode()
                finally:
                    prefill_closed.set()

            if mode == "prefill_error":
                prefill_started.set()
                await release_prefill.wait()
                return JSONResponse({"error": "prefill failed"}, status_code=503)
            return StreamingResponse(body(), media_type="application/json")

        @decode.post("/v1/" + endpoint)
        async def consume(request: Request):
            decode_seen.append((await request.json(), request.headers["x-request-id"]))
            if mode == "decode_error":
                return JSONResponse({"error": "decode failed"}, status_code=429)
            if mode == "json":
                return {"choices": [{"text": "answer"}], "usage": {"completion_tokens": 200}}

            async def body():
                try:
                    yield b'data: {"text":"answer"}\n\n'
                    if mode == "cancel":
                        await asyncio.Event().wait()
                    yield b"data: [DONE]\n\n"
                finally:
                    decode_closed.set()

            return StreamingResponse(body(), media_type="text/event-stream")

        async with AsyncExitStack() as stack:
            p_port = await stack.enter_async_context(serve(prefill))
            d_port = await stack.enter_async_context(serve(decode))
            proxy.global_args = SimpleNamespace(prefiller_instances=[("127.0.0.1", p_port)], decoder_instances=[("127.0.0.1", d_port)])
            proxy_port = await stack.enter_async_context(serve(proxy.app))
            client = await stack.enter_async_context(httpx.AsyncClient(base_url=f"http://127.0.0.1:{proxy_port}", timeout=5))
            original = {"model": "qwen", "stream": mode != "json", "max_tokens": 200, "min_tokens": 200, "ignore_eos": True, "temperature": 0, "stop": ["stop"], "stream_options": {"include_usage": True}}
            original.update({"messages": [{"role": "user", "content": "keep"}]} if endpoint.startswith("chat") else {"prompt": "keep"})
            task = asyncio.create_task(client.send(client.build_request("POST", "/v1/" + endpoint, json=original), stream=True))
            await asyncio.wait_for(prefill_started.wait(), 5)
            assert proxy.app.state.prefill_clients[0]["in_flight"] == 1
            assert decode_seen == []
            if mode == "prefill_cancel":
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                try:
                    await asyncio.wait_for(prefill_closed.wait(), 2)
                    assert proxy.app.state.prefill_clients[0]["in_flight"] == 0
                    assert decode_seen == []
                finally:
                    release_prefill.set()
                return
            release_prefill.set()
            response = await task
            try:
                assert proxy.app.state.prefill_clients[0]["in_flight"] == 0
                if mode.endswith("error"):
                    assert response.status_code == (503 if mode == "prefill_error" else 429)
                    assert json.loads(await response.aread())["error"].endswith("failed")
                elif mode == "bad_json":
                    assert response.status_code == 500
                elif mode == "json":
                    assert response.headers["content-type"].startswith("application/json")
                    assert json.loads(await response.aread())["usage"]["completion_tokens"] == 200
                elif mode == "cancel":
                    async for chunk in response.aiter_bytes():
                        assert b"answer" in chunk
                        break
                else:
                    assert await response.aread() == b'data: {"text":"answer"}\n\ndata: [DONE]\n\n'
            finally:
                await response.aclose()
            if mode in ("stream", "cancel"):
                await asyncio.wait_for(decode_closed.wait(), 5)
            p_data, p_id = prefill_seen[0]
            assert p_data["max_tokens"] == 1 and p_data["stream"] is False
            assert "min_tokens" not in p_data and "stream_options" not in p_data
            if mode not in ("prefill_error", "bad_json"):
                d_data, d_id = decode_seen[0]
                assert d_id == p_id
                assert d_data == {**original, "kv_transfer_params": metadata}
            else:
                assert decode_seen == []

    asyncio.run(exercise())
