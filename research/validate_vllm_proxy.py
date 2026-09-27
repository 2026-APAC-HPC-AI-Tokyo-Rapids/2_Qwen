"""CPU regression check for the pinned, patched vLLM NIXL proxy.

Run with: uv run --with fastapi --with httpx python research/validate_vllm_proxy.py /path/to/patched/proxy.py
No GPU or model is needed; fake P/D services preserve the actual proxy handlers.
"""

import asyncio
import importlib.util
import itertools
import json
import sys

import httpx


async def validate(path):
    spec = importlib.util.spec_from_file_location("vllm_proxy_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for endpoint in ("completions", "chat/completions"):
        for streaming in (False, True):
            requests = []
            payload = {"id": "test", "choices": [{"text": "hello"}]}
            body = b"data: " + json.dumps(payload).encode() + b"\n\ndata: [DONE]\n\n"

            async def prefill(request):
                data = json.loads(request.content)
                assert data["stream"] is False
                assert data["max_tokens"] == 1
                return httpx.Response(200, json={"kv_transfer_params": {"remote_engine_id": "test-prefill"}})

            async def decode(request, requests=requests, streaming=streaming, body=body, payload=payload):
                data = json.loads(request.content)
                requests.append(data)
                assert data["stream"] is streaming
                assert data["max_tokens"] == 200
                assert data["kv_transfer_params"]["remote_engine_id"] == "test-prefill"
                if streaming:
                    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
                return httpx.Response(200, json=payload)

            async with httpx.AsyncClient(transport=httpx.MockTransport(prefill), base_url="http://prefill/v1") as pclient, httpx.AsyncClient(transport=httpx.MockTransport(decode), base_url="http://decode/v1") as dclient:
                module.app.state.prefill_clients = [{"client": pclient, "id": 0}]
                module.app.state.decode_clients = [{"client": dclient, "id": 0}]
                module.app.state.prefill_iterator = itertools.cycle([0])
                module.app.state.decode_iterator = itertools.cycle([0])
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app), base_url="http://proxy") as client:
                    response = await client.post(f"/v1/{endpoint}", json={"stream": streaming, "max_tokens": 200})
                    assert response.status_code == 200
                    if streaming:
                        assert response.headers["content-type"].startswith("text/event-stream")
                        assert response.content == body
                    else:
                        assert response.headers["content-type"] == "application/json"
                        assert response.json() == payload
                    assert len(requests) == 1
            print(f"PASS {endpoint} stream={streaming}: content type, body, and P/D handoff")


if __name__ == "__main__":
    asyncio.run(validate(sys.argv[1]))
