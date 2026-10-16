# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools
import logging
import os
import uuid
from contextlib import asynccontextmanager

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []

    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f"http://{host}:{port}/v1"
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=prefiller_base_url,
                    limits=httpx.Limits(
                        max_connections=None,
                        # Avoid reusing a prefill connection closed by its server.
                        max_keepalive_connections=0,
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
                "in_flight": 0,
                "rank_in_flight": [0] * 4,
                "rank_iterator": itertools.cycle(range(4)),
                "engine_id": global_args.prefiller_engine_ids[i],
            }
        )

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f"http://{host}:{port}/v1"
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=decoder_base_url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    # Rotate ties between equally loaded prefills and round-robin decoders
    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    print(
        f"Initialized {len(app.state.prefill_clients)} prefill clients "
        f"and {len(app.state.decode_clients)} decode clients."
    )

    yield

    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()

    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    # Always use 127.0.0.1 as localhost binds to IPv6 which is blocked on CI
    parser.add_argument("--host", type=str, default="127.0.0.1")

    # For prefiller instances
    parser.add_argument(
        "--prefiller-hosts",
        "--prefiller-host",
        type=str,
        nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8100]
    )
    parser.add_argument("--prefiller-engine-ids", nargs="+", required=True)

    # For decoder instances
    parser.add_argument(
        "--decoder-hosts", "--decoder-host", type=str, nargs="+", default=["localhost"]
    )
    parser.add_argument(
        "--decoder-ports", "--decoder-port", type=int, nargs="+", default=[8200]
    )

    args = parser.parse_args()

    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports"
        )

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")

    # Create tuples of (host, port) for each service type
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    if len(args.prefiller_engine_ids) != len(args.prefiller_instances):
        raise ValueError("Each prefill group requires its explicit KV engine ID")
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    return args


def get_next_client(app, service_type: str):
    """
    Reserve the least-loaded prefiller; rotate ties and decoder selection.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == "prefill":
        clients = app.state.prefill_clients
        start = next(app.state.prefill_iterator)
        client_idx = min(range(len(clients)), key=lambda i: (clients[i]["in_flight"], (i - start) % len(clients)))
        clients[client_idx]["in_flight"] += 1
        return clients[client_idx]
    elif service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


async def send_request_to_service(
    client_info: dict, endpoint: str, req_data: dict, request_id: str, rank: int
):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    # P uses a copy; the original sampling limits are retained for D.
    req_data.pop("min_tokens", None)
    req_data.pop("min_completion_tokens", None)
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        "X-data-parallel-rank": str(rank),
    }

    client = client_info["client"]
    upstream = client.build_request("POST", endpoint, json=req_data, headers=headers)
    return await client.send(upstream, stream=True)


class UpstreamError(Exception):
    def __init__(self, response):
        self.response = response


def reserve_prefill_rank(client_info):
    start = next(client_info["rank_iterator"])
    counts = client_info["rank_in_flight"]
    rank = min(range(4), key=lambda i: (counts[i], (i - start) % 4))
    counts[rank] += 1
    return rank


def push_coordinates(client_info, rank, request_id):
    # The native DP engine appends _dpN and adds N to the side-channel port.
    return {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_engine_id": f'{client_info["engine_id"]}_dp{rank}',
        "remote_request_id": request_id,
        "remote_host": client_info["host"],
        "remote_port": client_info["port"] - 24500 + rank,
        "tp_size": 1,
        "pp_size": 1,
    }


async def open_decode_response(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Read D's headers before committing the downstream HTTP status.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    client = client_info["client"]
    request = client.build_request("POST", endpoint, json=req_data, headers=headers)
    return await client.send(request, stream=True)


async def _handle_completions(api: str, request: Request):
    # Consume the body before watching receive(), so only disconnects remain.
    await request.body()
    response = Response(status_code=499)
    async with anyio.create_task_group() as tasks:
        async def watch_disconnect():
            while True:
                if (await request.receive())["type"] == "http.disconnect":
                    tasks.cancel_scope.cancel()
                    return

        tasks.start_soon(watch_disconnect)
        response = await _prepare_completions(api, request)
        # StreamingResponse takes over disconnect handling after headers.
        tasks.cancel_scope.cancel()
    return response


async def _prepare_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())
        prefill_client_info = get_next_client(request.app, "prefill")
        rank = reserve_prefill_rank(prefill_client_info)
        coordinates = push_coordinates(prefill_client_info, rank, request_id)
        decode_client_info = get_next_client(request.app, "decode")
        decode_data = {**req_data, "kv_transfer_params": coordinates}
        released = False

        def release_prefill():
            nonlocal released
            if not released:
                prefill_client_info["in_flight"] -= 1
                prefill_client_info["rank_in_flight"][rank] -= 1
                released = True

        async def run_prefill():
            try:
                response = await send_request_to_service(prefill_client_info, api, req_data, request_id, rank)
                try:
                    await response.aread()
                    if response.is_error:
                        raise UpstreamError(Response(response.content, status_code=response.status_code, media_type=response.headers.get("content-type")))
                    metadata = response.json().get("kv_transfer_params", {})
                    if metadata.get("remote_engine_id") != coordinates["remote_engine_id"] or metadata.get("transfer_mode") != "push":
                        raise RuntimeError("prefill did not confirm the selected push engine")
                finally:
                    with anyio.CancelScope(shield=True):
                        await response.aclose()
            finally:
                release_prefill()

        async def run_decode():
            response = await open_decode_response(decode_client_info, api, decode_data, request_id=request_id)
            if response.is_error:
                try:
                    raise UpstreamError(Response(await response.aread(), status_code=response.status_code, media_type=response.headers.get("content-type")))
                finally:
                    await response.aclose()
            return response

        # D preallocates/registers its blocks while P computes, then waits for
        # the native producer WRITE. It does not need P's response block IDs.
        response = None
        upstream_error = None
        try:
            try:
                async with anyio.create_task_group() as upstreams:
                    upstreams.start_soon(run_prefill)
                    response = await run_decode()
            except* UpstreamError as errors:
                upstream_error = errors.exceptions[0]
            if upstream_error is not None:
                if response is not None:
                    with anyio.CancelScope(shield=True):
                        await response.aclose()
                return upstream_error.response
        except BaseException:
            if response is not None:
                with anyio.CancelScope(shield=True):
                    await response.aclose()
            raise
        finally:
            # Cancellation can occur before a newly-created P task starts.
            release_prefill()

        logger.debug("Using %s %s", prefill_client_info, decode_client_info)
        if not req_data.get("stream", False):
            try:
                return Response(await response.aread(), status_code=response.status_code, media_type=response.headers.get("content-type"))
            finally:
                await response.aclose()

        async def generate_stream():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()

        return StreamingResponse(generate_stream(), media_type="text/event-stream", background=BackgroundTask(response.aclose))

    except UpstreamError as error:
        return error.response
    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/v1/models")
async def get_models():
    decoder = app.state.decode_clients[0]
    response = await decoder["client"].get("/models")
    response.raise_for_status()
    return response.json()


@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }


if __name__ == "__main__":
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
