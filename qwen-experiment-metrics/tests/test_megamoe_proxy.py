"""Check group/rank reservations and native push endpoint identity."""

import importlib.util
import itertools
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location("pd_proxy", ROOT / "platforms/firmus/framework-comparison/vllm-proxy.py")
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)


def test_least_live_groups_and_ranks_rotate_ties():
    clients = [{"id": i, "in_flight": 0, "rank_in_flight": [0] * 4, "rank_iterator": itertools.cycle(range(4))} for i in range(3)]
    application = NS(state=NS(prefill_clients=clients, prefill_iterator=itertools.cycle(range(3))))
    assert [proxy.get_next_client(application, "prefill")["id"] for _ in range(4)] == [0, 1, 2, 0]
    assert [client["in_flight"] for client in clients] == [2, 1, 1]
    assert [proxy.reserve_prefill_rank(clients[0]) for _ in range(5)] == [0, 1, 2, 3, 0]
    assert clients[0]["rank_in_flight"] == [2, 1, 1, 1]
    clients[0]["rank_in_flight"][1] = 9
    assert proxy.reserve_prefill_rank(clients[0]) == 2


def test_native_dp_push_coordinates_do_not_require_finished_prefill_blocks():
    client = {"engine_id": "unique-run-prefill-1", "host": "192.0.2.1", "port": 30110}
    params = proxy.push_coordinates(client, 3, "same-id-for-both-legs")
    assert params == {"do_remote_decode": False, "do_remote_prefill": True, "remote_engine_id": "unique-run-prefill-1_dp3", "remote_request_id": "same-id-for-both-legs", "remote_host": "192.0.2.1", "remote_port": 5613, "tp_size": 1, "pp_size": 1}
    assert "remote_block_ids" not in params
