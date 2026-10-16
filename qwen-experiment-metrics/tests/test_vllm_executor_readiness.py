"""Exercise pinned MP Future semantics and non-consuming response inspection."""

import hashlib
import runpy
from collections import deque
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATCH = runpy.run_path(str(ROOT / "containers/fix-vllm-prefill-handoff.py"))
FIXTURES = Path(__file__).parent / "fixtures"


def patched_sources(monkeypatch):
    mp = (FIXTURES / "vllm-mp-completion.py.txt").read_bytes()
    mq = (FIXTURES / "vllm-mq-dequeue.py.txt").read_bytes()
    monkeypatch.setitem(PATCH["patch_multiproc"].__globals__, "MP_SOURCE_SHA256", hashlib.sha256(mp).hexdigest())
    monkeypatch.setitem(PATCH["patch_message_queue"].__globals__, "MQ_SOURCE_SHA256", hashlib.sha256(mq).hexdigest())
    return PATCH["patch_multiproc"](mp), PATCH["patch_message_queue"](mq)


def test_original_native_done_does_not_observe_available_response():
    source = (FIXTURES / "vllm-mp-completion.py.txt").read_bytes()
    wrapper = PATCH["selected_future_class"](source)
    responses, futures = deque(["physically available"]), deque()
    future = wrapper(futures, responses.popleft)
    assert not future.done() and len(responses) == 1
    assert future.result() == "physically available"
    assert future.done() and not responses and not futures


def test_patched_native_future_and_queue_contract(monkeypatch):
    mp, mq = patched_sources(monkeypatch)
    report = PATCH["validate_executor_readiness"](mp, mq)
    assert report["passed"] is True
    assert "injected CPU response state" in report["scope"]


def test_poll_preserves_aggregation_and_blocking_result(monkeypatch):
    mp, _ = patched_sources(monkeypatch)
    wrapper = PATCH["selected_future_class"](mp)
    responses, futures = deque(), deque()
    first = wrapper(futures, responses.popleft, aggregate=lambda value: value + 1, response_ready=lambda: bool(responses))
    second = wrapper(futures, responses.popleft, aggregate=lambda value: value * 2, response_ready=lambda: bool(responses))
    assert not second.poll()
    responses.extend([4, 6])
    assert second.result() == 12 and first.result() == 5
    assert first.poll() and second.poll() and not futures
    with pytest.raises(RuntimeError, match="timeout not implemented"):
        second.result(timeout=1)


@pytest.mark.parametrize("name", ["patch_multiproc", "patch_message_queue"])
def test_reject_unreviewed_source(name):
    with pytest.raises(ValueError, match="Unsupported pinned"):
        PATCH[name](b"unreviewed")


@pytest.mark.parametrize("name,fixture,key,old,new", [
    ("patch_multiproc", "vllm-mp-completion.py.txt", "MP_SOURCE_SHA256", b"    def result(self, timeout=None):", b"    def changed_result(self, timeout=None):"),
    ("patch_message_queue", "vllm-mq-dequeue.py.txt", "MQ_SOURCE_SHA256", b"    def dequeue(\n", b"    def changed_dequeue(\n"),
])
def test_reject_changed_anchor(monkeypatch, name, fixture, key, old, new):
    source = (FIXTURES / fixture).read_bytes().replace(old, new)
    monkeypatch.setitem(PATCH[name].__globals__, key, hashlib.sha256(source).hexdigest())
    with pytest.raises(ValueError, match="Nonunique"):
        PATCH[name](source)


@pytest.mark.parametrize("invalid", ["mp", "mq", "gpu"])
def test_installer_validates_all_sources_before_any_write(tmp_path, monkeypatch, invalid):
    patched_sources(monkeypatch)
    sources = {
        "engine": b"unchanged engine",
        "push": (FIXTURES / "vllm-push-completion.py.txt").read_bytes(),
        "mp": (FIXTURES / "vllm-mp-completion.py.txt").read_bytes(),
        "mq": (FIXTURES / "vllm-mq-dequeue.py.txt").read_bytes(),
        "gpu": (FIXTURES / "vllm-gpu-kv.py.txt").read_bytes(),
    }
    sources[invalid] = b"unsupported source"
    paths = {name: tmp_path / name for name in sources}
    for name, source in sources.items():
        paths[name].write_bytes(source)
    monkeypatch.setitem(PATCH["install"].__globals__, "patch", lambda source: source + b"modified")
    monkeypatch.setitem(PATCH["patch_push_worker"].__globals__, "PUSH_SOURCE_SHA256", hashlib.sha256(sources["push"]).hexdigest())
    with pytest.raises(ValueError, match="Unsupported pinned"):
        PATCH["install"](*(paths[name] for name in ("engine", "push", "mp", "mq", "gpu")))
    assert all(paths[name].read_bytes() == source for name, source in sources.items())
