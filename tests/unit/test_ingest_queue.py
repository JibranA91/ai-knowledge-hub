"""Unit tests for the durable ingest queue's pure guard logic (no DB/loop).

The queue is DB-backed now, so ordering/claiming is covered by integration
tests. Here we only cover the in-process drain guards, which short-circuit
before touching asyncio/DB.
"""
from app.services import ingest_queue as iq


def test_kick_noop_when_already_draining():
    iq._draining.add("org-x")
    before = len(iq._tasks)
    try:
        iq._kick("org-x")  # already draining → must not spawn a task
        assert len(iq._tasks) == before
    finally:
        iq._draining.discard("org-x")


def test_kick_noop_when_stopped():
    iq._stopped = True
    before = len(iq._tasks)
    try:
        iq._kick("org-y")  # shutting down → must not spawn a task
        assert len(iq._tasks) == before
    finally:
        iq._stopped = False
