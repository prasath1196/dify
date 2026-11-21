import queue
import time
from collections.abc import Iterator
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from core.workflow.enums import NodeType
from core.workflow.graph_engine.worker import Worker
from core.workflow.graph_events import NodeRunFailedEvent
from libs.datetime_utils import naive_utc_now


@pytest.fixture
def non_utc_timezone(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_unexpected_node_error_event_uses_utc_start_time(non_utc_timezone: None):
    event_queue: queue.Queue = queue.Queue()
    ready_queue = MagicMock()
    graph = MagicMock()
    graph.nodes = {"node-1": MagicMock(id="node-1", node_type=NodeType.CODE)}
    worker = Worker(ready_queue=ready_queue, event_queue=event_queue, graph=graph, layers=[])

    def next_node(timeout: float) -> str:
        if ready_queue.get.call_count > 1:
            worker.stop()
            raise queue.Empty
        return "node-1"

    ready_queue.get.side_effect = next_node

    with patch.object(Worker, "_execute_node", side_effect=RuntimeError("boom")):
        worker.run()

    event = event_queue.get_nowait()
    assert isinstance(event, NodeRunFailedEvent)
    assert event.error == "boom"
    assert abs(event.start_at - naive_utc_now()) < timedelta(minutes=1)
