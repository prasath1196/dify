import time
from collections.abc import Iterator
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from models import Account
from services.agent_service import AgentService

_MODULE = "services.agent_service"


@pytest.fixture
def non_utc_server_timezone(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _get_agent_logs(user_timezone: str) -> dict:
    app_model = MagicMock(id="app-1", tenant_id="tenant-1")
    app_model.app_model_config.agent_mode_dict = {"strategy": "function_call"}
    conversation = MagicMock(from_end_user_id=None, from_account_id="account-1")
    message = MagicMock(
        created_at=datetime(2025, 1, 1, 12, 0, 0),  # naive UTC, as stored in the database
        provider_response_latency=1.5,
        answer_tokens=10,
        message_tokens=5,
        agent_thoughts=[],
        message_files=[],
    )
    account = MagicMock(spec=Account)
    account.timezone = user_timezone

    with (
        patch(f"{_MODULE}.db") as db,
        patch(f"{_MODULE}.current_user", account),
        patch(f"{_MODULE}.AgentConfigManager.convert", return_value=MagicMock(tools=[])),
    ):
        db.session.query.return_value.where.return_value.first.side_effect = [
            conversation,
            message,
            MagicMock(name="executor"),
        ]
        return AgentService.get_agent_logs(app_model, "conversation-1", "message-1")


def test_agent_log_start_time_is_converted_from_utc_to_user_timezone(non_utc_server_timezone: None):
    result = _get_agent_logs("America/New_York")

    assert result["meta"]["start_time"] == "2025-01-01T07:00:00-05:00"


def test_agent_log_meta_summarises_message(non_utc_server_timezone: None):
    meta = _get_agent_logs("UTC")["meta"]

    assert meta["start_time"] == "2025-01-01T12:00:00+00:00"
    assert (meta["total_tokens"], meta["agent_mode"], meta["iterations"]) == (15, "function_call", 0)
