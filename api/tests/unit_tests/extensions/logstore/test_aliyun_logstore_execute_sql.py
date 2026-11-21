import logging
from unittest.mock import MagicMock

import pytest

from extensions.logstore.aliyun_logstore import AliyunLogStore


def _make_sdk_mode_logstore() -> AliyunLogStore:
    logstore = AliyunLogStore.__new__(AliyunLogStore)
    logstore._use_pg_protocol = False
    logstore._pg_client = None
    logstore.project_name = "project"
    logstore.log_enabled = True
    logstore.client = MagicMock()
    logstore.client.get_logs.return_value.get_logs.return_value = []
    return logstore


def test_execute_sql_sdk_mode_logs_combined_query(caplog: pytest.LogCaptureFixture):
    logstore = _make_sdk_mode_logstore()

    with caplog.at_level(logging.INFO, logger="extensions.logstore.aliyun_logstore"):
        result = logstore.execute_sql("SELECT count(*)", logstore="workflow", query="status:ok", from_time=1, to_time=2)

    assert result == []
    messages = [record.getMessage() for record in caplog.records]
    assert (
        "[LogStore-SDK] EXECUTE_SQL | logstore=workflow | project=project | from_time=1 | to_time=2 "
        "| full_query=status:ok | SELECT count(*)"
    ) in messages
