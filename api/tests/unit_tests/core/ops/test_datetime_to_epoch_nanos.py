import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta, timezone

import pytest

from core.ops.aliyun_trace.data_exporter.traceclient import convert_datetime_to_nanoseconds as aliyun_convert
from core.ops.arize_phoenix_trace.arize_phoenix_trace import datetime_to_nanos as arize_convert
from core.ops.mlflow_trace.mlflow_trace import datetime_to_nanoseconds as mlflow_convert
from core.ops.tencent_trace.utils import TencentTraceUtils
from core.ops.utils import datetime_to_epoch_nanos, measure_time

_NAIVE_UTC = datetime(2025, 1, 1, 12, 0, 0)
_EXPECTED_NS = int(datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC).timestamp()) * 1_000_000_000

_CONVERTERS: list[Callable[[datetime], int | None]] = [
    datetime_to_epoch_nanos,
    aliyun_convert,
    arize_convert,
    mlflow_convert,
    TencentTraceUtils.convert_datetime_to_nanoseconds,
]


@pytest.fixture
def non_utc_timezone(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.mark.parametrize("convert", _CONVERTERS)
def test_naive_datetimes_are_treated_as_utc(convert: Callable[[datetime], int | None], non_utc_timezone: None):
    assert convert(_NAIVE_UTC) == _EXPECTED_NS


def test_aware_datetimes_keep_their_offset(non_utc_timezone: None):
    aware = datetime(2025, 1, 1, 20, 0, 0, tzinfo=timezone(timedelta(hours=8)))

    assert datetime_to_epoch_nanos(aware) == _EXPECTED_NS


def test_none_handling_is_unchanged():
    assert aliyun_convert(None) is None
    assert mlflow_convert(None) is None
    now_ns = int(datetime.now(UTC).timestamp() * 1_000_000_000)
    assert abs(arize_convert(None) - now_ns) < 60 * 1_000_000_000
    assert abs(TencentTraceUtils.convert_datetime_to_nanoseconds(None) - now_ns) < 60 * 1_000_000_000


def test_measure_time_records_naive_utc(non_utc_timezone: None):
    with measure_time() as timer:
        pass

    utc_now = datetime.now(UTC).replace(tzinfo=None)
    assert timer["start"].tzinfo is None
    assert abs(timer["start"] - utc_now) < timedelta(minutes=1)
    assert timer["end"] >= timer["start"]
