from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from services.enterprise.enterprise_service import EnterpriseService

_SEND_REQUEST = "services.enterprise.enterprise_service.EnterpriseRequest.send_request"

_GETTERS = [
    EnterpriseService.get_app_sso_settings_last_update_time,
    EnterpriseService.get_workspace_sso_settings_last_update_time,
]


@pytest.mark.parametrize("getter", _GETTERS)
def test_timestamp_without_offset_is_treated_as_utc(getter):
    with patch(_SEND_REQUEST, return_value="2025-01-01T12:00:00"):
        last_update = getter()

    assert last_update == datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    # Comparable with the aware token grant time used by the web app auth check.
    assert datetime.fromtimestamp(0, tz=UTC) < last_update


@pytest.mark.parametrize("getter", _GETTERS)
def test_timestamp_with_offset_keeps_it(getter):
    with patch(_SEND_REQUEST, return_value="2025-01-01T20:00:00+08:00"):
        last_update = getter()

    assert last_update == datetime(2025, 1, 1, 20, 0, tzinfo=timezone(timedelta(hours=8)))


@pytest.mark.parametrize("getter", _GETTERS)
def test_invalid_timestamp_raises(getter):
    with patch(_SEND_REQUEST, return_value="not-a-date"):
        with pytest.raises(ValueError, match="Invalid date format"):
            getter()
