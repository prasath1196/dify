import json
from unittest.mock import MagicMock, patch

import pytest

from services.auth.api_key_auth_base import ApiKeyAuthBase
from services.auth.jina.jina import JinaAuth
from services.auth.watercrawl.watercrawl import WatercrawlAuth

_PROVIDERS = [
    pytest.param(
        JinaAuth,
        {"auth_type": "bearer", "config": {"api_key": "key"}},
        "services.auth.jina.jina.httpx.post",
        id="jina",
    ),
    pytest.param(
        WatercrawlAuth,
        {"auth_type": "x-api-key", "config": {"api_key": "key"}},
        "services.auth.watercrawl.watercrawl.httpx.get",
        id="watercrawl",
    ),
]


def _non_json_response(status_code: int, text: str) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.json.side_effect = json.JSONDecodeError("Expecting value", text, 0)
    return response


@pytest.mark.parametrize(("auth_cls", "credentials", "http_target"), _PROVIDERS)
@pytest.mark.parametrize("status_code", [403, 500])
def test_non_json_error_body_reports_status_and_text(
    auth_cls: type[ApiKeyAuthBase], credentials: dict, http_target: str, status_code: int
):
    with patch(http_target, return_value=_non_json_response(status_code, "<html>Bad Gateway</html>")):
        with pytest.raises(Exception) as exc_info:
            auth_cls(credentials).validate_credentials()

    assert str(exc_info.value) == f"Failed to authorize. Status code: {status_code}. Error: <html>Bad Gateway</html>"
