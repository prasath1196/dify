"""User-supplied schema URLs must be fetched through the SSRF proxy, never with a bare HTTP client."""

import json
from unittest.mock import MagicMock, patch

from core.tools.utils.parser import ApiBasedToolSchemaParser
from services.tools.api_tools_manage_service import ApiToolManageService

_OPENAPI_YAML = """
openapi: 3.0.0
info:
  title: Example
  version: 1.0.0
servers:
  - url: https://api.example.com
paths:
  /ping:
    get:
      operationId: ping
      summary: Ping
      responses:
        "200":
          description: ok
"""


def test_remote_tool_schema_is_fetched_through_ssrf_proxy():
    response = MagicMock(status_code=200, text=_OPENAPI_YAML)

    with (
        patch("services.tools.api_tools_manage_service.ssrf_proxy.get", return_value=response) as ssrf_get,
        patch("services.tools.api_tools_manage_service.ApiToolManageService.parser_api_schema"),
    ):
        result = ApiToolManageService.get_api_tool_provider_remote_schema(
            "user-1", "tenant-1", "http://169.254.169.254/latest/meta-data"
        )

    ssrf_get.assert_called_once()
    assert ssrf_get.call_args.args[0] == "http://169.254.169.254/latest/meta-data"
    assert result == {"schema": _OPENAPI_YAML}


def test_openai_plugin_api_url_is_fetched_through_ssrf_proxy():
    plugin_json = json.dumps({"api": {"url": "http://10.0.0.5/openapi.yaml", "type": "openapi"}})
    response = MagicMock(status_code=200, text=_OPENAPI_YAML)
    app = MagicMock()

    with (
        patch("core.tools.utils.parser.ssrf_proxy.get", return_value=response) as ssrf_get,
        patch("core.tools.utils.parser.request", app),
    ):
        app.headers.get.return_value = None
        bundles = ApiBasedToolSchemaParser.parse_openai_plugin_json_to_tool_bundle(plugin_json)

    ssrf_get.assert_called_once()
    assert ssrf_get.call_args.args[0] == "http://10.0.0.5/openapi.yaml"
    assert [bundle.operation_id for bundle in bundles] == ["ping"]
