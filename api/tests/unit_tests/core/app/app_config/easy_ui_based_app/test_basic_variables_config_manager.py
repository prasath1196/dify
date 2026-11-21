from unittest.mock import patch

import pytest

from core.app.app_config.easy_ui_based_app.variables.manager import BasicVariablesConfigManager
from core.app.app_config.entities import VariableEntityType

_FACTORY_VALIDATE = "core.app.app_config.easy_ui_based_app.variables.manager.ExternalDataToolFactory.validate_config"


def test_validate_and_set_defaults_returns_app_config_when_external_data_tool_enabled():
    tool_config = {"api_based_extension_id": "ext-1"}
    config = {
        "model": {"provider": "openai", "name": "gpt-4"},
        "user_input_form": [],
        "external_data_tools": [{"enabled": True, "type": "api", "variable": "weather", "config": tool_config}],
    }

    with patch(_FACTORY_VALIDATE) as validate_config:
        result, related_keys = BasicVariablesConfigManager.validate_and_set_defaults("tenant-1", config)

    validate_config.assert_called_once_with(name="api", tenant_id="tenant-1", config=tool_config)
    assert result is config
    assert result["model"] == {"provider": "openai", "name": "gpt-4"}
    assert related_keys == ["user_input_form", "external_data_tools"]


def test_validate_external_data_tools_marks_missing_enabled_as_disabled():
    config: dict = {"external_data_tools": [{"type": "api"}]}

    with patch(_FACTORY_VALIDATE) as validate_config:
        result, _ = BasicVariablesConfigManager.validate_external_data_tools_and_set_defaults("tenant-1", config)

    validate_config.assert_not_called()
    assert result["external_data_tools"] == [{"type": "api", "enabled": False}]


def test_validate_external_data_tools_requires_type_when_enabled():
    with pytest.raises(ValueError, match="type is required"):
        BasicVariablesConfigManager.validate_external_data_tools_and_set_defaults(
            "tenant-1", {"external_data_tools": [{"enabled": True, "config": {}}]}
        )


def test_validate_variables_sets_defaults():
    config: dict = {"user_input_form": [{"select": {"label": "Size", "variable": "size"}}]}

    result, _ = BasicVariablesConfigManager.validate_variables_and_set_defaults(config)

    assert result["user_input_form"] == [
        {"select": {"label": "Size", "variable": "size", "required": False, "options": []}}
    ]


@pytest.mark.parametrize(
    ("form_item", "message"),
    [
        ({"unknown": {"label": "x", "variable": "x"}}, "can only be"),
        ({"text-input": {"variable": "x"}}, "label is required"),
        ({"text-input": {"label": "x"}}, "variable is required"),
        ({"text-input": {"label": "x", "variable": "1abc"}}, "cannot start with a number"),
        ({"text-input": {"label": "x", "variable": "has space"}}, "cannot start with a number"),
        ({"select": {"label": "x", "variable": "x", "options": ["a"], "default": "b"}}, "must be in the options"),
    ],
)
def test_validate_variables_rejects_invalid_items(form_item: dict, message: str):
    with pytest.raises(ValueError, match=message):
        BasicVariablesConfigManager.validate_variables_and_set_defaults({"user_input_form": [form_item]})


def test_convert_collects_variables_and_enabled_external_data_tools():
    config = {
        "external_data_tools": [
            {"enabled": True, "variable": "legacy", "type": "api", "config": {"a": 1}},
            {"enabled": False, "variable": "off", "type": "api", "config": {}},
        ],
        "user_input_form": [
            {"text-input": {"variable": "name", "label": "Name", "required": True, "max_length": 20}},
            {"external_data_tool": {"variable": "weather", "type": "api", "config": {"b": 2}}},
            {"external_data_tool": {"variable": "no_config", "type": "api"}},
        ],
    }

    variables, external_data_variables = BasicVariablesConfigManager.convert(config)

    assert [(v.variable, v.type, v.required, v.max_length) for v in variables] == [
        ("name", VariableEntityType.TEXT_INPUT, True, 20)
    ]
    assert [(e.variable, e.config) for e in external_data_variables] == [("legacy", {"a": 1}), ("weather", {"b": 2})]
