import pytest
from pydantic import ValidationError

from core.workflow.nodes.tool.entities import ToolEntity

_BASE = {
    "provider_id": "provider",
    "provider_type": "builtin",
    "provider_name": "Provider",
    "tool_name": "tool",
    "tool_label": "Tool",
}


def test_tool_configurations_keeps_scalar_and_selector_values():
    configurations = {
        "query": "weather",
        "limit": 5,
        "verbose": True,
        "model": {"provider": "openai", "model": "gpt-4o", "mode": "chat"},
    }

    entity = ToolEntity(**_BASE, tool_configurations=configurations)

    assert entity.tool_configurations == configurations


def test_tool_configurations_must_be_a_dict():
    with pytest.raises(ValidationError, match="tool_configurations must be a dictionary"):
        ToolEntity(**_BASE, tool_configurations=["not", "a", "dict"])
