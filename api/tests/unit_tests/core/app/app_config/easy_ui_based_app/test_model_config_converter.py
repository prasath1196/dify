from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.app.app_config.easy_ui_based_app.model_config.converter import ModelConfigConverter
from core.app.app_config.entities import ModelConfigEntity
from core.entities.model_entities import ModelStatus

_MODULE = "core.app.app_config.easy_ui_based_app.model_config.converter"


def _convert(model_config: ModelConfigEntity) -> dict[str, Any]:
    app_config = MagicMock()
    app_config.tenant_id = "tenant-1"
    app_config.model = model_config

    with (
        patch(f"{_MODULE}.ProviderManager") as provider_manager,
        patch(f"{_MODULE}.ModelConfigWithCredentialsEntity") as entity_cls,
    ):
        configuration = provider_manager.return_value.get_provider_model_bundle.return_value.configuration
        configuration.get_current_credentials.return_value = {"api_key": "sk"}
        configuration.get_provider_model.return_value.status = ModelStatus.ACTIVE
        ModelConfigConverter.convert(app_config)

    return entity_cls.call_args.kwargs


def test_convert_passes_configured_stop_sequences():
    model_config = ModelConfigEntity(
        provider="langgenius/openai/openai", model="gpt-4o", mode="chat", parameters={"temperature": 0.1}, stop=["END"]
    )

    kwargs = _convert(model_config)

    assert kwargs["stop"] == ["END"]
    assert kwargs["parameters"] == {"temperature": 0.1}


def test_convert_moves_stop_left_in_parameters():
    model_config = ModelConfigEntity(
        provider="langgenius/openai/openai", model="gpt-4o", mode="chat", parameters={"stop": ["STOP"], "top_p": 1}
    )

    kwargs = _convert(model_config)

    assert kwargs["stop"] == ["STOP"]
    assert kwargs["parameters"] == {"top_p": 1}


def test_convert_does_not_mutate_app_config_model():
    model_config = ModelConfigEntity(
        provider="langgenius/openai/openai", model="gpt-4o", mode="chat", parameters={"stop": ["STOP"]}, stop=["END"]
    )

    kwargs = _convert(model_config)
    kwargs["stop"].append("Observation")

    assert model_config.parameters == {"stop": ["STOP"]}
    assert model_config.stop == ["END"]


@pytest.mark.parametrize(
    ("status", "error"),
    [(ModelStatus.NO_CONFIGURE, "credentials is not initialized"), (ModelStatus.QUOTA_EXCEEDED, "quota exceeded")],
)
def test_convert_rejects_unusable_models(status: ModelStatus, error: str):
    app_config = MagicMock()
    app_config.model = ModelConfigEntity(provider="p/p/p", model="m")

    with patch(f"{_MODULE}.ProviderManager") as provider_manager:
        configuration = provider_manager.return_value.get_provider_model_bundle.return_value.configuration
        configuration.get_current_credentials.return_value = {}
        configuration.get_provider_model.return_value.status = status
        with pytest.raises(Exception, match=error):
            ModelConfigConverter.convert(app_config)
