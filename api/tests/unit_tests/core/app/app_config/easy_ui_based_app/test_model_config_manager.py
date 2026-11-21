import copy

import pytest

from core.app.app_config.easy_ui_based_app.model_config.manager import ModelConfigManager


def _config() -> dict:
    return {
        "model": {
            "provider": "langgenius/openai/openai",
            "name": "gpt-4o",
            "mode": "chat",
            "completion_params": {"temperature": 0.2, "stop": ["Observation:"]},
        }
    }


def test_convert_splits_stop_from_parameters():
    entity = ModelConfigManager.convert(_config())

    assert entity.provider == "langgenius/openai/openai"
    assert entity.model == "gpt-4o"
    assert entity.mode == "chat"
    assert entity.parameters == {"temperature": 0.2}
    assert entity.stop == ["Observation:"]


def test_convert_does_not_mutate_config():
    config = _config()
    original = copy.deepcopy(config)

    ModelConfigManager.convert(config)

    assert config == original


def test_convert_is_repeatable_on_same_config():
    config = _config()

    ModelConfigManager.convert(config)
    entity = ModelConfigManager.convert(config)

    assert entity.stop == ["Observation:"]


def test_convert_requires_model():
    with pytest.raises(ValueError, match="model is required"):
        ModelConfigManager.convert({})


@pytest.mark.parametrize(
    ("stop", "message"), [("END", "must be of list type"), (["a", "b", "c", "d", "e"], "less than 4")]
)
def test_validate_model_completion_params_rejects_invalid_stop(stop: object, message: str):
    with pytest.raises(ValueError, match=message):
        ModelConfigManager.validate_model_completion_params({"stop": stop})


def test_validate_model_completion_params_defaults_stop():
    assert ModelConfigManager.validate_model_completion_params({"temperature": 1}) == {"temperature": 1, "stop": []}
