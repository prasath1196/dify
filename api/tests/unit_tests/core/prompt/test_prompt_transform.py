from unittest.mock import MagicMock, patch

import pytest

from core.app.entities.app_invoke_entities import ModelConfigWithCredentialsEntity
from core.model_runtime.entities.message_entities import UserPromptMessage
from core.model_runtime.entities.model_entities import AIModelEntity, ModelPropertyKey, ParameterRule
from core.prompt.prompt_transform import PromptTransform


def _make_parameter_rule(name: str, use_template: str | None = None) -> ParameterRule:
    rule = MagicMock(spec=ParameterRule)
    rule.name = name
    rule.use_template = use_template
    return rule


def _make_model_config(
    *,
    context_size: int | None,
    parameter_rules: list[ParameterRule],
    parameters: dict[str, int],
) -> ModelConfigWithCredentialsEntity:
    model_schema = MagicMock(spec=AIModelEntity)
    model_schema.model_properties = {} if context_size is None else {ModelPropertyKey.CONTEXT_SIZE: context_size}
    model_schema.parameter_rules = parameter_rules

    model_config = MagicMock(spec=ModelConfigWithCredentialsEntity)
    model_config.model = "gpt-4"
    model_config.model_schema = model_schema
    model_config.parameters = parameters
    model_config.provider_model_bundle = MagicMock()
    return model_config


def _calculate(model_config: ModelConfigWithCredentialsEntity, prompt_tokens: int) -> int:
    with patch("core.prompt.prompt_transform.ModelInstance") as model_instance_cls:
        model_instance_cls.return_value.get_llm_num_tokens.return_value = prompt_tokens
        return PromptTransform()._calculate_rest_token([UserPromptMessage(content="Hello")], model_config)


def test_calculate_rest_token_defaults_when_context_size_unknown():
    model_config = _make_model_config(context_size=None, parameter_rules=[], parameters={})

    assert _calculate(model_config, prompt_tokens=10) == 2000


@pytest.mark.parametrize(
    ("rule", "parameters"),
    [
        (_make_parameter_rule("max_tokens"), {"max_tokens": 50}),
        (_make_parameter_rule("max_output_tokens", use_template="max_tokens"), {"max_output_tokens": 50}),
        (_make_parameter_rule("max_output_tokens", use_template="max_tokens"), {"max_tokens": 50}),
    ],
    ids=["named_max_tokens", "templated_rule_name", "templated_template_key"],
)
def test_calculate_rest_token_subtracts_max_tokens_and_prompt(rule: ParameterRule, parameters: dict[str, int]):
    model_config = _make_model_config(context_size=200, parameter_rules=[rule], parameters=parameters)

    assert _calculate(model_config, prompt_tokens=6) == 200 - 50 - 6


def test_calculate_rest_token_treats_unset_max_tokens_as_zero():
    model_config = _make_model_config(
        context_size=200, parameter_rules=[_make_parameter_rule("max_tokens")], parameters={}
    )

    assert _calculate(model_config, prompt_tokens=6) == 194


def test_calculate_rest_token_never_negative():
    model_config = _make_model_config(
        context_size=62, parameter_rules=[_make_parameter_rule("max_tokens")], parameters={"max_tokens": 50}
    )

    assert _calculate(model_config, prompt_tokens=100) == 0
