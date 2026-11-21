import pytest

from core.model_runtime.entities.common_entities import I18nObject
from core.model_runtime.entities.provider_entities import (
    CredentialFormSchema,
    FormOption,
    FormShowOnObject,
    FormType,
)
from core.model_runtime.schema_validators.common_validator import CommonValidator


def _schema(variable: str, form_type: FormType = FormType.TEXT_INPUT, **kwargs) -> CredentialFormSchema:
    return CredentialFormSchema(variable=variable, label=I18nObject(en_US=variable), type=form_type, **kwargs)


def _validate(schemas: list[CredentialFormSchema], credentials: dict) -> dict:
    return CommonValidator()._validate_and_filter_credential_form_schemas(schemas, credentials)


@pytest.mark.parametrize(("raw", "expected"), [("true", True), ("TRUE", True), ("false", False), ("False", False)])
def test_switch_values_are_converted_and_kept(raw: str, expected: bool):
    schema = _schema("stream", FormType.SWITCH, required=False, default="true")

    assert _validate([schema], {"stream": raw}) == {"stream": expected}


def test_switch_rejects_non_boolean_text():
    with pytest.raises(ValueError, match="should be true or false"):
        _validate([_schema("stream", FormType.SWITCH)], {"stream": "yes"})


def test_missing_required_value_raises():
    with pytest.raises(ValueError, match="api_key is required"):
        _validate([_schema("api_key", required=True)], {"api_key": ""})


def test_missing_optional_value_uses_default_or_is_omitted():
    schemas = [_schema("base_url", required=False, default="https://api.example.com"), _schema("org", required=False)]

    assert _validate(schemas, {}) == {"base_url": "https://api.example.com"}


def test_unknown_credentials_are_filtered_out():
    assert _validate([_schema("api_key", required=True)], {"api_key": "sk", "extra": "x"}) == {"api_key": "sk"}


def test_max_length_is_enforced():
    with pytest.raises(ValueError, match="greater than 3"):
        _validate([_schema("code", max_length=3)], {"code": "abcd"})


def test_select_value_must_be_an_option():
    schema = _schema(
        "region",
        FormType.SELECT,
        options=[
            FormOption(label=I18nObject(en_US="US"), value="us"),
            FormOption(label=I18nObject(en_US="EU"), value="eu"),
        ],
    )

    assert _validate([schema], {"region": "eu"}) == {"region": "eu"}
    with pytest.raises(ValueError, match="not in options"):
        _validate([schema], {"region": "apac"})


def test_show_on_skips_hidden_fields():
    schema = _schema("endpoint", required=True, show_on=[FormShowOnObject(variable="mode", value="custom")])

    assert _validate([schema], {"mode": "default"}) == {}
    assert _validate([schema], {"mode": "custom", "endpoint": "https://x"}) == {"endpoint": "https://x"}
