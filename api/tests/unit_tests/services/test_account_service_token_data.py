from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest

from services.account_service import AccountService

_GENERATORS: list[tuple[str, Callable[..., tuple[str, str]]]] = [
    ("reset_password", AccountService.generate_reset_password_token),
    ("email_register", AccountService.generate_email_register_token),
    ("change_email", AccountService.generate_change_email_token),
    ("owner_transfer", AccountService.generate_owner_transfer_token),
]


@pytest.mark.parametrize(("token_type", "generate"), _GENERATORS, ids=[name for name, _ in _GENERATORS])
def test_interleaved_calls_keep_their_own_code(token_type: str, generate: Callable[..., tuple[str, str]]):
    stored: dict[str, dict[str, Any]] = {}

    def fake_generate_token(*, token_type: str, additional_data: dict[str, Any], **_: Any) -> str:
        if not stored:
            # Another request runs while this one is waiting on Redis, before its data is persisted.
            stored["second"] = {}
            generate("second@example.com", code="222222")
            stored["first"] = dict(additional_data)
            return "first-token"
        stored["second"] = dict(additional_data)
        return "second-token"

    with patch("services.account_service.TokenManager.generate_token", side_effect=fake_generate_token):
        code, _ = generate("first@example.com", code="111111")

    assert code == "111111"
    assert stored["first"]["code"] == "111111"
    assert stored["second"]["code"] == "222222"


@pytest.mark.parametrize(("token_type", "generate"), _GENERATORS, ids=[name for name, _ in _GENERATORS])
def test_caller_additional_data_is_not_mutated(token_type: str, generate: Callable[..., tuple[str, str]]):
    additional_data = {"phase": "reset"}

    with patch("services.account_service.TokenManager.generate_token", return_value="token") as generate_token:
        generate("user@example.com", code="123456", additional_data=additional_data)

    assert additional_data == {"phase": "reset"}
    passed = generate_token.call_args.kwargs["additional_data"]
    assert passed["phase"] == "reset"
    assert passed["code"] == "123456"
