from unittest.mock import MagicMock, patch

import pytest

from core.model_runtime.callbacks.base_callback import Callback
from core.model_runtime.entities.message_entities import UserPromptMessage
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel


class _StopInvokeError(Exception):
    pass


def test_invoke_does_not_add_logging_callback_to_caller_list():
    model = LargeLanguageModel.model_construct()
    caller_callback = MagicMock(spec=Callback)
    callbacks: list[Callback] = [caller_callback]

    with (
        patch("core.model_runtime.model_providers.__base.large_language_model.dify_config.DEBUG", True),
        patch.object(LargeLanguageModel, "_trigger_before_invoke_callbacks", side_effect=_StopInvokeError),
    ):
        with pytest.raises(_StopInvokeError):
            model.invoke(
                model="m", credentials={}, prompt_messages=[UserPromptMessage(content="hi")], callbacks=callbacks
            )

    assert callbacks == [caller_callback]
