from unittest.mock import MagicMock, patch

from core.llm_generator.llm_generator import LLMGenerator
from core.model_runtime.entities.llm_entities import LLMResult, LLMUsage
from core.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    TextPromptMessageContent,
)

_MODULE = "core.llm_generator.llm_generator"


def _result_with_list_content(*texts: str) -> LLMResult:
    image = ImagePromptMessageContent(url="https://example.com/a.png", format="png", mime_type="image/png")
    content = [image, *(TextPromptMessageContent(data=text) for text in texts)]
    return LLMResult(
        model="m",
        prompt_messages=[],
        message=AssistantPromptMessage(content=content),
        usage=LLMUsage.empty_usage(),
    )


def _patched_model_manager(result: LLMResult):
    manager = patch(f"{_MODULE}.ModelManager")
    mock = manager.start()
    instance = MagicMock()
    instance.invoke_llm.return_value = result
    mock.return_value.get_default_model_instance.return_value = instance
    mock.return_value.get_model_instance.return_value = instance
    return manager


def test_generate_qa_document_uses_text_parts_of_list_content():
    manager = _patched_model_manager(_result_with_list_content("Q1: a?\n", "A1: b  "))
    try:
        answer = LLMGenerator.generate_qa_document("tenant-1", "query", "English")
    finally:
        manager.stop()

    assert answer == "Q1: a?\nA1: b"


def test_generate_code_returns_text_for_list_content():
    manager = _patched_model_manager(_result_with_list_content("print(", "1)"))
    try:
        result = LLMGenerator.generate_code(
            "tenant-1",
            "print one",
            {"provider": "openai", "name": "gpt-4o", "completion_params": {}},
            "python3",
        )
    finally:
        manager.stop()

    assert result == {"code": "print(1)", "language": "python3", "error": ""}
