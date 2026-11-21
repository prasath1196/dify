from core.model_runtime.entities import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)
from core.prompt.utils.prompt_message_util import PromptMessageUtil

_TOOL_CALL = AssistantPromptMessage.ToolCall(
    id="call-1",
    type="function",
    function=AssistantPromptMessage.ToolCall.ToolCallFunction(name="get_weather", arguments='{"city": "Paris"}'),
)


def test_chat_mode_maps_roles_and_text():
    prompts = PromptMessageUtil.prompt_messages_to_prompt_for_saving(
        "chat",
        [
            SystemPromptMessage(content="be brief"),
            UserPromptMessage(content="hi"),
            AssistantPromptMessage(content="hello"),
        ],
    )

    assert prompts == [
        {"role": "system", "text": "be brief", "files": []},
        {"role": "user", "text": "hi", "files": []},
        {"role": "assistant", "text": "hello", "files": []},
    ]


def test_chat_mode_attaches_tool_calls_only_to_the_assistant_message():
    prompts = PromptMessageUtil.prompt_messages_to_prompt_for_saving(
        "chat",
        [
            UserPromptMessage(content="weather?"),
            AssistantPromptMessage(content="", tool_calls=[_TOOL_CALL]),
            ToolPromptMessage(content="sunny", tool_call_id="call-1"),
            UserPromptMessage(content="thanks"),
        ],
    )

    assert [("tool_calls" in prompt) for prompt in prompts] == [False, True, False, False]
    assert prompts[1]["tool_calls"] == [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
    ]
    assert prompts[2] == {"role": "tool", "text": "sunny", "files": []}


def test_chat_mode_joins_text_parts_and_truncates_image_data():
    image = ImagePromptMessageContent(format="png", mime_type="image/png", base64_data="A" * 10 + "MIDDLE" + "Z" * 10)

    prompts = PromptMessageUtil.prompt_messages_to_prompt_for_saving(
        "chat",
        [
            UserPromptMessage(
                content=[TextPromptMessageContent(data="look "), image, TextPromptMessageContent(data="here")]
            )
        ],
    )

    assert prompts == [
        {
            "role": "user",
            "text": "look here",
            "files": [
                {"type": "image", "data": image.data[:10] + "...[TRUNCATED]..." + image.data[-10:], "detail": "low"}
            ],
        }
    ]
    assert "MIDDLE" not in prompts[0]["files"][0]["data"]


def test_completion_mode_saves_only_first_message_as_user():
    prompts = PromptMessageUtil.prompt_messages_to_prompt_for_saving(
        "completion", [UserPromptMessage(content="complete this"), AssistantPromptMessage(content="ignored")]
    )

    assert prompts == [{"role": "user", "text": "complete this"}]
