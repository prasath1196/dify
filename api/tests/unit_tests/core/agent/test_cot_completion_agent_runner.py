from unittest.mock import MagicMock, patch

from core.agent.cot_completion_agent_runner import CotCompletionAgentRunner
from core.agent.entities import AgentPromptEntity, AgentScratchpadUnit
from core.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessage,
    TextPromptMessageContent,
    UserPromptMessage,
)


def _make_runner(history: list[PromptMessage], scratchpad: list[AgentScratchpadUnit]) -> CotCompletionAgentRunner:
    runner = CotCompletionAgentRunner.__new__(CotCompletionAgentRunner)
    runner._instruction = "Be helpful."
    runner._prompt_messages_tools = []
    runner._agent_scratchpad = scratchpad
    runner._query = "What now?"
    runner.app_config = MagicMock()
    runner.app_config.agent.prompt = AgentPromptEntity(
        first_prompt="{{instruction}}|{{historic_messages}}|{{agent_scratchpad}}|{{query}}",
        next_iteration="",
    )
    runner._organize_historic_prompt_messages = MagicMock(return_value=history)
    return runner


def test_historic_prompt_uses_text_of_user_messages_with_files():
    image = ImagePromptMessageContent(url="https://example.com/a.png", format="png", mime_type="image/png")
    history: list[PromptMessage] = [
        UserPromptMessage(content=[image, TextPromptMessageContent(data="What is this?")]),
        AssistantPromptMessage(content="A cat."),
    ]

    historic_prompt = _make_runner(history, [])._organize_historic_prompt()

    assert historic_prompt == "Question: What is this?\n\nA cat.\n\n"


def test_historic_prompt_formats_plain_messages():
    history: list[PromptMessage] = [
        UserPromptMessage(content="Hi"),
        AssistantPromptMessage(content="Hello"),
        AssistantPromptMessage(content=[TextPromptMessageContent(data="More")]),
    ]

    assert _make_runner(history, [])._organize_historic_prompt() == "Question: Hi\n\nHello\n\nMore\n\n"


def test_organize_prompt_messages_fills_template():
    scratchpad = [
        AgentScratchpadUnit(
            thought="look up",
            action_str='{"action": "search"}',
            action=AgentScratchpadUnit.Action(action_name="search", action_input={}),
            observation="found",
        ),
        AgentScratchpadUnit(agent_response="Done."),
    ]
    runner = _make_runner([UserPromptMessage(content="Hi"), AssistantPromptMessage(content="Hello")], scratchpad)

    with patch.object(CotCompletionAgentRunner, "_organize_historic_prompt", return_value="H"):
        messages = runner._organize_prompt_messages()

    assert messages == [
        UserPromptMessage(
            content=(
                "Be helpful.|H|"
                'Thought: look up\n\nAction: {"action": "search"}\n\nObservation: found\n\nFinal Answer: Done.'
                "|Question: What now?"
            )
        )
    ]
