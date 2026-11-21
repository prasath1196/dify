from unittest.mock import patch

from core.agent.cot_chat_agent_runner import CotChatAgentRunner
from core.agent.entities import AgentScratchpadUnit
from core.model_runtime.entities import AssistantPromptMessage, SystemPromptMessage, UserPromptMessage


def _make_runner(scratchpad: list[AgentScratchpadUnit]) -> CotChatAgentRunner:
    runner = CotChatAgentRunner.__new__(CotChatAgentRunner)
    runner._agent_scratchpad = scratchpad
    runner._query = "what is the weather?"
    runner.files = []
    return runner


def _organize(runner: CotChatAgentRunner):
    with (
        patch.object(CotChatAgentRunner, "_organize_system_prompt", return_value=SystemPromptMessage(content="sys")),
        patch.object(CotChatAgentRunner, "_organize_historic_prompt_messages", return_value=[]),
    ):
        return runner._organize_prompt_messages()


def test_organize_prompt_messages_without_scratchpad():
    messages = _organize(_make_runner([]))

    assert messages == [SystemPromptMessage(content="sys"), UserPromptMessage(content="what is the weather?")]


def test_organize_prompt_messages_renders_scratchpad_as_assistant_message():
    scratchpad = [
        AgentScratchpadUnit(
            thought="I should look it up",
            action_str='{"action": "weather", "action_input": {"city": "Paris"}}',
            action=AgentScratchpadUnit.Action(action_name="weather", action_input={"city": "Paris"}),
            observation="sunny",
        ),
        AgentScratchpadUnit(
            thought="no action yet", action=AgentScratchpadUnit.Action(action_name="x", action_input="")
        ),
        AgentScratchpadUnit(agent_response="It is sunny in Paris."),
    ]

    messages = _organize(_make_runner(scratchpad))

    assert messages == [
        SystemPromptMessage(content="sys"),
        UserPromptMessage(content="what is the weather?"),
        AssistantPromptMessage(
            content=(
                "Thought: I should look it up\n\n"
                'Action: {"action": "weather", "action_input": {"city": "Paris"}}\n\n'
                "Observation: sunny\n\n"
                "Thought: no action yet\n\n"
                "Final Answer: It is sunny in Paris."
            )
        ),
        UserPromptMessage(content="continue"),
    ]
