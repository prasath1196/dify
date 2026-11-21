import base64
import uuid
from collections.abc import Sequence
from typing import cast
from unittest import mock

import pytest

from core.app.entities.app_invoke_entities import InvokeFrom, ModelConfigWithCredentialsEntity
from core.entities.provider_configuration import ProviderConfiguration, ProviderModelBundle
from core.entities.provider_entities import CustomConfiguration, SystemConfiguration
from core.file import File, FileTransferMethod, FileType
from core.memory.token_buffer_memory import TokenBufferMemory
from core.model_runtime.entities.common_entities import I18nObject
from core.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageRole,
    SystemPromptMessage,
    TextPromptMessageContent,
    UserPromptMessage,
)
from core.model_runtime.entities.model_entities import AIModelEntity, FetchFrom, ModelFeature, ModelType
from core.model_runtime.model_providers.model_provider_factory import ModelProviderFactory
from core.prompt.entities.advanced_prompt_entities import MemoryConfig
from core.variables import ArrayAnySegment, ArrayFileSegment, NoneSegment
from core.workflow.entities import GraphInitParams
from core.workflow.nodes.llm import llm_utils
from core.workflow.nodes.llm.entities import (
    ContextConfig,
    LLMNodeChatModelMessage,
    LLMNodeData,
    ModelConfig,
    VisionConfig,
    VisionConfigOptions,
)
from core.workflow.nodes.llm.file_saver import LLMFileSaver
from core.workflow.nodes.llm.node import LLMNode
from core.workflow.runtime import GraphRuntimeState, VariablePool
from core.workflow.system_variable import SystemVariable
from models.enums import UserFrom
from models.provider import ProviderType


class MockTokenBufferMemory:
    def __init__(self, history_messages=None):
        self.history_messages = history_messages or []

    def get_history_prompt_messages(
        self, max_token_limit: int = 2000, message_limit: int | None = None
    ) -> Sequence[PromptMessage]:
        if message_limit is not None:
            return self.history_messages[-message_limit * 2 :]
        return self.history_messages


@pytest.fixture
def llm_node_data() -> LLMNodeData:
    return LLMNodeData(
        title="Test LLM",
        model=ModelConfig(provider="openai", name="gpt-3.5-turbo", mode="chat", completion_params={}),
        prompt_template=[],
        memory=None,
        context=ContextConfig(enabled=False),
        vision=VisionConfig(
            enabled=True,
            configs=VisionConfigOptions(
                variable_selector=["sys", "files"],
                detail=ImagePromptMessageContent.DETAIL.HIGH,
            ),
        ),
        reasoning_format="tagged",
    )


@pytest.fixture
def graph_init_params() -> GraphInitParams:
    return GraphInitParams(
        tenant_id="1",
        app_id="1",
        workflow_id="1",
        graph_config={},
        user_id="1",
        user_from=UserFrom.ACCOUNT,
        invoke_from=InvokeFrom.SERVICE_API,
        call_depth=0,
    )


@pytest.fixture
def graph_runtime_state() -> GraphRuntimeState:
    variable_pool = VariablePool(
        system_variables=SystemVariable.empty(),
        user_inputs={},
    )
    return GraphRuntimeState(
        variable_pool=variable_pool,
        start_at=0,
    )


@pytest.fixture
def llm_node(
    llm_node_data: LLMNodeData, graph_init_params: GraphInitParams, graph_runtime_state: GraphRuntimeState
) -> LLMNode:
    mock_file_saver = mock.MagicMock(spec=LLMFileSaver)
    node_config = {
        "id": "1",
        "data": llm_node_data.model_dump(),
    }
    node = LLMNode(
        id="1",
        config=node_config,
        graph_init_params=graph_init_params,
        graph_runtime_state=graph_runtime_state,
        llm_file_saver=mock_file_saver,
    )
    return node


@pytest.fixture
def model_config():
    # Create actual provider and model type instances
    model_provider_factory = ModelProviderFactory(tenant_id="test")
    provider_instance = model_provider_factory.get_plugin_model_provider("openai")
    model_type_instance = model_provider_factory.get_model_type_instance("openai", ModelType.LLM)

    # Create a ProviderModelBundle
    provider_model_bundle = ProviderModelBundle(
        configuration=ProviderConfiguration(
            tenant_id="1",
            provider=provider_instance,
            preferred_provider_type=ProviderType.CUSTOM,
            using_provider_type=ProviderType.CUSTOM,
            system_configuration=SystemConfiguration(enabled=False),
            custom_configuration=CustomConfiguration(provider=None),
            model_settings=[],
        ),
        model_type_instance=model_type_instance,
    )

    # Create and return a ModelConfigWithCredentialsEntity
    return ModelConfigWithCredentialsEntity(
        provider="openai",
        model="gpt-3.5-turbo",
        model_schema=AIModelEntity(
            model="gpt-3.5-turbo",
            label=I18nObject(en_US="GPT-3.5 Turbo"),
            model_type=ModelType.LLM,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={},
        ),
        mode="chat",
        credentials={},
        parameters={},
        provider_model_bundle=provider_model_bundle,
    )


def test_fetch_files_with_file_segment():
    file = File(
        id="1",
        tenant_id="test",
        type=FileType.IMAGE,
        filename="test.jpg",
        transfer_method=FileTransferMethod.LOCAL_FILE,
        related_id="1",
        storage_key="",
    )
    variable_pool = VariablePool.empty()
    variable_pool.add(["sys", "files"], file)

    result = llm_utils.fetch_files(variable_pool=variable_pool, selector=["sys", "files"])
    assert result == [file]


def test_fetch_files_with_array_file_segment():
    files = [
        File(
            id="1",
            tenant_id="test",
            type=FileType.IMAGE,
            filename="test1.jpg",
            transfer_method=FileTransferMethod.LOCAL_FILE,
            related_id="1",
            storage_key="",
        ),
        File(
            id="2",
            tenant_id="test",
            type=FileType.IMAGE,
            filename="test2.jpg",
            transfer_method=FileTransferMethod.LOCAL_FILE,
            related_id="2",
            storage_key="",
        ),
    ]
    variable_pool = VariablePool.empty()
    variable_pool.add(["sys", "files"], ArrayFileSegment(value=files))

    result = llm_utils.fetch_files(variable_pool=variable_pool, selector=["sys", "files"])
    assert result == files


def test_fetch_files_with_none_segment():
    variable_pool = VariablePool.empty()
    variable_pool.add(["sys", "files"], NoneSegment())

    result = llm_utils.fetch_files(variable_pool=variable_pool, selector=["sys", "files"])
    assert result == []


def test_fetch_files_with_array_any_segment():
    variable_pool = VariablePool.empty()
    variable_pool.add(["sys", "files"], ArrayAnySegment(value=[]))

    result = llm_utils.fetch_files(variable_pool=variable_pool, selector=["sys", "files"])
    assert result == []


def test_fetch_files_with_non_existent_variable():
    variable_pool = VariablePool.empty()
    result = llm_utils.fetch_files(variable_pool=variable_pool, selector=["sys", "files"])
    assert result == []


def _make_vision_model_config(features: list[ModelFeature]) -> ModelConfigWithCredentialsEntity:
    model_config = mock.MagicMock(spec=ModelConfigWithCredentialsEntity)
    model_config.provider = "openai"
    model_config.model = "gpt-4o"
    model_config.stop = None
    model_config.model_schema = mock.MagicMock(spec=AIModelEntity)
    model_config.model_schema.features = features
    return model_config


def _make_remote_image_file() -> File:
    return File(
        id="1",
        tenant_id="test",
        type=FileType.IMAGE,
        filename="test1.jpg",
        transfer_method=FileTransferMethod.REMOTE_URL,
        remote_url="https://example.com/test1.jpg",
        storage_key="",
    )


@pytest.mark.parametrize(
    ("vision_enabled", "features", "expect_image"),
    [
        (False, [ModelFeature.VISION], False),
        (True, [ModelFeature.VISION], True),
        (True, [], False),
    ],
    ids=["vision_disabled", "vision_enabled", "model_without_vision"],
)
def test_fetch_prompt_messages_attaches_sys_files_only_with_vision(
    vision_enabled: bool, features: list[ModelFeature], expect_image: bool
):
    image_content = ImagePromptMessageContent(url="https://example.com/test1.jpg", format="jpg", mime_type="image/jpeg")

    with (
        mock.patch("core.workflow.nodes.llm.node.ModelManager"),
        mock.patch("core.workflow.nodes.llm.node.file_manager.to_prompt_message_content", return_value=image_content),
    ):
        prompt_messages, stop = LLMNode.fetch_prompt_messages(
            sys_query="What is in this picture?",
            sys_files=[_make_remote_image_file()],
            model_config=_make_vision_model_config(features),
            prompt_template=[],
            vision_enabled=vision_enabled,
            vision_detail=ImagePromptMessageContent.DETAIL.HIGH,
            variable_pool=VariablePool.empty(),
            jinja2_variables=[],
            tenant_id="test",
        )

    assert stop is None
    if expect_image:
        assert prompt_messages == [
            UserPromptMessage(content=[image_content, TextPromptMessageContent(data="What is in this picture?")]),
        ]
    else:
        assert prompt_messages == [UserPromptMessage(content="What is in this picture?")]


_HISTORY: list[PromptMessage] = [
    UserPromptMessage(content="history user 1"),
    AssistantPromptMessage(content="history assistant 1"),
    UserPromptMessage(content="history user 2"),
    AssistantPromptMessage(content="history assistant 2"),
    UserPromptMessage(content="history user 3"),
    AssistantPromptMessage(content="history assistant 3"),
]


def _fetch_with_memory(
    prompt_template: Sequence[LLMNodeChatModelMessage],
    variable_pool: VariablePool,
    features: list[ModelFeature],
    window_size: int,
) -> Sequence[PromptMessage]:
    model_config = _make_vision_model_config(features)
    model_config.model_schema.model_properties = {}
    memory_config = MemoryConfig(window=MemoryConfig.WindowConfig(enabled=True, size=window_size))

    with mock.patch("core.workflow.nodes.llm.node.ModelManager"):
        prompt_messages, _ = LLMNode.fetch_prompt_messages(
            sys_query="current question",
            sys_files=[],
            context="retrieved context",
            memory=cast(TokenBufferMemory, MockTokenBufferMemory(history_messages=_HISTORY)),
            model_config=model_config,
            prompt_template=prompt_template,
            memory_config=memory_config,
            vision_enabled=False,
            vision_detail=ImagePromptMessageContent.DETAIL.HIGH,
            variable_pool=variable_pool,
            jinja2_variables=[],
            tenant_id="test",
        )
    return prompt_messages


@pytest.mark.parametrize("window_size", [1, 2, 3])
def test_fetch_prompt_messages_orders_template_memory_window_and_query(window_size: int):
    prompt_template = [
        LLMNodeChatModelMessage(text="system prompt", role=PromptMessageRole.SYSTEM, edition_type="basic"),
        LLMNodeChatModelMessage(text="{#context#}", role=PromptMessageRole.USER, edition_type="basic"),
        LLMNodeChatModelMessage(text="assistant prompt", role=PromptMessageRole.ASSISTANT, edition_type="basic"),
    ]

    prompt_messages = _fetch_with_memory(prompt_template, VariablePool.empty(), [], window_size)

    assert prompt_messages == [
        SystemPromptMessage(content="system prompt"),
        UserPromptMessage(content="retrieved context"),
        AssistantPromptMessage(content="assistant prompt"),
        *_HISTORY[window_size * -2 :],
        UserPromptMessage(content="current question"),
    ]


def test_fetch_prompt_messages_resolves_file_variable_in_template():
    image_content = ImagePromptMessageContent(url="https://example.com/test1.jpg", format="jpg", mime_type="image/jpeg")
    variable_pool = VariablePool.empty()
    variable_pool.add(["input", "image"], _make_remote_image_file())
    prompt_template = [
        LLMNodeChatModelMessage(text="{{#input.image#}}", role=PromptMessageRole.USER, edition_type="basic"),
    ]

    with mock.patch("core.workflow.nodes.llm.node.file_manager.to_prompt_message_content", return_value=image_content):
        prompt_messages = _fetch_with_memory(prompt_template, variable_pool, [ModelFeature.VISION], 1)

    assert prompt_messages == [
        UserPromptMessage(content=[image_content]),
        *_HISTORY[-2:],
        UserPromptMessage(content="current question"),
    ]


def test_handle_list_messages_basic(llm_node):
    messages = [
        LLMNodeChatModelMessage(
            text="Hello, {#context#}",
            role=PromptMessageRole.USER,
            edition_type="basic",
        )
    ]
    context = "world"
    jinja2_variables = []
    variable_pool = llm_node.graph_runtime_state.variable_pool
    vision_detail_config = ImagePromptMessageContent.DETAIL.HIGH

    result = llm_node.handle_list_messages(
        messages=messages,
        context=context,
        jinja2_variables=jinja2_variables,
        variable_pool=variable_pool,
        vision_detail_config=vision_detail_config,
    )

    assert len(result) == 1
    assert isinstance(result[0], UserPromptMessage)
    assert result[0].content == [TextPromptMessageContent(data="Hello, world")]


@pytest.fixture
def llm_node_for_multimodal(llm_node_data, graph_init_params, graph_runtime_state) -> tuple[LLMNode, LLMFileSaver]:
    mock_file_saver: LLMFileSaver = mock.MagicMock(spec=LLMFileSaver)
    node_config = {
        "id": "1",
        "data": llm_node_data.model_dump(),
    }
    node = LLMNode(
        id="1",
        config=node_config,
        graph_init_params=graph_init_params,
        graph_runtime_state=graph_runtime_state,
        llm_file_saver=mock_file_saver,
    )
    return node, mock_file_saver


class TestLLMNodeSaveMultiModalImageOutput:
    def test_llm_node_save_inline_output(self, llm_node_for_multimodal: tuple[LLMNode, LLMFileSaver]):
        llm_node, mock_file_saver = llm_node_for_multimodal
        content = ImagePromptMessageContent(
            format="png",
            base64_data=base64.b64encode(b"test-data").decode(),
            mime_type="image/png",
        )
        mock_file = File(
            id=str(uuid.uuid4()),
            tenant_id="1",
            type=FileType.IMAGE,
            transfer_method=FileTransferMethod.TOOL_FILE,
            related_id=str(uuid.uuid4()),
            filename="test-file.png",
            extension=".png",
            mime_type="image/png",
            size=9,
        )
        mock_file_saver.save_binary_string.return_value = mock_file
        file = llm_node.save_multimodal_image_output(
            content=content,
            file_saver=mock_file_saver,
        )
        # Manually append to _file_outputs since the static method doesn't do it
        llm_node._file_outputs.append(file)
        assert llm_node._file_outputs == [mock_file]
        assert file == mock_file
        mock_file_saver.save_binary_string.assert_called_once_with(
            data=b"test-data", mime_type="image/png", file_type=FileType.IMAGE
        )

    def test_llm_node_save_url_output(self, llm_node_for_multimodal: tuple[LLMNode, LLMFileSaver]):
        llm_node, mock_file_saver = llm_node_for_multimodal
        content = ImagePromptMessageContent(
            format="png",
            url="https://example.com/image.png",
            mime_type="image/jpg",
        )
        mock_file = File(
            id=str(uuid.uuid4()),
            tenant_id="1",
            type=FileType.IMAGE,
            transfer_method=FileTransferMethod.TOOL_FILE,
            related_id=str(uuid.uuid4()),
            filename="test-file.png",
            extension=".png",
            mime_type="image/png",
            size=9,
        )
        mock_file_saver.save_remote_url.return_value = mock_file
        file = llm_node.save_multimodal_image_output(
            content=content,
            file_saver=mock_file_saver,
        )
        # Manually append to _file_outputs since the static method doesn't do it
        llm_node._file_outputs.append(file)
        assert llm_node._file_outputs == [mock_file]
        assert file == mock_file
        mock_file_saver.save_remote_url.assert_called_once_with(content.url, FileType.IMAGE)


def test_llm_node_image_file_to_markdown(llm_node: LLMNode):
    mock_file = mock.MagicMock(spec=File)
    mock_file.generate_url.return_value = "https://example.com/image.png"
    markdown = llm_node._image_file_to_markdown(mock_file)
    assert markdown == "![](https://example.com/image.png)"


class TestSaveMultimodalOutputAndConvertResultToMarkdown:
    def test_str_content(self, llm_node_for_multimodal):
        llm_node, mock_file_saver = llm_node_for_multimodal
        gen = llm_node._save_multimodal_output_and_convert_result_to_markdown(
            contents="hello world", file_saver=mock_file_saver, file_outputs=[]
        )
        assert list(gen) == ["hello world"]
        mock_file_saver.save_binary_string.assert_not_called()
        mock_file_saver.save_remote_url.assert_not_called()

    def test_text_prompt_message_content(self, llm_node_for_multimodal):
        llm_node, mock_file_saver = llm_node_for_multimodal
        gen = llm_node._save_multimodal_output_and_convert_result_to_markdown(
            contents=[TextPromptMessageContent(data="hello world")], file_saver=mock_file_saver, file_outputs=[]
        )
        assert list(gen) == ["hello world"]
        mock_file_saver.save_binary_string.assert_not_called()
        mock_file_saver.save_remote_url.assert_not_called()

    def test_image_content_with_inline_data(self, llm_node_for_multimodal, monkeypatch):
        llm_node, mock_file_saver = llm_node_for_multimodal

        image_raw_data = b"PNG_DATA"
        image_b64_data = base64.b64encode(image_raw_data).decode()

        mock_saved_file = File(
            id=str(uuid.uuid4()),
            tenant_id="1",
            type=FileType.IMAGE,
            transfer_method=FileTransferMethod.TOOL_FILE,
            filename="test.png",
            extension=".png",
            size=len(image_raw_data),
            related_id=str(uuid.uuid4()),
            url="https://example.com/test.png",
            storage_key="test_storage_key",
        )
        mock_file_saver.save_binary_string.return_value = mock_saved_file
        gen = llm_node._save_multimodal_output_and_convert_result_to_markdown(
            contents=[
                ImagePromptMessageContent(
                    format="png",
                    base64_data=image_b64_data,
                    mime_type="image/png",
                )
            ],
            file_saver=mock_file_saver,
            file_outputs=llm_node._file_outputs,
        )
        yielded_strs = list(gen)
        assert len(yielded_strs) == 1

        # This assertion requires careful handling.
        # `FILES_URL` settings can vary across environments, which might lead to fragile tests.
        #
        # Rather than asserting the complete URL returned by _save_multimodal_output_and_convert_result_to_markdown,
        # we verify that the result includes the markdown image syntax and the expected file URL path.
        expected_file_url_path = f"/files/tools/{mock_saved_file.related_id}.png"
        assert yielded_strs[0].startswith("![](")
        assert expected_file_url_path in yielded_strs[0]
        assert yielded_strs[0].endswith(")")
        mock_file_saver.save_binary_string.assert_called_once_with(
            data=image_raw_data,
            mime_type="image/png",
            file_type=FileType.IMAGE,
        )
        assert mock_saved_file in llm_node._file_outputs

    def test_unknown_content_type(self, llm_node_for_multimodal):
        llm_node, mock_file_saver = llm_node_for_multimodal
        gen = llm_node._save_multimodal_output_and_convert_result_to_markdown(
            contents=frozenset(["hello world"]), file_saver=mock_file_saver, file_outputs=[]
        )
        assert list(gen) == ["hello world"]
        mock_file_saver.save_binary_string.assert_not_called()
        mock_file_saver.save_remote_url.assert_not_called()

    def test_unknown_item_type(self, llm_node_for_multimodal):
        llm_node, mock_file_saver = llm_node_for_multimodal
        gen = llm_node._save_multimodal_output_and_convert_result_to_markdown(
            contents=[frozenset(["hello world"])], file_saver=mock_file_saver, file_outputs=[]
        )
        assert list(gen) == ["frozenset({'hello world'})"]
        mock_file_saver.save_binary_string.assert_not_called()
        mock_file_saver.save_remote_url.assert_not_called()

    def test_none_content(self, llm_node_for_multimodal):
        llm_node, mock_file_saver = llm_node_for_multimodal
        gen = llm_node._save_multimodal_output_and_convert_result_to_markdown(
            contents=None, file_saver=mock_file_saver, file_outputs=[]
        )
        assert list(gen) == []
        mock_file_saver.save_binary_string.assert_not_called()
        mock_file_saver.save_remote_url.assert_not_called()


class TestReasoningFormat:
    """Test cases for reasoning_format functionality"""

    def test_split_reasoning_separated_mode(self):
        """Test separated mode: tags are removed and content is extracted"""

        text_with_think = """
        <think>I need to explain what Dify is. It's an open source AI platform.
        </think>Dify is an open source AI platform.
        """

        clean_text, reasoning_content = LLMNode._split_reasoning(text_with_think, "separated")

        assert clean_text == "Dify is an open source AI platform."
        assert reasoning_content == "I need to explain what Dify is. It's an open source AI platform."

    def test_split_reasoning_tagged_mode(self):
        """Test tagged mode: original text is preserved"""

        text_with_think = """
        <think>I need to explain what Dify is. It's an open source AI platform.
        </think>Dify is an open source AI platform.
        """

        clean_text, reasoning_content = LLMNode._split_reasoning(text_with_think, "tagged")

        # Original text unchanged
        assert clean_text == text_with_think
        # Empty reasoning content in tagged mode
        assert reasoning_content == ""

    def test_split_reasoning_no_think_blocks(self):
        """Test behavior when no <think> tags are present"""

        text_without_think = "This is a simple answer without any thinking blocks."

        clean_text, reasoning_content = LLMNode._split_reasoning(text_without_think, "separated")

        assert clean_text == text_without_think
        assert reasoning_content == ""

    def test_reasoning_format_default_value(self):
        """Test that reasoning_format defaults to 'tagged' for backward compatibility"""

        node_data = LLMNodeData(
            title="Test LLM",
            model=ModelConfig(provider="openai", name="gpt-3.5-turbo", mode="chat", completion_params={}),
            prompt_template=[],
            context=ContextConfig(enabled=False),
        )

        assert node_data.reasoning_format == "tagged"

        text_with_think = """
        <think>I need to explain what Dify is. It's an open source AI platform.
        </think>Dify is an open source AI platform.
        """
        clean_text, reasoning_content = LLMNode._split_reasoning(text_with_think, node_data.reasoning_format)

        assert clean_text == text_with_think
        assert reasoning_content == ""
