from typing import Annotated, Any, cast, Dict, List, Mapping, Optional, Sequence, TypedDict, Union
import operator
from uuid import uuid4

from langchain.tools import BaseTool
from langchain_core.language_models.base import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    MessageLikeRepresentation,
    _message_from_dict,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import ConfigurableField, RunnableBinding, chain
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END
from langgraph.graph.message import MessageGraph, Messages, add_messages
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.pregel import Pregel
from pydantic import Field

# Optional / late imports used by certain paths. Import at module top so
# import-time failures are visible early and so inner functions don't need to
# repeatedly import the same symbols.
try:
    from .main import looks_like_math, safe_eval  # type: ignore
except Exception:
    def looks_like_math(_s: str) -> bool:  # fallback: don't treat anything as math
        return False

    def safe_eval(_s: str) -> str:  # placeholder if safe_eval isn't available
        raise RuntimeError("safe_eval not available")

try:
    from langchain.agents import create_agent  # type: ignore
except Exception:
    create_agent = None

try:
    from langchain.schema.messages import HumanMessage  # type: ignore
except Exception:
    HumanMessage = None

# Local application imports used by executor logic
from enum import Enum
from typing import Mapping, Optional, Sequence, Union

from app.agent_types.xml_agent import get_xml_agent_executor
from app.checkpoint import AsyncPostgresCheckpoint
from app.llms import (
    get_anthropic_llm,
    get_google_llm,
    get_mixtral_fireworks,
    get_ollama_llm,
    get_openai_llm,
)

# local get_retrieval_executor is defined later in this module
from .basic_tools import (
    RETRIEVAL_DESCRIPTION,
    TOOLS,
    ActionServer,
    Arxiv,
    AvailableTools,
    Connery,
    DallE,
    DDGSearch,
    PressReleases,
    PubMed,
    Retrieval,
    SecFilings,
    Tavily,
    TavilyAnswer,
    Wikipedia,
    YouSearch,
    get_retrieval_tool,
    get_retriever,
)



class LiberalFunctionMessage(FunctionMessage):
    content: Any = Field(default="")


class LiberalToolMessage(ToolMessage):
    content: Any = Field(default="")


def _convert_pydantic_dict_to_message(
    data: MessageLikeRepresentation,
) -> MessageLikeRepresentation:
    """Convert a dictionary to a message object if it matches message format."""
    if isinstance(data, dict) and "content" in data and isinstance(data.get("type"), str):
        # Don't mutate the original dict (caller may reuse it); copy before popping.
        data_copy = dict(data)
        _type = data_copy.pop("type")
        return _message_from_dict({"data": data_copy, "type": _type})
    return data


def add_messages_liberal(left: Messages, right: Messages) -> Messages:
    # coerce to list
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return add_messages(
        [_convert_pydantic_dict_to_message(m) for m in left],
        [_convert_pydantic_dict_to_message(m) for m in right],
    )


def get_tools_agent_executor(
    tools: list[BaseTool],
    llm: LanguageModelLike,
    system_message: str,
    interrupt_before_action: bool,
    checkpoint: BaseCheckpointSaver,
):
    async def _get_messages(messages):
        msgs = []
        for m in messages:
            if isinstance(m, LiberalToolMessage):
                _dict = m.model_dump()
                _dict["content"] = str(_dict["content"])
                m_c = ToolMessage(**_dict)
                msgs.append(m_c)
            elif isinstance(m, FunctionMessage):
                # anthropic doesn't like function messages
                msgs.append(HumanMessage(content=str(m.content)))
            else:
                msgs.append(m)

        return [SystemMessage(content=system_message)] + msgs

    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm
    agent = _get_messages | llm_with_tools
    tool_executor = ToolExecutor(tools)

    # Define the function that determines whether to continue or not
    def should_continue(messages):
        last_message = messages[-1]
        # If there is no function call, then we finish
        tool_calls = getattr(last_message, "tool_calls", None)
        if not tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    # Define the function to execute tools
    async def call_tool(messages):
        actions: list[ToolInvocation] = []
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = cast(AIMessage, messages[-1])
        for tool_call in last_message.tool_calls:
            # We construct a ToolInvocation from the function_call
            actions.append(
                ToolInvocation(
                    tool=tool_call["name"],
                    tool_input=tool_call["args"],
                )
            )
        # We call the tool_executor and get back a response
        responses = await tool_executor.abatch(actions)
        # We use the response to create a ToolMessage
        tool_messages = [
            LiberalToolMessage(
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
                content=response,
            )
            for tool_call, response in zip(last_message.tool_calls, responses)
        ]
        return tool_messages

    workflow = MessageGraph()

    # Define the two nodes we will cycle between
    workflow.add_node("agent", agent)
    workflow.add_node("action", call_tool)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "action",
            # Otherwise we finish.
            "end": END,
        },
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("action", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    return workflow.compile(
        checkpointer=checkpoint,
        interrupt_before=["action"] if interrupt_before_action else None,
    )

Tool = Union[
    ActionServer,
    Connery,
    DDGSearch,
    Arxiv,
    YouSearch,
    SecFilings,
    PressReleases,
    PubMed,
    Wikipedia,
    Tavily,
    TavilyAnswer,
    Retrieval,
    DallE,
]


class AgentType(str, Enum):
    GPT_35_TURBO = "GPT 3.5 Turbo"
    GPT_4 = "GPT 4 Turbo"
    GPT_4O = "GPT 4o"
    AZURE_OPENAI = "GPT 4 (Azure OpenAI)"
    CLAUDE2 = "Claude 2"
    BEDROCK_CLAUDE2 = "Claude 2 (Amazon Bedrock)"
    GEMINI = "GEMINI"
    OLLAMA = "Ollama"


DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

CHECKPOINTER = AsyncPostgresCheckpoint()


def get_agent_executor(
    tools: list,
    agent: AgentType,
    system_message: str,
    interrupt_before_action: bool,
):
    if agent == AgentType.GPT_35_TURBO:
        llm = get_openai_llm()
        return get_tools_agent_executor(
            tools, llm, system_message, interrupt_before_action, CHECKPOINTER
        )
    elif agent == AgentType.GPT_4:
        llm = get_openai_llm(model="gpt-4-turbo")
        return get_tools_agent_executor(
            tools, llm, system_message, interrupt_before_action, CHECKPOINTER
        )
    elif agent == AgentType.GPT_4O:
        llm = get_openai_llm(model="gpt-4o")
        return get_tools_agent_executor(
            tools, llm, system_message, interrupt_before_action, CHECKPOINTER
        )
    elif agent == AgentType.AZURE_OPENAI:
        llm = get_openai_llm(azure=True)
        return get_tools_agent_executor(
            tools, llm, system_message, interrupt_before_action, CHECKPOINTER
        )
    elif agent == AgentType.CLAUDE2:
        llm = get_anthropic_llm()
        return get_tools_agent_executor(
            tools, llm, system_message, interrupt_before_action, CHECKPOINTER
        )
    elif agent == AgentType.BEDROCK_CLAUDE2:
        llm = get_anthropic_llm(bedrock=True)
        return get_xml_agent_executor(
            tools, llm, system_message, interrupt_before_action, CHECKPOINTER
        )
    elif agent == AgentType.GEMINI:
        llm = get_google_llm()
        return get_tools_agent_executor(
            tools, llm, system_message, interrupt_before_action, CHECKPOINTER
        )
    elif agent == AgentType.OLLAMA:
        llm = get_ollama_llm()
        return get_tools_agent_executor(
            tools, llm, system_message, interrupt_before_action, CHECKPOINTER
        )
    else:
        raise ValueError("Unexpected agent type")


class ConfigurableAgent(RunnableBinding):
    tools: Sequence[Tool]
    agent: AgentType
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    retrieval_description: str = RETRIEVAL_DESCRIPTION
    interrupt_before_action: bool = False
    assistant_id: Optional[str] = None
    thread_id: Optional[str] = ""
    user_id: Optional[str] = None

    def __init__(
        self,
        *,
        tools: Sequence[Tool],
        agent: AgentType = AgentType.GPT_35_TURBO,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        assistant_id: Optional[str] = None,
        thread_id: Optional[str] = "",
        retrieval_description: str = RETRIEVAL_DESCRIPTION,
        interrupt_before_action: bool = False,
        kwargs: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
        **others: Any,
    ) -> None:
        others.pop("bound", None)
        _tools = []
        for _tool in tools:
            if _tool["type"] == AvailableTools.RETRIEVAL:
                if assistant_id is None or thread_id is None:
                    raise ValueError(
                        "Both assistant_id and thread_id must be provided if Retrieval tool is used"
                    )
                _tools.append(
                    get_retrieval_tool(assistant_id, thread_id, retrieval_description)
                )
            else:
                tool_config = _tool.get("config", {})
                _returned_tools = TOOLS[_tool["type"]](**tool_config)
                if isinstance(_returned_tools, list):
                    _tools.extend(_returned_tools)
                else:
                    _tools.append(_returned_tools)
        _agent = get_agent_executor(
            _tools, agent, system_message, interrupt_before_action
        )
        agent_executor = _agent.with_config({"recursion_limit": 50})
        super().__init__(
            tools=tools,
            agent=agent,
            system_message=system_message,
            retrieval_description=retrieval_description,
            bound=agent_executor,
            kwargs=kwargs or {},
            config=config or {},
        )


class LLMType(str, Enum):
    GPT_35_TURBO = "GPT 3.5 Turbo"
    GPT_4 = "GPT 4 Turbo"
    GPT_4O = "GPT 4o"
    AZURE_OPENAI = "GPT 4 (Azure OpenAI)"
    CLAUDE2 = "Claude 2"
    BEDROCK_CLAUDE2 = "Claude 2 (Amazon Bedrock)"
    GEMINI = "GEMINI"
    MIXTRAL = "Mixtral"
    OLLAMA = "Ollama"


def get_chatbot(
    llm_type: LLMType,
    system_message: str,
):
    if llm_type == LLMType.GPT_35_TURBO:
        llm = get_openai_llm()
    elif llm_type == LLMType.GPT_4:
        llm = get_openai_llm(model="gpt-4")
    elif llm_type == LLMType.GPT_4O:
        llm = get_openai_llm(model="gpt-4o")
    elif llm_type == LLMType.AZURE_OPENAI:
        llm = get_openai_llm(azure=True)
    elif llm_type == LLMType.CLAUDE2:
        llm = get_anthropic_llm()
    elif llm_type == LLMType.BEDROCK_CLAUDE2:
        llm = get_anthropic_llm(bedrock=True)
    elif llm_type == LLMType.GEMINI:
        llm = get_google_llm()
    elif llm_type == LLMType.MIXTRAL:
        llm = get_mixtral_fireworks()
    elif llm_type == LLMType.OLLAMA:
        llm = get_ollama_llm()
    else:
        raise ValueError("Unexpected llm type")
    return get_chatbot_executor(llm, system_message, CHECKPOINTER)


class ConfigurableChatBot(RunnableBinding):
    llm: LLMType
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    user_id: Optional[str] = None

    def __init__(
        self,
        *,
        llm: LLMType = LLMType.GPT_35_TURBO,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        kwargs: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
        **others: Any,
    ) -> None:
        others.pop("bound", None)

        chatbot = get_chatbot(llm, system_message)
        super().__init__(
            llm=llm,
            system_message=system_message,
            bound=chatbot,
            kwargs=kwargs or {},
            config=config or {},
        )


chatbot = (
    ConfigurableChatBot(llm=LLMType.GPT_35_TURBO, checkpoint=CHECKPOINTER)
    .configurable_fields(
        llm=ConfigurableField(id="llm_type", name="LLM Type"),
        system_message=ConfigurableField(id="system_message", name="Instructions"),
    )
    .with_types(
        input_type=Messages,
        output_type=Sequence[AnyMessage],
    )
)


class ConfigurableRetrieval(RunnableBinding):
    llm_type: LLMType
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    assistant_id: Optional[str] = None
    thread_id: Optional[str] = ""
    user_id: Optional[str] = None

    def __init__(
        self,
        *,
        llm_type: LLMType = LLMType.GPT_35_TURBO,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        assistant_id: Optional[str] = None,
        thread_id: Optional[str] = "",
        kwargs: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
        **others: Any,
    ) -> None:
        others.pop("bound", None)
        retriever = get_retriever(assistant_id, thread_id)
        if llm_type == LLMType.GPT_35_TURBO:
            llm = get_openai_llm()
        elif llm_type == LLMType.GPT_4:
            llm = get_openai_llm(model="gpt-4-turbo")
        elif llm_type == LLMType.GPT_4O:
            llm = get_openai_llm(model="gpt-4o")
        elif llm_type == LLMType.AZURE_OPENAI:
            llm = get_openai_llm(azure=True)
        elif llm_type == LLMType.CLAUDE2:
            llm = get_anthropic_llm()
        elif llm_type == LLMType.BEDROCK_CLAUDE2:
            llm = get_anthropic_llm(bedrock=True)
        elif llm_type == LLMType.GEMINI:
            llm = get_google_llm()
        elif llm_type == LLMType.MIXTRAL:
            llm = get_mixtral_fireworks()
        elif llm_type == LLMType.OLLAMA:
            llm = get_ollama_llm()
        else:
            raise ValueError("Unexpected llm type")
        chatbot = get_retrieval_executor(llm, retriever, system_message, CHECKPOINTER)
        super().__init__(
            llm_type=llm_type,
            system_message=system_message,
            bound=chatbot,
            kwargs=kwargs or {},
            config=config or {},
        )


chat_retrieval = (
    ConfigurableRetrieval(llm_type=LLMType.GPT_35_TURBO, checkpoint=CHECKPOINTER)
    .configurable_fields(
        llm_type=ConfigurableField(id="llm_type", name="LLM Type"),
        system_message=ConfigurableField(id="system_message", name="Instructions"),
        assistant_id=ConfigurableField(
            id="assistant_id", name="Assistant ID", is_shared=True
        ),
        thread_id=ConfigurableField(
            id="thread_id", name="Thread ID", annotation=str, is_shared=True
        ),
    )
    .with_types(
        input_type=Dict[str, Any],
        output_type=Dict[str, Any],
    )
)


agent: Pregel = (
    ConfigurableAgent(
        agent=AgentType.GPT_35_TURBO,
        tools=[],
        system_message=DEFAULT_SYSTEM_MESSAGE,
        retrieval_description=RETRIEVAL_DESCRIPTION,
        assistant_id=None,
        thread_id="",
    )
    .configurable_fields(
        agent=ConfigurableField(id="agent_type", name="Agent Type"),
        system_message=ConfigurableField(id="system_message", name="Instructions"),
        interrupt_before_action=ConfigurableField(
            id="interrupt_before_action",
            name="Tool Confirmation",
            description="If Yes, you'll be prompted to continue before each tool is executed.\nIf No, tools will be executed automatically by the agent.",
        ),
        assistant_id=ConfigurableField(
            id="assistant_id", name="Assistant ID", is_shared=True
        ),
        thread_id=ConfigurableField(
            id="thread_id", name="Thread ID", annotation=str, is_shared=True
        ),
        tools=ConfigurableField(id="tools", name="Tools"),
        retrieval_description=ConfigurableField(
            id="retrieval_description", name="Retrieval Description"
        ),
    )
    .configurable_alternatives(
        ConfigurableField(id="type", name="Bot Type"),
        default_key="agent",
        prefix_keys=True,
        chatbot=chatbot,
        chat_retrieval=chat_retrieval,
    )
    .with_types(
        input_type=Messages,
        output_type=Sequence[AnyMessage],
    )
)

if __name__ == "__main__":
    import asyncio
    # HumanMessage may not be available in all environments (optional dep).
    if HumanMessage is None:
        print("HumanMessage not available; demo run skipped.")
    else:
        async def run():
            async for m in agent.astream_events(
                HumanMessage(content="whats your name"),
                config={"configurable": {"user_id": "2", "thread_id": "test1"}},
                version="v1",
            ):
                print(m)

        asyncio.run(run())

# use local add_messages_liberal defined above


def get_chatbot_executor(
    llm: LanguageModelLike,
    system_message: str,
    checkpoint: BaseCheckpointSaver,
):
    def _get_messages(messages):
        return [SystemMessage(content=system_message)] + messages

    chatbot = _get_messages | llm

    workflow = StateGraph(Annotated[List[BaseMessage], add_messages_liberal])
    workflow.add_node("chatbot", chatbot)
    workflow.set_entry_point("chatbot")
    workflow.set_finish_point("chatbot")
    app = workflow.compile(checkpointer=checkpoint)
    return app

# use local LiberalToolMessage and add_messages_liberal defined above

search_prompt = PromptTemplate.from_template(
    """Given the conversation below, come up with a search query to look up.

This search query can be either a few words or question

Return ONLY this search query, nothing more.

>>> Conversation:
{conversation}
>>> END OF CONVERSATION

Remember, return ONLY the search query that will help you when formulating a response to the above conversation."""
)


response_prompt_template = """{instructions}

Respond to the user using ONLY the context provided below. Do not make anything up.

{context}"""


def get_retrieval_executor(
    llm: LanguageModelLike,
    retriever: BaseRetriever,
    system_message: str,
    checkpoint: BaseCheckpointSaver,
):
    class AgentState(TypedDict):
        messages: Annotated[List[BaseMessage], add_messages_liberal]
        msg_count: Annotated[int, operator.add]

    def _get_messages(messages):
        chat_history = []
        for m in messages:
            if isinstance(m, AIMessage):
                if not m.tool_calls:
                    chat_history.append(m)
            if isinstance(m, HumanMessage):
                chat_history.append(m)
        response = messages[-1].content
        content = "\n".join([d["page_content"] for d in response])
        return [
            SystemMessage(
                content=response_prompt_template.format(
                    instructions=system_message, context=content
                )
            )
        ] + chat_history

    @chain
    async def get_search_query(messages: Sequence[BaseMessage]):
        convo = []
        for m in messages:
            if isinstance(m, AIMessage):
                if "function_call" not in m.additional_kwargs:
                    convo.append(f"AI: {m.content}")
            if isinstance(m, HumanMessage):
                convo.append(f"Human: {m.content}")
        conversation = "\n".join(convo)
        prompt = await search_prompt.ainvoke({"conversation": conversation})
        response = await llm.ainvoke(prompt, {"tags": ["nostream"]})
        return response

    async def invoke_retrieval(state: AgentState):
        messages = state["messages"]
        if len(messages) == 1:
            human_input = messages[-1].content
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": uuid4().hex,
                                "name": "retrieval",
                                "args": {"query": human_input},
                            }
                        ],
                    )
                ]
            }
        else:
            search_query = await get_search_query.ainvoke(messages)
            return {
                "messages": [
                    AIMessage(
                        id=search_query.id,
                        content="",
                        tool_calls=[
                            {
                                "id": uuid4().hex,
                                "name": "retrieval",
                                "args": {"query": search_query.content},
                            }
                        ],
                    )
                ]
            }

    async def retrieve(state: AgentState):
        messages = state["messages"]
        params = messages[-1].tool_calls[0]
        query = params["args"]["query"]
        response = await retriever.ainvoke(query)
        response = [doc.model_dump() for doc in response]
        msg = LiberalToolMessage(
            name="retrieval", content=response, tool_call_id=params["id"]
        )
        return {"messages": [msg], "msg_count": 1}

    def call_model(state: AgentState):
        messages = state["messages"]
        response = llm.invoke(_get_messages(messages))
        return {"messages": [response], "msg_count": 1}

    workflow = StateGraph(AgentState)
    workflow.add_node("invoke_retrieval", invoke_retrieval)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("response", call_model)
    workflow.set_entry_point("invoke_retrieval")
    workflow.add_edge("invoke_retrieval", "retrieve")
    workflow.add_edge("retrieve", "response")
    workflow.add_edge("response", END)
    app = workflow.compile(checkpointer=checkpoint)
    return app

