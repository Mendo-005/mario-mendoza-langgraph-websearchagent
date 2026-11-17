import json
import operator
import os
from datetime import date
from typing import Annotated, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

load_dotenv()

KNOWLEDGE_CUTOFF = "July 2025"


def check_api_keys(use_search: bool = True) -> None:
    required_keys = ["ANTHROPIC_API_KEY"]
    if use_search:
        required_keys.append("TAVILY_API_KEY")

    missing_keys = [key for key in required_keys if not os.environ.get(key)]
    if missing_keys:
        keys = ", ".join(missing_keys)
        raise RuntimeError(
            f"Faltan las claves API: {keys}. Configúralas en el archivo .env."
        )


def build_system_prompt(use_search: bool = True) -> str:
    today = date.today().strftime("%Y-%m-%d")
    intro = (
        "You are a helpful assistant and an expert on LangChain and LangGraph.\n"
        f"Today's date is {today}. Your training knowledge ends in {KNOWLEDGE_CUTOFF}.\n"
        "Always answer in the same language as the user's question.\n\n"
    )

    if not use_search:
        return intro + (
            "Answer from your own knowledge. If a question concerns information "
            "after your knowledge cutoff, say so clearly instead of guessing.\n"
            "Use the multiply tool for multiplications."
        )

    return intro + (
        "Use web search for current or changing information, or whenever you are "
        "not confident your knowledge is up to date. For stable knowledge, code, "
        "and simple arithmetic, answer directly; use the multiply tool for "
        "multiplications. Make one focused search first, and search again only "
        "if needed. Cite the URLs you relied on. If you did not search, say so "
        "briefly and do not invent sources."
    )


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def create_agent(
    model_name: str = "claude-haiku-4-5",
    model_provider: str = "anthropic",
    temperature: float = 0,
    max_results: int = 5,
    search_depth: str = "basic",
    include_raw_content: bool = False,
    include_domains: Optional[list[str]] = None,
    exclude_domains: Optional[list[str]] = None,
    use_search: bool = True,
):
    check_api_keys(use_search)

    model_options = (
        {} if model_provider == "anthropic" else {"temperature": temperature}
    )
    model = init_chat_model(model_name, model_provider=model_provider, **model_options)

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    tools = [multiply]
    if use_search:
        tools.append(
            TavilySearch(
                max_results=max_results,
                topic="general",
                search_depth=search_depth,
                include_raw_content=include_raw_content,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )
        )

    tools_by_name = {selected_tool.name: selected_tool for selected_tool in tools}
    model_with_tools = model.bind_tools(tools)
    system_message = SystemMessage(content=build_system_prompt(use_search))

    def llm_call(state: MessagesState) -> dict:
        response = model_with_tools.invoke([system_message, *state["messages"]])
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def tool_node(state: MessagesState) -> dict:
        results = []
        for tool_call in state["messages"][-1].tool_calls:
            selected_tool = tools_by_name[tool_call["name"]]
            observation = selected_tool.invoke(tool_call["args"])
            if not isinstance(observation, str):
                observation = json.dumps(observation, ensure_ascii=False)
            results.append(
                ToolMessage(
                    content=observation,
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
        return {"messages": results}

    def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
        if state["messages"][-1].tool_calls:
            return "tool_node"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    graph.add_edge("tool_node", "llm_call")
    return graph.compile()


def run_agent(agent, user_input: str) -> dict:
    result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    for message in result["messages"]:
        message.pretty_print()
    return result
