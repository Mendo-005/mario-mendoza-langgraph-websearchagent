"""Agente LangGraph con búsqueda web (Tavily), optimizado para gastar pocos tokens.

Novedades respecto a la versión anterior:
- Tarea 1: system prompt que decide cuándo buscar (conocimiento estático vs. actual).
- Tarea 2: get_token_usage / print_token_usage -> tokens totales y tokens del tool call.
- Tarea 3: print_tool_messages -> tool messages legibles.
- create_agent(use_search=False) crea el mismo agente SIN búsqueda (para comparar).
"""

import getpass
import json
import operator
import os
from datetime import date
from typing import List, Literal, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

load_dotenv()

# Fecha hasta la que claude-haiku-4-5 tiene conocimiento (ver documentación del modelo).
KNOWLEDGE_CUTOFF = "July 2025"

# Nombre interno de la herramienta de Tavily (lo usamos para contar las búsquedas).
SEARCH_TOOL_NAME = "tavily_search"


def check_api_keys(use_search: bool = True):
    """Pide las API keys que falten (solo pide Tavily si se usa la búsqueda)."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Anthropic API key: ")

    if use_search and not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key: ")


# ---------------------------------------------------------------------------
# Tarea 1: system prompt
# ---------------------------------------------------------------------------
def build_system_prompt(use_search: bool = True) -> str:
    """Devuelve el system prompt. Con use_search=False no menciona la búsqueda."""
    today = date.today().strftime("%Y-%m-%d")

    intro = (
        "You are an expert on LangChain and LangGraph and a helpful general assistant.\n"
        f"Today's date is {today}. Your training knowledge ends in {KNOWLEDGE_CUTOFF}.\n"
        "Always answer in the same language as the user's question.\n\n"
    )

    if not use_search:
        return intro + (
            "You have no search tool. Answer from your own knowledge.\n"
            "If the question is about something after your knowledge cutoff, or you are "
            "not sure your information is up to date, say so clearly instead of guessing.\n"
            "Use the multiply tool for multiplications."
        )

    return intro + (
        "Web searches are expensive, so first decide whether you really need one.\n\n"
        "Answer directly from your own knowledge, WITHOUT searching, when the question is about:\n"
        f"- Stable knowledge from before {KNOWLEDGE_CUTOFF}: concepts, definitions, history, "
        "science, math, and how established libraries and APIs work.\n"
        "- Writing or explaining code, text or reasoning that needs no external data.\n"
        "- Simple arithmetic (use the multiply tool for multiplications).\n\n"
        "Use the search tool ONLY when:\n"
        f"- The question is about events, releases or versions after {KNOWLEDGE_CUTOFF}, "
        "or about data that changes over time (news, prices, weather, 'latest', 'current', 'today').\n"
        "- You do not know the answer, or you are not confident your knowledge is up to date.\n\n"
        "When you search: make ONE focused search with a short query. "
        "Search again only if the results do not answer the question.\n\n"
        "Sources: if you used search results, cite the URLs you relied on. "
        "If you answered from your own knowledge, say so in one short sentence "
        "and do not invent sources."
    )


def create_agent(
    model_name: str = "claude-haiku-4-5",
    model_provider: str = "anthropic",  # ODER: "ollama"
    temperature: float = 0,
    max_results: int = 5,
    search_depth: str = "basic",
    include_raw_content: bool = False,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    use_search: bool = True,
):
    """Creates the agent. With use_search=False the agent has no search tool."""

    check_api_keys(use_search)

    model = init_chat_model(
        model_name,
        model_provider=model_provider,
        temperature=temperature,
    )

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply `a` and `b`.

        Args:
            a: First int
            b: Second int
        """
        return a * b

    # Augment the LLM with tools
    tools = [multiply]
    if use_search:
        search_tool = TavilySearch(
            max_results=max_results,
            topic="general",
            search_depth=search_depth,
            include_raw_content=include_raw_content,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
        tools.append(search_tool)

    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    system_prompt = build_system_prompt(use_search)

    # Step 2: Define state
    class MessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
        llm_calls: int

    # Step 3: Define model node
    def llm_call(state: MessagesState):
        """LLM decides whether to call a tool or not"""

        return {
            "messages": [
                model_with_tools.invoke(
                    [SystemMessage(content=system_prompt)] + state["messages"]
                )
            ],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    # Step 4: Define tool node
    def tool_node(state: MessagesState):
        """Performs the tool call"""

        result = []
        for tool_call in state["messages"][-1].tool_calls:
            selected_tool = tools_by_name[tool_call["name"]]
            observation = selected_tool.invoke(tool_call["args"])

            # Tavily devuelve un dict. Lo guardamos como JSON (texto) para que
            # el ToolMessage sea siempre un string y se pueda leer después.
            if not isinstance(observation, str):
                observation = json.dumps(observation, ensure_ascii=False)

            result.append(
                ToolMessage(
                    content=observation,
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
        return {"messages": result}

    # Step 5: Define logic to determine whether to end
    def should_continue(state: MessagesState) -> Literal["tool_node", END]:
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        last_message = state["messages"][-1]

        # If the LLM makes a tool call, then perform an action
        if last_message.tool_calls:
            return "tool_node"

        # Otherwise, we stop (reply to the user)
        return END

    # Step 6: Build agent
    agent_builder = StateGraph(MessagesState)

    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    agent_builder.add_edge("tool_node", "llm_call")

    return agent_builder.compile()


# ---------------------------------------------------------------------------
# Tarea 2: tokens
# ---------------------------------------------------------------------------
def get_token_usage(messages) -> dict:
    """Calcula los tokens usados en una ejecución del agente.

    Cada AIMessage trae `usage_metadata` con los tokens de esa llamada al LLM.
    Los tokens del resultado de una herramienta no vienen por separado: se
    cuentan como input de la SIGUIENTE llamada al LLM. Por eso los calculamos así:

        input de la llamada siguiente
        - input de la llamada anterior
        - output de la llamada anterior (el propio tool call)
        = tokens del resultado de la herramienta

    Devuelve un dict con:
        llm_calls              -> nº de llamadas al LLM
        search_calls           -> nº de búsquedas web
        input_tokens           -> input total (incluye los resultados de las herramientas)
        output_tokens          -> output total
        total_tokens           -> input + output
        tool_result_tokens     -> tokens de los resultados de herramientas (contados una vez)
        tool_input_tokens_billed -> igual, pero contando cada vez que se reenvían al LLM
        tool_share_of_input    -> % del input que son resultados de herramientas
    """
    llm_calls = 0
    search_calls = 0
    input_tokens = 0
    output_tokens = 0
    tool_result_tokens = 0
    tool_input_tokens_billed = 0

    tool_tokens_in_context = 0  # tokens de herramientas que ya están en el historial
    prev_input = 0
    prev_output = 0
    tool_message_seen = False

    for message in messages:
        if isinstance(message, ToolMessage):
            tool_message_seen = True
            if message.name == SEARCH_TOOL_NAME:
                search_calls += 1

        elif isinstance(message, AIMessage):
            usage = message.usage_metadata or {}
            current_input = usage.get("input_tokens", 0)
            current_output = usage.get("output_tokens", 0)

            llm_calls += 1
            input_tokens += current_input
            output_tokens += current_output

            # Si entre la llamada anterior y esta hubo herramientas, la diferencia
            # de input son los tokens de sus resultados.
            if tool_message_seen:
                new_tool_tokens = max(current_input - prev_input - prev_output, 0)
                tool_result_tokens += new_tool_tokens
                tool_tokens_in_context += new_tool_tokens
                tool_message_seen = False

            # En cada llamada se reenvían todos los resultados anteriores.
            tool_input_tokens_billed += tool_tokens_in_context

            prev_input = current_input
            prev_output = current_output

    if input_tokens > 0:
        tool_share = round(100 * tool_input_tokens_billed / input_tokens, 1)
    else:
        tool_share = 0.0

    return {
        "llm_calls": llm_calls,
        "search_calls": search_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "tool_result_tokens": tool_result_tokens,
        "tool_input_tokens_billed": tool_input_tokens_billed,
        "tool_share_of_input": tool_share,
    }


def print_token_usage(messages):
    """Imprime el resumen de tokens de get_token_usage."""
    usage = get_token_usage(messages)

    print("-" * 50)
    print("TOKEN USAGE")
    print("-" * 50)
    print(f"LLM calls:                 {usage['llm_calls']}")
    print(f"Web searches:              {usage['search_calls']}")
    print(f"Input tokens:              {usage['input_tokens']}")
    print(f"Output tokens:             {usage['output_tokens']}")
    print(f"TOTAL tokens:              {usage['total_tokens']}")
    print(f"Tool result tokens:        {usage['tool_result_tokens']}")
    print(f"Tool tokens billed (all):  {usage['tool_input_tokens_billed']}")
    print(f"Tool share of input:       {usage['tool_share_of_input']} %")
    return usage


# ---------------------------------------------------------------------------
# Tarea 3: tool messages legibles
# ---------------------------------------------------------------------------
def _shorten(text, max_chars) -> str:
    """Quita saltos de línea y recorta el texto a max_chars (None = sin recortar)."""
    text = " ".join(str(text).split())
    if max_chars and len(text) > max_chars:
        hidden = len(text) - max_chars
        return f"{text[:max_chars]}... [+{hidden} chars]"
    return text


def print_tool_message(message: ToolMessage, number: int = 1, max_chars: Optional[int] = 500):
    """Imprime un ToolMessage. Los resultados de Tavily se muestran uno a uno."""
    content = message.content

    print("=" * 70)
    print(f"TOOL MESSAGE {number}: {message.name or 'unknown'} ({len(str(content))} chars)")
    print("=" * 70)

    # El contenido es un JSON en texto (ver tool_node). Si no lo es, se imprime tal cual.
    data = content
    if isinstance(content, str):
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = content

    if isinstance(data, dict) and "results" in data:
        print(f"Query:         {data.get('query')}")
        print(f"Results:       {len(data['results'])}")
        print(f"Response time: {data.get('response_time')} s")
        if data.get("answer"):
            print(f"Answer:        {_shorten(data['answer'], max_chars)}")

        for i, item in enumerate(data["results"], start=1):
            item_content = item.get("content") or ""
            print()
            print(f"[{i}] {item.get('title')}")
            print(f"    URL:     {item.get('url')}")
            print(f"    Score:   {item.get('score')}")
            print(f"    Content ({len(item_content)} chars): {_shorten(item_content, max_chars)}")
    else:
        print(_shorten(data, max_chars))
    print()


def print_tool_messages(messages, max_chars: Optional[int] = 500):
    """Imprime todos los ToolMessages de una lista de mensajes.

    max_chars: máximo de caracteres por resultado (None para verlo completo).
    """
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    if not tool_messages:
        print("No tool messages (the agent answered without using tools).")
        return

    for number, message in enumerate(tool_messages, start=1):
        print_tool_message(message, number, max_chars)


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
def run_agent(
    agent,
    user_input,
    show_tools: bool = True,
    show_tokens: bool = True,
    tool_max_chars: Optional[int] = 500,
):
    """Ejecuta el agente, imprime la conversación y devuelve el resultado del grafo."""

    messages = [HumanMessage(content=user_input)]
    result = agent.invoke({"messages": messages})

    tool_number = 0
    for m in result["messages"]:
        if isinstance(m, ToolMessage):
            if show_tools:
                tool_number += 1
                print_tool_message(m, tool_number, tool_max_chars)
        else:
            m.pretty_print()

    if show_tokens:
        print_token_usage(result["messages"])

    return result
