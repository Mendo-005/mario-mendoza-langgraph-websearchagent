"""Funciones de análisis para el notebook (Paso 2 y Paso 3)."""

import pandas as pd
from anthropic import Anthropic
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from .agent import JUDGE_USAGE, SEARCH_TOOL_NAME, build_tools, create_agent, get_token_usage

# USD por millón de tokens, para prompts de hasta 100K tokens (verifica en la página de precios de Anthropic)
PRICES = {
    "claude-haiku-4-5": {"input": 1.00, "cache_read": 0.10, "cache_write": 1.25, "output": 5.00},
    "claude-haiku-5-5": {"input": 0.10, "cache_read": 0.01, "cache_write": 0.125, "output": 0.50},
}

_client = None


def count_tokens(text: str, model: str = "claude-haiku-4-5") -> int:
    """Cuenta tokens con la API count_tokens de Anthropic (incluye unos pocos tokens de overhead de formato)."""
    global _client
    if _client is None:
        _client = Anthropic()
    response = _client.messages.count_tokens(model=model, messages=[{"role": "user", "content": text}])
    return response.input_tokens


def tool_results_to_df(messages, model: str = "claude-haiku-4-5", count: bool = True) -> pd.DataFrame:
    """Una fila por resultado de Tavily: título, url, score, tamaño del snippet."""
    import json

    rows = []
    for m in messages:
        if not (isinstance(m, ToolMessage) and m.name == SEARCH_TOOL_NAME):
            continue
        try:
            data = json.loads(m.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        for r in data.get("results", []):
            content = r.get("content") or ""
            rows.append(
                {
                    "query": data.get("query"),
                    "title": (r.get("title") or "")[:60],
                    "url": r.get("url"),
                    "score": r.get("score"),
                    "content_chars": len(content),
                    "content_tokens": count_tokens(content, model) if count and content else 0,
                }
            )
    return pd.DataFrame(rows)


def show_llm_tools(model_name: str = "claude-haiku-4-5", model_provider: str = "anthropic", **tool_kwargs) -> list:
    """Devuelve las tools tal y como se envían a la API (lo que 've' el LLM tras bind_tools)."""
    model = init_chat_model(model_name, model_provider=model_provider, temperature=0)
    tools = build_tools(**tool_kwargs)
    return model.bind_tools(tools).kwargs["tools"]


def schema_tokens(model_name: str = "claude-haiku-4-5", model_provider: str = "anthropic", **tool_kwargs) -> int:
    """
    Tokens fijos que añade bind_tools() a cada llamada: input con tools menos input sin tools.
    Incluye el overhead que Anthropic añade al activar tools (es coste real).
    """
    model = init_chat_model(model_name, model_provider=model_provider, temperature=0)
    tools = build_tools(**tool_kwargs)
    without_tools = model.invoke("Hola").usage_metadata["input_tokens"]
    with_tools = model.bind_tools(tools).invoke("Hola").usage_metadata["input_tokens"]
    return with_tools - without_tools


class Grounding(BaseModel):
    supported: bool = Field(description="True si toda la respuesta está respaldada por el contexto")
    unsupported_claims: list[str] = Field(description="Afirmaciones que no aparecen en el contexto")


def make_grounding_judge(model_name: str = "claude-haiku-4-5", model_provider: str = "anthropic"):
    """Modelo que comprueba si la respuesta final está respaldada por los resultados de búsqueda."""
    model = init_chat_model(model_name, model_provider=model_provider, temperature=0)
    return model.with_structured_output(Grounding)


def judge_grounding(messages, judge):
    """Devuelve un Grounding, o None si el agente no usó herramientas de búsqueda."""
    contexts = [str(m.content) for m in messages if isinstance(m, ToolMessage) and m.name == SEARCH_TOOL_NAME]
    if not contexts:
        return None
    prompt = (
        "Search context:\n" + "\n".join(contexts)
        + f"\n\nAssistant answer:\n{messages[-1].content}\n\n"
        "Is every factual claim in the answer supported by the search context?"
    )
    return judge.invoke(prompt)


def estimate_cost(usage: dict, model: str = "claude-haiku-4-5") -> float:
    """Coste en USD. usage necesita input_tokens (incluye caché) y output_tokens; la caché es opcional."""
    p = PRICES[model]
    read = usage.get("cache_read_tokens", 0)
    write = usage.get("cache_creation_tokens", 0)
    normal = usage["input_tokens"] - read - write
    return (
        normal * p["input"] + read * p["cache_read"] + write * p["cache_write"] + usage["output_tokens"] * p["output"]
    ) / 1_000_000


def run_experiment(name: str, agent_kwargs: dict, questions: list, grounding_judge=None) -> dict:
    """Ejecuta un agente con una configuración sobre todas las preguntas y resume tokens, coste y respaldo."""
    model = agent_kwargs.get("model_name", "claude-haiku-4-5")
    agent = create_agent(**agent_kwargs)
    JUDGE_USAGE.clear()

    searches = input_tokens = output_tokens = tool_tokens = billed_tool_tokens = 0
    cost = 0.0
    backed = judged = 0

    for question in questions:
        result = agent.invoke({"messages": [HumanMessage(content=question)]})
        usage = get_token_usage(result["messages"])
        searches += usage["search_calls"]
        input_tokens += usage["input_tokens"]
        output_tokens += usage["output_tokens"]
        tool_tokens += usage["tool_result_tokens"]
        billed_tool_tokens += usage["tool_input_tokens_billed"]
        cost += estimate_cost(usage, model)

        if grounding_judge is not None:
            verdict = judge_grounding(result["messages"], grounding_judge)
            if verdict is not None:
                judged += 1
                backed += int(verdict.supported)

    judge_cost = sum(estimate_cost(u, model) for u in JUDGE_USAGE)
    return {
        "config": name,
        "searches": searches,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tool_result_tokens": tool_tokens,
        "tool_tokens_billed": billed_tool_tokens,
        "agent_cost_usd": round(cost, 5),
        "judge_cost_usd": round(judge_cost, 5),
        "total_cost_usd": round(cost + judge_cost, 5),
        "answers_supported": f"{backed}/{judged}",
    }