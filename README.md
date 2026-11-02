# Websearch Agent: evaluación de Tavily como `search_tool`

Evaluación de Tavily como herramienta de búsqueda para un chatbot de consultas web, con un agente LangGraph (`claude-haiku-4-5`). Se mide el coste en tokens (con foco en los `ToolMessage`), la relevancia de los resultados y se prueban mejoras.

## Estructura

```
websearch-agent/
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── agent.py       # agente LangGraph, system prompt, get_token_usage, print_tool_messages, opciones del Paso 3
│   └── analysis.py    # count_tokens, schema de tools, juez de fundamentación, coste y experimentos
└── notebooks/
    └── tavily_evaluation.ipynb
```

## Instalación

```bash
git clone https://github.com/Mendo-005/mario-mendoza-langgraph-websearchagent.git
cd mario-mendoza-langgraph-websearchagent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
```

Edita `.env` y añade `ANTHROPIC_API_KEY` y `TAVILY_API_KEY`.

## Ejecución

Prueba rápida del agente (desde la raíz del proyecto):

```bash
python -m src.agent
```

Notebook completo:

```bash
jupyter notebook notebooks/tavily_evaluation.ipynb
```

> 📸 *Captura: salida de `python -m src.agent` (ToolMessage + TOKEN USAGE)*

## Qué se evalúa

**Paso 1: comportamiento y tokens.** El system prompt inyecta fecha actual y cutoff, y decide cuándo buscar. `get_token_usage()` calcula los tokens del resultado de la tool por diferencia entre llamadas consecutivas:

```
tool_result_tokens = input(llamada N+1) - input(llamada N) - output(llamada N)
```

El `AIMessage` con `tool_calls` lo genera el modelo y cuenta en `output_tokens` de la llamada N. El `ToolMessage` lo devuelve la tool y solo cuenta como input desde la llamada N+1, y se reenvía en cada llamada posterior (`tool_input_tokens_billed`). El resultado se valida con la API `count_tokens` de Anthropic. Ojo: `output_tokens` incluye los tokens de thinking, y `input_tokens` ya incluye los de caché.

**Paso 2: anatomía de Tavily.** Estructura de la respuesta, parámetros, schema que ve el LLM tras `bind_tools()`, calidad de snippets, relación score-relevancia y comprobación de alucinaciones con un juez LLM.

**Paso 3: mejoras.** Opciones de `create_agent(...)`:

| Opción | Qué hace |
|---|---|
| `max_results`, `search_depth`, `include_domains` | Parámetros de Tavily |
| `compact_results=True` | JSON compacto y solo `title`, `url`, `content`, `score` |
| `min_score=0.5` | Descarta resultados con score bajo (si no queda ninguno, mensaje corto) |
| `relevance_judge=True` | Un LLM filtra con esquema Pydantic (`relevant`, `explanation`) |
| `cache_system_prompt=True` | Prompt caching explícito de Anthropic (tools + system prompt) |

> 📸 *Captura: tabla comparativa final del notebook*

## Resultados

| Configuración | Input tokens | Coste (USD) | Respuestas respaldadas |
|---|---|---|---|
| Línea base | — | — | — |
| JSON compacto | — | — | — |
| 3 resultados + compacto | — | — | — |
| score >= 0.5 | — | — | — |
| Juez LLM de relevancia | — | — | — |

## Conclusiones

*(Completar tras ejecutar el notebook.)*

## Notas

- Precios por modelo en `PRICES` (`src/analysis.py`), para prompts de hasta 100K tokens. Verifícalos en la página de precios de Anthropic.
- Las respuestas pueden variar entre ejecuciones: repite los experimentos varias veces antes de concluir.
- El `.env` no se sube al repositorio (`.gitignore`).