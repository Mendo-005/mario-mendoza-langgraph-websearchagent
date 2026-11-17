# Websearch

Aplicación web sencilla para hacer preguntas a un agente LangGraph. El agente usa Claude para responder y Tavily cuando necesita información actualizada; la web muestra la respuesta y sus fuentes.

## Requisitos

- Python 3.12 o superior
- Una clave de Anthropic (`ANTHROPIC_API_KEY`)
- Una clave de Tavily (`TAVILY_API_KEY`)

## Instalación y configuración

```powershell
uv sync
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edita `.env` y añade tus claves:

```text
ANTHROPIC_API_KEY=tu_clave_de_anthropic
TAVILY_API_KEY=tu_clave_de_tavily
```

## Ejecutar

```powershell
uv run mario-mendoza-langgraph-websearchagent
```

Abre [http://127.0.0.1:8000](http://127.0.0.1:8000), escribe una pregunta y pulsa **Enviar**. Para detener el servidor, pulsa `Ctrl+C` en la terminal.

El servidor ofrece `POST /api/chat` con un cuerpo JSON `{"message":"Tu pregunta"}` y devuelve `{"answer":"Respuesta del agente"}`. La aplicación y la API se sirven desde el mismo origen.
