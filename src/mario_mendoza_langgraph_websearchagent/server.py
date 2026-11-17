import json
import logging
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from langchain.messages import HumanMessage

if __package__:
    from .websearch.agent import create_agent
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mario_mendoza_langgraph_websearchagent.websearch.agent import create_agent

WEB_DIRECTORY = Path(__file__).with_name("web")
MAX_MESSAGE_LENGTH = 4_000
MAX_REQUEST_BYTES = 16_384

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    raise TypeError("El agente devolvió una respuesta con un formato no compatible.")


def create_handler(agent):
    class RequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_DIRECTORY), **kwargs)

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/api/health":
                self._send_json(HTTPStatus.OK, {"status": "ok"})
                return
            if path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            super().do_GET()

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/chat":
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            if self.headers.get_content_type() != "application/json":
                self._send_json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    {"error": "La petición debe ser JSON."},
                )
                return

            content_length = self.headers.get("Content-Length")
            try:
                body_size = int(content_length) if content_length is not None else -1
            except ValueError:
                body_size = -1
            if body_size < 0:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "La petición no tiene un tamaño válido."},
                )
                return
            if body_size > MAX_REQUEST_BYTES:
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "La petición es demasiado grande."},
                )
                return

            try:
                payload = json.loads(self.rfile.read(body_size))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "El JSON de la petición no es válido."},
                )
                return

            message = payload.get("message") if isinstance(payload, dict) else None
            if not isinstance(message, str) or not message.strip():
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Escribe una pregunta antes de enviarla."},
                )
                return
            message = message.strip()
            if len(message) > MAX_MESSAGE_LENGTH:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"La pregunta no puede superar {MAX_MESSAGE_LENGTH} caracteres."},
                )
                return

            try:
                result = agent.invoke(
                    {"messages": [HumanMessage(content=message)]}
                )
                messages = result.get("messages")
                if not messages:
                    raise ValueError("El agente no devolvió ningún mensaje.")
                answer = message_text(messages[-1].content)
                if not answer.strip():
                    raise ValueError("El agente devolvió una respuesta vacía.")
            except Exception:
                logger.exception("No se pudo completar la consulta del agente.")
                self._send_json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": "No se pudo completar la consulta. Revisa el log del servidor."},
                )
                return

            self._send_json(HTTPStatus.OK, {"answer": answer})

        def _send_json(self, status: HTTPStatus, payload: dict) -> None:
            response = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(response)

    return RequestHandler


def main() -> None:
    agent = create_agent()
    server = ThreadingHTTPServer(("127.0.0.1", 8000), create_handler(agent))
    print("Websearch listo en http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
