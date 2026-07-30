import uuid

from starlette.requests import Request
from starlette.types import ASGIApp

from app.core.logging import request_id_var

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware:
    """ASGI middleware binding a per-request id to the logging correlation contextvar."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request = Request(scope)
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        token = request_id_var.set(request_id)

        async def send_with_header(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((REQUEST_ID_HEADER.lower().encode(), request_id.encode()))
            await send(message)

        try:
            await self._app(scope, receive, send_with_header)
        finally:
            request_id_var.reset(token)
