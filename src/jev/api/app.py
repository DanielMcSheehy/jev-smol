"""FastAPI app exposing the Jev / CLEF ("System One") decision API.

Routes:
    POST /v1/systemone  — main decision endpoint (POST /v1/decision is an alias)
    GET  /healthz       — {"status", "backend", "model"}
    GET  /v1/models     — {"data": [{"id", "object": "model"}]}

The engine is created lazily on the first request so importing this module
(and running the test-suite) never pulls in torch. ``create_app`` builds an
explicit app; the module-level ``app`` is configured from the JEV_BACKEND /
JEV_MODEL / JEV_DEVICE environment variables so ``uvicorn jev.api.app:app``
works out of the box.

Error mapping: ImageError/UnsupportedError -> 400, EngineNotReadyError -> 503,
each as {"error": {"type", "message"}}; request validation -> FastAPI's 422;
bodies over 13 MiB -> 413.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..engine import get_engine
from ..schema import JevRequest, JevResponse
from .errors import EngineNotReadyError, JevError, UnsupportedError

MAX_BODY_BYTES = 13 * 1024 * 1024  # 13 MiB request cap (CLEF contract)


class _BodyTooLarge(Exception):
    pass


def _resolved_model_name(backend: str, model: Optional[str]) -> str:
    if model:
        return model
    return "jev-mini" if backend == "vlm" else "jev-mock"


def _engine_kwargs(backend: str, model: Optional[str], device: str) -> dict:
    if backend == "vlm":
        kwargs: dict = {"device": device}
        if model:
            kwargs["model"] = model
        return kwargs
    kwargs = {"device": device} if backend not in ("mock",) else {}
    if model:
        kwargs["model_id"] = model
    return kwargs


def create_app(backend: str = "mock", model: Optional[str] = None, device: str = "auto") -> FastAPI:
    app = FastAPI(title="jev", version="0.1.0")
    state = {"engine": None}

    def _body_too_large() -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content={"error": {"type": "body_too_large",
                               "message": f"request body exceeds {MAX_BODY_BYTES} bytes"}},
        )

    @app.exception_handler(JevError)
    async def _jev_error_handler(request: Request, exc: JevError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"type": exc.error_type, "message": exc.message}},
        )

    @app.middleware("http")
    async def _limit_body_size(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None and content_length.isdigit() and int(content_length) > MAX_BODY_BYTES:
            return _body_too_large()
        if content_length is None:
            # Streaming fallback: count bytes as they arrive and abort with 413
            # once the cap is exceeded. (BaseHTTPMiddleware GET scopes carry no
            # receive callable, and bodyless requests cannot exceed the cap.)
            receive = request.scope.get("receive")
            if receive is not None:
                received = 0

                async def limited_receive():
                    nonlocal received
                    message = await receive()
                    if message["type"] == "http.request":
                        received += len(message.get("body", b""))
                        if received > MAX_BODY_BYTES:
                            raise _BodyTooLarge()
                    return message

                request.scope["receive"] = limited_receive
        try:
            return await call_next(request)
        except _BodyTooLarge:
            return _body_too_large()

    def _engine():
        if state["engine"] is None:
            try:
                state["engine"] = get_engine(backend, **_engine_kwargs(backend, model, device))
            except Exception as exc:  # loading failures -> 503
                raise EngineNotReadyError(f"engine backend {backend!r} failed to load: {exc}") from exc
        return state["engine"]

    async def _systemone(request: JevRequest) -> JevResponse:
        # Enforced here (not only in the engine) so every backend, including
        # the mock, rejects videos with a clean 400 per the CLEF contract.
        if request.videos:
            raise UnsupportedError("this model does not process videos; send images instead")
        return _engine().decide(request)

    app.add_api_route("/v1/systemone", _systemone, methods=["POST"], response_model=None)
    app.add_api_route("/v1/decision", _systemone, methods=["POST"], response_model=None)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "backend": backend, "model": _resolved_model_name(backend, model)}

    @app.get("/v1/models")
    async def models() -> dict:
        return {"data": [{"id": _resolved_model_name(backend, model), "object": "model"}]}

    return app


app = create_app(
    backend=os.environ.get("JEV_BACKEND", "mock"),
    model=os.environ.get("JEV_MODEL") or None,
    device=os.environ.get("JEV_DEVICE", "auto"),
)
