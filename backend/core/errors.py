from __future__ import annotations
from backend.services.inference import InvalidRequestError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.services.model_registry import CheckpointError, ModelLimitError, ModelNotFoundError


def install_error_handlers(app: FastAPI) -> None:
    """Map registry exceptions to HTTP status codes with a {'detail': ...} body."""

    def make(status: int):
        async def handler(request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse(status_code=status, content={"detail": str(exc)})

        return handler

    app.add_exception_handler(ModelNotFoundError, make(404))
    app.add_exception_handler(CheckpointError, make(422))
    app.add_exception_handler(ModelLimitError, make(409))
    app.add_exception_handler(InvalidRequestError, make(422))