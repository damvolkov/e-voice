"""Error response models — OpenAI-compatible and simple formats."""

from pydantic import BaseModel
from robyn import Response

##### OPENAI COMPATIBLE #####


class OpenAIErrorDetail(BaseModel):
    """Inner error object matching OpenAI's API error format."""

    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class OpenAIErrorResponse(BaseModel):
    """OpenAI-compatible error wrapper: {"error": {"message": ...}}."""

    error: OpenAIErrorDetail


##### TAXONOMY / INTERNAL #####


class ErrorResponse(BaseModel):
    """Simple error response for taxonomy and internal endpoints."""

    error: str
    detail: str | None = None


##### FACTORY #####

_TAXONOMY_SEGMENTS: frozenset[str] = frozenset(("/stt/", "/tts/"))


def error_response(
    path: str,
    status_code: int,
    message: str,
    *,
    error_type: str = "invalid_request_error",
    detail: str | None = None,
) -> Response:
    """Build error Response — OpenAI format unless path is a taxonomy alias."""
    if any(seg in path for seg in _TAXONOMY_SEGMENTS):
        body = ErrorResponse(error=message, detail=detail).model_dump_json()
    else:
        body = OpenAIErrorResponse(
            error=OpenAIErrorDetail(message=message, type=error_type),
        ).model_dump_json()

    return Response(status_code=status_code, headers={"content-type": "application/json"}, description=body)
