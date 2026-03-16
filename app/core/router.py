"""Router with automatic body parsing, validation and response handling."""

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any

import orjson
from pydantic import BaseModel, ValidationError
from robyn import Request, Response, SubRouter, status_codes
from robyn.robyn import HttpMethod
from robyn.types import Body

from app.models.core import BodyType, UploadFile

FILE_UPLOAD_ENDPOINTS: set[str] = set()


def parse_endpoint_signature(
    sig: inspect.Signature,
) -> tuple[dict[str, tuple[BodyType, type | None]], set[str]]:
    """Parse function signature for body and file parameters."""
    parsed: dict[str, tuple[BodyType, type | None]] = {}
    file_params: set[str] = set()

    for name, param in sig.parameters.items():
        annotation = param.annotation

        if annotation is UploadFile:
            file_params.add(name)
            continue

        match annotation:
            case type() if issubclass(annotation, BaseModel):
                parsed[name] = (BodyType.PYDANTIC, type(annotation.__name__, (annotation, Body), {}))
            case type() if issubclass(annotation, Body):
                parsed[name] = (BodyType.JSONABLE, annotation)
            case type() if annotation is dict:
                parsed[name] = (BodyType.JSONABLE, None)
            case _ if name == "body":
                parsed[name] = (BodyType.JSONABLE, None)

    return parsed, file_params


def parse_request_body(
    body_config: dict[str, tuple[BodyType, type | None]],
    kwargs: dict[str, Any],
) -> Response | None:
    """Parse JSON/Pydantic body parameters."""
    for param_name, (body_type, model_cls) in body_config.items():
        if param_name not in kwargs:
            continue
        raw = kwargs[param_name]
        if not isinstance(raw, (str, bytes)):
            continue

        match body_type:
            case BodyType.PYDANTIC if model_cls:
                try:
                    kwargs[param_name] = model_cls.model_validate_json(raw)  # type: ignore[union-attr]
                except ValidationError as ex:
                    return Response(status_code=422, headers={}, description=ex.json())
            case BodyType.JSONABLE:
                try:
                    kwargs[param_name] = orjson.loads(raw)
                except orjson.JSONDecodeError as ex:
                    return Response(status_code=422, headers={}, description=str(ex))
            case BodyType.RAW:
                pass
    return None


def parse_request_files(
    file_params: set[str],
    request: Request,
    kwargs: dict[str, Any],
) -> Response | None:
    """Transfer request.files to UploadFile kwargs."""
    if not file_params:
        return None

    files = getattr(request, "files", None)
    if not files:
        return Response(
            status_code=status_codes.HTTP_422_UNPROCESSABLE_ENTITY,
            headers={"content-type": "application/json"},
            description=orjson.dumps({"error": "missing_files", "required": list(file_params)}).decode(),
        )

    for param_name in file_params:
        kwargs[param_name] = UploadFile(files=dict(files))

    return None


def parse_response(result: Any) -> Response:
    """Convert handler result to Response."""
    match result:
        case Response():
            return result
        case BaseModel():
            return Response(
                status_code=status_codes.HTTP_200_OK,
                headers={"content-type": "application/json"},
                description=result.model_dump_json(indent=4),
            )
        case dict():
            return Response(
                status_code=status_codes.HTTP_200_OK,
                headers={"content-type": "application/json"},
                description=orjson.dumps(result).decode(),
            )
        case _:
            return Response(
                status_code=status_codes.HTTP_200_OK,
                headers={},
                description=str(result),
            )


HTTP_METHODS = (
    HttpMethod.GET,
    HttpMethod.POST,
    HttpMethod.PUT,
    HttpMethod.DELETE,
    HttpMethod.PATCH,
    HttpMethod.HEAD,
    HttpMethod.OPTIONS,
    HttpMethod.TRACE,
    HttpMethod.CONNECT,
)


def _create_method_wrapper(original_method: Callable, router_prefix: str = "") -> Callable:
    @wraps(original_method)
    def method_wrapper(*args, **kwargs) -> Callable:
        endpoint = args[0] if args else kwargs.get("endpoint", "")
        decorator = original_method(*args, **kwargs)

        def handler_decorator(handler: Callable) -> Callable:
            sig = inspect.signature(handler)
            body_config, file_params = parse_endpoint_signature(sig)
            has_request_param = "request" in sig.parameters

            if file_params:
                full_path = f"{router_prefix}{endpoint}".replace("//", "/")
                FILE_UPLOAD_ENDPOINTS.add(full_path)

            @wraps(handler)
            async def wrapped_handler(request: Request, **h_kwargs):
                if error := parse_request_body(body_config, h_kwargs):
                    return error

                if file_params and (error := parse_request_files(file_params, request, h_kwargs)):
                    return error

                # Pass request to handler only if it declared it
                if has_request_param:
                    h_kwargs["request"] = request

                result = await handler(**h_kwargs)
                return parse_response(result)

            # Build signature: always include request for Robyn injection
            new_params = [inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Request)]
            for name, param in sig.parameters.items():
                if name == "request" or name in file_params:
                    continue
                if name in body_config:
                    new_params.append(param.replace(annotation=body_config[name][1]))
                else:
                    new_params.append(param)

            wrapped_handler.__signature__ = sig.replace(parameters=new_params)  # type: ignore[attr-defined]
            return decorator(wrapped_handler)

        return handler_decorator

    return method_wrapper


class Router(SubRouter):
    """Enhanced SubRouter with automatic body/file parsing and response handling."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._prefix = kwargs.get("prefix", "")
        self._wrap_methods()

    def _wrap_methods(self) -> None:
        """Wrap HTTP methods with parsing logic."""
        for method in HTTP_METHODS:
            method_name = str(method).split(".")[-1].lower()
            if hasattr(self, method_name):
                original_method = getattr(self, method_name)
                wrapped_method = _create_method_wrapper(original_method, self._prefix)
                setattr(self, method_name, wrapped_method)
