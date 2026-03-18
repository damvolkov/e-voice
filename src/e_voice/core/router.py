"""Router with automatic body parsing, validation and response handling."""

import inspect
import re
from collections.abc import Callable
from functools import wraps
from typing import Any

import orjson
from pydantic import BaseModel, ValidationError
from robyn import Request, Response, StreamingResponse, SubRouter, status_codes
from robyn.robyn import HttpMethod
from robyn.types import Body, PathParams

from e_voice.models.core import BodyType, UploadFile

_PATH_PARAM_RE = re.compile(r":(\w+)")

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
                    return Response(
                        status_code=status_codes.HTTP_422_UNPROCESSABLE_ENTITY,
                        headers={"content-type": "application/json"},
                        description=orjson.dumps(
                            {"error": {"message": str(ex), "type": "invalid_request_error"}}
                        ).decode(),
                    )
            case BodyType.JSONABLE:
                try:
                    kwargs[param_name] = orjson.loads(raw)
                except orjson.JSONDecodeError as ex:
                    return Response(
                        status_code=status_codes.HTTP_422_UNPROCESSABLE_ENTITY,
                        headers={"content-type": "application/json"},
                        description=orjson.dumps(
                            {"error": {"message": str(ex), "type": "invalid_request_error"}}
                        ).decode(),
                    )
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


def parse_response(result: Any) -> Response | StreamingResponse:
    """Convert handler result to Response. Passes StreamingResponse through."""
    match result:
        case StreamingResponse():
            return result
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


def _create_method_wrapper(
    original_method: Callable,
    router_prefix: str = "",
    handler_registry: dict | None = None,
    method_name: str = "",
) -> Callable:
    @wraps(original_method)
    def method_wrapper(*args, **kwargs) -> Callable:
        endpoint = args[0] if args else kwargs.get("endpoint", "")
        decorator = original_method(*args, **kwargs)

        def handler_decorator(handler: Callable) -> Callable:
            sig = inspect.signature(handler)
            body_config, file_params = parse_endpoint_signature(sig)
            has_request_param = "request" in sig.parameters
            path_param_names = frozenset(_PATH_PARAM_RE.findall(endpoint)) & set(sig.parameters)

            if file_params:
                full_path = f"{router_prefix}{endpoint}".replace("//", "/")
                FILE_UPLOAD_ENDPOINTS.add(full_path)

            @wraps(handler)
            async def wrapped_handler(request: Request, path_params: PathParams | None = None, **h_kwargs):
                if path_param_names and path_params:
                    for pp_name in path_param_names:
                        h_kwargs[pp_name] = path_params.get(pp_name, "")

                if error := parse_request_body(body_config, h_kwargs):
                    return error

                if file_params and (error := parse_request_files(file_params, request, h_kwargs)):
                    return error

                if has_request_param:
                    h_kwargs["request"] = request

                result = await handler(**h_kwargs)
                return parse_response(result)

            new_params = [
                inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Request),
            ]
            if path_param_names:
                new_params.append(
                    inspect.Parameter("path_params", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=PathParams)
                )
            for name, param in sig.parameters.items():
                if name == "request" or name in file_params or name in path_param_names:
                    continue
                if name in body_config:
                    new_params.append(param.replace(annotation=body_config[name][1]))
                else:
                    new_params.append(param)

            wrapped_handler.__signature__ = sig.replace(parameters=new_params)  # type: ignore[attr-defined]

            if handler_registry is not None:
                full_path = f"{router_prefix}{endpoint}"
                handler_registry[full_path] = (method_name, wrapped_handler)

            return decorator(wrapped_handler)

        return handler_decorator

    return method_wrapper


class Router(SubRouter):
    """Enhanced SubRouter with automatic body/file parsing, response handling, and aliases."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._prefix = kwargs.get("prefix", "")
        self._originals: dict[str, Callable] = {}
        self._handlers: dict[str, tuple[str, Callable]] = {}
        self._wrap_methods()

    def _wrap_methods(self) -> None:
        """Wrap HTTP methods with parsing logic."""
        for method in HTTP_METHODS:
            method_name = str(method).split(".")[-1].lower()
            if hasattr(self, method_name):
                original_method = getattr(self, method_name)
                self._originals[method_name] = original_method
                setattr(
                    self,
                    method_name,
                    _create_method_wrapper(original_method, self._prefix, self._handlers, method_name),
                )

    def alias(self, source: str, *aliases: str) -> None:
        """Register existing endpoint's wrapped handler on additional paths (no re-wrapping)."""
        full_source = f"{self._prefix}{source}"
        if full_source not in self._handlers:
            raise ValueError(f"No handler registered for {full_source}. Call alias() after the handler is defined.")
        method_name, wrapped_handler = self._handlers[full_source]
        original_method = self._originals[method_name]
        for alias_path in aliases:
            original_method(alias_path)(wrapped_handler)
