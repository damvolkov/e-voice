import inspect

import pytest
from pydantic import BaseModel
from robyn import Headers, Response, StreamingResponse

from e_voice.core.router import (
    Router,
    _create_method_wrapper,
    parse_endpoint_signature,
    parse_request_body,
    parse_request_files,
    parse_response,
)
from e_voice.models.core import BodyType, UploadFile


class SampleModel(BaseModel):
    name: str
    value: int


##### PARSE ENDPOINT SIGNATURE #####


async def test_parse_endpoint_signature_pydantic() -> None:
    async def handler(body: SampleModel) -> None:
        pass

    body_config, file_params = parse_endpoint_signature(inspect.signature(handler))
    assert "body" in body_config
    assert body_config["body"][0] == BodyType.PYDANTIC
    assert file_params == set()


async def test_parse_endpoint_signature_dict() -> None:
    async def handler(data: dict) -> None:
        pass

    body_config, file_params = parse_endpoint_signature(inspect.signature(handler))
    assert "data" in body_config
    assert body_config["data"][0] == BodyType.JSONABLE


async def test_parse_endpoint_signature_body_named() -> None:
    async def handler(body) -> None:
        pass

    body_config, _ = parse_endpoint_signature(inspect.signature(handler))
    assert "body" in body_config
    assert body_config["body"][0] == BodyType.JSONABLE


async def test_parse_endpoint_signature_no_body() -> None:
    async def handler(request, global_dependencies) -> None:
        pass

    body_config, file_params = parse_endpoint_signature(inspect.signature(handler))
    assert body_config == {}
    assert file_params == set()


async def test_parse_endpoint_signature_upload_file() -> None:
    async def handler(files: UploadFile) -> None:
        pass

    body_config, file_params = parse_endpoint_signature(inspect.signature(handler))
    assert body_config == {}
    assert "files" in file_params


##### PARSE REQUEST BODY #####


async def test_parse_request_body_pydantic_valid() -> None:
    body_config = {"body": (BodyType.PYDANTIC, SampleModel)}
    kwargs = {"body": '{"name": "test", "value": 42}'}

    assert parse_request_body(body_config, kwargs) is None
    assert isinstance(kwargs["body"], SampleModel)
    assert kwargs["body"].name == "test"
    assert kwargs["body"].value == 42


async def test_parse_request_body_pydantic_invalid() -> None:
    body_config = {"body": (BodyType.PYDANTIC, SampleModel)}
    kwargs = {"body": '{"name": "test"}'}

    error = parse_request_body(body_config, kwargs)
    assert isinstance(error, Response)
    assert error.status_code == 422


async def test_parse_request_body_jsonable_valid() -> None:
    body_config = {"data": (BodyType.JSONABLE, None)}
    kwargs = {"data": '{"key": "value"}'}

    assert parse_request_body(body_config, kwargs) is None
    assert kwargs["data"] == {"key": "value"}


async def test_parse_request_body_jsonable_invalid() -> None:
    body_config = {"data": (BodyType.JSONABLE, None)}
    kwargs = {"data": "not valid json"}

    error = parse_request_body(body_config, kwargs)
    assert isinstance(error, Response)
    assert error.status_code == 422


async def test_parse_request_body_raw_unchanged() -> None:
    body_config = {"data": (BodyType.RAW, None)}
    kwargs = {"data": b"raw bytes"}

    assert parse_request_body(body_config, kwargs) is None
    assert kwargs["data"] == b"raw bytes"


async def test_parse_request_body_missing_param_ignored() -> None:
    body_config = {"body": (BodyType.PYDANTIC, SampleModel)}

    assert parse_request_body(body_config, {}) is None


##### PARSE RESPONSE #####


async def test_parse_response_response_passthrough() -> None:
    original = Response(status_code=201, headers={}, description="created")
    assert parse_response(original) is original


async def test_parse_response_streaming_passthrough() -> None:
    sr = StreamingResponse(content=iter(["chunk"]), media_type="audio/pcm")
    result = parse_response(sr)
    assert result is sr
    assert isinstance(result, StreamingResponse)


async def test_parse_response_sse_passthrough() -> None:
    sr = StreamingResponse(content=iter(["data: test\n\n"]), media_type="text/event-stream")
    result = parse_response(sr)
    assert result is sr
    assert isinstance(result, StreamingResponse)
    assert sr.headers.get("Content-Type") == "text/event-stream"


async def test_parse_response_streaming_preserves_headers() -> None:
    headers = Headers({"Content-Type": "audio/wav"})
    sr = StreamingResponse(content=iter([b"data"]), headers=headers, media_type="audio/wav")
    result = parse_response(sr)
    assert result is sr
    assert sr.headers.get("Content-Type") == "audio/wav"


async def test_parse_response_pydantic_to_json() -> None:
    model = SampleModel(name="test", value=123)
    result = parse_response(model)

    assert isinstance(result, Response)
    assert result.status_code == 200
    assert result.headers["content-type"] == "application/json"
    assert "test" in str(result.description)
    assert "123" in str(result.description)


async def test_parse_response_dict_to_json() -> None:
    result = parse_response({"key": "value", "num": 42})
    assert isinstance(result, Response)
    assert result.status_code == 200
    assert result.headers["content-type"] == "application/json"
    assert "key" in str(result.description)


async def test_parse_response_other_to_string() -> None:
    result = parse_response("plain text")
    assert isinstance(result, Response)
    assert result.status_code == 200
    assert result.description == "plain text"


@pytest.mark.parametrize(
    ("input_val", "expected_type"),
    [
        (Response(status_code=200, headers={}, description=""), Response),
        (StreamingResponse(content=iter([""]), media_type="text/event-stream"), StreamingResponse),
        (StreamingResponse(content=iter([b""]), media_type="audio/pcm"), StreamingResponse),
        (SampleModel(name="x", value=1), Response),
        ({"a": 1}, Response),
        ("text", Response),
        (123, Response),
    ],
    ids=["response", "sse-stream", "audio-stream", "pydantic", "dict", "str", "int"],
)
async def test_parse_response_return_type(input_val, expected_type) -> None:
    result = parse_response(input_val)
    assert isinstance(result, expected_type)


##### UPLOAD FILE #####


async def test_upload_file_empty_is_falsy() -> None:
    assert not UploadFile()


async def test_upload_file_with_data_is_truthy() -> None:
    assert UploadFile(files={"file": b"content"})


async def test_upload_file_iteration() -> None:
    upload = UploadFile(files={"a": b"1", "b": b"2"})
    assert len(list(upload)) == 2


async def test_upload_file_get() -> None:
    upload = UploadFile(files={"file": b"content"})
    assert upload.get("file") == b"content"
    assert upload.get("missing") is None


##### PARSE REQUEST FILES #####


async def test_parse_request_files_empty_params() -> None:
    assert parse_request_files(set(), None, {}) is None  # ty: ignore[invalid-argument-type]


async def test_parse_request_files_missing_files() -> None:
    class FakeRequest:
        files = None

    error = parse_request_files({"audio"}, FakeRequest(), {})  # ty: ignore[invalid-argument-type]
    assert isinstance(error, Response)
    assert error.status_code == 422


async def test_parse_request_files_no_files_attr() -> None:
    class FakeRequest:
        pass

    error = parse_request_files({"audio"}, FakeRequest(), {})  # ty: ignore[invalid-argument-type]
    assert isinstance(error, Response)
    assert error.status_code == 422


async def test_parse_request_files_success() -> None:
    class FakeRequest:
        files = [("audio.wav", b"fake audio bytes")]

    kwargs: dict = {}
    result = parse_request_files({"audio"}, FakeRequest(), kwargs)  # ty: ignore[invalid-argument-type]
    assert result is None
    assert "audio" in kwargs
    assert isinstance(kwargs["audio"], UploadFile)


##### PARSE REQUEST BODY EDGE CASES #####


async def test_parse_request_body_already_parsed_skipped() -> None:
    body_config = {"body": (BodyType.PYDANTIC, SampleModel)}
    model = SampleModel(name="test", value=1)
    kwargs = {"body": model}

    assert parse_request_body(body_config, kwargs) is None
    assert kwargs["body"] is model


##### CREATE METHOD WRAPPER #####


async def test_create_method_wrapper_registers_handler() -> None:
    calls: list = []

    def original_method(endpoint):
        def decorator(handler):
            calls.append((endpoint, handler))
            return handler

        return decorator

    registry: dict = {}
    wrapped = _create_method_wrapper(original_method, "/api", registry, "post")

    @wrapped("/users")
    async def handler(body: SampleModel) -> None:
        return {"ok": True}  # ty: ignore[invalid-return-type]

    assert len(calls) == 1
    assert calls[0][0] == "/users"
    assert "/api/users" in registry
    assert registry["/api/users"][0] == "post"


async def test_create_method_wrapper_wraps_handler_with_parsing(mocker) -> None:
    def original_method(endpoint):
        def decorator(handler):
            return handler

        return decorator

    wrapped = _create_method_wrapper(original_method, "/api", {}, "post")

    @wrapped("/items")
    async def handler(body: SampleModel) -> Response:
        return Response(status_code=200, headers={}, description=body.name)

    mock_request = mocker.MagicMock()
    mock_request.files = None

    result = await handler(mock_request, body='{"name": "test", "value": 42}')
    assert isinstance(result, Response)
    assert result.status_code == 200


##### ROUTER INIT #####


async def test_router_init_wraps_methods() -> None:
    router = Router(__file__, prefix="/test")
    assert router._prefix == "/test"
    assert "get" in router._originals
    assert "post" in router._originals


async def test_router_alias(mocker) -> None:
    router = Router(__file__, prefix="/v1")

    @router.post("/source")
    async def handler() -> Response:
        return Response(status_code=200, headers={}, description="ok")

    router.alias("/source", "/alias1", "/alias2")
