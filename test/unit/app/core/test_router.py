"""Tests for custom router with body parsing and response handling."""

import inspect

import pytest
from pydantic import BaseModel
from robyn import Response

from app.core.router import parse_endpoint_signature, parse_request_body, parse_response
from app.models.core import BodyType, UploadFile


# -----------------------------------------------------------------------------
# Test Models
# -----------------------------------------------------------------------------


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int


# -----------------------------------------------------------------------------
# pre_parse_body Tests
# -----------------------------------------------------------------------------


class TestPreParseBody:
    """Tests for pre_parse_body function."""

    def test_pydantic_model_annotation(self) -> None:
        """Verify Pydantic model annotations are detected."""

        async def handler(body: SampleModel) -> None:
            pass

        sig = inspect.signature(handler)
        body_config, file_params = parse_endpoint_signature(sig)

        assert "body" in body_config
        assert body_config["body"][0] == BodyType.PYDANTIC
        assert file_params == set()

    def test_dict_annotation(self) -> None:
        """Verify dict annotations are detected as JSONABLE."""

        async def handler(data: dict) -> None:
            pass

        sig = inspect.signature(handler)
        body_config, file_params = parse_endpoint_signature(sig)

        assert "data" in body_config
        assert body_config["data"][0] == BodyType.JSONABLE
        assert file_params == set()

    def test_body_named_parameter(self) -> None:
        """Verify parameter named 'body' is detected as JSONABLE."""

        async def handler(body) -> None:
            pass

        sig = inspect.signature(handler)
        body_config, file_params = parse_endpoint_signature(sig)

        assert "body" in body_config
        assert body_config["body"][0] == BodyType.JSONABLE
        assert file_params == set()

    def test_no_body_parameters(self) -> None:
        """Verify handlers without body params return empty dict."""

        async def handler(request, global_dependencies) -> None:
            pass

        sig = inspect.signature(handler)
        body_config, file_params = parse_endpoint_signature(sig)

        assert body_config == {}
        assert file_params == set()

    def test_upload_file_annotation(self) -> None:
        """Verify UploadFile annotations are detected as file params."""

        async def handler(files: UploadFile) -> None:
            pass

        sig = inspect.signature(handler)
        body_config, file_params = parse_endpoint_signature(sig)

        assert body_config == {}
        assert "files" in file_params


# -----------------------------------------------------------------------------
# post_parse_body Tests
# -----------------------------------------------------------------------------


class TestPostParseBody:
    """Tests for post_parse_body function."""

    def test_pydantic_valid_json(self) -> None:
        """Verify valid JSON is parsed into Pydantic model."""
        body_config = {"body": (BodyType.PYDANTIC, SampleModel)}
        kwargs = {"body": '{"name": "test", "value": 42}'}

        error = parse_request_body(body_config, kwargs)

        assert error is None
        assert isinstance(kwargs["body"], SampleModel)
        assert kwargs["body"].name == "test"
        assert kwargs["body"].value == 42

    def test_pydantic_invalid_json(self) -> None:
        """Verify invalid JSON returns 422 Response."""
        body_config = {"body": (BodyType.PYDANTIC, SampleModel)}
        kwargs = {"body": '{"name": "test"}'}  # missing 'value'

        error = parse_request_body(body_config, kwargs)

        assert isinstance(error, Response)
        assert error.status_code == 422

    def test_jsonable_valid_json(self) -> None:
        """Verify valid JSON is parsed to dict."""
        body_config = {"data": (BodyType.JSONABLE, None)}
        kwargs = {"data": '{"key": "value"}'}

        error = parse_request_body(body_config, kwargs)

        assert error is None
        assert kwargs["data"] == {"key": "value"}

    def test_jsonable_invalid_json(self) -> None:
        """Verify invalid JSON returns 422 Response."""
        body_config = {"data": (BodyType.JSONABLE, None)}
        kwargs = {"data": "not valid json"}

        error = parse_request_body(body_config, kwargs)

        assert isinstance(error, Response)
        assert error.status_code == 422

    def test_raw_body_unchanged(self) -> None:
        """Verify RAW body type leaves data unchanged."""
        body_config = {"data": (BodyType.RAW, None)}
        original = b"raw bytes"
        kwargs = {"data": original}

        error = parse_request_body(body_config, kwargs)

        assert error is None
        assert kwargs["data"] == original

    def test_missing_param_ignored(self) -> None:
        """Verify missing parameters don't cause errors."""
        body_config = {"body": (BodyType.PYDANTIC, SampleModel)}
        kwargs = {}  # body not in kwargs

        error = parse_request_body(body_config, kwargs)

        assert error is None


# -----------------------------------------------------------------------------
# parse_response Tests
# -----------------------------------------------------------------------------


class TestParseResponse:
    """Tests for parse_response function."""

    def test_response_passthrough(self) -> None:
        """Verify Response objects pass through unchanged."""
        original = Response(status_code=201, headers={}, description="created")
        result = parse_response(original)
        assert result is original

    def test_pydantic_model_to_json(self) -> None:
        """Verify Pydantic models are serialized to JSON."""
        model = SampleModel(name="test", value=123)
        result = parse_response(model)

        assert result.status_code == 200
        assert result.headers["content-type"] == "application/json"
        description = str(result.description)
        assert "test" in description
        assert "123" in description

    def test_dict_to_json(self) -> None:
        """Verify dicts are serialized to JSON."""
        data = {"key": "value", "num": 42}
        result = parse_response(data)

        assert result.status_code == 200
        assert result.headers["content-type"] == "application/json"
        description = str(result.description)
        assert "key" in description
        assert "value" in description

    def test_other_to_string(self) -> None:
        """Verify other types are converted to string."""
        result = parse_response("plain text")
        assert result.status_code == 200
        assert result.description == "plain text"

    @pytest.mark.parametrize(
        ("input_val", "expected_type"),
        [
            (Response(status_code=200, headers={}, description=""), Response),
            (SampleModel(name="x", value=1), Response),
            ({"a": 1}, Response),
            ("text", Response),
            (123, Response),
        ],
    )
    def test_always_returns_response(self, input_val, expected_type) -> None:
        """Verify parse_response always returns a Response."""
        result = parse_response(input_val)
        assert isinstance(result, expected_type)


# -----------------------------------------------------------------------------
# UploadFile Model Tests
# -----------------------------------------------------------------------------


class TestUploadFileModel:
    """Tests for UploadFile model."""

    def test_empty_upload_file(self) -> None:
        """Verify empty UploadFile is falsy."""
        upload = UploadFile()
        assert not upload

    def test_upload_file_with_data(self) -> None:
        """Verify UploadFile with data is truthy."""
        upload = UploadFile(files={"file": b"content"})
        assert upload

    def test_upload_file_iteration(self) -> None:
        """Verify UploadFile can be iterated."""
        upload = UploadFile(files={"a": b"1", "b": b"2"})
        items = list(upload)
        assert len(items) == 2

    def test_upload_file_get(self) -> None:
        """Verify UploadFile.get() works."""
        upload = UploadFile(files={"file": b"content"})
        assert upload.get("file") == b"content"
        assert upload.get("missing") is None
