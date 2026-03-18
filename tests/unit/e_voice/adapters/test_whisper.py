"""Unit tests for the redesigned WhisperAdapter — lifecycle, inference, formatting."""

import numpy as np
import orjson
import pytest
from faster_whisper.transcribe import Segment, TranscriptionInfo
from tests.unit.e_voice.adapters.conftest import make_segment

from e_voice.adapters.whisper import (
    WhisperAdapter,
    build_response,
    format_segment,
    segment_to_model,
)
from e_voice.core.settings import ComputeType, DeviceType, STTConfig
from e_voice.core.settings import settings as st
from e_voice.models.stt import InferenceParams, ModelSpec

##### MODEL SPEC #####


async def test_model_spec_frozen() -> None:
    spec = ModelSpec(model_id="test", device="cuda", compute_type="float16")
    with pytest.raises(AttributeError):
        spec.model_id = "other"  # ty: ignore[invalid-assignment]


async def test_model_spec_defaults() -> None:
    spec = ModelSpec(model_id="test")
    assert spec.device == "cpu"
    assert spec.compute_type == "auto"


async def test_model_spec_equality() -> None:
    a = ModelSpec(model_id="m", device="cuda", compute_type="float16")
    b = ModelSpec(model_id="m", device="cuda", compute_type="float16")
    c = ModelSpec(model_id="m", device="cpu")
    assert a == b
    assert a != c


async def test_model_spec_hashable() -> None:
    specs = {
        ModelSpec(model_id="m", device="cuda"),
        ModelSpec(model_id="m", device="cpu"),
        ModelSpec(model_id="m", device="cuda"),
    }
    assert len(specs) == 2


##### INFERENCE PARAMS #####


async def test_inference_params_defaults() -> None:
    params = InferenceParams()
    assert params.language is None
    assert params.temperature == 0.0
    assert params.word_timestamps is False
    assert params.vad_filter is False
    assert params.hotwords is None


async def test_inference_params_frozen() -> None:
    params = InferenceParams(language="en")
    with pytest.raises(AttributeError):
        params.language = "es"  # ty: ignore[invalid-assignment]


##### ADAPTER INIT #####


async def test_adapter_init_default_config() -> None:
    adapter = WhisperAdapter()
    assert adapter.loaded_models() == []


@pytest.mark.parametrize(
    ("device", "compute"),
    [
        (DeviceType.CUDA, ComputeType.FLOAT16),
        (DeviceType.CPU, ComputeType.INT8),
        (DeviceType.AUTO, ComputeType.DEFAULT),
    ],
    ids=["cuda-fp16", "cpu-int8", "auto-default"],
)
async def test_adapter_respects_device_config(device: DeviceType, compute: ComputeType) -> None:
    config = STTConfig(device=device, compute_type=compute)
    adapter = WhisperAdapter(config=config)
    assert adapter._config.device == device
    assert adapter._config.compute_type == compute


##### MODEL LIFECYCLE #####


async def test_load_stores_model(mocker) -> None:
    adapter = WhisperAdapter()
    spec = ModelSpec(model_id="test-model", device="cpu")
    mocker.patch.object(adapter, "_create_model", return_value=mocker.MagicMock())
    await adapter.load(spec)
    assert await adapter.is_loaded(spec)
    assert spec in adapter.loaded_models()


async def test_load_idempotent(mocker) -> None:
    adapter = WhisperAdapter()
    spec = ModelSpec(model_id="test-model", device="cpu")
    create = mocker.patch.object(adapter, "_create_model", return_value=mocker.MagicMock())
    await adapter.load(spec)
    await adapter.load(spec)
    create.assert_called_once()


async def test_unload_removes_model(mocker) -> None:
    adapter = WhisperAdapter()
    spec = ModelSpec(model_id="test-model", device="cpu")
    mocker.patch.object(adapter, "_create_model", return_value=mocker.MagicMock())
    await adapter.load(spec)
    assert await adapter.unload(spec) is True
    assert not await adapter.is_loaded(spec)
    assert adapter.loaded_models() == []


async def test_unload_nonexistent() -> None:
    adapter = WhisperAdapter()
    assert await adapter.unload(ModelSpec(model_id="ghost")) is False


async def test_is_loaded_false() -> None:
    adapter = WhisperAdapter()
    assert await adapter.is_loaded(ModelSpec(model_id="ghost")) is False


async def test_resolve_raises_not_loaded() -> None:
    adapter = WhisperAdapter()
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter._resolve(ModelSpec(model_id="ghost"))


async def test_resolve_falls_back_to_default_config(mocker) -> None:
    config = STTConfig(device=DeviceType.CUDA, compute_type=ComputeType.FLOAT16)
    adapter = WhisperAdapter(config=config)
    spec = ModelSpec(model_id=st.stt.model, device="cuda", compute_type="float16")
    mocker.patch.object(adapter, "_create_model", return_value=mocker.MagicMock())
    await adapter.load(spec)
    model = adapter._resolve(None)
    assert model is not None


async def test_download_calls_snapshot(tmp_path, mocker) -> None:
    adapter = WhisperAdapter()
    dl = mocker.patch("e_voice.adapters.whisper.snapshot_download", return_value=str(tmp_path))
    path = await adapter.download("org/model")
    dl.assert_called_once()
    assert path == tmp_path


##### BATCH INFERENCE #####


async def test_transcribe_calls_model(mocker) -> None:
    adapter = WhisperAdapter()
    spec = ModelSpec(model_id="test-model", device="cpu")
    mock_model = mocker.MagicMock()
    mock_segments = [make_segment(text=" hello")]
    mock_info = mocker.MagicMock(language="en", duration=1.0)
    mock_model.transcribe.return_value = (iter(mock_segments), mock_info)
    mocker.patch.object(adapter, "_create_model", return_value=mock_model)
    await adapter.load(spec)

    result = await adapter.transcribe(np.zeros(16_000, dtype=np.float32), spec=spec)
    segments, info = result
    assert not isinstance(segments, str)
    assert not isinstance(info, str)
    assert len(segments) == 1
    assert info.language == "en"


async def test_translate_uses_translate_task(mocker) -> None:
    adapter = WhisperAdapter()
    spec = ModelSpec(model_id="test-model", device="cpu")
    mock_model = mocker.MagicMock()
    mock_model.transcribe.return_value = (iter([make_segment(text=" hola")]), mocker.MagicMock())
    mocker.patch.object(adapter, "_create_model", return_value=mock_model)
    await adapter.load(spec)

    await adapter.translate(np.zeros(16_000, dtype=np.float32), spec=spec)
    mock_model.transcribe.assert_called_once()
    call_kwargs = mock_model.transcribe.call_args
    assert call_kwargs[1]["task"] == "translate"


##### FORMATTED BATCH INFERENCE #####


async def test_transcribe_formatted_json(mocker, info: TranscriptionInfo) -> None:
    adapter = WhisperAdapter()
    spec = ModelSpec(model_id="test-model", device="cpu")
    mock_model = mocker.MagicMock()
    mock_model.transcribe.return_value = (iter([make_segment(text=" hello")]), info)
    mocker.patch.object(adapter, "_create_model", return_value=mock_model)
    await adapter.load(spec)

    body, ct = await adapter.transcribe(np.zeros(16_000, dtype=np.float32), spec=spec, response_format="json")
    assert ct == "application/json"
    assert "hello" in body


async def test_translate_formatted_verbose_json(mocker) -> None:
    adapter = WhisperAdapter()
    spec = ModelSpec(model_id="test-model", device="cpu")
    mock_model = mocker.MagicMock()
    real_info = TranscriptionInfo(
        language="es",
        language_probability=0.99,
        duration=1.0,
        duration_after_vad=1.0,
        all_language_probs=None,
        transcription_options=None,  # ty: ignore[invalid-argument-type]
        vad_options=None,  # ty: ignore[invalid-argument-type]
    )
    mock_model.transcribe.return_value = (iter([make_segment(text=" hola")]), real_info)
    mocker.patch.object(adapter, "_create_model", return_value=mock_model)
    await adapter.load(spec)

    result = await adapter.translate(np.zeros(16_000, dtype=np.float32), spec=spec, response_format="verbose_json")
    body, ct = result
    assert isinstance(body, str)
    assert ct == "application/json"
    parsed = orjson.loads(body)
    assert parsed["task"] == "translate"


##### STREAMING INFERENCE #####


async def test_transcribe_stream_yields_formatted_text(mocker) -> None:
    adapter = WhisperAdapter()
    spec = ModelSpec(model_id="test-model", device="cpu")
    mock_model = mocker.MagicMock()
    segs = [make_segment(text=" hello"), make_segment(text=" world")]
    mock_model.transcribe.return_value = (iter(segs), mocker.MagicMock())
    mocker.patch.object(adapter, "_create_model", return_value=mock_model)
    await adapter.load(spec)

    collected = []
    async for chunk in adapter.transcribe_stream(np.zeros(16_000, dtype=np.float32), spec=spec, response_format="text"):
        collected.append(chunk)

    assert len(collected) == 2
    assert collected[0] == " hello"
    assert collected[1] == " world"


async def test_transcribe_stream_yields_formatted_json(mocker) -> None:
    adapter = WhisperAdapter()
    spec = ModelSpec(model_id="test-model", device="cpu")
    mock_model = mocker.MagicMock()
    mock_model.transcribe.return_value = (iter([make_segment(text=" hello")]), mocker.MagicMock())
    mocker.patch.object(adapter, "_create_model", return_value=mock_model)
    await adapter.load(spec)

    collected = []
    async for chunk in adapter.transcribe_stream(np.zeros(16_000, dtype=np.float32), spec=spec, response_format="json"):
        collected.append(chunk)

    assert len(collected) == 1
    assert '"text"' in collected[0]


##### FORMAT_SEGMENT (ovld) #####


@pytest.mark.parametrize(
    ("fmt", "expected_substr"),
    [
        ("text", " hello"),
        ("json", '"text"'),
        ("verbose_json", '"text"'),
        ("srt", "-->"),
        ("vtt", "-->"),
    ],
    ids=["text", "json", "verbose_json", "srt", "vtt"],
)
async def test_format_segment_dispatch(segment: Segment, fmt: str, expected_substr: str) -> None:
    result = format_segment(segment, fmt)  # ty: ignore[no-matching-overload]
    assert expected_substr in result


async def test_format_segment_srt_index(segment: Segment) -> None:
    result = format_segment(segment, "srt", 5)
    assert result.startswith("6\n")


##### BUILD_RESPONSE (ovld) #####


async def test_build_response_text(segment: Segment, info: TranscriptionInfo, sample_audio: np.ndarray) -> None:
    body, ct = build_response([segment], info, sample_audio, "text")
    assert body == "hello"
    assert ct == "text/plain"


async def test_build_response_json(segment: Segment, info: TranscriptionInfo, sample_audio: np.ndarray) -> None:
    body, ct = build_response([segment], info, sample_audio, "json")
    assert ct == "application/json"
    assert "hello" in body


async def test_build_response_verbose_json(segment: Segment, info: TranscriptionInfo, sample_audio: np.ndarray) -> None:
    body, ct = build_response([segment], info, sample_audio, "verbose_json")
    assert ct == "application/json"
    parsed = orjson.loads(body)
    assert "segments" in parsed
    assert "language" in parsed
    assert "duration" in parsed


async def test_build_response_srt(segment: Segment, info: TranscriptionInfo, sample_audio: np.ndarray) -> None:
    body, ct = build_response([segment], info, sample_audio, "srt")
    assert ct == "text/plain"
    assert "-->" in body


async def test_build_response_vtt(segment: Segment, info: TranscriptionInfo, sample_audio: np.ndarray) -> None:
    body, ct = build_response([segment], info, sample_audio, "vtt")
    assert ct == "text/plain"
    assert "WEBVTT" in body
    assert "-->" in body


async def test_build_response_verbose_json_with_words(
    segment_with_words: Segment,
    info: TranscriptionInfo,
    sample_audio: np.ndarray,
) -> None:
    body, _ = build_response([segment_with_words], info, sample_audio, "verbose_json", True)
    parsed = orjson.loads(body)
    assert parsed["words"] is not None
    assert len(parsed["words"]) == 2


async def test_build_response_translate_task(
    segment: Segment, info: TranscriptionInfo, sample_audio: np.ndarray
) -> None:
    body, _ = build_response([segment], info, sample_audio, "verbose_json", False, "translate")
    parsed = orjson.loads(body)
    assert parsed["task"] == "translate"


##### SEGMENT_TO_MODEL #####


async def test_segment_to_model_basic(segment: Segment) -> None:
    model = segment_to_model(segment)
    assert model.text == " hello"
    assert model.start == 0.0
    assert model.end == 1.0
    assert model.words is None


async def test_segment_to_model_with_words(segment_with_words: Segment) -> None:
    model = segment_to_model(segment_with_words, word_timestamps=True)
    assert model.words is not None
    assert len(model.words) == 2
    assert model.words[0].word == "hello"
