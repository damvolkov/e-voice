"""Unit tests for WhisperAdapter — lifecycle, inference, domain type conversion."""

import numpy as np
import pytest
from tests.unit.e_voice.adapters.conftest import make_segment

from e_voice.adapters.stt.whisper import WhisperAdapter
from e_voice.core.settings import ComputeType, DeviceType, STTConfig
from e_voice.models.stt import InferenceParams, ModelSpec, Span, Transcript

_TEST_CONFIG = STTConfig(model="test-model", device=DeviceType.CPU, compute_type=ComputeType.AUTO)
_TEST_SPEC = ModelSpec(model_id="test-model", device="cpu", compute_type="auto")

##### MODEL SPEC #####


async def test_model_spec_frozen() -> None:
    spec = ModelSpec(model_id="test", device="gpu", compute_type="float16")
    with pytest.raises(AttributeError):
        spec.model_id = "other"  # ty: ignore[invalid-assignment]


async def test_model_spec_defaults() -> None:
    spec = ModelSpec(model_id="test")
    assert spec.device == "cpu"
    assert spec.compute_type == "auto"


async def test_model_spec_equality() -> None:
    a = ModelSpec(model_id="m", device="gpu", compute_type="float16")
    b = ModelSpec(model_id="m", device="gpu", compute_type="float16")
    c = ModelSpec(model_id="m", device="cpu")
    assert a == b
    assert a != c


async def test_model_spec_hashable() -> None:
    specs = {
        ModelSpec(model_id="m", device="gpu"),
        ModelSpec(model_id="m", device="cpu"),
        ModelSpec(model_id="m", device="gpu"),
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
        (DeviceType.GPU, ComputeType.FLOAT16),
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


async def test_adapter_supported_devices() -> None:
    adapter = WhisperAdapter()
    assert DeviceType.CPU in adapter.supported_devices
    assert DeviceType.GPU in adapter.supported_devices


##### MODEL LIFECYCLE #####


async def test_load_stores_model(mocker) -> None:
    adapter = WhisperAdapter()
    spec = ModelSpec(model_id="test-model", device="cpu")
    mocker.patch.object(adapter, "_wa_create_model", return_value=mocker.MagicMock())
    await adapter.load(spec)
    assert await adapter.is_loaded(spec)
    assert spec in adapter.loaded_models()


async def test_load_idempotent(mocker) -> None:
    adapter = WhisperAdapter()
    spec = ModelSpec(model_id="test-model", device="cpu")
    create = mocker.patch.object(adapter, "_wa_create_model", return_value=mocker.MagicMock())
    await adapter.load(spec)
    await adapter.load(spec)
    create.assert_called_once()


async def test_unload_removes_model(mocker) -> None:
    adapter = WhisperAdapter()
    spec = ModelSpec(model_id="test-model", device="cpu")
    mocker.patch.object(adapter, "_wa_create_model", return_value=mocker.MagicMock())
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
        adapter._wa_resolve()


async def test_download_calls_snapshot(tmp_path, mocker) -> None:
    adapter = WhisperAdapter()
    dl = mocker.patch("e_voice.adapters.stt.whisper.snapshot_download", return_value=str(tmp_path))
    path = await adapter.download("org/model")
    dl.assert_called_once()
    assert path == tmp_path


##### BATCH INFERENCE — RETURNS TRANSCRIPT #####


async def test_transcribe_returns_transcript(mocker) -> None:
    adapter = WhisperAdapter(config=_TEST_CONFIG)
    mock_model = mocker.MagicMock()
    mock_segments = [make_segment(text=" hello")]
    mock_info = mocker.MagicMock(language="en", duration=1.0)
    mock_model.transcribe.return_value = (iter(mock_segments), mock_info)
    mocker.patch.object(adapter, "_wa_create_model", return_value=mock_model)
    await adapter.load(_TEST_SPEC)

    result = await adapter.transcribe(np.zeros(16_000, dtype=np.float32))
    assert isinstance(result, Transcript)
    assert len(result.spans) == 1
    assert isinstance(result.spans[0], Span)
    assert result.spans[0].text == " hello"
    assert result.language == "en"


async def test_translate_returns_transcript(mocker) -> None:
    adapter = WhisperAdapter(config=_TEST_CONFIG)
    mock_model = mocker.MagicMock()
    mock_model.transcribe.return_value = (
        iter([make_segment(text=" hola")]),
        mocker.MagicMock(language="es", duration=1.0),
    )
    mocker.patch.object(adapter, "_wa_create_model", return_value=mock_model)
    await adapter.load(_TEST_SPEC)

    result = await adapter.translate(np.zeros(16_000, dtype=np.float32))
    assert isinstance(result, Transcript)
    assert result.spans[0].text == " hola"
    mock_model.transcribe.assert_called_once()
    assert mock_model.transcribe.call_args[1]["task"] == "translate"


##### STREAMING INFERENCE — YIELDS SPANS #####


async def test_transcribe_stream_yields_spans(mocker) -> None:
    adapter = WhisperAdapter(config=_TEST_CONFIG)
    mock_model = mocker.MagicMock()
    segs = [make_segment(text=" hello"), make_segment(text=" world")]
    mock_model.transcribe.return_value = (iter(segs), mocker.MagicMock())
    mocker.patch.object(adapter, "_wa_create_model", return_value=mock_model)
    await adapter.load(_TEST_SPEC)

    collected: list[Span] = []
    async for span in adapter.transcribe_stream(np.zeros(16_000, dtype=np.float32)):
        collected.append(span)

    assert len(collected) == 2
    assert all(isinstance(s, Span) for s in collected)
    assert collected[0].text == " hello"
    assert collected[1].text == " world"


async def test_translate_stream_yields_spans(mocker) -> None:
    adapter = WhisperAdapter(config=_TEST_CONFIG)
    mock_model = mocker.MagicMock()
    segs = [make_segment(text=" hola"), make_segment(text=" mundo")]
    mock_model.transcribe.return_value = (iter(segs), mocker.MagicMock())
    mocker.patch.object(adapter, "_wa_create_model", return_value=mock_model)
    await adapter.load(_TEST_SPEC)

    collected: list[Span] = []
    async for span in adapter.translate_stream(np.zeros(16_000, dtype=np.float32)):
        collected.append(span)

    assert len(collected) == 2
    assert mock_model.transcribe.call_args[1]["task"] == "translate"


##### STREAM ERROR PROPAGATION #####


async def test_run_stream_propagates_error(mocker) -> None:
    adapter = WhisperAdapter(config=_TEST_CONFIG)
    mock_model = mocker.MagicMock()

    def _explode(*args, **kwargs):
        raise RuntimeError("GPU exploded")

    mock_model.transcribe.side_effect = _explode
    mocker.patch.object(adapter, "_wa_create_model", return_value=mock_model)
    await adapter.load(_TEST_SPEC)

    with pytest.raises(RuntimeError, match="GPU exploded"):
        async for _ in adapter.transcribe_stream(np.zeros(16_000, dtype=np.float32)):
            pass


##### _WA_CREATE_MODEL #####


async def test_create_model_calls_whisper_model(mocker) -> None:
    mock_cls = mocker.patch("e_voice.adapters.stt.whisper.WhisperModel")
    adapter = WhisperAdapter()
    spec = ModelSpec(model_id="openai/whisper-tiny", device="cpu", compute_type="int8")

    adapter._wa_create_model(spec)

    mock_cls.assert_called_once()
    assert mock_cls.call_args[0][0] == "openai/whisper-tiny"
    assert mock_cls.call_args[1]["device"] == "cpu"
