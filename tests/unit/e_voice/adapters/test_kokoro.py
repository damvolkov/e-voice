"""Unit tests for KokoroAdapter — lifecycle, synthesis, voice resolution."""

import numpy as np
import pytest

from e_voice.adapters.kokoro import KokoroAdapter, _resolve_provider
from e_voice.core.settings import DeviceType
from e_voice.models.tts import OnnxProvider, SynthesisParams, TTSModelSpec

##### VOICE LANG RESOLUTION #####


@pytest.mark.parametrize(
    ("voice", "lang", "expected"),
    [
        ("af_heart", None, "en-us"),
        ("bf_emma", None, "en-gb"),
        ("ef_spanish", None, "es"),
        ("jf_japanese", None, "ja"),
        ("af_heart", "fr", "fr"),
        ("bf_emma", "de", "de"),
    ],
    ids=["en-us-prefix", "en-gb-prefix", "es-prefix", "ja-prefix", "explicit-fr", "explicit-de"],
)
async def test_synthesis_params_resolved_lang(voice: str, lang: str | None, expected: str) -> None:
    assert SynthesisParams(voice=voice, lang=lang).resolved_lang == expected


async def test_synthesis_params_unknown_prefix_defaults_en_us() -> None:
    assert SynthesisParams(voice="xf_unknown").resolved_lang == "en-us"


async def test_synthesis_params_empty_voice_defaults_en_us() -> None:
    assert SynthesisParams(voice="").resolved_lang == "en-us"


##### ONNX PROVIDER RESOLUTION #####


async def test_resolve_provider_cuda(mocker) -> None:
    mocker.patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert _resolve_provider("cuda") == OnnxProvider.CUDA


async def test_resolve_provider_cpu(mocker) -> None:
    mocker.patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"])
    assert _resolve_provider("cpu") == OnnxProvider.CPU


async def test_resolve_provider_cuda_fallback_to_cpu(mocker) -> None:
    mocker.patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"])
    assert _resolve_provider("cuda") == OnnxProvider.CPU


##### MODEL SPEC #####


async def test_tts_model_spec_defaults() -> None:
    spec = TTSModelSpec()
    assert spec.model_id == "kokoro"
    assert spec.device == DeviceType.CPU


async def test_tts_model_spec_hashable() -> None:
    specs = {
        TTSModelSpec(device=DeviceType.CUDA),
        TTSModelSpec(device=DeviceType.CPU),
        TTSModelSpec(device=DeviceType.CUDA),
    }
    assert len(specs) == 2


##### ADAPTER LIFECYCLE #####


async def test_adapter_init() -> None:
    adapter = KokoroAdapter()
    assert adapter.loaded == []


async def test_adapter_is_loaded_false() -> None:
    adapter = KokoroAdapter()
    assert await adapter.is_loaded() is False


async def test_adapter_unload_not_loaded() -> None:
    adapter = KokoroAdapter()
    assert await adapter.unload() is False


async def test_adapter_resolve_raises_not_loaded() -> None:
    adapter = KokoroAdapter()
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter._resolve()


##### LIFECYCLE WITH MOCKS #####


async def test_load_stores_model(mocker) -> None:
    adapter = KokoroAdapter()
    spec = TTSModelSpec(device=DeviceType.CPU)
    adapter._models[spec] = mocker.MagicMock()
    assert await adapter.is_loaded(spec)
    assert spec in adapter.loaded


async def test_load_idempotent(mocker) -> None:
    adapter = KokoroAdapter()
    spec = TTSModelSpec(device=DeviceType.CPU)
    first_model = mocker.MagicMock()
    adapter._models[spec] = first_model
    await adapter.load(spec)
    assert adapter._models[spec] is first_model


async def test_unload_clears_model(mocker) -> None:
    adapter = KokoroAdapter()
    spec = TTSModelSpec(device=DeviceType.CPU)
    adapter._models[spec] = mocker.MagicMock()
    assert await adapter.unload(spec) is True
    assert not await adapter.is_loaded(spec)


##### SYNTHESIZE #####


async def test_synthesize_calls_create(mocker) -> None:
    adapter = KokoroAdapter()
    mock_kokoro = mocker.MagicMock()
    mock_kokoro.create.return_value = (np.zeros(24_000, dtype=np.float32), 24_000)
    spec = TTSModelSpec(device=DeviceType.CPU)
    adapter._models[spec] = mock_kokoro

    samples, sr = await adapter.synthesize("hello", spec=spec)
    assert sr == 24_000
    assert len(samples) == 24_000
    mock_kokoro.create.assert_called_once()


##### SYNTHESIZE STREAM #####


async def test_synthesize_stream_yields_chunks(mocker) -> None:
    adapter = KokoroAdapter()
    mock_kokoro = mocker.MagicMock()
    chunks = [
        (np.zeros(4800, dtype=np.float32), 24_000),
        (np.zeros(4800, dtype=np.float32), 24_000),
    ]

    async def fake_stream(*args, **kwargs):
        for c in chunks:
            yield c

    mock_kokoro.create_stream = fake_stream
    spec = TTSModelSpec(device=DeviceType.CPU)
    adapter._models[spec] = mock_kokoro

    collected = []
    async for samples, sr in adapter.synthesize_stream("hello", spec=spec):
        collected.append((samples, sr))

    assert len(collected) == 2
    assert collected[0][1] == 24_000


##### VOICES #####


async def test_voices_cached_after_load(mocker) -> None:
    adapter = KokoroAdapter()
    adapter._voices = ["af_heart", "bf_emma"]
    assert adapter.voices == ["af_heart", "bf_emma"]


async def test_voices_empty_before_load() -> None:
    adapter = KokoroAdapter()
    assert adapter.voices == []
