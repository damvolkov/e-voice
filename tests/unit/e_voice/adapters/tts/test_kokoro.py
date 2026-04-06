"""Unit tests for KokoroAdapter — lifecycle, synthesis, voice resolution."""

import numpy as np
import pytest

from e_voice.adapters.tts.kokoro import KokoroAdapter, _resolve_provider
from e_voice.core.settings import DeviceType
from e_voice.core.settings import settings as st
from e_voice.models.tts import OnnxProvider, SynthesisParams, TTSModelSpec, resolve_voice_lang

##### VOICE LANG RESOLUTION — resolve_voice_lang #####


@pytest.mark.parametrize(
    ("voice", "expected"),
    [
        ("af_heart", "en-us"),
        ("bf_emma", "en-gb"),
        ("ef_dora", "es"),
        ("ff_siwis", "fr"),
        ("hf_alpha", "hi"),
        ("if_sara", "it"),
        ("jf_alpha", "ja"),
        ("pf_dora", "pt-br"),
        ("zf_xiaobei", "zh"),
    ],
    ids=["en-us", "en-gb", "es", "fr", "hi", "it", "ja", "pt-br", "zh"],
)
async def test_resolve_voice_lang_valid_prefix(voice: str, expected: str) -> None:
    assert resolve_voice_lang(voice) == expected


async def test_resolve_voice_lang_unknown_prefix_raises() -> None:
    with pytest.raises(ValueError, match="Unknown voice prefix"):
        resolve_voice_lang("xf_unknown")


async def test_resolve_voice_lang_empty_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_voice_lang("")


##### SYNTHESIS PARAMS #####


async def test_synthesis_params_defaults() -> None:
    params = SynthesisParams()
    assert params.voice == "af_heart"
    assert params.speed == 1.0
    assert params.lang == "en-us"


async def test_synthesis_params_frozen() -> None:
    params = SynthesisParams()
    with pytest.raises(AttributeError):
        params.voice = "bf_emma"  # ty: ignore[invalid-assignment]


##### ONNX PROVIDER RESOLUTION #####


async def test_resolve_provider_gpu(mocker) -> None:
    mocker.patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert _resolve_provider("gpu") == OnnxProvider.CUDA


async def test_resolve_provider_cpu(mocker) -> None:
    mocker.patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"])
    assert _resolve_provider("cpu") == OnnxProvider.CPU


async def test_resolve_provider_gpu_fallback_to_cpu(mocker) -> None:
    mocker.patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"])
    assert _resolve_provider("gpu") == OnnxProvider.CPU


##### MODEL SPEC #####


async def test_tts_model_spec_defaults() -> None:
    spec = TTSModelSpec()
    assert spec.model_id == "kokoro"
    assert spec.device == DeviceType.CPU


async def test_tts_model_spec_hashable() -> None:
    specs = {
        TTSModelSpec(device=DeviceType.GPU),
        TTSModelSpec(device=DeviceType.CPU),
        TTSModelSpec(device=DeviceType.GPU),
    }
    assert len(specs) == 2


##### ADAPTER LIFECYCLE #####


async def test_adapter_init() -> None:
    adapter = KokoroAdapter()
    assert adapter.loaded_models() == []


async def test_adapter_is_loaded_false() -> None:
    adapter = KokoroAdapter()
    assert await adapter.is_loaded() is False


async def test_adapter_unload_not_loaded() -> None:
    adapter = KokoroAdapter()
    assert await adapter.unload() is False


async def test_adapter_resolve_raises_not_loaded() -> None:
    adapter = KokoroAdapter()
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter._ka_resolve()


##### LIFECYCLE WITH MOCKS #####


async def test_load_stores_model(mocker) -> None:
    adapter = KokoroAdapter()
    spec = TTSModelSpec(device=DeviceType.CPU)
    adapter._models[spec] = mocker.MagicMock()
    assert await adapter.is_loaded(spec)
    assert spec in adapter.loaded_models()


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
    spec = TTSModelSpec(device=st.tts.device)
    adapter._models[spec] = mock_kokoro

    samples, sr = await adapter.synthesize("hello")
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
    spec = TTSModelSpec(device=st.tts.device)
    adapter._models[spec] = mock_kokoro

    collected = []
    async for samples, sr in adapter.synthesize_stream("hello"):
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


##### LOAD WITH DOWNLOAD #####


async def test_load_downloads_when_files_missing(tmp_path, mocker) -> None:
    mocker.patch("e_voice.core.settings.Settings.MODELS_PATH", tmp_path, create=True)
    mock_kokoro_cls = mocker.patch("e_voice.adapters.tts.kokoro.Kokoro")
    mock_instance = mocker.MagicMock()
    mock_instance.get_voices.return_value = ["af_heart"]
    mock_kokoro_cls.return_value = mock_instance
    mock_download = mocker.patch.object(KokoroAdapter, "_ka_download_files", new_callable=mocker.AsyncMock)

    adapter = KokoroAdapter()
    await adapter.load(TTSModelSpec(device=DeviceType.CPU))

    mock_download.assert_awaited_once()
    assert await adapter.is_loaded(TTSModelSpec(device=DeviceType.CPU))
    assert adapter.voices == ["af_heart"]


##### _DOWNLOAD #####


async def test_download_fetches_files(tmp_path, mocker) -> None:
    model_path = tmp_path / "model.onnx"
    voices_path = tmp_path / "voices.bin"

    class FakeResp:
        def raise_for_status(self):
            pass

        async def aiter_bytes(self, chunk_size=0):
            yield b"\x00" * 100

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

    class FakeClient:
        def stream(self, method, url):
            return FakeResp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

    mocker.patch("e_voice.adapters.tts.kokoro.httpx.AsyncClient", return_value=FakeClient())

    await KokoroAdapter._ka_download_files(model_path, voices_path)

    assert model_path.exists()
    assert voices_path.exists()


##### DOWNLOAD METHOD #####


async def test_download_returns_model_dir(tmp_path, mocker) -> None:
    mocker.patch("e_voice.core.settings.Settings.MODELS_PATH", tmp_path, create=True)
    mock_download = mocker.patch.object(KokoroAdapter, "_ka_download_files", new_callable=mocker.AsyncMock)

    adapter = KokoroAdapter()
    result = await adapter.download()

    mock_download.assert_awaited_once()
    assert result == tmp_path / "tts" / st.tts.backend
