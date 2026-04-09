"""Unit tests for QwenAdapter — lifecycle, synthesis routing, streaming, clone, capabilities.

All torch/qwen-tts calls are mocked since they are optional dependencies.
Skipped entirely in CI when torch is not installed.
"""

from importlib.util import find_spec
from unittest.mock import MagicMock

import numpy as np
import pytest

from e_voice.core.settings import DeviceType
from e_voice.models.error import BackendCapabilityError
from e_voice.models.tts import SynthesisParams, TTSModelSpec

pytestmark = pytest.mark.skipif(find_spec("torch") is None, reason="torch not installed (optional dep)")


@pytest.fixture
def _qwen_cls():
    """Import QwenAdapter class."""
    from e_voice.adapters.tts.qwen import QwenAdapter

    return QwenAdapter


@pytest.fixture
def loaded_adapter(_qwen_cls, mocker, tmp_path):
    """Return a QwenAdapter with mocked models loaded."""
    adapter = _qwen_cls()
    adapter._batch_model = MagicMock()
    adapter._stream_model = MagicMock()
    adapter._loaded_spec = TTSModelSpec(model_id="qwen", device=DeviceType.GPU)
    adapter._preset_voices = ["serena", "aiden"]
    adapter._voices = ["serena", "aiden", "tatan"]
    adapter._voice_entries = []
    adapter._can_clone = True
    mocker.patch("e_voice.core.settings.Settings.MODELS_PATH", tmp_path, create=True)
    return adapter


##### INIT #####


async def test_adapter_init(_qwen_cls) -> None:
    adapter = _qwen_cls()
    assert adapter.voices == []
    assert adapter.voice_entries == []
    assert adapter.loaded_models() == []
    assert adapter.supports_voice_clone is False


async def test_adapter_supported_devices_gpu_only(_qwen_cls) -> None:
    adapter = _qwen_cls()
    assert adapter.supported_devices == frozenset({DeviceType.GPU})
    assert DeviceType.CPU not in adapter.supported_devices


##### LOAD #####


async def test_load_rejects_cpu(_qwen_cls) -> None:
    adapter = _qwen_cls()
    with pytest.raises(BackendCapabilityError, match="requires GPU"):
        await adapter.load(TTSModelSpec(model_id="qwen", device=DeviceType.CPU))


async def test_load_creates_models(_qwen_cls, mocker, tmp_path) -> None:
    adapter = _qwen_cls()
    mocker.patch.object(
        type(adapter),
        "_qa_create_models",
        return_value=(MagicMock(), MagicMock(), ["preset1"], True),
    )
    mocker.patch.object(type(adapter), "_qa_discover_voice_entries", return_value=[])
    mocker.patch("e_voice.core.settings.Settings.MODELS_PATH", tmp_path, create=True)

    await adapter.load(TTSModelSpec(model_id="qwen", device=DeviceType.GPU))
    assert await adapter.is_loaded()
    assert len(adapter.loaded_models()) == 1
    assert adapter.supports_voice_clone is True
    assert "preset1" in adapter.voices


async def test_load_idempotent(_qwen_cls, mocker, tmp_path) -> None:
    adapter = _qwen_cls()
    create = mocker.patch.object(
        type(adapter),
        "_qa_create_models",
        return_value=(MagicMock(), MagicMock(), [], False),
    )
    mocker.patch.object(type(adapter), "_qa_discover_voice_entries", return_value=[])
    mocker.patch("e_voice.core.settings.Settings.MODELS_PATH", tmp_path, create=True)

    spec = TTSModelSpec(model_id="qwen", device=DeviceType.GPU)
    await adapter.load(spec)
    await adapter.load(spec)
    create.assert_called_once()


##### UNLOAD #####


async def test_unload_not_loaded(_qwen_cls) -> None:
    adapter = _qwen_cls()
    assert await adapter.unload() is False


async def test_unload_releases_resources(_qwen_cls, mocker, tmp_path) -> None:
    adapter = _qwen_cls()
    mocker.patch.object(
        type(adapter),
        "_qa_create_models",
        return_value=(MagicMock(), MagicMock(), [], False),
    )
    mocker.patch.object(type(adapter), "_qa_discover_voice_entries", return_value=[])
    mocker.patch("e_voice.core.settings.Settings.MODELS_PATH", tmp_path, create=True)
    mock_torch = MagicMock()
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    await adapter.load(TTSModelSpec(model_id="qwen", device=DeviceType.GPU))
    result = await adapter.unload()
    assert result is True
    assert adapter.loaded_models() == []
    assert adapter.voices == []
    assert adapter.supports_voice_clone is False


##### VOICE DISCOVERY #####


async def test_discover_voice_entries(tmp_path, _qwen_cls, mocker) -> None:
    voice_dir = tmp_path / "tts" / "qwen" / "voices"
    voice_dir.mkdir(parents=True)
    (voice_dir / "serena_en.pt").touch()
    (voice_dir / "tatan_es.pt").touch()

    mocker.patch("e_voice.core.settings.Settings.MODELS_PATH", tmp_path, create=True)

    entries = _qwen_cls._qa_discover_voice_entries()
    assert len(entries) == 2
    ids = {e.id for e in entries}
    assert "serena" in ids
    assert "tatan" in ids
    langs = {e.language for e in entries}
    assert "en" in langs
    assert "es" in langs


async def test_discover_voice_entries_empty(tmp_path, _qwen_cls, mocker) -> None:
    mocker.patch("e_voice.core.settings.Settings.MODELS_PATH", tmp_path, create=True)
    assert _qwen_cls._qa_discover_voice_entries() == []


##### SYNTHESIZE — VOICE ROUTING #####


async def test_synthesize_with_preset_voice(loaded_adapter) -> None:
    mock_model = loaded_adapter._batch_model
    mock_model.generate_custom_voice.return_value = ([np.zeros(24000, dtype=np.float32)], 24000)

    params = SynthesisParams(voice="serena", lang="en")
    samples, sr = await loaded_adapter.synthesize("hello", params=params)

    assert sr == 24000
    assert len(samples) == 24000
    mock_model.generate_custom_voice.assert_called_once()


async def test_synthesize_with_cloned_voice(loaded_adapter, mocker) -> None:
    mock_model = loaded_adapter._batch_model
    mock_model.generate_voice_clone.return_value = ([np.zeros(24000, dtype=np.float32)], 24000)

    fake_prompt = {"ref_code": "fake"}
    mocker.patch.object(loaded_adapter, "_qa_load_clone_prompt", return_value=fake_prompt)

    params = SynthesisParams(voice="tatan", lang="es")
    samples, sr = await loaded_adapter.synthesize("hola", params=params)

    assert sr == 24000
    mock_model.generate_voice_clone.assert_called_once()


async def test_synthesize_unknown_voice_raises(loaded_adapter, mocker) -> None:
    mocker.patch.object(loaded_adapter, "_qa_load_clone_prompt", return_value=None)
    loaded_adapter._preset_voices = []

    params = SynthesisParams(voice="ghost")
    with pytest.raises(RuntimeError, match="not found"):
        await loaded_adapter.synthesize("test", params=params)


##### SYNTHESIZE STREAM — VOICE ROUTING #####


async def test_synthesize_stream_with_preset(loaded_adapter) -> None:
    chunks = [(np.zeros(4800, dtype=np.float32), 24000, {})]
    loaded_adapter._stream_model.generate_custom_voice_streaming.return_value = iter(chunks)

    params = SynthesisParams(voice="aiden", lang="en")
    collected = []
    async for samples, sr in loaded_adapter.synthesize_stream("test", params=params):
        collected.append((samples, sr))

    assert len(collected) == 1
    assert collected[0][1] == 24000


async def test_synthesize_stream_with_clone(loaded_adapter, mocker) -> None:
    chunks = [(np.zeros(4800, dtype=np.float32), 24000, {})]
    loaded_adapter._stream_model.generate_voice_clone_streaming.return_value = iter(chunks)

    fake_prompt = {"ref_code": "fake"}
    mocker.patch.object(loaded_adapter, "_qa_load_clone_prompt", return_value=fake_prompt)

    params = SynthesisParams(voice="tatan", lang="es")
    collected = []
    async for samples, sr in loaded_adapter.synthesize_stream("hola", params=params):
        collected.append((samples, sr))

    assert len(collected) == 1


async def test_synthesize_stream_unknown_voice_raises(loaded_adapter, mocker) -> None:
    mocker.patch.object(loaded_adapter, "_qa_load_clone_prompt", return_value=None)
    loaded_adapter._preset_voices = []

    params = SynthesisParams(voice="ghost")
    with pytest.raises(RuntimeError, match="not found"):
        async for _ in loaded_adapter.synthesize_stream("test", params=params):
            pass


async def test_synthesize_stream_propagates_error(loaded_adapter) -> None:
    def _explode(**kwargs):
        raise RuntimeError("GPU exploded")

    loaded_adapter._stream_model.generate_custom_voice_streaming.side_effect = _explode

    params = SynthesisParams(voice="serena", lang="en")
    with pytest.raises(RuntimeError, match="GPU exploded"):
        async for _ in loaded_adapter.synthesize_stream("test", params=params):
            pass


##### CLONE VOICE #####


async def test_clone_voice_saves_prompt(loaded_adapter, mocker, tmp_path) -> None:
    mock_model = loaded_adapter._batch_model
    mock_model.create_voice_clone_prompt.return_value = {"ref_code": "fake"}
    mock_torch = MagicMock()
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    voice_dir = tmp_path / "tts" / "qwen" / "voices"
    voice_dir.mkdir(parents=True)

    voice_id = await loaded_adapter.clone_voice("newvoice", tmp_path / "ref.wav", "hello", language="fr")

    assert voice_id == "newvoice"
    assert "newvoice" in loaded_adapter.voices
    mock_torch.save.assert_called_once()
    save_path = mock_torch.save.call_args[0][1]
    assert "newvoice_fr.pt" in str(save_path)


async def test_clone_voice_default_language(loaded_adapter, mocker, tmp_path) -> None:
    loaded_adapter._batch_model.create_voice_clone_prompt.return_value = {}
    mock_torch = MagicMock()
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    voice_dir = tmp_path / "tts" / "qwen" / "voices"
    voice_dir.mkdir(parents=True)

    await loaded_adapter.clone_voice("test", tmp_path / "ref.wav", "hello")

    save_path = mock_torch.save.call_args[0][1]
    assert "test_en.pt" in str(save_path)


##### CLONE PROMPT LOADING #####


async def test_load_clone_prompt_exists(loaded_adapter, mocker, tmp_path) -> None:
    voice_dir = tmp_path / "tts" / "qwen" / "voices"
    voice_dir.mkdir(parents=True)
    (voice_dir / "myvoice_es.pt").write_bytes(b"fake")

    mock_torch = MagicMock()
    mock_torch.load.return_value = {"ref_code": "loaded"}
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    result = loaded_adapter._qa_load_clone_prompt("myvoice")
    assert result is not None
    mock_torch.load.assert_called_once()


async def test_load_clone_prompt_not_found(loaded_adapter, tmp_path) -> None:
    result = loaded_adapter._qa_load_clone_prompt("nonexistent")
    assert result is None


##### RESOLVE #####


async def test_resolve_batch_not_loaded(_qwen_cls) -> None:
    adapter = _qwen_cls()
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter._qa_resolve_batch()


async def test_resolve_stream_not_loaded(_qwen_cls) -> None:
    adapter = _qwen_cls()
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter._qa_resolve_stream()


##### TO_FLOAT32 #####


async def test_to_float32_numpy(_qwen_cls) -> None:
    arr = np.array([0.5, -0.5], dtype=np.float64)
    result = _qwen_cls._qa_to_float32(arr)
    assert result.dtype == np.float32


async def test_to_float32_already_f32(_qwen_cls) -> None:
    arr = np.array([0.5, -0.5], dtype=np.float32)
    result = _qwen_cls._qa_to_float32(arr)
    assert result is arr


##### CLONE CAPABILITY #####


async def test_supports_voice_clone_dynamic(_qwen_cls) -> None:
    adapter = _qwen_cls()
    assert adapter.supports_voice_clone is False
    adapter._can_clone = True
    assert adapter.supports_voice_clone is True


##### DOWNLOAD #####


async def test_download_calls_snapshot(tmp_path, _qwen_cls, mocker) -> None:
    mocker.patch("e_voice.core.settings.Settings.MODELS_PATH", tmp_path, create=True)
    mock_dl = mocker.patch("huggingface_hub.snapshot_download", return_value=str(tmp_path))

    adapter = _qwen_cls()
    result = await adapter.download("Qwen/model")

    mock_dl.assert_called_once()
    assert result == tmp_path


##### LANGUAGE MAPPING #####


async def test_resolve_language() -> None:
    from e_voice.adapters.tts.qwen import _resolve_language

    assert _resolve_language("en-us") == "English"
    assert _resolve_language("es") == "Spanish"
    assert _resolve_language("ja") == "Japanese"
    assert _resolve_language("unknown") == "English"
