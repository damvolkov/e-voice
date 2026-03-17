"""Unit tests for adapters/kokoro.py — device config, language resolution, lifecycle."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from e_voice.adapters.kokoro import _ONNX_PROVIDERS, KokoroAdapter, _resolve_lang
from e_voice.core.settings import DeviceType

##### LANGUAGE RESOLUTION #####


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
async def test_resolve_lang(voice: str, lang: str | None, expected: str) -> None:
    assert _resolve_lang(voice, lang) == expected


async def test_resolve_lang_unknown_prefix() -> None:
    assert _resolve_lang("xf_unknown", None) == "en-us"


async def test_resolve_lang_empty_voice() -> None:
    assert _resolve_lang("", None) == "en-us"


##### ONNX PROVIDER MAPPING #####


async def test_onnx_provider_cuda() -> None:
    assert _ONNX_PROVIDERS[DeviceType.CUDA] == "CUDAExecutionProvider"


async def test_onnx_provider_cpu() -> None:
    assert _ONNX_PROVIDERS[DeviceType.CPU] == "CPUExecutionProvider"


async def test_onnx_provider_auto_not_mapped() -> None:
    assert DeviceType.AUTO not in _ONNX_PROVIDERS


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
        adapter._lc_resolve()


##### LIFECYCLE WITH MOCKS #####


async def test_load_sets_model(tmp_path) -> None:
    model_path = tmp_path / "kokoro-v1.0.onnx"
    voices_path = tmp_path / "voices-v1.0.bin"
    model_path.write_bytes(b"fake")
    voices_path.write_bytes(b"fake")

    adapter = KokoroAdapter(model_dir=tmp_path)
    mock_kokoro = MagicMock()
    with patch("e_voice.adapters.kokoro.Kokoro", return_value=mock_kokoro):
        await adapter.load()

    assert await adapter.is_loaded()
    assert adapter.loaded_models() == ["kokoro"]


async def test_load_skips_if_already_loaded(tmp_path) -> None:
    model_path = tmp_path / "kokoro-v1.0.onnx"
    voices_path = tmp_path / "voices-v1.0.bin"
    model_path.write_bytes(b"fake")
    voices_path.write_bytes(b"fake")

    adapter = KokoroAdapter(model_dir=tmp_path)
    mock_kokoro = MagicMock()
    with patch("e_voice.adapters.kokoro.Kokoro", return_value=mock_kokoro) as ctor:
        await adapter.load()
        await adapter.load()
    ctor.assert_called_once()


async def test_unload_clears_model(tmp_path) -> None:
    model_path = tmp_path / "kokoro-v1.0.onnx"
    voices_path = tmp_path / "voices-v1.0.bin"
    model_path.write_bytes(b"fake")
    voices_path.write_bytes(b"fake")

    adapter = KokoroAdapter(model_dir=tmp_path)
    with patch("e_voice.adapters.kokoro.Kokoro", return_value=MagicMock()):
        await adapter.load()
    assert await adapter.unload() is True
    assert not await adapter.is_loaded()


async def test_synthesize_calls_create(tmp_path) -> None:
    model_path = tmp_path / "kokoro-v1.0.onnx"
    voices_path = tmp_path / "voices-v1.0.bin"
    model_path.write_bytes(b"fake")
    voices_path.write_bytes(b"fake")

    mock_kokoro = MagicMock()
    mock_kokoro.create.return_value = (np.zeros(24000, dtype=np.float32), 24000)

    adapter = KokoroAdapter(model_dir=tmp_path)
    with patch("e_voice.adapters.kokoro.Kokoro", return_value=mock_kokoro):
        await adapter.load()
    samples, sr = await adapter.synthesize("hello", voice="af_heart")
    assert sr == 24000
    assert len(samples) == 24000


async def test_download_creates_files(tmp_path) -> None:
    adapter = KokoroAdapter(model_dir=tmp_path)
    with patch.object(adapter, "_lc_download_files") as dl:
        path = await adapter.download()
    dl.assert_called_once()
    assert path == tmp_path


##### SYNTHESIZE STREAM #####


async def test_synthesize_stream_yields_chunks(tmp_path) -> None:
    model_path = tmp_path / "kokoro-v1.0.onnx"
    voices_path = tmp_path / "voices-v1.0.bin"
    model_path.write_bytes(b"fake")
    voices_path.write_bytes(b"fake")

    chunks = [
        (np.zeros(4800, dtype=np.float32), 24000),
        (np.zeros(4800, dtype=np.float32), 24000),
    ]

    mock_kokoro = MagicMock()

    async def fake_stream(*args, **kwargs):
        for c in chunks:
            yield c

    mock_kokoro.create_stream = fake_stream

    adapter = KokoroAdapter(model_dir=tmp_path)
    with patch("e_voice.adapters.kokoro.Kokoro", return_value=mock_kokoro):
        await adapter.load()

    collected = []
    async for samples, sr in adapter.synthesize_stream("hello", voice="af_heart"):
        collected.append((samples, sr))

    assert len(collected) == 2
    assert collected[0][1] == 24000


##### GET VOICES #####


async def test_get_voices(tmp_path) -> None:
    model_path = tmp_path / "kokoro-v1.0.onnx"
    voices_path = tmp_path / "voices-v1.0.bin"
    model_path.write_bytes(b"fake")
    voices_path.write_bytes(b"fake")

    mock_kokoro = MagicMock()
    mock_kokoro.get_voices.return_value = ["af_heart", "bf_emma", "jf_alpha"]

    adapter = KokoroAdapter(model_dir=tmp_path)
    with patch("e_voice.adapters.kokoro.Kokoro", return_value=mock_kokoro):
        await adapter.load()

    voices = adapter.get_voices()
    assert voices == ["af_heart", "bf_emma", "jf_alpha"]


async def test_get_voices_not_loaded() -> None:
    adapter = KokoroAdapter()
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.get_voices()
