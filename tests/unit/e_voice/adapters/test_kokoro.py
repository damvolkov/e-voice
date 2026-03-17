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
