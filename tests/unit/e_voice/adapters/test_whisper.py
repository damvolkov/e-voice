"""Unit tests for adapters/whisper.py — static formatters and subtitle generation."""

from dataclasses import dataclass

import numpy as np
import pytest

from e_voice.adapters.whisper import WhisperAdapter, _format_srt, _format_srt_segment, _format_vtt, _format_vtt_segment
from e_voice.core.settings import ComputeType, DeviceType, STTConfig
from e_voice.models.transcription import ResponseFormat


@dataclass
class MockWord:
    start: float = 0.0
    end: float = 0.5
    word: str = "hello"
    probability: float = 0.9


@dataclass
class MockSegment:
    id: int = 0
    seek: int = 0
    start: float = 0.0
    end: float = 1.0
    text: str = " hello"
    tokens: tuple = (1, 2, 3)
    temperature: float = 0.0
    avg_logprob: float = -0.3
    compression_ratio: float = 1.2
    no_speech_prob: float = 0.1
    words: list = None


##### SEGMENTS_TO_TEXT #####


async def test_segments_to_text_single() -> None:
    seg = MockSegment(text=" hello world")
    assert WhisperAdapter.segments_to_text([seg]) == "hello world"


async def test_segments_to_text_multiple() -> None:
    segs = [MockSegment(text=" hello"), MockSegment(text=" world")]
    assert WhisperAdapter.segments_to_text(segs) == "hello world"


async def test_segments_to_text_empty() -> None:
    assert WhisperAdapter.segments_to_text([]) == ""


##### SEGMENT_TO_MODEL #####


async def test_segment_to_model_basic() -> None:
    seg = MockSegment(text=" hello", start=0.0, end=1.0, no_speech_prob=0.1)
    model = WhisperAdapter.segment_to_model(seg)
    assert model.text == " hello"
    assert model.start == 0.0
    assert model.end == 1.0
    assert model.words is None


async def test_segment_to_model_with_words() -> None:
    seg = MockSegment(
        text=" hello world",
        words=[MockWord(word="hello", start=0.0, end=0.5), MockWord(word="world", start=0.5, end=1.0)],
    )
    model = WhisperAdapter.segment_to_model(seg, word_timestamps=True)
    assert model.words is not None
    assert len(model.words) == 2
    assert model.words[0].word == "hello"


##### FORMAT_SEGMENT_FOR_STREAMING #####


async def test_format_streaming_text() -> None:
    seg = MockSegment(text=" hello")
    result = WhisperAdapter.format_segment_for_streaming(seg, ResponseFormat.TEXT)
    assert result == " hello"


async def test_format_streaming_json() -> None:
    seg = MockSegment(text=" hello")
    result = WhisperAdapter.format_segment_for_streaming(seg, ResponseFormat.JSON)
    assert "hello" in result


async def test_format_streaming_verbose_json() -> None:
    seg = MockSegment(text=" hello", start=0.0, end=1.0)
    result = WhisperAdapter.format_segment_for_streaming(seg, ResponseFormat.VERBOSE_JSON)
    assert "hello" in result


async def test_format_streaming_srt() -> None:
    seg = MockSegment(text=" hello", start=0.0, end=1.0)
    result = WhisperAdapter.format_segment_for_streaming(seg, ResponseFormat.SRT, segment_index=0)
    assert "1" in result
    assert "-->" in result
    assert "hello" in result


async def test_format_streaming_vtt() -> None:
    seg = MockSegment(text=" hello", start=0.0, end=1.5)
    result = WhisperAdapter.format_segment_for_streaming(seg, ResponseFormat.VTT)
    assert "-->" in result
    assert "hello" in result


##### WHISPER_ADAPTER INIT #####


async def test_adapter_init_default_config() -> None:
    adapter = WhisperAdapter()
    assert adapter.loaded_models() == []


async def test_adapter_is_loaded_false() -> None:
    adapter = WhisperAdapter()
    result = await adapter.is_loaded("nonexistent")
    assert result is False


async def test_adapter_unload_nonexistent() -> None:
    adapter = WhisperAdapter()
    result = await adapter.unload("nonexistent")
    assert result is False


async def test_adapter_resolve_raises_not_loaded() -> None:
    adapter = WhisperAdapter()
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter._lc_resolve("nonexistent")


##### BUILD_RESPONSE #####


async def test_build_response_text() -> None:
    seg = MockSegment(text=" hello world")
    audio = np.zeros(16000, dtype=np.float32)
    info = type("Info", (), {"language": "en", "duration": 1.0})()
    body, ct = WhisperAdapter.build_response([seg], info, audio, ResponseFormat.TEXT)
    assert body == "hello world"
    assert ct == "text/plain"


async def test_build_response_json() -> None:
    seg = MockSegment(text=" hello")
    audio = np.zeros(16000, dtype=np.float32)
    info = type("Info", (), {"language": "en", "duration": 1.0})()
    body, ct = WhisperAdapter.build_response([seg], info, audio, ResponseFormat.JSON)
    assert "hello" in body
    assert ct == "application/json"


async def test_build_response_verbose_json() -> None:
    seg = MockSegment(text=" hello", start=0.0, end=1.0)
    audio = np.zeros(16000, dtype=np.float32)
    info = type("Info", (), {"language": "en", "duration": 1.0})()
    body, ct = WhisperAdapter.build_response([seg], info, audio, ResponseFormat.VERBOSE_JSON)
    assert "segments" in body
    assert ct == "application/json"


async def test_build_response_srt() -> None:
    seg = MockSegment(text=" hello", start=0.0, end=1.0)
    audio = np.zeros(16000, dtype=np.float32)
    info = type("Info", (), {"language": "en", "duration": 1.0})()
    body, ct = WhisperAdapter.build_response([seg], info, audio, ResponseFormat.SRT)
    assert "-->" in body
    assert ct == "text/plain"


async def test_build_response_vtt() -> None:
    seg = MockSegment(text=" hello", start=0.0, end=1.0)
    audio = np.zeros(16000, dtype=np.float32)
    info = type("Info", (), {"language": "en", "duration": 1.0})()
    body, ct = WhisperAdapter.build_response([seg], info, audio, ResponseFormat.VTT)
    assert "WEBVTT" in body
    assert ct == "text/plain"


##### SUBTITLE FORMATTERS #####


async def test_format_srt_segment() -> None:
    seg = MockSegment(text=" hello world", start=1.0, end=2.5)
    result = _format_srt_segment(seg, 0)
    assert result.startswith("1\n")
    assert "-->" in result
    assert "hello world" in result


async def test_format_srt_multiple() -> None:
    segs = [MockSegment(text=" hi", start=0.0, end=1.0), MockSegment(text=" bye", start=1.0, end=2.0)]
    result = _format_srt(segs)
    assert "1\n" in result
    assert "2\n" in result


async def test_format_vtt_segment() -> None:
    seg = MockSegment(text=" hello", start=0.0, end=1.5)
    result = _format_vtt_segment(seg)
    assert "-->" in result
    assert "." in result


async def test_format_vtt_full() -> None:
    segs = [MockSegment(text=" hi", start=0.0, end=1.0)]
    result = _format_vtt(segs)
    assert result.startswith("WEBVTT")


##### DEVICE CONFIG #####


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
    """WhisperAdapter stores the device config from STTConfig."""
    config = STTConfig(device=device, compute_type=compute)
    adapter = WhisperAdapter(config=config)
    assert adapter._config.device == device
    assert adapter._config.compute_type == compute
