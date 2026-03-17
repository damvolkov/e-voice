"""Unit tests for core/helpers.py — audio utilities."""

import base64
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from e_voice.core.helpers import (
    WHISPER_SAMPLE_RATE,
    audio_duration,
    encode_audio,
    encode_audio_chunk,
    float32_to_base64_pcm16,
    float32_to_pcm16,
    format_timestamp,
    format_timestamp_vtt,
    resample_audio,
)

##### AUDIO_DURATION #####


@pytest.mark.parametrize(
    ("samples", "rate", "expected"),
    [
        (16000, 16000, 1.0),
        (32000, 16000, 2.0),
        (8000, 16000, 0.5),
        (0, 16000, 0.0),
        (48000, 48000, 1.0),
    ],
    ids=["1s", "2s", "half", "empty", "48khz"],
)
async def test_audio_duration(samples: int, rate: int, expected: float) -> None:
    data = np.zeros(samples, dtype=np.float32)
    assert audio_duration(data, rate) == pytest.approx(expected)


##### FORMAT_TIMESTAMP #####


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00,000"),
        (1.5, "00:00:01,500"),
        (61.5, "00:01:01,500"),
        (3661.0, "01:01:01,000"),
    ],
    ids=["zero", "1.5s", "1m1s", "1h1m1s"],
)
async def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


##### FORMAT_TIMESTAMP_VTT #####


async def test_format_timestamp_vtt_uses_dot() -> None:
    assert format_timestamp_vtt(1.5) == "00:00:01.500"
    assert "." in format_timestamp_vtt(0.0)
    assert "," not in format_timestamp_vtt(0.0)


##### FLOAT32_TO_PCM16 #####


async def test_pcm16_conversion_length() -> None:
    data = np.zeros(100, dtype=np.float32)
    result = float32_to_pcm16(data)
    assert len(result) == 200


async def test_pcm16_clipping() -> None:
    data = np.array([2.0, -2.0], dtype=np.float32)
    result = float32_to_pcm16(data)
    pcm = np.frombuffer(result, dtype=np.int16)
    assert pcm[0] == 32767
    assert pcm[1] == -32767


async def test_pcm16_silence() -> None:
    data = np.zeros(10, dtype=np.float32)
    result = float32_to_pcm16(data)
    pcm = np.frombuffer(result, dtype=np.int16)
    assert np.all(pcm == 0)


##### FLOAT32_TO_BASE64_PCM16 #####


async def test_base64_pcm16_roundtrip() -> None:
    data = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    b64 = float32_to_base64_pcm16(data)
    decoded = base64.b64decode(b64)
    assert len(decoded) == 6


##### RESAMPLE_AUDIO #####


async def test_resample_downsample() -> None:
    data = np.zeros(48000, dtype=np.float32)
    result = resample_audio(data, 48000, 16000)
    assert len(result) == pytest.approx(16000, abs=10)
    assert result.dtype == np.float32


async def test_resample_upsample() -> None:
    data = np.zeros(8000, dtype=np.float32)
    result = resample_audio(data, 8000, 16000)
    assert len(result) == pytest.approx(16000, abs=10)


##### ENCODE_AUDIO #####


async def test_encode_audio_pcm() -> None:
    data = np.zeros(1600, dtype=np.float32)
    result = encode_audio(data, WHISPER_SAMPLE_RATE, "pcm")
    assert len(result) == 3200


@pytest.mark.parametrize("fmt", ["wav", "mp3", "flac"], ids=["wav", "mp3", "flac"])
async def test_encode_audio_formats(fmt: str) -> None:
    data = np.random.randn(16000).astype(np.float32) * 0.1
    result = encode_audio(data, WHISPER_SAMPLE_RATE, fmt)
    assert len(result) > 0


##### ENCODE_AUDIO_CHUNK #####


async def test_encode_audio_chunk_pcm() -> None:
    data = np.zeros(800, dtype=np.float32)
    result = encode_audio_chunk(data, WHISPER_SAMPLE_RATE, "pcm")
    assert len(result) == 1600


##### AUDIO_SAMPLES_FROM_FILE #####


async def test_audio_samples_from_file_wav() -> None:
    """Decode a WAV file to float32 mono."""
    from e_voice.core.helpers import audio_samples_from_file

    buf = BytesIO()
    data = np.random.randn(16000).astype(np.float32) * 0.1
    sf.write(buf, data, 16000, format="WAV")
    wav_bytes = buf.getvalue()

    result = audio_samples_from_file(wav_bytes, target_sample_rate=16000)
    assert result.dtype == np.float32
    assert len(result) == pytest.approx(16000, abs=10)


async def test_audio_samples_from_file_resamples() -> None:
    """Resamples 48kHz to 16kHz."""
    from e_voice.core.helpers import audio_samples_from_file

    buf = BytesIO()
    data = np.random.randn(48000).astype(np.float32) * 0.1
    sf.write(buf, data, 48000, format="WAV")
    wav_bytes = buf.getvalue()

    result = audio_samples_from_file(wav_bytes, target_sample_rate=16000)
    assert len(result) == pytest.approx(16000, abs=100)
