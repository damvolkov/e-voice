import base64
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from e_voice.core.audio import Audio
from e_voice.core.settings import settings as st

##### DURATION #####


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
    assert Audio.duration(data, rate) == pytest.approx(expected)


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
async def test_audio_format_timestamp(seconds: float, expected: str) -> None:
    assert Audio.format_timestamp(seconds) == expected


##### FORMAT_TIMESTAMP_VTT #####


async def test_audio_format_timestamp_vtt_uses_dot() -> None:
    assert Audio.format_timestamp_vtt(1.5) == "00:00:01.500"
    assert "." in Audio.format_timestamp_vtt(0.0)
    assert "," not in Audio.format_timestamp_vtt(0.0)


##### FLOAT32_TO_PCM16 #####


async def test_audio_float32_to_pcm16_length() -> None:
    data = np.zeros(100, dtype=np.float32)
    result = Audio.float32_to_pcm16(data)
    assert len(result) == 200


async def test_audio_float32_to_pcm16_clipping() -> None:
    data = np.array([2.0, -2.0], dtype=np.float32)
    result = Audio.float32_to_pcm16(data)
    pcm = np.frombuffer(result, dtype=np.int16)
    assert pcm[0] == 32767
    assert pcm[1] == -32767


async def test_audio_float32_to_pcm16_silence() -> None:
    data = np.zeros(10, dtype=np.float32)
    result = Audio.float32_to_pcm16(data)
    pcm = np.frombuffer(result, dtype=np.int16)
    assert np.all(pcm == 0)


##### FLOAT32_TO_BASE64_PCM16 #####


async def test_audio_float32_to_base64_pcm16_roundtrip() -> None:
    data = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    b64 = Audio.float32_to_base64_pcm16(data)
    decoded = base64.b64decode(b64)
    assert len(decoded) == 6


##### RESAMPLE #####


async def test_audio_resample_downsample() -> None:
    data = np.zeros(48000, dtype=np.float32)
    result = Audio.resample(data, 48000, 16000)
    assert len(result) == pytest.approx(16000, abs=10)
    assert result.dtype == np.float32


async def test_audio_resample_upsample() -> None:
    data = np.zeros(8000, dtype=np.float32)
    result = Audio.resample(data, 8000, 16000)
    assert len(result) == pytest.approx(16000, abs=10)


##### ENCODE #####


async def test_audio_encode_pcm() -> None:
    data = np.zeros(1600, dtype=np.float32)
    result = Audio.encode(data, st.stt.sample_rate, "pcm")
    assert len(result) == 3200


@pytest.mark.parametrize("fmt", ["wav", "mp3", "flac", "opus", "aac"], ids=["wav", "mp3", "flac", "opus", "aac"])
async def test_audio_encode_formats(fmt: str) -> None:
    data = np.random.randn(16000).astype(np.float32) * 0.1
    result = Audio.encode(data, st.stt.sample_rate, fmt)
    assert len(result) > 0


##### ENCODE_CHUNK #####


async def test_audio_encode_chunk_pcm() -> None:
    data = np.zeros(800, dtype=np.float32)
    result = Audio.encode_chunk(data, st.stt.sample_rate, "pcm")
    assert len(result) == 1600


##### SAMPLES_FROM_FILE #####


async def test_audio_samples_from_file_wav() -> None:
    buf = BytesIO()
    data = np.random.randn(16000).astype(np.float32) * 0.1
    sf.write(buf, data, 16000, format="WAV")

    result = Audio.samples_from_file(buf.getvalue(), target_sample_rate=16000)
    assert result.dtype == np.float32
    assert len(result) == pytest.approx(16000, abs=10)


async def test_audio_samples_from_file_resamples() -> None:
    buf = BytesIO()
    data = np.random.randn(48000).astype(np.float32) * 0.1
    sf.write(buf, data, 48000, format="WAV")

    result = Audio.samples_from_file(buf.getvalue(), target_sample_rate=16000)
    assert len(result) == pytest.approx(16000, abs=100)


##### PCM16_TO_FLOAT32 #####


async def test_audio_pcm16_to_float32_length() -> None:
    pcm_bytes = np.zeros(160, dtype=np.int16).tobytes()
    result = Audio.pcm16_to_float32(pcm_bytes)
    assert len(result) == 160
    assert result.dtype == np.float32


async def test_audio_pcm16_to_float32_silence() -> None:
    pcm_bytes = np.zeros(100, dtype=np.int16).tobytes()
    result = Audio.pcm16_to_float32(pcm_bytes)
    assert np.allclose(result, 0.0)


async def test_audio_pcm16_to_float32_roundtrip() -> None:
    original = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    pcm_bytes = Audio.float32_to_pcm16(original)
    recovered = Audio.pcm16_to_float32(pcm_bytes)
    assert len(recovered) == len(original)
    assert np.allclose(recovered, original, atol=1e-4)


async def test_audio_pcm16_to_float32_custom_sample_rate() -> None:
    pcm_bytes = np.zeros(480, dtype=np.int16).tobytes()
    result = Audio.pcm16_to_float32(pcm_bytes, sample_rate=48000)
    assert len(result) == 480


##### STT SAMPLE RATE #####


async def test_stt_sample_rate_from_settings() -> None:
    assert st.stt.sample_rate == 16_000
