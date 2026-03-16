"""Audio processing utilities for STT and TTS pipelines."""

import base64
from io import BytesIO

import av
import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly

WHISPER_SAMPLE_RATE = 16_000

##### CODEC MAP #####

_CODEC_MAP: dict[str, tuple[str, str, dict]] = {
    "mp3": ("mp3", "mp3", {"audio_bitrate": "128k"}),
    "wav": ("wav", "pcm_s16le", {}),
    "flac": ("flac", "flac", {}),
    "opus": ("ogg", "libopus", {"audio_bitrate": "128k"}),
    "aac": ("adts", "aac", {"audio_bitrate": "128k"}),
}


def audio_samples_from_file(file_bytes: bytes, target_sample_rate: int = WHISPER_SAMPLE_RATE) -> NDArray[np.float32]:
    """Decode audio file bytes to float32 mono at target sample rate."""
    data, sample_rate = sf.read(BytesIO(file_bytes), dtype="float32", always_2d=True)

    data = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]

    if sample_rate != target_sample_rate:
        data = resample_audio(data, sample_rate, target_sample_rate)

    return data.astype(np.float32)


def resample_audio(
    data: NDArray[np.float32],
    sample_rate: int,
    target_sample_rate: int,
) -> NDArray[np.float32]:
    """Resample audio data using polyphase filtering. O(n)."""
    gcd = np.gcd(sample_rate, target_sample_rate)
    up = target_sample_rate // gcd
    down = sample_rate // gcd
    return resample_poly(data, up, down).astype(np.float32)


def audio_duration(data: NDArray[np.float32], sample_rate: int = WHISPER_SAMPLE_RATE) -> float:
    """Duration in seconds."""
    return len(data) / sample_rate


def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT/VTT timestamp HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds to VTT timestamp HH:MM:SS.mmm."""
    return format_timestamp(seconds).replace(",", ".")


##### TTS AUDIO ENCODING #####


def float32_to_pcm16(data: NDArray[np.float32]) -> bytes:
    """Convert float32 [-1,1] samples to PCM16 little-endian bytes."""
    return (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def float32_to_base64_pcm16(data: NDArray[np.float32]) -> str:
    """Convert float32 samples to base64-encoded PCM16."""
    return base64.b64encode(float32_to_pcm16(data)).decode()


def encode_audio(data: NDArray[np.float32], sample_rate: int, fmt: str) -> bytes:
    """Encode float32 audio to target format using pyav. Returns full audio bytes."""
    if fmt == "pcm":
        return float32_to_pcm16(data)

    container_fmt, codec_name, codec_opts = _CODEC_MAP[fmt]
    buf = BytesIO()

    output = av.open(buf, mode="w", format=container_fmt)
    stream = output.add_stream(codec_name, rate=sample_rate, layout="mono")
    for k, v in codec_opts.items():
        setattr(stream, k, v) if hasattr(stream, k) else stream.options.update({k: v})

    pcm16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    frame = av.AudioFrame.from_ndarray(pcm16.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = sample_rate

    for packet in stream.encode(frame):
        output.mux(packet)
    for packet in stream.encode():
        output.mux(packet)

    output.close()
    return buf.getvalue()


def encode_audio_chunk(data: NDArray[np.float32], sample_rate: int, fmt: str) -> bytes:
    """Encode a single audio chunk for streaming. For PCM returns raw bytes, else full encode."""
    if fmt == "pcm":
        return float32_to_pcm16(data)
    return encode_audio(data, sample_rate, fmt)
