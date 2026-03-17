"""Audio processing utilities for STT and TTS pipelines."""

import base64
from io import BytesIO
from typing import ClassVar

import av
import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly


class Audio:
    """Codec, resampling, and encoding operations for audio data."""

    WHISPER_SAMPLE_RATE: ClassVar[int] = 16_000

    _CODEC_MAP: ClassVar[dict[str, tuple[str, str, dict]]] = {
        "mp3": ("mp3", "mp3", {"audio_bitrate": "128k"}),
        "wav": ("wav", "pcm_s16le", {}),
        "flac": ("flac", "flac", {}),
        "opus": ("ogg", "libopus", {"audio_bitrate": "128k"}),
        "aac": ("adts", "aac", {"audio_bitrate": "128k"}),
    }

    ##### DECODING #####

    @staticmethod
    def samples_from_file(file_bytes: bytes, target_sample_rate: int = 16_000) -> NDArray[np.float32]:
        """Decode audio file bytes to float32 mono at target sample rate."""
        data, sample_rate = sf.read(BytesIO(file_bytes), dtype="float32", always_2d=True)
        data = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        if sample_rate != target_sample_rate:
            data = Audio.resample(data, sample_rate, target_sample_rate)
        return data.astype(np.float32)

    @staticmethod
    def pcm16_to_float32(raw: bytes, sample_rate: int = 16_000) -> NDArray[np.float32]:
        """Decode raw PCM16-LE bytes to float32 samples."""
        audio, _ = sf.read(
            BytesIO(raw),
            format="RAW",
            channels=1,
            samplerate=sample_rate,
            subtype="PCM_16",
            dtype="float32",
            endian="LITTLE",
        )
        return audio

    ##### ENCODING #####

    @staticmethod
    def float32_to_pcm16(data: NDArray[np.float32]) -> bytes:
        """Convert float32 [-1,1] samples to PCM16 little-endian bytes."""
        return (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    @staticmethod
    def float32_to_base64_pcm16(data: NDArray[np.float32]) -> str:
        """Convert float32 samples to base64-encoded PCM16."""
        return base64.b64encode(Audio.float32_to_pcm16(data)).decode()

    @staticmethod
    def encode(data: NDArray[np.float32], sample_rate: int, fmt: str) -> bytes:
        """Encode float32 audio to target format using pyav."""
        if fmt == "pcm":
            return Audio.float32_to_pcm16(data)

        container_fmt, codec_name, codec_opts = Audio._CODEC_MAP[fmt]
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

    @staticmethod
    def encode_chunk(data: NDArray[np.float32], sample_rate: int, fmt: str) -> bytes:
        """Encode a single audio chunk for streaming."""
        if fmt == "pcm":
            return Audio.float32_to_pcm16(data)
        return Audio.encode(data, sample_rate, fmt)

    ##### RESAMPLING #####

    @staticmethod
    def resample(
        data: NDArray[np.float32],
        sample_rate: int,
        target_sample_rate: int,
    ) -> NDArray[np.float32]:
        """Resample audio data using polyphase filtering. O(n)."""
        gcd = np.gcd(sample_rate, target_sample_rate)
        up = target_sample_rate // gcd
        down = sample_rate // gcd
        return resample_poly(data, up, down).astype(np.float32)

    ##### METRICS #####

    @staticmethod
    def duration(data: NDArray[np.float32], sample_rate: int = 16_000) -> float:
        """Duration in seconds."""
        return len(data) / sample_rate

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Format seconds to SRT/VTT timestamp HH:MM:SS,mmm."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def format_timestamp_vtt(seconds: float) -> str:
        """Format seconds to VTT timestamp HH:MM:SS.mmm."""
        return Audio.format_timestamp(seconds).replace(",", ".")
