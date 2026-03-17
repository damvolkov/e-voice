"""Bounded circular audio buffer for streaming STT."""

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16_000


class AudioBuffer:
    """Bounded audio buffer. Max duration configurable (default 45s).

    When buffer exceeds max, discards oldest audio (default 30s) and updates offset.
    All timestamps are absolute (offset-aware).
    """

    __slots__ = ("_data", "_offset", "_max_samples", "_trim_samples", "_sample_rate")

    def __init__(
        self,
        max_duration_s: float = 45.0,
        trim_duration_s: float = 30.0,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._data: NDArray[np.float32] = np.array([], dtype=np.float32)
        self._offset: float = 0.0
        self._max_samples = int(max_duration_s * sample_rate)
        self._trim_samples = int(trim_duration_s * sample_rate)
        self._sample_rate = sample_rate

    @property
    def duration(self) -> float:
        """Current buffer duration in seconds."""
        return len(self._data) / self._sample_rate

    @property
    def offset(self) -> float:
        """Seconds of audio discarded from the front (global timeline base)."""
        return self._offset

    @property
    def total_duration(self) -> float:
        """Total audio received (offset + current buffer duration)."""
        return self._offset + self.duration

    @property
    def samples(self) -> int:
        return len(self._data)

    def append(self, samples: NDArray[np.float32]) -> None:
        """Append samples. Trims oldest audio if buffer exceeds max."""
        self._data = np.concatenate((self._data, samples)) if len(self._data) > 0 else samples.copy()
        if len(self._data) > self._max_samples:
            self._ab_trim()

    def slice_from(self, timestamp_s: float) -> NDArray[np.float32]:
        """Return audio from absolute timestamp to end of buffer.

        If timestamp precedes buffer start (already trimmed), returns all available audio.
        """
        relative_s = max(0.0, timestamp_s - self._offset)
        start_idx = min(int(relative_s * self._sample_rate), len(self._data))
        return self._data[start_idx:]

    def new_samples_since(self, timestamp_s: float) -> float:
        """Seconds of new audio since absolute timestamp."""
        return max(0.0, self.total_duration - timestamp_s)

    def clear(self) -> None:
        self._data = np.array([], dtype=np.float32)
        self._offset = 0.0

    def _ab_trim(self) -> None:
        """Discard oldest audio to stay within bounds."""
        trim = min(self._trim_samples, len(self._data) - (self._max_samples - self._trim_samples))
        if trim > 0:
            self._offset += trim / self._sample_rate
            self._data = self._data[trim:]
