import numpy as np
import pytest

from e_voice.core.settings import settings as st
from e_voice.streaming.audio import AudioBuffer

SAMPLE_RATE = st.stt.sample_rate


def _seconds_of_audio(seconds: float, rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(seconds * rate), dtype=np.float32)


##### BASIC PROPERTIES #####


async def test_buffer_empty_on_init() -> None:
    buf = AudioBuffer()
    assert buf.duration == 0.0
    assert buf.offset == 0.0
    assert buf.total_duration == 0.0
    assert buf.samples == 0


async def test_buffer_append_single_chunk() -> None:
    buf = AudioBuffer()
    buf.append(_seconds_of_audio(1.0))
    assert buf.duration == pytest.approx(1.0)
    assert buf.offset == 0.0
    assert buf.total_duration == pytest.approx(1.0)
    assert buf.samples == SAMPLE_RATE


async def test_buffer_append_multiple_chunks() -> None:
    buf = AudioBuffer()
    buf.append(_seconds_of_audio(1.0))
    buf.append(_seconds_of_audio(2.0))
    assert buf.duration == pytest.approx(3.0)
    assert buf.total_duration == pytest.approx(3.0)


##### TRIM BEHAVIOR #####


async def test_buffer_trims_when_exceeding_max() -> None:
    buf = AudioBuffer(max_duration_s=5.0, trim_duration_s=3.0)
    buf.append(_seconds_of_audio(6.0))
    assert buf.duration < 5.0
    assert buf.offset > 0.0
    assert buf.total_duration == pytest.approx(6.0, abs=0.1)


async def test_buffer_offset_increases_after_trim() -> None:
    buf = AudioBuffer(max_duration_s=4.0, trim_duration_s=2.0)
    buf.append(_seconds_of_audio(2.0))
    assert buf.offset == 0.0
    buf.append(_seconds_of_audio(3.0))
    assert buf.offset > 0.0


async def test_buffer_total_duration_consistent_after_trim() -> None:
    buf = AudioBuffer(max_duration_s=3.0, trim_duration_s=2.0)
    for _ in range(10):
        buf.append(_seconds_of_audio(1.0))
    assert buf.total_duration == pytest.approx(10.0, abs=0.1)
    assert buf.duration <= 3.0


##### SLICE_FROM #####


async def test_slice_from_start() -> None:
    buf = AudioBuffer()
    buf.append(_seconds_of_audio(3.0))
    sliced = buf.slice_from(0.0)
    assert len(sliced) == 3 * SAMPLE_RATE


async def test_slice_from_middle() -> None:
    buf = AudioBuffer()
    buf.append(_seconds_of_audio(4.0))
    sliced = buf.slice_from(2.0)
    assert len(sliced) == pytest.approx(2 * SAMPLE_RATE, abs=1)


async def test_slice_from_beyond_buffer() -> None:
    buf = AudioBuffer()
    buf.append(_seconds_of_audio(2.0))
    sliced = buf.slice_from(5.0)
    assert len(sliced) == 0


async def test_slice_from_after_trim() -> None:
    buf = AudioBuffer(max_duration_s=3.0, trim_duration_s=2.0)
    buf.append(_seconds_of_audio(5.0))
    offset = buf.offset
    sliced = buf.slice_from(offset)
    assert len(sliced) == buf.samples


async def test_slice_from_before_offset_returns_all() -> None:
    buf = AudioBuffer(max_duration_s=3.0, trim_duration_s=2.0)
    buf.append(_seconds_of_audio(5.0))
    sliced = buf.slice_from(0.0)
    assert len(sliced) == buf.samples


##### NEW_SAMPLES_SINCE #####


async def test_new_samples_since() -> None:
    buf = AudioBuffer()
    buf.append(_seconds_of_audio(3.0))
    assert buf.new_samples_since(1.0) == pytest.approx(2.0)
    assert buf.new_samples_since(3.0) == pytest.approx(0.0)
    assert buf.new_samples_since(0.0) == pytest.approx(3.0)


async def test_new_samples_since_future() -> None:
    buf = AudioBuffer()
    buf.append(_seconds_of_audio(1.0))
    assert buf.new_samples_since(5.0) == 0.0


##### CLEAR #####


async def test_clear() -> None:
    buf = AudioBuffer()
    buf.append(_seconds_of_audio(3.0))
    buf.clear()
    assert buf.duration == 0.0
    assert buf.offset == 0.0
    assert buf.samples == 0
