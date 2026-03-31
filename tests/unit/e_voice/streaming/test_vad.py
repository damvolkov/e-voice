"""Unit tests for streaming/vad.py — frame-level SpeechStateTracker."""

import numpy as np
import pytest

from e_voice.core.settings import VADConfig
from e_voice.streaming.vad import _CONTEXT_SIZE, _FRAME_SIZE, SpeechState, SpeechStateTracker


def _mock_session_run(prob: float):
    """Create a mock session.run that returns a fixed probability."""

    def _run(_output_names, inputs):
        return np.array([prob], dtype=np.float32), inputs["h"], inputs["c"]

    return _run


##### CONSTRUCTION #####


async def test_tracker_initial_state_idle() -> None:
    tracker = SpeechStateTracker(VADConfig())
    assert tracker.state == SpeechState.IDLE


async def test_tracker_default_neg_threshold() -> None:
    config = VADConfig(threshold=0.65)
    tracker = SpeechStateTracker(config)
    assert tracker._st_neg_threshold == pytest.approx(0.50)


async def test_tracker_explicit_neg_threshold() -> None:
    config = VADConfig(threshold=0.65, neg_threshold=0.3)
    tracker = SpeechStateTracker(config)
    assert tracker._st_neg_threshold == pytest.approx(0.3)


##### FRAME INFERENCE #####


async def test_infer_frame_returns_float() -> None:
    tracker = SpeechStateTracker(VADConfig())
    frame = np.zeros(_FRAME_SIZE, dtype=np.float32)
    prob = tracker._st_infer_frame(frame)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


async def test_infer_frame_updates_context() -> None:
    tracker = SpeechStateTracker(VADConfig())
    frame = np.ones(_FRAME_SIZE, dtype=np.float32) * 0.5
    tracker._st_infer_frame(frame)
    assert np.allclose(tracker._st_context, frame[-_CONTEXT_SIZE:])


async def test_infer_frame_updates_lstm_state() -> None:
    tracker = SpeechStateTracker(VADConfig())
    h_before = tracker._st_h.copy()
    c_before = tracker._st_c.copy()
    frame = np.ones(_FRAME_SIZE, dtype=np.float32) * 0.1
    tracker._st_infer_frame(frame)
    assert not np.allclose(h_before, tracker._st_h) or not np.allclose(c_before, tracker._st_c)


##### UPDATE — SILENCE #####


async def test_update_silence_stays_idle() -> None:
    tracker = SpeechStateTracker(VADConfig())
    silence = np.zeros(_FRAME_SIZE * 10, dtype=np.float32)
    triggered = tracker.update(silence)
    assert triggered is False
    assert tracker.state == SpeechState.IDLE


async def test_update_silence_returns_false() -> None:
    tracker = SpeechStateTracker(VADConfig())
    silence = np.zeros(1600, dtype=np.float32)
    assert tracker.update(silence) is False


##### UPDATE — PENDING BUFFER #####


async def test_update_buffers_partial_frames() -> None:
    tracker = SpeechStateTracker(VADConfig())
    partial = np.zeros(100, dtype=np.float32)
    tracker.update(partial)
    assert len(tracker._st_pending) == 100


async def test_update_processes_full_frames_from_accumulated() -> None:
    tracker = SpeechStateTracker(VADConfig())
    tracker.update(np.zeros(500, dtype=np.float32))
    assert len(tracker._st_pending) == 500
    tracker.update(np.zeros(100, dtype=np.float32))
    assert len(tracker._st_pending) == 600 - _FRAME_SIZE


async def test_update_exact_frame_no_pending() -> None:
    tracker = SpeechStateTracker(VADConfig())
    exact = np.zeros(_FRAME_SIZE, dtype=np.float32)
    tracker.update(exact)
    assert len(tracker._st_pending) == 0


##### STATE MACHINE — TRANSITIONS #####


async def test_state_machine_idle_to_speaking(mocker) -> None:
    config = VADConfig(threshold=0.5, min_speech_duration_ms=32)
    tracker = SpeechStateTracker(config, sample_rate=16_000)
    mocker.patch.object(tracker._st_session, "run", side_effect=_mock_session_run(0.8))
    audio = np.zeros(_FRAME_SIZE * 2, dtype=np.float32)
    tracker.update(audio)
    assert tracker.state == SpeechState.SPEAKING


async def test_state_machine_stays_idle_below_threshold(mocker) -> None:
    config = VADConfig(threshold=0.5, min_speech_duration_ms=32)
    tracker = SpeechStateTracker(config, sample_rate=16_000)
    mocker.patch.object(tracker._st_session, "run", side_effect=_mock_session_run(0.3))
    audio = np.zeros(_FRAME_SIZE * 5, dtype=np.float32)
    tracker.update(audio)
    assert tracker.state == SpeechState.IDLE


async def test_state_machine_speaking_to_idle_triggers(mocker) -> None:
    config = VADConfig(threshold=0.5, neg_threshold=0.3, min_speech_duration_ms=32, min_silence_duration_ms=64)
    tracker = SpeechStateTracker(config, sample_rate=16_000)

    mocker.patch.object(tracker._st_session, "run", side_effect=_mock_session_run(0.8))
    tracker.update(np.zeros(_FRAME_SIZE * 2, dtype=np.float32))
    assert tracker.state == SpeechState.SPEAKING

    mocker.patch.object(tracker._st_session, "run", side_effect=_mock_session_run(0.1))
    triggered = tracker.update(np.zeros(_FRAME_SIZE * 5, dtype=np.float32))
    assert triggered is True
    assert tracker.state == SpeechState.IDLE


async def test_state_machine_no_trigger_during_speech(mocker) -> None:
    config = VADConfig(threshold=0.5, min_speech_duration_ms=32, min_silence_duration_ms=64)
    tracker = SpeechStateTracker(config, sample_rate=16_000)

    mocker.patch.object(tracker._st_session, "run", side_effect=_mock_session_run(0.8))
    triggered = tracker.update(np.zeros(_FRAME_SIZE * 10, dtype=np.float32))
    assert triggered is False
    assert tracker.state == SpeechState.SPEAKING


async def test_state_machine_brief_silence_no_trigger(mocker) -> None:
    config = VADConfig(threshold=0.5, neg_threshold=0.3, min_speech_duration_ms=32, min_silence_duration_ms=320)
    tracker = SpeechStateTracker(config, sample_rate=16_000)

    mocker.patch.object(tracker._st_session, "run", side_effect=_mock_session_run(0.8))
    tracker.update(np.zeros(_FRAME_SIZE * 2, dtype=np.float32))
    assert tracker.state == SpeechState.SPEAKING

    mocker.patch.object(tracker._st_session, "run", side_effect=_mock_session_run(0.1))
    triggered = tracker.update(np.zeros(_FRAME_SIZE * 2, dtype=np.float32))
    assert triggered is False
    assert tracker.state == SpeechState.SPEAKING


async def test_state_machine_silence_reset_on_speech_resume(mocker) -> None:
    config = VADConfig(threshold=0.5, neg_threshold=0.3, min_speech_duration_ms=32, min_silence_duration_ms=320)
    tracker = SpeechStateTracker(config, sample_rate=16_000)

    mocker.patch.object(tracker._st_session, "run", side_effect=_mock_session_run(0.8))
    tracker.update(np.zeros(_FRAME_SIZE * 2, dtype=np.float32))
    assert tracker.state == SpeechState.SPEAKING

    mocker.patch.object(tracker._st_session, "run", side_effect=_mock_session_run(0.1))
    tracker.update(np.zeros(_FRAME_SIZE * 2, dtype=np.float32))
    assert tracker._st_silence_frames > 0

    mocker.patch.object(tracker._st_session, "run", side_effect=_mock_session_run(0.8))
    tracker.update(np.zeros(_FRAME_SIZE, dtype=np.float32))
    assert tracker._st_silence_frames == 0
    assert tracker.state == SpeechState.SPEAKING


##### MIN FRAMES CALCULATION #####


@pytest.mark.parametrize(
    ("duration_ms", "sample_rate", "expected_frames"),
    [(32, 16_000, 1), (300, 16_000, 9), (2000, 16_000, 62), (0, 16_000, 1)],
    ids=["one-frame", "300ms", "2sec", "zero-clamped"],
)
async def test_min_frames_calculation(duration_ms: int, sample_rate: int, expected_frames: int) -> None:
    config = VADConfig(min_speech_duration_ms=duration_ms, min_silence_duration_ms=duration_ms)
    tracker = SpeechStateTracker(config, sample_rate=sample_rate)
    assert tracker._st_min_speech_frames == expected_frames
    assert tracker._st_min_silence_frames == expected_frames
