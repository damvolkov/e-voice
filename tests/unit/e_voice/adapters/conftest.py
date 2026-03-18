"""Shared fixtures for adapter unit tests."""

import numpy as np
import pytest
from faster_whisper.transcribe import Segment, TranscriptionInfo


@pytest.fixture
def sample_audio() -> np.ndarray:
    """1 second of silence at 16kHz."""
    return np.zeros(16_000, dtype=np.float32)


@pytest.fixture
def segment() -> Segment:
    return Segment(
        id=0,
        seek=0,
        start=0.0,
        end=1.0,
        text=" hello",
        tokens=(1, 2, 3),
        avg_logprob=-0.3,
        compression_ratio=1.2,
        no_speech_prob=0.1,
        words=None,
        temperature=0.0,
    )


@pytest.fixture
def segment_with_words() -> Segment:
    from faster_whisper.transcribe import Word

    return Segment(
        id=0,
        seek=0,
        start=0.0,
        end=1.0,
        text=" hello world",
        tokens=(1, 2, 3),
        avg_logprob=-0.3,
        compression_ratio=1.2,
        no_speech_prob=0.1,
        temperature=0.0,
        words=[
            Word(start=0.0, end=0.5, word="hello", probability=0.9),
            Word(start=0.5, end=1.0, word="world", probability=0.85),
        ],
    )


@pytest.fixture
def info() -> TranscriptionInfo:
    return TranscriptionInfo(
        language="en",
        language_probability=0.99,
        duration=1.0,
        duration_after_vad=1.0,
        all_language_probs=None,
        transcription_options=None,
        vad_options=None,
    )


def make_segment(
    text: str = " hello",
    start: float = 0.0,
    end: float = 1.0,
    no_speech_prob: float = 0.1,
) -> Segment:
    """Factory for creating test Segments with defaults."""
    return Segment(
        id=0,
        seek=0,
        start=start,
        end=end,
        text=text,
        tokens=(1, 2, 3),
        avg_logprob=-0.3,
        compression_ratio=1.2,
        no_speech_prob=no_speech_prob,
        words=None,
        temperature=0.0,
    )
