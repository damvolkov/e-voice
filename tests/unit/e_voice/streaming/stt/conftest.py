"""Shared fixtures for streaming STT unit tests — uses domain types."""

import pytest

from e_voice.models.stt import Span, Transcript, Word


@pytest.fixture
def mock_segments_with_words() -> tuple[Span, ...]:
    return (
        Span(
            text=" Hola mundo.",
            start=0.0,
            end=1.5,
            no_speech_prob=0.1,
            words=(
                Word(text="Hola", start=0.0, end=0.5, probability=0.9),
                Word(text="mundo.", start=0.5, end=1.0, probability=0.9),
            ),
        ),
    )


@pytest.fixture
def mock_segments_silence() -> tuple[Span, ...]:
    return (
        Span(
            text=" ",
            start=0.0,
            end=1.0,
            no_speech_prob=0.9,
            words=(Word(text=" ", start=0.0, end=1.0, probability=0.9),),
        ),
    )


@pytest.fixture
def mock_transcript(mock_segments_with_words: tuple[Span, ...]) -> Transcript:
    return Transcript(spans=mock_segments_with_words, language="es", duration=1.5)


@pytest.fixture
def mock_transcript_silence(mock_segments_silence: tuple[Span, ...]) -> Transcript:
    return Transcript(spans=mock_segments_silence, language="es", duration=1.0)
