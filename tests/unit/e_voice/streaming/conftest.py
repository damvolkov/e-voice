from dataclasses import dataclass

import pytest


@dataclass(slots=True)
class MockSegmentWord:
    word: str
    start: float
    end: float
    probability: float = 0.9


@dataclass(slots=True)
class MockSegment:
    id: int = 0
    seek: int = 0
    start: float = 0.0
    end: float = 1.0
    text: str = ""
    tokens: list[int] = None
    temperature: float = 0.0
    avg_logprob: float = -0.3
    compression_ratio: float = 1.2
    no_speech_prob: float = 0.1
    words: list[MockSegmentWord] = None

    def __post_init__(self) -> None:
        if self.tokens is None:
            self.tokens = []


@dataclass(slots=True)
class MockTranscriptionInfo:
    language: str = "es"
    language_probability: float = 0.9
    duration: float = 1.0
    duration_after_vad: float = 1.0
    all_language_probs: list = None
    transcription_options: dict = None
    vad_options: dict = None

    def __post_init__(self) -> None:
        if self.all_language_probs is None:
            self.all_language_probs = []


@pytest.fixture
def mock_segments_with_words() -> list[MockSegment]:
    return [
        MockSegment(
            text=" Hola mundo.",
            start=0.0,
            end=1.5,
            no_speech_prob=0.1,
            words=[
                MockSegmentWord(word=" Hola", start=0.0, end=0.5),
                MockSegmentWord(word=" mundo.", start=0.5, end=1.0),
            ],
        ),
    ]


@pytest.fixture
def mock_segments_silence() -> list[MockSegment]:
    return [
        MockSegment(
            text=" ",
            start=0.0,
            end=1.0,
            no_speech_prob=0.9,
            words=[MockSegmentWord(word=" ", start=0.0, end=1.0)],
        ),
    ]


@pytest.fixture
def mock_info() -> MockTranscriptionInfo:
    return MockTranscriptionInfo()
