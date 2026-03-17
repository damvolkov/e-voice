"""Word-level text utilities for streaming STT — LocalAgreement primitives."""

import re
from dataclasses import dataclass, field

_SENTENCE_ENDINGS: frozenset[str] = frozenset(".?!")
_CANONICALIZE_RE: re.Pattern[str] = re.compile(r"[^a-z\d]")


@dataclass(slots=True)
class StreamingWord:
    """A single word with timing and probability from whisper."""

    word: str
    start: float
    end: float
    probability: float = 0.0

    def offset(self, seconds: float) -> None:
        """Shift timestamps by absolute offset (for buffer-relative → global conversion)."""
        self.start += seconds
        self.end += seconds


@dataclass(slots=True)
class WordBuffer:
    """Accumulated confirmed words with text/timing access."""

    words: list[StreamingWord] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words).strip()

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0

    def after(self, seconds: float) -> list[StreamingWord]:
        """Words starting at or after `seconds`."""
        return [w for w in self.words if w.start >= seconds - 0.1]

    def extend(self, new_words: list[StreamingWord]) -> None:
        self.words.extend(new_words)

    def __len__(self) -> int:
        return len(self.words)

    def __bool__(self) -> bool:
        return bool(self.words)


def canonicalize_word(text: str) -> str:
    """Lowercase, strip non-alphanumeric for comparison. O(n)."""
    return _CANONICALIZE_RE.sub("", text.lower())


def common_prefix(a: list[StreamingWord], b: list[StreamingWord]) -> list[StreamingWord]:
    """Longest common prefix by canonicalized word comparison. Returns words from `a`. O(min(|a|,|b|))."""
    prefix: list[StreamingWord] = []
    for wa, wb in zip(a, b, strict=False):
        if canonicalize_word(wa.word) != canonicalize_word(wb.word):
            break
        prefix.append(wa)
    return prefix


def is_eos(text: str) -> bool:
    """Check if text ends with sentence-ending punctuation (not ellipsis)."""
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped.endswith("..."):
        return False
    return stripped[-1] in _SENTENCE_ENDINGS


def to_full_sentences(words: list[StreamingWord]) -> list[list[StreamingWord]]:
    """Split words into complete sentences (ending with .?!). Incomplete trailing sentence excluded."""
    sentences: list[list[StreamingWord]] = []
    current: list[StreamingWord] = []
    for w in words:
        current.append(w)
        if is_eos(w.word):
            sentences.append(current)
            current = []
    return sentences


def last_full_sentence_end(words: list[StreamingWord]) -> float:
    """End timestamp of the last complete sentence. Returns 0.0 if no complete sentence."""
    if sentences := to_full_sentences(words):
        return sentences[-1][-1].end
    return 0.0


def last_full_sentence_text(words: list[StreamingWord]) -> str | None:
    """Text of the last complete sentence. Returns None if no complete sentence."""
    if sentences := to_full_sentences(words):
        return " ".join(w.word for w in sentences[-1]).strip()
    return None


def words_to_text(words: list[StreamingWord]) -> str:
    """Join words into text."""
    return " ".join(w.word for w in words).strip()
