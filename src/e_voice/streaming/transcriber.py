"""Streaming STT state machine — LocalAgreement + per-connection session management."""

import time
from dataclasses import dataclass
from enum import StrEnum, auto

import numpy as np
from faster_whisper.transcribe import Segment
from numpy.typing import NDArray

from e_voice.adapters.whisper import WhisperAdapter
from e_voice.core.settings import settings as st
from e_voice.models.stt import InferenceParams
from e_voice.streaming.audio import AudioBuffer
from e_voice.streaming.text import (
    StreamingWord,
    WordBuffer,
    common_prefix,
    last_full_sentence_end,
    last_full_sentence_text,
    words_to_text,
)

##### EVENTS #####


class StreamingEventType(StrEnum):
    TRANSCRIPT_UPDATE = auto()
    TRANSCRIPT_FINAL = auto()
    SESSION_END = auto()


@dataclass(slots=True, frozen=True)
class StreamingEvent:
    """Event sent back to the WebSocket client."""

    type: StreamingEventType
    confirmed_text: str
    unconfirmed_text: str
    new_confirmed: str
    is_final: bool = False


##### LOCAL AGREEMENT #####


class LocalAgreement:
    """Word-level stability: only emits words appearing in consecutive transcriptions."""

    __slots__ = ("_la_unconfirmed",)

    def __init__(self) -> None:
        self._la_unconfirmed: list[StreamingWord] = []

    def merge(
        self,
        confirmed: WordBuffer,
        incoming_words: list[StreamingWord],
    ) -> list[StreamingWord]:
        """Compare incoming against previous unconfirmed. Returns newly confirmed words.

        Algorithm:
        1. Filter incoming to words after confirmed.end - 0.1s (overlap margin)
        2. Find common_prefix between filtered incoming and self._la_unconfirmed
        3. Remainder becomes new unconfirmed
        4. Return prefix as newly confirmed
        """
        overlap_margin = 0.1
        incoming_after = (
            [w for w in incoming_words if w.start >= confirmed.end - overlap_margin] if confirmed else incoming_words
        )

        prefix = common_prefix(incoming_after, self._la_unconfirmed)

        if len(incoming_after) > len(prefix):
            self._la_unconfirmed = incoming_after[len(prefix) :]
        else:
            self._la_unconfirmed = []

        return prefix

    @property
    def unconfirmed(self) -> list[StreamingWord]:
        return self._la_unconfirmed

    @property
    def unconfirmed_text(self) -> str:
        return words_to_text(self._la_unconfirmed)

    def flush(self) -> list[StreamingWord]:
        """Return and clear all unconfirmed words (session end / silence flush)."""
        words = self._la_unconfirmed
        self._la_unconfirmed = []
        return words


##### SESSION STATE #####


class SessionState:
    """Per-connection streaming transcription state."""

    __slots__ = (
        "audio_buffer",
        "confirmed",
        "agreement",
        "language",
        "model_id",
        "response_format",
        "_ss_last_transcribe_end",
        "_ss_last_activity_ts",
        "_ss_prev_unconfirmed",
        "_ss_same_output_count",
    )

    def __init__(
        self,
        language: str | None = None,
        model_id: str | None = None,
        response_format: str = "text",
    ) -> None:
        self.audio_buffer = AudioBuffer(
            max_duration_s=st.streaming.max_buffer_seconds,
            trim_duration_s=st.streaming.trim_seconds,
        )
        self.confirmed = WordBuffer()
        self.agreement = LocalAgreement()
        self.language = language
        self.model_id = model_id or st.stt.model
        self.response_format = response_format
        self._ss_last_transcribe_end: float = 0.0
        self._ss_last_activity_ts: float = time.monotonic()
        self._ss_prev_unconfirmed: str = ""
        self._ss_same_output_count: int = 0


##### PROCESSING #####


async def process_audio_chunk(
    session: SessionState,
    whisper: WhisperAdapter,
    audio_samples: NDArray[np.float32],
) -> StreamingEvent | None:
    """Core processing: append audio, transcribe if enough accumulated, apply LocalAgreement."""
    session.audio_buffer.append(audio_samples)

    new_audio = session.audio_buffer.new_samples_since(session._ss_last_transcribe_end)
    if new_audio < st.streaming.min_duration:
        return None

    audio_start = _needs_audio_after(session.confirmed)
    audio_slice = session.audio_buffer.slice_from(audio_start)

    if len(audio_slice) < 1600:
        return None

    prompt_text = _build_prompt(session.confirmed)

    segments, info = await whisper.transcribe(
        audio_slice,
        params=InferenceParams(
            language=session.language,
            prompt=prompt_text,
            temperature=0.0,
            word_timestamps=True,
            vad_filter=False,
        ),
    )

    session._ss_last_transcribe_end = session.audio_buffer.total_duration

    incoming_words = _extract_words(segments, audio_start, st.streaming.no_speech_threshold)

    if not incoming_words:
        return None

    prev_confirmed_len = len(session.confirmed)

    newly_confirmed = session.agreement.merge(session.confirmed, incoming_words)

    event_type = StreamingEventType.TRANSCRIPT_UPDATE

    if newly_confirmed:
        session.confirmed.extend(newly_confirmed)
        session._ss_last_activity_ts = time.monotonic()

    if _check_same_output(session):
        flushed = session.agreement.flush()
        if flushed:
            session.confirmed.extend(flushed)
            event_type = StreamingEventType.TRANSCRIPT_FINAL

    new_words = session.confirmed.words[prev_confirmed_len:]
    new_confirmed_text = words_to_text(new_words) if new_words else ""
    unconfirmed_text = session.agreement.unconfirmed_text

    if not new_confirmed_text and not unconfirmed_text:
        return None

    return StreamingEvent(
        type=event_type,
        confirmed_text=session.confirmed.text,
        unconfirmed_text=unconfirmed_text,
        new_confirmed=new_confirmed_text,
    )


def flush_session(session: SessionState) -> StreamingEvent:
    """Flush remaining unconfirmed words on disconnect. Sync — no whisper call needed."""
    remaining = session.agreement.flush()
    flushed_text = words_to_text(remaining) if remaining else ""
    if remaining:
        session.confirmed.extend(remaining)

    return StreamingEvent(
        type=StreamingEventType.SESSION_END,
        confirmed_text=session.confirmed.text,
        unconfirmed_text="",
        new_confirmed=flushed_text,
        is_final=True,
    )


##### HELPERS #####


def _needs_audio_after(confirmed: WordBuffer) -> float:
    """Audio start point for re-transcription: end of last complete sentence, or 0.0."""
    if not confirmed:
        return 0.0
    return last_full_sentence_end(confirmed.words)


def _build_prompt(confirmed: WordBuffer) -> str | None:
    """Last confirmed sentence as initial_prompt for whisper continuity."""
    if not confirmed:
        return None
    return last_full_sentence_text(confirmed.words)


def _extract_words(
    segments: list[Segment],
    audio_offset: float,
    no_speech_threshold: float,
) -> list[StreamingWord]:
    """Convert faster-whisper Segments to StreamingWord list. Filters by no_speech_prob."""
    words: list[StreamingWord] = []
    for seg in segments:
        if seg.no_speech_prob > no_speech_threshold:
            continue
        if not seg.words:
            continue
        for w in seg.words:
            words.append(
                StreamingWord(
                    word=w.word.strip(),
                    start=w.start + audio_offset,
                    end=w.end + audio_offset,
                    probability=w.probability,
                )
            )
    return words


def _check_same_output(session: SessionState) -> bool:
    """Detect repeated identical unconfirmed output. Returns True if threshold exceeded."""
    current = session.agreement.unconfirmed_text
    if not current:
        session._ss_same_output_count = 0
        session._ss_prev_unconfirmed = ""
        return False

    if current == session._ss_prev_unconfirmed:
        session._ss_same_output_count += 1
    else:
        session._ss_same_output_count = 1
        session._ss_prev_unconfirmed = current

    return session._ss_same_output_count >= st.streaming.same_output_threshold
