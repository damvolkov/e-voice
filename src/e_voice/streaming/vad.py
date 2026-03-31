"""Frame-level Voice Activity Detection using Silero VAD ONNX session.

Runs per-frame inference (~44us/frame) with persistent LSTM state,
matching the approach used by Google Cloud STT and Deepgram.
"""

from enum import StrEnum, auto

import numpy as np
from faster_whisper.vad import get_vad_model
from numpy.typing import NDArray

from e_voice.core.settings import VADConfig

_FRAME_SIZE: int = 512
_CONTEXT_SIZE: int = 64
_DEFAULT_NEG_OFFSET: float = 0.15


class SpeechState(StrEnum):
    IDLE = auto()
    SPEAKING = auto()


class SpeechStateTracker:
    """Per-session frame-level VAD with Silero ONNX.

    Processes audio in 512-sample frames (32ms @ 16kHz). Maintains LSTM hidden
    state across frames for continuous speech/silence detection.
    """

    __slots__ = (
        "_st_c",
        "_st_config",
        "_st_context",
        "_st_h",
        "_st_min_silence_frames",
        "_st_min_speech_frames",
        "_st_neg_threshold",
        "_st_pending",
        "_st_session",
        "_st_silence_frames",
        "_st_speech_frames",
        "_st_state",
    )

    def __init__(self, config: VADConfig, sample_rate: int = 16_000) -> None:
        self._st_session = get_vad_model().session
        self._st_h = np.zeros((1, 1, 128), dtype=np.float32)
        self._st_c = np.zeros((1, 1, 128), dtype=np.float32)
        self._st_context = np.zeros(_CONTEXT_SIZE, dtype=np.float32)
        self._st_pending: NDArray[np.float32] = np.array([], dtype=np.float32)
        self._st_state = SpeechState.IDLE
        self._st_speech_frames: int = 0
        self._st_silence_frames: int = 0
        self._st_config = config
        self._st_neg_threshold = (
            config.neg_threshold if config.neg_threshold is not None else config.threshold - _DEFAULT_NEG_OFFSET
        )
        frame_duration_s = _FRAME_SIZE / sample_rate
        self._st_min_speech_frames = max(1, int(config.min_speech_duration_ms / 1000 / frame_duration_s))
        self._st_min_silence_frames = max(1, int(config.min_silence_duration_ms / 1000 / frame_duration_s))

    @property
    def state(self) -> SpeechState:
        return self._st_state

    def update(self, audio: NDArray[np.float32]) -> bool:
        """Process audio samples frame-by-frame. Returns True if end-of-speech detected."""
        self._st_pending = np.concatenate([self._st_pending, audio]) if len(self._st_pending) > 0 else audio.copy()

        triggered = False
        while len(self._st_pending) >= _FRAME_SIZE:
            frame = self._st_pending[:_FRAME_SIZE]
            self._st_pending = self._st_pending[_FRAME_SIZE:]

            prob = self._st_infer_frame(frame)

            match self._st_state:
                case SpeechState.IDLE:
                    if prob >= self._st_config.threshold:
                        self._st_speech_frames += 1
                        if self._st_speech_frames >= self._st_min_speech_frames:
                            self._st_state = SpeechState.SPEAKING
                            self._st_silence_frames = 0
                    else:
                        self._st_speech_frames = 0

                case SpeechState.SPEAKING:
                    if prob < self._st_neg_threshold:
                        self._st_silence_frames += 1
                        if self._st_silence_frames >= self._st_min_silence_frames:
                            self._st_state = SpeechState.IDLE
                            self._st_speech_frames = 0
                            self._st_silence_frames = 0
                            triggered = True
                    else:
                        self._st_silence_frames = 0

        return triggered

    def _st_infer_frame(self, frame: NDArray[np.float32]) -> float:
        """Run single-frame ONNX inference. ~44us on CPU."""
        inp = np.concatenate([self._st_context, frame]).reshape(1, -1)
        output, self._st_h, self._st_c = self._st_session.run(None, {"input": inp, "h": self._st_h, "c": self._st_c})
        self._st_context = frame[-_CONTEXT_SIZE:]
        return float(output.item())
