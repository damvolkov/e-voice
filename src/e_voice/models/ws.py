"""Pydantic models for WebSocket query parameters and per-message request bodies."""

from typing import Annotated, Self

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, field_validator, model_validator

from e_voice.core.settings import ResponseFormatType
from e_voice.core.settings import settings as st
from e_voice.core.websocket import BaseWSParams
from e_voice.models.tts import resolve_voice_lang

##### FIELD TYPES #####

type NormalizedLanguage = Annotated[
    str | None,
    BeforeValidator(lambda v: st.stt.default_language if not v or v == "auto" else v),
]

type NormalizedResponseFormat = Annotated[
    str | ResponseFormatType,
    BeforeValidator(lambda v: v if v else st.stt.default_response_format.value),
    AfterValidator(lambda v: ResponseFormatType(v)),
]

type NormalizedModel = Annotated[
    str,
    BeforeValidator(lambda v: v if v else st.stt.model),
]

type CoercedBool = Annotated[
    bool,
    BeforeValidator(lambda v: v.lower() == "true" if isinstance(v, str) else bool(v or False)),
]

type NormalizedVoice = Annotated[
    str,
    BeforeValidator(lambda v: (resolve_voice_lang(v), v)[1] if v else st.tts.default_voice),
]

type NormalizedLang = Annotated[
    str,
    BeforeValidator(lambda v: v or ""),
]

##### STT PARAMS #####


class STTParams(BaseWSParams):
    """Validated STT WebSocket query parameters."""

    language: NormalizedLanguage = st.stt.default_language
    response_format: NormalizedResponseFormat = st.stt.default_response_format
    model: NormalizedModel = st.stt.model
    segmentation: CoercedBool = False


##### TTS PARAMS #####


class TTSParams(BaseWSParams):
    """TTS WebSocket query parameters — body carries the request."""


##### TTS PER-MESSAGE REQUEST #####


class WSSpeechRequest(BaseModel):
    """Per-message TTS request body over WebSocket."""

    model_config = ConfigDict(frozen=True)

    input: str
    voice: NormalizedVoice = st.tts.default_voice
    speed: float = st.tts.default_speed
    lang: NormalizedLang = ""

    @field_validator("input")
    @classmethod
    def _validate_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Input text must not be empty")
        return v

    @model_validator(mode="after")
    def _resolve_lang(self) -> Self:
        """Infer lang from voice prefix; reject explicit lang that conflicts."""
        voice_lang = resolve_voice_lang(self.voice)
        if not self.lang:
            object.__setattr__(self, "lang", voice_lang)
        elif self.lang != voice_lang:
            raise ValueError(f"Language '{self.lang}' conflicts with voice '{self.voice}' (expected '{voice_lang}')")
        return self
