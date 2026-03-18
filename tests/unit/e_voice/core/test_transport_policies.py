"""Transport policy tests — verify HTTP, SSE, streaming, and WebSocket contracts.

Mocks model output and validates that the transport layer (headers, content-type,
body format, SSE protocol, streaming chunks, WS messages) is correct.
Suffixes: _http, _sse, _stream, _ws match the taxonomical endpoint aliases.
"""

import base64
from collections.abc import AsyncGenerator

import numpy as np
import orjson
import pytest
from faster_whisper.transcribe import Segment, TranscriptionInfo
from robyn import Headers, Response, SSEMessage, SSEResponse, StreamingResponse

from e_voice.adapters.whisper import build_response, format_segment
from e_voice.core.audio import Audio
from e_voice.core.router import parse_response

##### STUBS #####

_FAKE_AUDIO_F32 = np.zeros(24_000, dtype=np.float32)


def _segment(
    text: str = " Hello world.",
    start: float = 0.0,
    end: float = 1.0,
) -> Segment:
    return Segment(
        id=0,
        seek=0,
        start=start,
        end=end,
        text=text,
        tokens=(1, 2, 3),
        avg_logprob=-0.5,
        compression_ratio=1.0,
        no_speech_prob=0.01,
        words=None,
        temperature=0.0,
    )


def _info(language: str = "en") -> TranscriptionInfo:
    return TranscriptionInfo(
        language=language,
        language_probability=0.99,
        duration=1.0,
        duration_after_vad=1.0,
        all_language_probs=None,
        transcription_options=None,
        vad_options=None,
    )


##### ROUTER — TRANSPORT TYPE DISPATCH #####


async def test_parse_response_passthrough_http() -> None:
    resp = Response(status_code=200, headers={"content-type": "application/json"}, description='{"ok":true}')
    assert parse_response(resp) is resp


async def test_parse_response_passthrough_sse() -> None:
    async def gen():
        yield "data: test\n\n"

    sr = SSEResponse(gen())
    result = parse_response(sr)
    assert result is sr
    assert isinstance(result, StreamingResponse)


async def test_parse_response_passthrough_stream() -> None:
    async def gen():
        yield b"\x00" * 4800

    headers = Headers({"Content-Type": "audio/pcm"})
    sr = StreamingResponse(content=gen(), headers=headers, media_type="audio/pcm")
    assert parse_response(sr) is sr


async def test_parse_response_does_not_stringify_sse() -> None:
    async def gen():
        yield "data: chunk\n\n"

    sr = SSEResponse(gen())
    result = parse_response(sr)
    assert isinstance(result, StreamingResponse)


@pytest.mark.parametrize(
    ("input_val", "expected_type"),
    [
        (Response(status_code=200, headers={}, description=""), Response),
        (StreamingResponse(content=iter([""]), media_type="text/event-stream"), StreamingResponse),
        (StreamingResponse(content=iter([b""]), media_type="audio/pcm"), StreamingResponse),
        ({"a": 1}, Response),
        ("text", Response),
    ],
    ids=["http", "sse", "stream", "dict-http", "str-http"],
)
async def test_parse_response_type_dispatch(input_val, expected_type) -> None:
    assert isinstance(parse_response(input_val), expected_type)


##### STT — HTTP #####


async def test_stt_http_text_format() -> None:
    body, ct = build_response([_segment()], _info(), np.zeros(16000, dtype=np.float32), "text")
    assert ct == "text/plain"
    assert body == "Hello world."


async def test_stt_http_json_format() -> None:
    body, ct = build_response([_segment()], _info(), np.zeros(16000, dtype=np.float32), "json")
    assert ct == "application/json"
    assert orjson.loads(body)["text"] == "Hello world."


async def test_stt_http_verbose_json_format() -> None:
    body, ct = build_response([_segment()], _info(), np.zeros(16000, dtype=np.float32), "verbose_json")
    assert ct == "application/json"
    parsed = orjson.loads(body)
    assert "segments" in parsed
    assert "language" in parsed
    assert "duration" in parsed


@pytest.mark.parametrize("fmt", ["srt", "vtt"], ids=["srt", "vtt"])
async def test_stt_http_subtitle_formats(fmt: str) -> None:
    body, ct = build_response(
        [_segment(text=" Hello.", start=0.0, end=1.0)],
        _info(),
        np.zeros(16000, dtype=np.float32),
        fmt,
    )
    assert ct == "text/plain"
    assert "-->" in body


##### STT — SSE #####


async def test_stt_sse_content_type() -> None:
    sr = SSEResponse(iter([""]))
    assert sr.media_type == "text/event-stream"
    assert sr.headers.get("Content-Type") == "text/event-stream"


async def test_stt_sse_message_format() -> None:
    msg = SSEMessage(data="test payload")
    assert "data:" in msg
    assert "test payload" in msg
    assert msg.endswith("\n\n")


async def test_stt_sse_message_with_event_type() -> None:
    msg = SSEMessage(data="payload", event="custom")
    assert "event:" in msg and "data:" in msg


async def test_stt_sse_segment_message() -> None:
    msg = SSEMessage(data=format_segment(_segment(), "text"))
    assert "data:" in msg
    assert "Hello world." in msg


async def test_stt_sse_done_marker() -> None:
    msg = SSEMessage(data="[DONE]")
    assert "[DONE]" in msg


async def test_stt_sse_status_code() -> None:
    sr = SSEResponse(iter(["data: x\n\n"]))
    assert sr.status_code == 200


##### STT — WS #####


async def test_stt_ws_event_json_format() -> None:
    from e_voice.streaming.transcriber import StreamingEvent, StreamingEventType
    from e_voice.websockets.stt import format_event

    event = StreamingEvent(
        type=StreamingEventType.TRANSCRIPT_UPDATE,
        confirmed_text="Hello",
        unconfirmed_text="world",
        new_confirmed="Hello",
    )
    parsed = orjson.loads(format_event(event, "json"))
    assert parsed["type"] == "transcript_update"
    assert parsed["text"] == "Hello"
    assert parsed["partial"] == "world"
    assert parsed["is_final"] is False


async def test_stt_ws_event_text_format() -> None:
    from e_voice.streaming.transcriber import StreamingEvent, StreamingEventType
    from e_voice.websockets.stt import format_event

    event = StreamingEvent(
        type=StreamingEventType.TRANSCRIPT_UPDATE,
        confirmed_text="Hello world",
        unconfirmed_text="",
        new_confirmed="world",
    )
    assert format_event(event, "text") == "Hello world"


##### TTS — HTTP #####


async def test_tts_http_encodes_wav() -> None:
    audio_bytes = Audio.encode(_FAKE_AUDIO_F32, 24000, "wav")
    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 44
    assert audio_bytes[:4] == b"RIFF"


##### TTS — SSE #####


async def test_tts_sse_delta_event() -> None:
    from e_voice.core.audio import Audio
    from e_voice.models.tts import SpeechAudioDeltaEvent

    b64 = Audio.float32_to_base64_pcm16(_FAKE_AUDIO_F32[:4800])
    event = SpeechAudioDeltaEvent(audio=b64)
    parsed = orjson.loads(event.model_dump_json())

    assert parsed["type"] == "speech.audio.delta"
    decoded = base64.b64decode(parsed["audio"])
    assert len(decoded) == 4800 * 2
    assert len(decoded) % 2 == 0


async def test_tts_sse_done_event() -> None:
    from e_voice.models.tts import SpeechAudioDoneEvent

    parsed = orjson.loads(SpeechAudioDoneEvent().model_dump_json())
    assert parsed["type"] == "speech.audio.done"
    assert len(parsed) == 1


async def test_tts_sse_full_sequence() -> None:
    from e_voice.core.audio import Audio
    from e_voice.models.tts import SpeechAudioDeltaEvent, SpeechAudioDoneEvent

    events = [
        SpeechAudioDeltaEvent(audio=Audio.float32_to_base64_pcm16(_FAKE_AUDIO_F32[:4800])).model_dump_json(),
        SpeechAudioDeltaEvent(audio=Audio.float32_to_base64_pcm16(_FAKE_AUDIO_F32[:4800])).model_dump_json(),
        SpeechAudioDoneEvent().model_dump_json(),
    ]
    assert orjson.loads(events[0])["type"] == "speech.audio.delta"
    assert orjson.loads(events[1])["type"] == "speech.audio.delta"
    assert orjson.loads(events[2])["type"] == "speech.audio.done"


##### TTS — STREAM (CHUNKED) #####


async def test_tts_stream_preserves_content_type() -> None:
    async def gen() -> AsyncGenerator[bytes]:
        yield b"\x00" * 4800

    headers = Headers({"Content-Type": "audio/pcm"})
    sr = StreamingResponse(content=gen(), headers=headers, media_type="audio/pcm")
    result = parse_response(sr)
    assert isinstance(result, StreamingResponse)
    assert result.headers.get("Content-Type") == "audio/pcm"


@pytest.mark.parametrize(
    ("media_type", "expected_ct"),
    [
        ("audio/pcm", "audio/pcm"),
        ("audio/mpeg", "audio/mpeg"),
        ("audio/wav", "audio/wav"),
        ("audio/flac", "audio/flac"),
        ("audio/ogg", "audio/ogg"),
        ("audio/aac", "audio/aac"),
    ],
    ids=["pcm", "mp3", "wav", "flac", "opus", "aac"],
)
async def test_tts_stream_audio_formats(media_type: str, expected_ct: str) -> None:
    async def gen():
        yield b"\x00" * 100

    headers = Headers({"Content-Type": media_type})
    sr = StreamingResponse(content=gen(), headers=headers, media_type=media_type)
    assert sr.headers.get("Content-Type") == expected_ct


async def test_tts_stream_headers_are_robyn_headers() -> None:
    async def gen():
        yield b""

    headers = Headers({"Content-Type": "audio/pcm"})
    sr = StreamingResponse(content=gen(), headers=headers, media_type="audio/pcm")
    assert hasattr(sr.headers, "set")


##### TTS — WS #####


async def test_tts_ws_delta_message() -> None:
    audio_b64 = Audio.float32_to_base64_pcm16(_FAKE_AUDIO_F32[:4800])
    parsed = orjson.loads(orjson.dumps({"type": "speech.audio.delta", "audio": audio_b64}))
    assert parsed["type"] == "speech.audio.delta"
    assert len(base64.b64decode(parsed["audio"])) == 4800 * 2


async def test_tts_ws_done_message() -> None:
    parsed = orjson.loads(orjson.dumps({"type": "speech.audio.done"}))
    assert parsed == {"type": "speech.audio.done"}


##### PCM16 ENCODING #####


async def test_pcm16_base64_roundtrip() -> None:
    original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    decoded = base64.b64decode(Audio.float32_to_base64_pcm16(original))
    assert len(decoded) == len(original) * 2

    pcm16 = np.frombuffer(decoded, dtype=np.int16)
    assert pcm16[0] == 0
    assert pcm16[3] == 32767
    assert pcm16[4] == -32767


async def test_pcm16_little_endian() -> None:
    raw = base64.b64decode(Audio.float32_to_base64_pcm16(np.array([0.5], dtype=np.float32)))
    assert int.from_bytes(raw, byteorder="little", signed=True) == int(0.5 * 32767)
