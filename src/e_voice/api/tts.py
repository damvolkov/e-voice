"""OpenAI-compatible Text-to-Speech API — speech synthesis, voice listing."""

from robyn import Request, Response, SSEMessage, SSEResponse, status_codes

from e_voice.adapters.kokoro import KokoroAdapter
from e_voice.core.audio import Audio
from e_voice.core.logger import logger
from e_voice.core.router import Router
from e_voice.models.tts import (
    ListVoicesResponse,
    SpeechAudioDeltaEvent,
    SpeechAudioDoneEvent,
    SpeechRequest,
    StreamFormat,
    SynthesisParams,
    VoiceLang,
    VoiceObject,
)

router = Router(__file__, prefix="/v1")

_CONTENT_TYPE_MAP: dict[str, str] = {
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/ogg",
    "aac": "audio/aac",
}


##### POST /v1/audio/speech #####


@router.post("/audio/speech")
async def speech(request: Request, body: SpeechRequest, global_dependencies):
    """Synthesize speech from text (OpenAI-compatible). Supports batch, SSE, and WebSocket."""
    kokoro: KokoroAdapter = global_dependencies.get("state").kokoro
    fmt = body.response_format.value
    content_type = _CONTENT_TYPE_MAP.get(fmt, "application/octet-stream")
    params = SynthesisParams(voice=body.voice, speed=body.speed, lang=body.lang)

    logger.info("speech request", step="TTS", voice=body.voice, stream=body.stream, format=fmt)

    if body.stream and body.stream_format == StreamFormat.SSE:

        async def sse_generator():
            async for samples, _sr in kokoro.synthesize_stream(body.input, params=params):
                audio_b64 = Audio.float32_to_base64_pcm16(samples)
                yield SSEMessage(data=SpeechAudioDeltaEvent(audio=audio_b64).model_dump_json())
            yield SSEMessage(data=SpeechAudioDoneEvent().model_dump_json())

        return SSEResponse(sse_generator())

    # Robyn 0.72 StreamingResponse only accepts str — Rust layer re-encodes as UTF-8,
    # corrupting binary audio (bytes ≥0x80 expand to 2-byte UTF-8 sequences).
    # All HTTP binary audio goes through batch synthesis until Robyn adds bytes support.
    # True incremental streaming available via SSE (base64) and WebSocket (binary frames).
    samples, sr = await kokoro.synthesize(body.input, params=params)
    audio_bytes = Audio.encode(samples, sr, fmt)

    logger.info("speech complete", step="TTS", size=len(audio_bytes))
    return Response(
        status_code=status_codes.HTTP_200_OK,
        headers={"content-type": content_type},
        description=audio_bytes,
    )


##### TAXONOMICAL ALIASES #####

router.alias("/audio/speech", "/tts/http", "/tts/sse", "/tts/stream")

##### GET /v1/audio/voices #####


@router.get("/audio/voices")
async def list_voices(global_dependencies):
    """List available TTS voices."""
    kokoro: KokoroAdapter = global_dependencies.get("state").kokoro

    voices = [
        VoiceObject(
            id=vid,
            name=vid,
            language=VoiceLang[vid[0].upper()] if vid and vid[0].upper() in VoiceLang.__members__ else "en-us",
        )
        for vid in sorted(kokoro.voices)
    ]

    return ListVoicesResponse(voices=voices)
