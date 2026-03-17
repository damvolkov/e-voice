"""OpenAI-compatible Text-to-Speech API — speech synthesis, voice listing."""

from robyn import Headers, Request, Response, SSEMessage, SSEResponse, StreamingResponse, status_codes

from e_voice.adapters.kokoro import _VOICE_LANG_MAP, KokoroAdapter
from e_voice.core.audio import Audio
from e_voice.core.logger import logger
from e_voice.core.router import Router
from e_voice.models.tts import (
    ListVoicesResponse,
    SpeechAudioDeltaEvent,
    SpeechAudioDoneEvent,
    SpeechRequest,
    StreamFormat,
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
    """Synthesize speech from text (OpenAI-compatible). Supports streaming."""
    kokoro: KokoroAdapter = global_dependencies.get("state").kokoro
    fmt = body.response_format.value
    content_type = _CONTENT_TYPE_MAP.get(fmt, "application/octet-stream")

    logger.info("speech request", step="TTS", voice=body.voice, stream=body.stream, format=fmt)

    if body.stream and body.stream_format == StreamFormat.SSE:

        async def sse_generator():
            async for samples, _sr in kokoro.synthesize_stream(
                body.input,
                voice=body.voice,
                speed=body.speed,
                lang=body.lang,
            ):
                audio_b64 = Audio.float32_to_base64_pcm16(samples)
                yield SSEMessage(data=SpeechAudioDeltaEvent(audio=audio_b64).model_dump_json())
            yield SSEMessage(data=SpeechAudioDoneEvent().model_dump_json())

        return SSEResponse(sse_generator())

    if body.stream:

        async def audio_generator():
            async for samples, sr in kokoro.synthesize_stream(
                body.input,
                voice=body.voice,
                speed=body.speed,
                lang=body.lang,
            ):
                yield Audio.encode_chunk(samples, sr, fmt)

        headers = Headers({"Content-Type": content_type})
        return StreamingResponse(
            audio_generator(),
            status_code=status_codes.HTTP_200_OK,
            headers=headers,
            media_type=content_type,
        )

    samples, sr = await kokoro.synthesize(
        body.input,
        voice=body.voice,
        speed=body.speed,
        lang=body.lang,
    )
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
    voice_ids = kokoro.get_voices()

    voices = [
        VoiceObject(
            id=vid,
            name=vid,
            language=_VOICE_LANG_MAP.get(vid[0], "en-us") if vid else "en-us",
        )
        for vid in sorted(voice_ids)
    ]

    return ListVoicesResponse(voices=voices)
