"""OpenAI SDK compatibility tests — STT endpoints.

Validates that e-voice is a drop-in replacement for OpenAI's Audio API
using the official openai Python SDK.
"""

from openai import AsyncOpenAI
from pytest_audioeval.metrics.text import TextMetrics

##### POST /v1/audio/transcriptions (SDK) #####


async def test_transcribe_json(openai_client: AsyncOpenAI, en_sample) -> None:
    """SDK transcription returns text in JSON format."""
    result = await openai_client.audio.transcriptions.create(
        model="whisper",
        file=("audio.wav", en_sample.audio_bytes(), "audio/wav"),
        response_format="json",
    )
    assert hasattr(result, "text")
    assert len(result.text.strip()) > 0


async def test_transcribe_text(openai_client: AsyncOpenAI, en_sample) -> None:
    """SDK transcription with text response_format."""
    result = await openai_client.audio.transcriptions.create(
        model="whisper",
        file=("audio.wav", en_sample.audio_bytes(), "audio/wav"),
        response_format="text",
    )
    assert len(result.strip()) > 0


async def test_transcribe_verbose_json(openai_client: AsyncOpenAI, en_sample) -> None:
    """SDK transcription with verbose_json response_format."""
    result = await openai_client.audio.transcriptions.create(
        model="whisper",
        file=("audio.wav", en_sample.audio_bytes(), "audio/wav"),
        response_format="verbose_json",
    )
    assert hasattr(result, "text")
    assert hasattr(result, "segments")
    assert hasattr(result, "language")
    assert hasattr(result, "duration")


async def test_transcribe_srt(openai_client: AsyncOpenAI, en_sample) -> None:
    """SDK transcription with SRT subtitle format."""
    result = await openai_client.audio.transcriptions.create(
        model="whisper",
        file=("audio.wav", en_sample.audio_bytes(), "audio/wav"),
        response_format="srt",
    )
    assert "-->" in result


async def test_transcribe_vtt(openai_client: AsyncOpenAI, en_sample) -> None:
    """SDK transcription with VTT subtitle format."""
    result = await openai_client.audio.transcriptions.create(
        model="whisper",
        file=("audio.wav", en_sample.audio_bytes(), "audio/wav"),
        response_format="vtt",
    )
    assert "WEBVTT" in result
    assert "-->" in result


async def test_transcribe_quality(
    openai_client: AsyncOpenAI,
    en_sample,
    audioeval_thresholds: dict[str, float],
) -> None:
    """SDK transcription meets WER/CER quality thresholds."""
    result = await openai_client.audio.transcriptions.create(
        model="whisper",
        file=("audio.wav", en_sample.audio_bytes(), "audio/wav"),
        response_format="json",
    )
    metrics = TextMetrics.compute(en_sample.reference_text, result.text)
    metrics.assert_quality(
        max_wer=audioeval_thresholds["max_wer"],
        max_cer=audioeval_thresholds["max_cer"],
    )


async def test_transcribe_with_language(openai_client: AsyncOpenAI, es_sample) -> None:
    """SDK transcription with explicit language parameter."""
    result = await openai_client.audio.transcriptions.create(
        model="whisper",
        file=("audio.wav", es_sample.audio_bytes(), "audio/wav"),
        response_format="json",
        language="es",
    )
    assert len(result.text.strip()) > 0


async def test_transcribe_with_prompt(openai_client: AsyncOpenAI, en_sample) -> None:
    """SDK transcription with prompt hint."""
    result = await openai_client.audio.transcriptions.create(
        model="whisper",
        file=("audio.wav", en_sample.audio_bytes(), "audio/wav"),
        response_format="json",
        prompt="greeting",
    )
    assert len(result.text.strip()) > 0


##### POST /v1/audio/translations (SDK) #####


async def test_translate_es_to_en(openai_client: AsyncOpenAI, es_sample) -> None:
    """SDK translation produces English text from Spanish audio."""
    result = await openai_client.audio.translations.create(
        model="whisper",
        file=("audio.wav", es_sample.audio_bytes(), "audio/wav"),
        response_format="json",
    )
    assert hasattr(result, "text")
    assert len(result.text.strip()) > 0


async def test_translate_text_format(openai_client: AsyncOpenAI, es_sample) -> None:
    """SDK translation with text response format."""
    result = await openai_client.audio.translations.create(
        model="whisper",
        file=("audio.wav", es_sample.audio_bytes(), "audio/wav"),
        response_format="text",
    )
    assert len(result.strip()) > 0
