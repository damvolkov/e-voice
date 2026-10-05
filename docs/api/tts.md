# Text to speech

The TTS service (`e-voice-tts serve`, port 5600) streams audio while it is generated: with Pocket the
first 80 ms chunk leaves about 70–120 ms after the text arrives on a warm stream (Qwen3: ≈ 0.4 s). Every endpoint takes a learned
`voice` (see [Voices](../guide/voices.md)); without one, the configured default `tts.voice` speaks.

## Native stream — `GET /v1/stream`

WebSocket. Query: `lang` (`es`, `en`), `voice`, `format` (`pcm` s16le, default, or `f32`). The voice is
primed before the upgrade, so a full server (503) or an unreadable voice fails as plain HTTP.

| Direction | Frame | Meaning |
|---|---|---|
| ← | `{"type":"ready","rate":24000,"format":"pcm","lang":"es","voice":"damien"}` | stream open |
| → | `{"type":"text","text":"Hola, "}` | any amount of text, e.g. one LLM token |
| → | `{"type":"flush"}` | speak what is buffered even without a sentence end |
| → | `{"type":"cancel"}` | barge-in: stop the sentence now, drop what is queued |
| → | `{"type":"close"}` | speak the rest, then close |
| ← | `{"type":"start","sentence":0,"text":"Hola, soy Demian."}` | a sentence begins |
| ← | binary | audio, one frame per 80 ms of speech |
| ← | `{"type":"end","sentence":0,"error":null}` | the sentence ended (`error.kind = "cancelled"` on barge-in) |
| ← | `{"type":"closed"}` | last frame |

Text is cut into sentences server-side (Unicode sentence bounds; shorter than `tts.text.min` chars merge,
longer than `tts.text.max` are cut at `, ; :` then spaces), so a client can forward LLM tokens as they
come.

```bash
ecli tts "Hola, ¿qué tal?" --voice damien          # speaks through the default output device
ecli tts --voice damien                            # every typed line is spoken; an empty line barges in
ecli tts "…" --mute --out speech.wav               # record only
```

## OpenAI — `POST /v1/audio/speech`

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5600/v1", api_key="unused")
with client.audio.speech.with_streaming_response.create(
    model="pocket", voice="damien", input="Hola, soy Demian.", response_format="wav"
) as response:
    response.stream_to_file("speech.wav")
```

- `response_format`: `wav` (default; streaming header), `pcm` (24 kHz s16le, as OpenAI's), `f32`.
  Compressed formats (`mp3`, `opus`, `aac`, `flac`) answer 400.
- `stream_format`: `audio` (default, bytes as generated) or `sse` (`speech.audio.delta` events with
  base64 audio, then `speech.audio.done`).
- `voice`: a learned voice id; any other name (`alloy`, …) speaks with the default voice.
- `language` (extension): `es` or `en`; the default is `tts.lang`.
- `model`, `speed` and `instructions` are accepted and ignored.

## Voices — `/v1/voices`

| Request | Does |
|---|---|
| `GET /v1/voices` | `{"voices": ["damien"]}` |
| `POST /v1/voices` (multipart: `voice_id`, optional `text`, one or more `file`) | learns a voice from the clips → 201 `{"voice_id","seconds"}` |
| `DELETE /v1/voices/{voice_id}` | 204, or 404 |

```bash
curl -F voice_id=damien -F file=@me.mp3 -F "text=Lo que digo en el clip." http://127.0.0.1:5600/v1/voices
```

## Errors

OpenAI shape, `{"error": {"message", "type"}}`: 400 invalid request, 404 unknown voice, 422 a voice
the loaded model cannot use, 501 a backend that cannot learn voices, 503 every worker busy.
