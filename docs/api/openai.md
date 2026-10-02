# OpenAI

## Transcriptions

`POST /v1/audio/transcriptions`, multipart:

| Field | Values |
|---|---|
| `file` | wav, mp3, m4a, flac, ogg, opus, webm |
| `model` | any (accepted for compatibility) |
| `language` | `es`, `en` |
| `response_format` | `json` (default), `text`, `srt`, `vtt`, `verbose_json` |
| `stream` | `true` → server-sent events |
| `emotion` | `field`, `tag`, `off` (extension) |

`prompt`, `temperature` and `timestamp_granularities[]` are accepted and ignored.

- **`json`:** `{"text", "usage", "emotion"}`.
- **`verbose_json`:** adds `language`, `duration` and `segments[]` with `start`/`end` in seconds and a
  per-segment `emotion`.
- **`stream=true`:** one `transcript.text.delta` per final, then `transcript.text.done` with `usage`.
- **Errors:** OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`.
- **Failure policy:** a request whose segments fail is a 500, never silently partial text.

## Realtime

`GET /v1/realtime?intent=transcription&language=es&emotion=field` upgrades to the Realtime transcription
protocol. Both the GA (`session.*`) and beta (`transcription_session.*`) event names are served.

| Client event | Effect |
|---|---|
| `session.update` / `transcription_session.update` | sets language (`audio.input.transcription.language` or `input_audio_transcription.language`) and rate (`audio.input.format.rate`, default 24000) before audio starts |
| `input_audio_buffer.append` | base64 pcm16 audio |
| `input_audio_buffer.commit` / `clear` | acknowledged; server VAD always segments |
| `session.close` | drains pending segments, then closes with 1000 |

Server events:

- `session.created` / `session.updated`;
- `input_audio_buffer.speech_started` / `speech_stopped` / `committed`;
- `conversation.item.input_audio_transcription.delta` (from streaming partials);
- `…completed` (with the `emotion` extension) and `…failed`;
- `error`.
