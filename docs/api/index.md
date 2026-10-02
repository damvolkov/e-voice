# Protocols

Every transport shares one `Runner`, so a backend behaves the same behind every protocol.

- **Uploads** go through the lossless file intake: input pauses while segment work queues, so no
  segment is dropped.
- **Sockets** go through the live intake: the overload policy sheds work, and the wake word gate
  applies.

| Protocol | Endpoint | Transport |
|---|---|---|
| [Native](native.md) | `GET /v1/stream` | WebSocket, `view=struct` (events) or `flat` (text) |
| [OpenAI](openai.md) | `POST /v1/audio/transcriptions` · `GET /v1/realtime` | multipart REST (+SSE) · Realtime WebSocket |
| [Deepgram](deepgram.md) | `POST /v1/listen` · `GET /v1/listen` | raw-body REST · live WebSocket |
| [ElevenLabs](elevenlabs.md) | `POST /v1/speech-to-text` | multipart REST |
| Service | `GET /health` · `GET /v1/models` · `GET /openapi.json` · `GET /docs` | |

## Choosing the engine

Uploads (OpenAI transcriptions, Deepgram prerecorded, ElevenLabs Scribe) select the ASR engine through
their model field — `model`, or `model_id` for ElevenLabs. Selectable engines are:

- the default file engine (`stt.pipeline.asr.offline.backend`);
- every engine listed in `offline.choices`, loaded at startup.

`GET /v1/models` lists them, default first. A value naming an engine (`parakeet`, `canary`, `cohere`,
`whisper`) or its model id (`whisper-turbo`) selects it. Any other value — the `whisper-1`, `nova-2`,
`scribe_v1` clients send by default — uses the default engine. Naming an engine that is not loaded is a
400 that lists the loaded ones.

```toml
[stt.pipeline.asr]
offline = { backend = "parakeet", choices = ["whisper", "cohere"] }
```

Live sockets always use the configured live backend.

Anthropic has no speech-to-text API, so there is none to mirror. The running service serves its own
[reference](reference.md) at `/docs`.

## Emotion

`server.emotion` sets the default; every endpoint accepts `emotion=field|tag|off` (query parameter or
form field).

| Mode | Effect |
|---|---|
| `field` | an `emotion` object on segments, finals and summaries — clients that don't know it ignore it |
| `tag` | the object, plus `[angry] …` before each segment's text, for clients that only read `text` |
| `off` | neither (SER still runs unless `stt.pipeline.ser.backend = "off"`) |

```json
{"label": "neutral", "scores": {"angry": 0.01, "happy": 0.02, "neutral": 0.95, "…": 0.0}, "model": "emotion2vec-plus-base"}
```

Labels: `angry`, `disgusted`, `fearful`, `happy`, `neutral`, `other`, `sad`, `surprised`, `unknown`.
`unknown` means SER was late, failed, off, or the segment was shorter than `ser.min`.

## Languages

`es` and `en`. Regions (`es-ES`, `en_US`, `es-419`) and ElevenLabs' `spa`/`eng` are accepted. Any other
language is rejected before work starts: HTTP 400, or no WebSocket upgrade.
