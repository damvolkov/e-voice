<p align="center">
  <img src="assets/e-voice-landscape.svg" alt="e-voice" width="420">
</p>

<p align="center">
  <strong>Production-grade Speech API</strong> — STT (faster-whisper) + TTS (Kokoro-ONNX)<br>
  Streaming, WebSocket, and OpenAI-compatible endpoints
</p>

<p align="center">
  Powered by <a href="https://github.com/sparckles/Robyn">Robyn</a> (Rust-backed async Python)
</p>

---

## Quick Start

```bash
# Local development
make install
make dev              # API on :5500, Gradio UI on :5600

# Docker (nginx gateway unifies everything on :5500)
docker compose up -d  # → localhost:5500 (UI + API + docs)
```

| Service | Local | Docker |
|---------|-------|--------|
| API | `localhost:5500/v1/...` | `localhost:5500/v1/...` |
| Docs | `localhost:5500/docs` | `localhost:5500/docs` |
| Gradio UI | `localhost:5600` | `localhost:5500` (root) |
| WebSocket STT | `ws://localhost:5500/v1/audio/transcriptions` | Same |
| WebSocket TTS | `ws://localhost:5500/v1/audio/speech` | Same |

## API Reference

Base URL: `http://localhost:5500`

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + service info |

### Speech-to-Text (STT)

#### HTTP (OpenAI-compatible)

| Method | Endpoint | Description | OpenAI Compatible |
|--------|----------|-------------|:-:|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio file | Yes |
| `POST` | `/v1/audio/translations` | Translate audio to English | Yes |

**Request**: `multipart/form-data`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | binary | required | Audio file (WAV, MP3, FLAC, etc.) |
| `model` | string | `faster-whisper-large-v3-turbo` | Whisper model ID |
| `language` | string | auto-detect | ISO 639-1 language code |
| `prompt` | string | — | Context hint for transcription |
| `response_format` | string | `json` | `json`, `text`, `verbose_json`, `srt`, `vtt` |
| `temperature` | float | `0.0` | Sampling temperature (0.0–1.0) |
| `stream` | bool | `false` | Enable SSE streaming |
| `vad_filter` | bool | `false` | Voice Activity Detection |
| `hotwords` | string | — | Bias transcription toward specific words |
| `timestamp_granularities[]` | string | `segment` | `segment` or `word` |

**Response formats**:

```json
// json
{"text": "Hello world."}

// verbose_json
{"task": "transcribe", "language": "en", "duration": 2.5, "text": "Hello world.", "segments": [...], "words": [...]}
```

```
// text
Hello world.

// srt
1
00:00:00,000 --> 00:00:02,500
Hello world.

// vtt
WEBVTT

00:00:00.000 --> 00:00:02.500
Hello world.
```

#### SSE Streaming

Same endpoint with `stream=true`. Returns `text/event-stream`:

```
data: {"text": "Hello"}
data: {"text": " world."}
```

#### WebSocket (Real-time Streaming)

| Endpoint | Description |
|----------|-------------|
| `WS /v1/audio/transcriptions` | Real-time streaming STT with LocalAgreement |

**Query parameters**: `?language=es&response_format=text&model=...`

**Protocol**:
1. Connect with optional query params
2. Send base64-encoded PCM16 audio (16kHz mono) as text frames
3. Receive streaming transcription events

**Response** (when `response_format=json`):

```json
{"type": "transcript_update", "text": "Hola mundo", "partial": "qué tal", "is_final": false}
{"type": "transcript_final", "text": "Hola mundo, qué tal.", "partial": "", "is_final": false}
{"type": "session_end", "text": "Hola mundo, qué tal.", "partial": "", "is_final": true}
```

When `response_format=text`: returns confirmed text as plain string.

**Features**:
- LocalAgreement algorithm — words are never retracted once confirmed
- Sentence-boundary finalization + same-output detection
- Bounded audio buffer (45s max) with context re-transcription
- Per-connection state with automatic cleanup on disconnect

### Text-to-Speech (TTS)

#### HTTP (OpenAI-compatible)

| Method | Endpoint | Description | OpenAI Compatible |
|--------|----------|-------------|:-:|
| `POST` | `/v1/audio/speech` | Synthesize speech from text | Yes |
| `GET` | `/v1/audio/voices` | List available voices | — |

**Request**: `application/json`

```json
{
  "input": "Hello world.",
  "model": "kokoro",
  "voice": "af_heart",
  "response_format": "mp3",
  "speed": 1.0,
  "stream": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string | required | Text to synthesize |
| `model` | string | `kokoro` | TTS model |
| `voice` | string | `af_heart` | Voice ID (prefix = language) |
| `response_format` | string | `mp3` | `pcm`, `mp3`, `wav`, `flac`, `opus`, `aac` |
| `speed` | float | `1.0` | Speaking speed multiplier |
| `stream` | bool | `false` | Enable streaming response |

#### SSE Streaming

Same endpoint with `stream=true` + `stream_format=sse`.

#### Audio Streaming

Same endpoint with `stream=true` + `stream_format=audio`. Returns chunked binary audio.

#### WebSocket (Real-time Streaming)

| Endpoint | Description |
|----------|-------------|
| `WS /v1/audio/speech` | Real-time streaming TTS |

**Protocol**:
1. Connect
2. Send JSON: `{"input": "Hello", "voice": "af_heart", "speed": 1.0, "lang": "en-us"}`
3. Receive streaming audio chunks:

```json
{"type": "speech.audio.delta", "audio": "<base64_pcm16_24khz>"}
{"type": "speech.audio.done"}
```

### Model Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/models` | List loaded models |
| `GET` | `/v1/models/:model_id` | Get model info |
| `POST` | `/v1/api/ps/:model_id` | Load model into memory |
| `DELETE` | `/v1/api/ps/:model_id` | Unload model |
| `GET` | `/v1/api/ps` | List loaded models (internal) |
| `POST` | `/v1/api/pull/:model_id` | Download model from HuggingFace |

## Transport Summary

| Transport | STT | TTS |
|-----------|:---:|:---:|
| HTTP POST (batch) | `/v1/audio/transcriptions` | `/v1/audio/speech` |
| SSE (streaming) | `stream=true` | `stream=true` + `stream_format=sse` |
| Audio stream | — | `stream=true` + `stream_format=audio` |
| WebSocket | `WS /v1/audio/transcriptions` | `WS /v1/audio/speech` |

## OpenAI Compatibility

e-voice implements the [OpenAI Audio API](https://platform.openai.com/docs/api-reference/audio) specification:

| OpenAI Endpoint | e-voice | Status |
|-----------------|---------|--------|
| `POST /v1/audio/transcriptions` | Same | Full |
| `POST /v1/audio/translations` | Same | Full |
| `POST /v1/audio/speech` | Same | Full |
| `GET /v1/models` | Same | Full |
| `GET /v1/audio/voices` | Extension | Extension |
| `WS /v1/audio/transcriptions` | Extension | Extension |
| `WS /v1/audio/speech` | Extension | Extension |

Drop-in replacement for any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5500/v1", api_key="unused")

# Transcribe
result = client.audio.transcriptions.create(
    model="whisper-1", file=open("audio.wav", "rb")
)

# Synthesize
response = client.audio.speech.create(
    model="kokoro", voice="af_heart", input="Hello world."
)
```

## Development

```bash
make lint          # ruff check --fix + format
make type          # ty type check
make test          # unit tests (parallel, coverage >90%)
make check         # lint + type + test
make stt           # mic -> WebSocket STT (ffmpeg + websocat)
make tts           # text -> WebSocket TTS (websocat)
```

## Web UI (Gradio)

A Gradio playground launches automatically alongside the API:

- **Speech-to-Text** — upload audio, select model/language, transcribe or translate (with SSE streaming)
- **Text-to-Speech** — enter text, pick voice/speed, synthesize audio
- **Models** — view downloaded models, download new STT/TTS models

| Environment | URL |
|-------------|-----|
| Local | `http://localhost:5600` |
| Docker | `http://localhost:5500` (nginx proxies root → Gradio) |

Disable via `front.enabled: false` in `data/config/config.yaml`.

## Configuration

All settings in `data/config/config.yaml` (YAML-based, typed via `pydantic-settings`):

```yaml
system:
  port: 5500
  debug: true

stt:
  model: "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
  device: cuda

tts:
  device: cuda
  default_voice: af_heart

front:
  enabled: true
  port: 7860
```

## Data Layout

```
data/
├── config/
│   └── config.yaml   # Tracked in git — all app configuration
└── models/
    ├── stt/           # Gitignored — downloaded Whisper models
    └── tts/           # Gitignored — downloaded Kokoro models
```
