# Clients

## OpenAI SDKs

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5500/v1", api_key="unused")
result = client.audio.transcriptions.create(
    model="whisper-1", file=open("a.wav", "rb"), language="es", response_format="verbose_json",
)
```

`stream=True` yields `transcript.text.delta` events; the Realtime API works on `/v1/realtime`.

## OpenHuman

Verified against OpenHuman `tinyhumansai/openhuman@ce1a859` with its own STT client
(`tinyinference-voice`). The test matrix is 3 API styles × WAV, MediaRecorder WebM/Opus, fragmented
MP4/AAC and Ogg/Opus × no language, `es`, `en`: 36 of 36 pass.

Add a voice provider and route STT to it in OpenHuman's `config.toml`:

```toml
stt_provider = "evoice:parakeet"        # "<slug>:<model>"; the model selects the engine

[[voice_providers]]
id = "vp_evoice_local"
slug = "evoice"
label = "e-voice"
endpoint = "http://127.0.0.1:5500/v1"
auth_style = "bearer"                   # no key needed; an empty one is sent and ignored
capability = "stt"
stt_api_style = "openaiaudio"           # or "deepgram", "elevenlabs" — all three are served
tts_api_style = "openaiaudio"
default_stt_model = "parakeet"
```

- **Spelling of the style.** In the TOML the OpenAI style is spelled `openaiaudio`; the settings UI and
  RPC spell it `openai_audio`.
- **Requests.** Each style posts to the matching endpoint:
  - `openaiaudio`: `{endpoint}/audio/transcriptions`;
  - `deepgram`: `{endpoint}/listen?model=…`;
  - `elevenlabs`: `{endpoint}/speech-to-text`.
- **Responses.** OpenHuman reads only the text. Set `server.emotion = "tag"` to keep emotion in what
  it reads (`[angry] …`).
- **Engine.** `evoice:whisper` selects Whisper for that route when it is in `offline.choices`.
- **Language.** Dictation sends no language, so the server's `stt.lang` applies. Parakeet and Whisper
  detect it anyway; with Canary or Cohere, enable `lid`.

## Hermes

Use its OpenAI speech-to-text provider with `stt.openai.base_url = "http://127.0.0.1:5500/v1"`.

## ecli

The bundled terminal tester streams the microphone (or a WAV in real time) to `/v1/stream`:

```bash
ecli devices                                   # inputs; * marks the system default
ecli stt --flat --device "Q20i"                # live text, level meter, latency per final
ecli stt --struct --lang en --record sent.wav  # every event as JSON; keep what was sent
ecli stt --flat --wav sample.wav               # replay a file instead of a microphone
ecli stt --url ws://host:5500                  # another gateway
```

`make stt ARGS="…"` runs it from the repository.

On Linux, inputs come from PulseAudio (served by PipeWire), so device names match the system's sound
settings. `ecli` warns about two common mistakes:

- a "Monitor of …" input is what the speakers play, not a microphone;
- digital silence means the input is muted or wrong.

Bluetooth headsets expose their microphone only in the headset (HFP) profile.
