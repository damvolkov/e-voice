<p align="center">
  <img src="assets/e-voice.svg" alt="e-voice" width="160" />
</p>

<h1 align="center">e-voice</h1>

<p align="center">
  <strong>CPU-first speech service in Rust — streaming STT with emotion on every segment, behind OpenAI, Deepgram and ElevenLabs compatible APIs.</strong>
</p>

<p align="center">
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-%3E%3D1.94-orange?logo=rust&logoColor=white" alt="Rust"></a>
  <a href="https://github.com/k2-fsa/sherpa-onnx"><img src="https://img.shields.io/badge/runtime-sherpa--onnx%201.13.8-blue" alt="sherpa-onnx"></a>
  <a href="https://ubuntu.com/"><img src="https://img.shields.io/badge/platform-Linux%20x86__64-E95420?logo=linux&logoColor=white" alt="Linux"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://github.com/j178/prek"><img src="https://img.shields.io/badge/hooks-prek-blueviolet" alt="prek"></a>
</p>

---

One pipeline, every node swappable from config. Full documentation:
**[damvolkov.github.io/e-voice](https://damvolkov.github.io/e-voice/)** (built locally with `make docs`).

```
audio ─▶ [denoise] ─▶ AGC ─▶ [ww] ─▶ VAD ─▶ [lid] ─▶ ASR (streaming partials) ─┬─▶ join ─▶ final {text, lang, emotion}
                                                                               └─▶ SER ─┘
```

| Node | Backends | Runtime |
|---|---|---|
| denoise | `off` (default), `gtcrn` | sherpa-onnx |
| ww | `off` (default), `kws` (open vocabulary, phrase `hey eager`), `oww` (openWakeWord) | sherpa-onnx · `ort` |
| vad | `silero` (default), `ten` | sherpa-onnx |
| lid | `off` (default), `whisper` (tiny) | sherpa-onnx |
| asr | streaming: `nemotron` 3.5 (live default), `kroko` (CC-BY-SA, research only) · per segment: `parakeet` TDT v3 (file default), `canary` 180M flash, `cohere` Transcribe, `whisper` turbo | sherpa-onnx |
| ser | `emotion2vec` plus base (default), `emotion2vec-large` (local export), `off` | `ort` |

Spanish and English; sherpa-onnx and `ort` share one `libonnxruntime.so`; no GPU.

## Quickstart

```bash
make install             # OS packages, pinned Rust (rust-toolchain.toml), uv, prek, llvm-cov, git hooks
make setup               # sherpa toolkit + the models the config declares → data/stt
make serve               # gateway on :5500, docs at http://127.0.0.1:5500/docs
make stt ARGS="--flat"   # another terminal: talk; Ctrl+C to finish (--struct for JSON events)
```

With Docker — everything under [`docker/`](docker), one Dockerfile and one compose per deployment:

```bash
make up                  # MODE=full (default): every service in one container on :5500
make up MODE=stt         # the STT service alone on :5500
make up MODE=split       # stt on :5500 + tts on :5600 (tts is a template until its crate exists)
make down [MODE=…]
```

| Mode | Dockerfile | Compose | Image |
|---|---|---|---|
| `full` | `docker/Dockerfile.full` | `docker/compose.full.yml` | `e-voice:full` |
| `stt` | `docker/Dockerfile.stt` | `docker/compose.stt.yml` | `e-voice:stt` |
| `tts` (template) | `docker/Dockerfile.tts` | `docker/compose.tts.yml` | `e-voice:tts` |

Each Dockerfile is two-stage (cargo-chef builder → debian-slim runtime, non-root, tini, healthcheck)
with its own `.dockerignore` beside it. A one-shot `pull` service installs the configured models into
the `e-voice-stt` volume (shared by `full` and `stt`; `EVOICE_STT_DATA=/abs/path` for a host directory)
before the service starts.

`make install` detects the platform; force it with `OS=ubuntu|mac|windows` (ubuntu = any apt
distribution, windows = Git Bash with winget). Its parts run alone: `system` (compiler, cmake,
pkg-config, curl, bzip2, ALSA headers on Linux), `toolchain`, `tools`. The Dockerfiles run the same
`make system`, `make sherpa` and `make dist`, and CI runs `make system toolchain sherpa` then
`make hooks` — one definition of every step. Prebuilt sherpa-onnx libraries are pinned for
linux-x64, macOS (universal2) and windows-x64; only Linux is exercised by CI.

`ecli` and every client below point at `127.0.0.1:5500` either way.

## API

Interactive reference: `GET /docs` (Scalar), spec at `GET /openapi.json`. All transports share one
`Runner`: uploaded files go through the lossless file intake (input pauses instead of dropping
segments), sockets through the live intake.

| Protocol | Endpoint | Shape |
|---|---|---|
| OpenAI transcriptions | `POST /v1/audio/transcriptions` | multipart; `json`, `text`, `srt`, `vtt`, `verbose_json`; `stream=true` → SSE `transcript.text.delta` / `done` |
| OpenAI Realtime | `GET /v1/realtime` (WebSocket) | `session.update`, `input_audio_buffer.append` (base64 pcm16, 24 kHz unless set) → `speech_started` / `stopped` / `committed`, `…transcription.delta` / `completed` |
| Deepgram | `POST /v1/listen` · `GET /v1/listen` (WebSocket) | prerecorded `results.channels[].alternatives[]`; live `Results` (interim + final), `SpeechStarted`, `UtteranceEnd`, `Metadata`; `linear16` / `pcm_f32le` |
| ElevenLabs Scribe | `POST /v1/speech-to-text` | multipart `file`, `language_code` (`spa`, `eng`, `es`, `en`) |
| Native | `GET /v1/stream` (WebSocket) | `?lang&rate&encoding=s16le\|f32le&view=struct\|flat`; binary PCM in, events out |
| — | `GET /v1/models`, `GET /health` | |

Anthropic has no speech-to-text API, so there is nothing to mirror.

### Emotion

`server.emotion` sets the default; every endpoint accepts `emotion=field|tag|off` per request.

| Mode | Effect |
|---|---|
| `field` | an `emotion` object on segments, finals and summaries (clients that don't know it ignore it) |
| `tag` | the object, plus `[angry] …` prefixed to each segment's text, for clients that only read `text` |
| `off` | neither; SER still runs unless `stt.pipeline.ser.backend = "off"` |

### Native stream

`view=struct` (default) sends one JSON event per frame; `view=flat` sends only each final's text.
Text frame `{"type":"end"}` drains pending segments, then `{"type":"closed"}` and close code 1000.

```json
{"type":"speech","segment":0,"state":"started","at":8288}
{"type":"partial","segment":0,"text":"No preguntes qué"}
{"type":"speech","segment":0,"state":"stopped","at":91840}
{"type":"final","segment":0,"span":{"start":8288,"end":91840},"lang":"es","text":"…","emotion":{"label":"neutral","scores":{…},"model":"emotion2vec-plus-base"},"error":null}
{"type":"closed"}
```

Every segment gets exactly one `final`, in order. A late or failed SER yields `unknown`; a failed ASR
sets `error`. Closing the socket cancels; SIGTERM finalizes open segments.

### Clients

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:5500/v1", api_key="unused")
print(client.audio.transcriptions.create(model="whisper-1", file=open("a.mp3", "rb"), language="es").text)
```

- **OpenHuman** — voice provider with `endpoint = "http://127.0.0.1:5500/v1"`, `capability = "stt"` and
  `stt_api_style` = `openaiaudio` (default), `deepgram` or `elevenlabs`. It reads only `text`: set
  `server.emotion = "tag"` to keep emotion in the transcript.
- **Hermes** — its OpenAI STT provider with `stt.openai.base_url = "http://127.0.0.1:5500/v1"`.

## ecli

```bash
ecli devices                                   # inputs; * marks the system default
ecli stt --flat --device "Q20i"                # live text with a level meter and latency per final
ecli stt --struct --lang en --record sent.wav  # JSON events; keep exactly what was sent
ecli stt --flat --wav sample.wav               # replay a file in real time instead of a microphone
ecli stt --url ws://host:5500                  # another gateway
```

Inputs come from PulseAudio (served by PipeWire), so names match the system's sound settings. A
"Monitor of …" input is what the speakers play, not a microphone; `ecli` warns about it, and about
digital silence. Bluetooth headsets expose their microphone only in the headset (HFP) profile.

## Configuration

`evoice.toml` in the working directory (or `--config path`), overridden by `EVOICE_<SECTION>__<KEY>`
(e.g. `EVOICE_STT__PIPELINE__ASR__CHUNK=560ms`). [`evoice.example.toml`](evoice.example.toml) lists
every key with its default and every accepted value:

```
[server]                     host, port, upload, emotion, log
[stt]                        lang
[stt.pipeline]               pending, overload, stall, tick, preroll, jobs, gate, gain
[stt.pipeline.ww|vad|asr|ser] backend + its knobs
[stt.ops]                    data, manifest, verify
[tts]                        reserved
```

Choices are closed enums. An unknown backend, key or out-of-range value stops the service at startup.

### Wake word

The pipeline runs ungated by default (`ww.backend = "off"`). With `kws` any English phrase of two or more
words works without training; `make wake WORD="hey eager"` measures it — synthetic positives over three
voices, decoys and real speech as negatives, every threshold × boost — and prints the
`[stt.pipeline.ww]` block to paste. Single words fire on everyday speech; see
[the calibration notes](docs/history/2026-10-benchmark.md#wake-word-kws-open-vocabulary).

## Models

[`stt/models.toml`](stt/models.toml) pins every artifact by URL (Hugging Face by commit) and sha256.
`e-voice pull` installs atomically with a per-file digest stamp: pipeline models into
`data/stt/models` (the Docker volume), tool and test models into `data/stt/ops/models`. `serve` never
downloads and refuses to start on a missing or stale model; `e-voice verify --full` re-hashes everything.

## Layout

```
stt/                    the service (crate e-voice-stt, binary e-voice)
├── src/
│   ├── main.rs         serve | pull | verify | bench | wake
│   ├── api/            gateway: live bridge, batch, openai/ deepgram/ elevenlabs/ native/, docs
│   ├── config/         one typed section per node and domain
│   ├── core/           settings, models, audio, runtime, logger
│   ├── schema/         contracts: audio, lang, emotion, segment, event, transcript, error
│   ├── workflow/       session (sans-IO state machine), runner, nodes; ww/ vad/ asr/ ser/
│   └── ops/            internal tooling: bench, wake calibration, voice synthesis
├── tests/              workflow, api, ops
├── ops/                sherpa.sh, bench.sh, bench/*.toml
└── models.toml
cli/                    ecli: microphone → /v1/stream tester
docker/                 Dockerfile.{full,stt,tts} (+ .dockerignore each), compose.{full,stt,tts}.yml
eval/                   Python (uv): dataset fetch, scoring, report
data/                   ignored: ops/target (cargo), stt/models, stt/ops/{sherpa,models,datasets,results}
docs/history/           dated decisions and measurements
```

Adding a backend: a file in its node folder, a variant in `config/<node>.rs`, an arm in the registry.

## Benchmark

```bash
make setup ARGS=--all    # tool and test models too
make datasets            # FLEURS es/en, MESD es, CREMA-D en, pinned by commit
make bench               # stt/ops/bench/*.toml × datasets, file and live modes
make report              # WER, CER, SER accuracy / F1 / angry recall, RTF, latency, CPU, RSS
```

Current numbers and why the defaults are what they are: [docs/history/2026-10-benchmark.md](docs/history/2026-10-benchmark.md).
Parakeet on files: 4.5% WER es / 8.1% en at 0.25 CPU s per audio second; Nemotron live: finals ~0.74 s
after speech ends.

## Quality gates

| Command | Runs |
|---|---|
| `make check` | rustfmt, clippy (pedantic, `-D warnings`), layer bounds, tests without models |
| `make backends` | model-backed tests: each backend in isolation, the pipeline, the gateway, the tools |
| `make coverage` | every test with line coverage ≥ 90% |
| `make hooks` | every prek hook (pre-commit: the `check` set, gitleaks, zizmor; pre-push: coverage) |

The session core is property-tested: one final per segment, in order; bounded pending work; exactly one
`closed`.
