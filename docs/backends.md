# Backends

Every backend runs on CPU through sherpa-onnx (native) or `ort` (custom graphs). Licences are the
models'; entries marked "see the model card" were not verified here — check them before shipping.

## ASR

| Backend | Kind | Languages | Model | Licence |
|---|---|---|---|---|
| `nemotron` | streaming (160/560/1120 ms chunks) | 40 locales, prompt per stream | Nemotron 3.5 0.6B int8 | see the model card |
| `kroko` | streaming, one model per language | es, en | Kroko zipformer 2025-08 | CC-BY-SA, hobby/research — never a default |
| `parakeet` | per segment, detects language | 25 European | Parakeet TDT 0.6B v3 int8 | CC-BY-4.0 |
| `canary` | per segment, language given | en, es, de, fr | Canary 180M flash int8 | CC-BY-4.0 |
| `cohere` | per segment, language given | 14 | Cohere Transcribe 2B int8 (1.6 GB) | Apache-2.0 |
| `whisper` | per segment, detects language | 99 | Whisper large-v3 turbo int8 | MIT |

Canary keeps one recognizer per language, because sherpa fixes its source language per recognizer.
Cohere sets the language per stream. Parakeet and Whisper ignore the requested language: they
transcribe what they hear.

## Other nodes

| Node | Backend | Model | Licence |
|---|---|---|---|
| denoise | `gtcrn` | GTCRN simple (48k params) | see the model card |
| ww | `kws` | Zipformer KWS GigaSpeech 3.3M | see the model card |
| ww | `oww` | openWakeWord melspectrogram + embedding + `hey_jarvis` | Apache-2.0 / CC-BY-NC-SA (classifier) |
| vad | `silero`, `ten` | Silero VAD v5, TEN VAD | MIT, Apache-2.0 (TEN terms) |
| lid | `whisper` | Whisper tiny | MIT |
| ser | `emotion2vec` | emotion2vec+ base (768-dim) | FunASR model licence |
| ser | `emotion2vec-large` | emotion2vec+ large (1024-dim), local export | FunASR model licence |

## Considered and not added

| Candidate | Why not |
|---|---|
| SenseVoice emotion | sherpa's Rust result drops `emotion`/`lang`; reading them needs FFI `unsafe`, which the crate forbids. Waits for an upstream field. |
| Moonshine (es) | non-commercial licence; Moonshine streaming is not in sherpa |
| Qwen3-ASR, Omnilingual ASR | weaker on Spanish for their CPU cost |
| Canary 1B v2 | no sherpa export published |
| microWakeWord | TFLite only |
| Diarization, livekit-wakeword | later |

## Adding a backend

1. A file in `workflow/<node>/` implementing the node's trait.
2. A variant in `config/<node>.rs`, with its model id.
3. An arm in `workflow/<node>/registry.rs`.
4. Its artifact in `stt/models.toml`, pinned by sha256.
5. A model-backed test in `stt/tests/workflow/` and a benchmark config in `stt/ops/bench/`.
