# Backends

## STT

Every backend runs on CPU through sherpa-onnx (native) or `ort` (custom graphs). Licences are the
models'; entries marked "see the model card" were not verified here — check them before shipping.

### ASR

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

### Other nodes

| Node | Backend | Model | Licence |
|---|---|---|---|
| denoise | `gtcrn` | GTCRN simple (48k params) | see the model card |
| ww | `kws` | Zipformer KWS GigaSpeech 3.3M | see the model card |
| ww | `oww` | openWakeWord melspectrogram + embedding + `hey_jarvis` | Apache-2.0 / CC-BY-NC-SA (classifier) |
| vad | `silero`, `ten` | Silero VAD v5, TEN VAD | MIT, Apache-2.0 (TEN terms) |
| lid | `whisper` | Whisper tiny | MIT |
| ser | `emotion2vec` | emotion2vec+ base (768-dim) | FunASR model licence |
| ser | `emotion2vec-large` | emotion2vec+ large (1024-dim), local export | FunASR model licence |

### Considered and not added

| Candidate | Why not |
|---|---|
| SenseVoice emotion | sherpa's Rust result drops `emotion`/`lang`; reading them needs FFI `unsafe`, which the crate forbids. Waits for an upstream field. |
| Moonshine (es) | non-commercial licence; Moonshine streaming is not in sherpa |
| Qwen3-ASR, Omnilingual ASR | weaker on Spanish for their CPU cost |
| Canary 1B v2 | no sherpa export published |
| microWakeWord | TFLite only |
| Diarization, livekit-wakeword | later |

### Adding a backend

1. A file in `workflow/<node>/` implementing the node's trait.
2. A variant in `config/<node>.rs`, with its model id.
3. An arm in `workflow/<node>/registry.rs`.
4. Its artifact in `evoice/stt/models.toml`, pinned by sha256.
5. A model-backed test in `evoice/stt/tests/workflow/` and a benchmark config in `evoice/stt/ops/bench/`.

## TTS

Only backends that stream for real: audio leaves while a sentence is still being generated (the
conformance test in `evoice/tts/tests/workflow/conformance.rs` enforces it).

| Backend | Model | Languages | Streaming | Voices | Runtime | Licence |
|---|---|---|---|---|---|---|
| `pocket` (default) | Kyutai Pocket TTS 100M, `base` 6 layers / `large` 24 (Spanish) | es, en | frame by frame, 80 ms | zero-shot from a 10–30 s clip | our autoregressive loop on `ort` over a local ONNX export | code MIT · weights CC-BY-4.0, gated (no cloning without consent) |
| `qwen3` | Qwen3-TTS 12Hz 0.6B Base (int4 talker and code predictor) | es, en (+8) | 80 ms frames, voiced 4 then 16 at a time (causal vocoder, 50 frames of left context, on its own thread) | x-vector from the clip; in-context with the reference codes when the clip is ≤ 10 s and has its transcript | our talker + code-predictor loop on `ort` over wavekat's ONNX graphs | Apache-2.0 |
| `neutts` | NeuTTS Nano (es, en backbones, 24-layer Llama) + NeuCodec | es, en | 25 codes (0.5 s) per window, overlap-add | in-context: reference codes + phonemized transcript (required) | backbone int4 GQA on `ort` (our export), espeak-ng for phonemes | NeuTTS Open License 1.0: free under $5M annual revenue, outputs included; gated |

Pocket runs on our own loop rather than sherpa-onnx: sherpa 1.13.8 ships English Pocket only, cannot
load the current Spanish checkpoints (#3755), and generates a whole sentence before decoding it — a
first chunk only after the full sentence. The loop reproduces the Python reference bundle to < 1e-3
(`test_pocket_reproduces_the_python_reference_loop`), and its output matches upstream torch.

Qwen3's preprocessing (speaker-encoder mel, tokenizer, text projection, x-vector, reference codes) is
checked against the official code (`make qwen3` writes the golden, `qwen3::tests` compare). Its cost is
the 15-step code predictor per frame (≈ 40 ms on 4 threads, memory-bound: the fp32 graph is 3× slower),
so one stream runs at about real time.

NeuTTS needs the neuphonic terms accepted on Hugging Face (backbones and decoder are gated), `espeak-ng`
installed (`make system`), and `make neutts` to export the backbones. Its model-backed verification is
pending that access; the pure parts (phonemizer punctuation runs, overlap-add) are unit-tested.

| | Pocket base | Qwen3 (x-vector) |
|---|---|---|
| WER es / en, round trip | 9.9 / **13.6** % | **8.2** / 16.4 % |
| similarity to the speaker, es / en | 0.69 / 0.72 | **0.74 / 0.74** |
| first audio | **161 ms** | 380 ms |
| RTF, one stream | **0.35** | 0.96 |
| aggregate throughput (8 streams) | **4.8×** | 1.8× |

40 FLEURS sentences per language, voice learned from a 20 s sample (2026-10-03, i9-14900K). In-context
Qwen3 cloning on that 20 s clip measured worse than the x-vector (similarity 0.71, RTF 1.36, first audio
1.9 s), hence its 10 s limit. Qwen3 is the closer voice; Pocket is the one that scales.

### Considered and not added

| Candidate | Why not |
|---|---|
| Supertonic 3, Kokoro, Piper/VITS, Matcha, MeloTTS | non-autoregressive, a sentence at once: not streaming |
| Chatterbox Multilingual (es-es) | best Castilian fit, but RTF ≥ 4 on CPU even in ggml |
| CosyVoice 3, VoxCPM 2, Orpheus, Higgs | GPU-class on CPU (RTF 2–4 and worse) |
| XTTS v2, F5-TTS, Fish/OpenAudio, Voxtral TTS | non-commercial weights |

### Adding a backend

1. A file in `evoice/tts/src/workflow/synth/` implementing `Synth` (and `train` if it can learn voices).
2. A variant in `evoice/tts/src/config/synth.rs`, with its model ids.
3. An arm in `evoice/tts/src/workflow/synth/registry.rs`.
4. Its artifacts in `evoice/tts/models.toml`, pinned by sha256.
5. The conformance test passing on it, a model-backed test, and a config in `evoice/tts/ops/bench/`.
