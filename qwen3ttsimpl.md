# e-voice — Qwen3-TTS Backend Integration

> Implementation guide for adding Qwen3-TTS as an optional TTS backend alongside Kokoro-ONNX.

---

## 1. Context & Motivation

e-voice currently ships with **Kokoro-ONNX** (82M params) as its sole TTS backend. Kokoro excels at CPU inference (~7x real-time), zero-config deployment, and minimal dependencies (`kokoro-onnx` + ONNX Runtime). However, it lacks voice cloning, fine-tuning support, and instruction-driven voice control.

**Qwen3-TTS** (0.6B / 1.7B params, Apache 2.0) fills these gaps: 3-second voice cloning, official single-speaker fine-tuning, natural language voice design, and 10-language coverage. The tradeoff: it requires PyTorch and performs best on GPU.

### Design Principles

- **Kokoro remains the default** — zero-config, CPU-first, no new deps.
- **Qwen3 is opt-in** — installed via optional extras, activated via config.
- **Transport layer is untouched** — WS, SSE, chunked, HTTP all work as-is.
- **Backend Protocol enforces uniformity** — both backends implement the same interface.
- **Dependency isolation** — `torch` never enters the core dependency tree.

---

## 2. Comparative Analysis

| Aspect | Kokoro-ONNX | Qwen3-TTS 0.6B | Qwen3-TTS 1.7B |
|---|---|---|---|
| Parameters | 82M | 600M | 1.7B |
| Runtime | ONNX Runtime | PyTorch | PyTorch |
| VRAM (GPU) | ~200MB | ~4-6GB | ~6-8GB |
| CPU real-time factor | ~7x (excellent) | ~0.3-0.5x (unusable for streaming) | ~0.1-0.2x (batch only) |
| GPU real-time factor | N/A | ~3-5x | ~2-3x |
| Streaming | Sentence-chunk (sync) | Token-level autoregressive | Token-level autoregressive |
| Voice clone | No | Yes (3s ref audio) | Yes (3s ref audio) |
| Fine-tune | No | Yes (official) | Yes (official) |
| Voice design | No | Yes (text description) | Yes (text description) |
| Languages | 9 (preset voices) | 10 | 10 |
| Instruction control | No | No (Base model) | Yes (CustomVoice) |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| Key deps | `kokoro-onnx` | `qwen-tts`, `torch`, `torchaudio` | Same |

### CPU Verdict

Qwen3-TTS on CPU generates audio **slower than playback speed**. This makes real-time streaming impossible. For CPU deployments, Qwen3 can still be used in **batch mode** (generate full audio → deliver as chunked response), but the latency will be seconds, not milliseconds.

**Recommendation**: Expose Qwen3 CPU as `stream=false` only. If `stream=true` is requested and device is CPU, return `422` with guidance to use Kokoro or switch to GPU.

---

## 3. Architecture

### 3.1 Backend Protocol

```
src/e_voice/tts/
├── protocol.py          # TTSBackend Protocol + VoiceInfo model
├── kokoro_backend.py    # Existing Kokoro implementation (refactored)
├── qwen3_backend.py     # New Qwen3-TTS backend
├── factory.py           # Backend factory (config-driven)
├── voice_store.py       # Cloned voice persistence (Qwen3 only)
└── __init__.py
```

### 3.2 Protocol Definition

```python
# src/e_voice/tts/protocol.py

from typing import Protocol, AsyncIterator, runtime_checkable
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class VoiceInfo:
    """Voice metadata exposed via GET /v1/audio/voices."""
    id: str
    name: str
    language: str
    gender: str | None = None
    is_cloned: bool = False
    preview_url: str | None = None


@dataclass(slots=True, frozen=True)
class TTSResult:
    """Single synthesis result."""
    audio: bytes          # raw PCM16-LE
    sample_rate: int
    duration_ms: float


@runtime_checkable
class TTSBackend(Protocol):
    """Contract every TTS backend must fulfill."""

    @property
    def name(self) -> str: ...

    @property
    def sample_rate(self) -> int: ...

    @property
    def supports_streaming(self) -> bool: ...

    @property
    def supports_voice_clone(self) -> bool: ...

    async def load(self) -> None:
        """Initialize model, download weights if needed."""
        ...

    async def unload(self) -> None:
        """Release GPU/CPU resources."""
        ...

    async def synthesize(
        self,
        text: str,
        voice: str,
        *,
        speed: float = 1.0,
        language: str | None = None,
    ) -> TTSResult:
        """Full synthesis — returns complete audio."""
        ...

    async def stream_chunks(
        self,
        text: str,
        voice: str,
        *,
        speed: float = 1.0,
        language: str | None = None,
        chunk_size: int = 8,
    ) -> AsyncIterator[bytes]:
        """Yield PCM16 audio chunks progressively."""
        ...

    async def list_voices(self) -> list[VoiceInfo]:
        """Return available voices for this backend."""
        ...
```

### 3.3 Backend Factory

```python
# src/e_voice/tts/factory.py

from e_voice.core.settings import settings as st


async def create_tts_backend() -> TTSBackend:
    """Instantiate the configured TTS backend. O(1)."""
    match st.TTS_BACKEND:
        case "kokoro":
            from e_voice.tts.kokoro_backend import KokoroBackend
            backend = KokoroBackend(device=st.TTS_DEVICE)

        case "qwen3":
            try:
                from e_voice.tts.qwen3_backend import Qwen3Backend
            except ImportError as exc:
                msg = (
                    "Qwen3 backend requires optional deps. "
                    "Install with: uv pip install e-voice[qwen]"
                )
                raise RuntimeError(msg) from exc
            backend = Qwen3Backend(
                model_name=st.QWEN3_MODEL,
                device=st.TTS_DEVICE,
                dtype=st.QWEN3_DTYPE,
            )

        case _:
            raise ValueError(f"Unknown TTS backend: {st.TTS_BACKEND}")

    await backend.load()
    return backend
```

### 3.4 Streaming Strategy (per backend)

#### Kokoro (unchanged)

```
Input text
    │
    ▼
Sentence splitter (misaki/regex)
    │
    ▼
┌─────────────────────────┐
│ For each sentence:       │
│   ONNX forward pass     │  ← ~5-15ms per sentence (CPU)
│   Yield PCM chunk        │
└─────────────────────────┘
    │
    ▼
Transport layer (WS/SSE/chunked)
```

**Latency to first chunk**: ~50ms (CPU), ~20ms (GPU).

#### Qwen3 GPU — Token Streaming

```
Input text
    │
    ▼
Tokenizer (qwen-tts)
    │
    ▼
┌──────────────────────────────────────┐
│ Autoregressive LM generation          │
│   ▼                                   │
│ Every N tokens (chunk_size=8):        │
│   Code2Wav decode (12Hz tokenizer)   │  ← ~8-12ms per chunk
│   Yield PCM chunk (~640ms of audio)  │
└──────────────────────────────────────┘
    │
    ▼
Transport layer (WS/SSE/chunked)
```

**Latency to first chunk**: ~97ms (GPU). Real-time factor: ~3-5x.

#### Qwen3 CPU — Batch Fallback

```
Input text
    │
    ▼
Full synthesis (non-streaming)        ← 2-10s depending on length
    │
    ▼
Split output audio into fixed-size chunks (e.g. 4096 samples)
    │
    ▼
Yield chunks as "simulated streaming"
    │
    ▼
Transport layer (WS/SSE/chunked)
```

**Latency to first chunk**: 2-10s. NOT real-time. Acceptable for HTTP batch responses.

---

## 4. Configuration

### 4.1 config.yaml additions

```yaml
tts:
  backend: kokoro              # kokoro | qwen3
  device: auto                 # gpu | cpu | auto
  default_voice: af_heart

  # Kokoro-specific (existing)
  kokoro:
    model_path: null           # auto-download if null

  # Qwen3-specific (new)
  qwen3:
    model: "Qwen/Qwen3-TTS-12Hz-0.6B-Base"    # HF id or local path
    dtype: bfloat16            # bfloat16 (GPU) | float32 (CPU)
    attn: auto                 # auto | flash_attention_2 | sdpa | eager
    max_new_tokens: 2048
    voice_store_dir: data/voices    # cloned voice prompts persist here
    cpu_streaming: false       # if true, allows (slow) batch-chunked delivery on CPU
```

### 4.2 Settings model (pydantic-settings)

```python
# In existing settings hierarchy

class Qwen3Settings(BaseSettings):
    """Qwen3-TTS backend configuration."""
    QWEN3_MODEL: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    QWEN3_DTYPE: str = "bfloat16"
    QWEN3_ATTN: str = "auto"
    QWEN3_MAX_NEW_TOKENS: int = 2048
    QWEN3_VOICE_STORE_DIR: Path = Path("data/voices")
    QWEN3_CPU_STREAMING: bool = False
```

### 4.3 pyproject.toml — Optional Dependencies

```toml
[project.optional-dependencies]
qwen = [
    "qwen-tts>=0.2.0",
    "torch>=2.5.1",
    "torchaudio>=2.5.1",
    "soundfile>=0.12.1",
]
# For streaming acceleration (GPU only):
qwen-fast = [
    "e-voice[qwen]",
    "faster-qwen3-tts>=0.1.0",
]
```

Core deps remain untouched — `kokoro-onnx`, `onnxruntime`, etc.

---

## 5. Qwen3 Backend Implementation Guide

### 5.1 Model Loading

```python
# src/e_voice/tts/qwen3_backend.py — load() sketch

async def load(self) -> None:
    import torch
    from qwen_tts import Qwen3TTSModel

    # Resolve dtype
    match self._dtype_str:
        case "bfloat16" if self._device != "cpu":
            dtype = torch.bfloat16
        case _:
            dtype = torch.float32

    # Resolve attention
    match self._attn:
        case "auto":
            attn = self._detect_best_attn()
        case other:
            attn = other

    self._model = Qwen3TTSModel.from_pretrained(
        self._model_name,
        device_map=self._device,
        dtype=dtype,
        attn_implementation=attn,
    )
    self._loaded = True
```

### 5.2 Streaming Synthesis (GPU)

Two options, pick one based on deps:

**Option A — `faster-qwen3-tts` (preferred, best perf)**:

```python
async def stream_chunks(self, text, voice, *, speed=1.0, language=None, chunk_size=8):
    if not self._is_gpu:
        raise TTSStreamingUnavailable("Qwen3 streaming requires GPU")

    from faster_qwen3_tts import FasterQwen3TTS

    # FasterQwen3TTS wraps the same checkpoint
    for pcm_chunk in self._model.generate_custom_voice_streaming(
        text=text,
        speaker=voice,
        language=language or "English",
        chunk_size=chunk_size,
    ):
        yield self._to_pcm16(pcm_chunk)
```

**Option B — `qwen3-tts-streaming` fork (simpler)**:

```python
async def stream_chunks(self, text, voice, *, speed=1.0, language=None, chunk_size=8):
    for pcm_chunk in self._model.stream_generate_pcm(
        text=text,
        speaker=voice,
        language=language or "English",
    ):
        yield self._to_pcm16(pcm_chunk)
```

### 5.3 Voice Clone Flow

```python
# src/e_voice/tts/voice_store.py

async def clone_voice(
    self,
    voice_id: str,
    ref_audio_path: Path,
    ref_text: str,
) -> VoiceInfo:
    """Create a reusable cloned voice prompt. O(n) on audio length."""

    # Generate voice clone prompt (extracts x-vector + codec tokens)
    prompt = self._model.create_voice_clone_prompt(
        ref_wav=str(ref_audio_path),
        ref_text=ref_text,
    )

    # Persist as safetensors for reuse
    save_path = self._voice_dir / f"{voice_id}.safetensors"
    torch.save(prompt, save_path)

    return VoiceInfo(
        id=voice_id,
        name=voice_id,
        language="multilingual",
        is_cloned=True,
    )
```

### 5.4 Using Cloned / Fine-tuned Voices

```python
async def synthesize(self, text, voice, *, speed=1.0, language=None):
    # Check if voice is a cloned prompt
    clone_path = self._voice_dir / f"{voice}.safetensors"

    if clone_path.exists():
        # Load persisted clone prompt
        voice_clone_prompt = torch.load(clone_path, weights_only=True)
        wavs, sr = self._model.generate_voice_clone(
            text=text,
            voice_clone_prompt=voice_clone_prompt,
            language=language or "English",
        )
    else:
        # Use as speaker name (preset or fine-tuned)
        wavs, sr = self._model.generate_custom_voice(
            text=text,
            speaker=voice,
            language=language or "English",
        )

    return TTSResult(audio=self._to_pcm16(wavs[0]), sample_rate=sr, ...)
```

---

## 6. API Surface Changes

### 6.1 New Endpoints

| Method | Endpoint | Description | Backend |
|---|---|---|---|
| `POST` | `/v1/audio/voices/clone` | Clone voice from ref audio | qwen3 only |
| `DELETE` | `/v1/audio/voices/:voice_id` | Delete cloned voice | qwen3 only |
| `GET` | `/v1/audio/voices` | List voices (extended) | all |

### 6.2 Voice Clone Endpoint

```
POST /v1/audio/voices/clone
Content-Type: multipart/form-data

Fields:
  voice_id    string    required    Unique identifier for the cloned voice
  file        binary    required    Reference audio (3-25s, WAV/MP3/FLAC)
  ref_text    string    required    Transcript of the reference audio
  language    string    optional    ISO code (auto-detect if omitted)

Response: 201 Created
{
  "id": "my_voice",
  "name": "my_voice",
  "language": "multilingual",
  "is_cloned": true
}
```

### 6.3 Extended Speech Request

The existing `POST /v1/audio/speech` body gains an optional field:

```json
{
  "input": "Hola mundo.",
  "model": "qwen3",
  "voice": "my_cloned_voice",
  "language": "Spanish",
  "response_format": "wav",
  "speed": 1.0,
  "stream": true
}
```

`language` is new — Kokoro ignores it (inferred from voice prefix). Qwen3 uses it for explicit language routing.

### 6.4 Error Handling

| Condition | Status | Body |
|---|---|---|
| `stream=true` + qwen3 + CPU | `422` | `{"error": "Qwen3 streaming requires GPU. Use stream=false or switch to kokoro backend."}` |
| Voice clone on kokoro backend | `422` | `{"error": "Voice cloning requires qwen3 backend."}` |
| Qwen3 deps not installed | `503` | `{"error": "Qwen3 backend unavailable. Install with: uv pip install e-voice[qwen]"}` |
| Clone ref audio too short (<2s) | `400` | `{"error": "Reference audio must be at least 2 seconds."}` |

---

## 7. Refactoring Existing Kokoro Code

The current Kokoro implementation lives inline in the TTS handlers. Extract it into `KokoroBackend` implementing `TTSBackend`:

### Key Changes

1. **Extract** all `kokoro-onnx` calls into `KokoroBackend.synthesize()` and `stream_chunks()`.
2. **Move** voice listing logic into `KokoroBackend.list_voices()`.
3. **Handler functions** become backend-agnostic — they call `app.state.tts_backend.synthesize(...)` instead of Kokoro directly.
4. **Sentence splitting** stays inside KokoroBackend (it's a Kokoro-specific streaming strategy).

### Before (handler, conceptual)

```python
async def handle_tts(request):
    kokoro = app.state.kokoro_model
    audio = kokoro.create(text, voice, speed)
    return Response(audio, content_type="audio/wav")
```

### After (handler, backend-agnostic)

```python
async def handle_tts(request):
    backend: TTSBackend = app.state.tts_backend
    result = await backend.synthesize(text, voice, speed=speed)
    return Response(result.audio, content_type="audio/wav")
```

Streaming handlers similarly delegate to `backend.stream_chunks()`.

---

## 8. Docker Strategy

### 8.1 Multi-stage Images

```dockerfile
# ── Base (shared) ─────────────────────
FROM python:3.13-slim AS base
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

# ── CPU image (Kokoro only, default) ──
FROM base AS cpu
# No torch, no CUDA — small image (~800MB)
CMD ["python", "-m", "e_voice"]

# ── GPU image (Kokoro + Qwen3) ────────
FROM nvidia/cuda:12.4-runtime AS gpu
WORKDIR /app
COPY --from=base /app/.venv /app/.venv
RUN uv pip install e-voice[qwen]
CMD ["python", "-m", "e_voice"]
```

### 8.2 Compose profiles

```yaml
services:
  evoice-cpu:
    image: ghcr.io/damvolkov/e-voice:cpu
    profiles: ["cpu"]
    ports:
      - "${EVOICE_PORT:-5500}:80"

  evoice-gpu:
    image: ghcr.io/damvolkov/e-voice:gpu
    profiles: ["gpu"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "${EVOICE_PORT:-5500}:80"
    volumes:
      - voices:/app/data/voices    # persisted cloned voices
```

---

## 9. Dependency Tree Comparison

### CPU install (default)

```
e-voice
├── robyn
├── kokoro-onnx
│   └── onnxruntime
├── faster-whisper (STT)
├── pydantic-settings
└── (standard lib)
```

**Total**: ~200MB installed.

### GPU install (with Qwen3)

```
e-voice[qwen]
├── (all CPU deps)
├── qwen-tts
│   ├── torch (~2GB)
│   ├── torchaudio
│   ├── transformers
│   └── soundfile
└── faster-qwen3-tts (optional, for CUDA graph streaming)
```

**Total**: ~4GB installed (torch dominates).

---

## 10. Implementation Roadmap

### Phase 1 — Protocol & Refactor (no new deps)

1. Define `TTSBackend` protocol in `protocol.py`.
2. Extract current Kokoro code into `KokoroBackend` implementing the protocol.
3. Create `factory.py` with config-driven backend instantiation.
4. Refactor all TTS handlers to use `app.state.tts_backend` (generic).
5. Verify all 4 transports (HTTP, SSE, WS, chunked) work unchanged.

**Risk**: Zero — pure refactor, Kokoro behavior identical.

### Phase 2 — Qwen3 Backend (batch mode)

1. Implement `Qwen3Backend` with `synthesize()` (non-streaming).
2. Add `list_voices()` with preset + cloned voice resolution.
3. Add voice clone endpoint + `VoiceStore` persistence.
4. Add `qwen` optional dependency group to `pyproject.toml`.
5. Config schema additions + validation.

**Deliverable**: Qwen3 works for HTTP batch requests (no streaming yet).

### Phase 3 — Qwen3 Streaming (GPU)

1. Implement `stream_chunks()` using `faster-qwen3-tts` or streaming fork.
2. Add device-aware guard (GPU → stream, CPU → 422 or batch fallback).
3. Verify WS + SSE streaming with Qwen3.
4. Benchmark TTFA and RTF on target hardware.

**Deliverable**: Full streaming parity with Kokoro on GPU.

### Phase 4 — Docker & CI

1. Multi-stage Dockerfile (cpu / gpu variants).
2. GitHub Actions matrix build for both images.
3. Compose profiles.
4. Smoke tests per backend in CI.

---

## 11. Testing Strategy

### Unit Tests

```
tests/unit/tts/
├── test_protocol.py           # Protocol compliance checks
├── test_kokoro_backend.py     # Kokoro synthesis + streaming
├── test_qwen3_backend.py      # Qwen3 synthesis (mocked model)
├── test_factory.py            # Backend instantiation logic
└── test_voice_store.py        # Clone persistence CRUD
```

Qwen3 unit tests **mock** the model — no torch in CI for unit tests.

### Integration Tests

```
tests/integration/tts/
├── test_kokoro_e2e.py         # Real Kokoro synthesis
├── test_qwen3_e2e.py          # Real Qwen3 synthesis (GPU CI only)
└── test_transport_agnostic.py # Same request → HTTP/SSE/WS/chunked
```

Mark Qwen3 integration tests with `@pytest.mark.gpu`.

### Key Assertions

- `stream_chunks()` yields valid PCM16-LE (parseable by soundfile).
- Cloned voice produces audio with speaker similarity > threshold.
- Backend switch via config does not affect transport behavior.
- CPU + Qwen3 + `stream=true` → `422`.

---

## 12. Open Questions & Decisions

| # | Question | Options | Recommendation |
|---|---|---|---|
| 1 | Which streaming fork? | `faster-qwen3-tts` vs `qwen3-tts-streaming` | `faster-qwen3-tts` — CUDA graphs, better TTFA, no `torch.compile` dep |
| 2 | 0.6B or 1.7B default? | 0.6B lighter, 1.7B better quality | 0.6B as default, 1.7B configurable |
| 3 | CPU batch fallback? | Allow slow batch delivery or hard-reject streaming | Config flag `cpu_streaming: false` (default reject) |
| 4 | Voice clone format? | safetensors vs pickle | safetensors (safe, fast, no arbitrary code exec) |
| 5 | Fine-tuned model loading? | Same `model` field points to local path | Yes — `qwen3.model: "./data/models/my-voice"` |
| 6 | Sample rate normalization? | Kokoro=24kHz, Qwen3=24kHz | Both 24kHz — no resampling needed |
| 7 | VoiceDesign model support? | Adds text-described voice creation | Phase 5 (future) — separate model download, separate config |

---

## 13. File Manifest

New and modified files:

```
src/e_voice/
├── tts/
│   ├── __init__.py              # (existing, updated exports)
│   ├── protocol.py              # NEW — TTSBackend protocol + models
│   ├── kokoro_backend.py        # NEW — extracted from existing handlers
│   ├── qwen3_backend.py         # NEW — Qwen3-TTS implementation
│   ├── factory.py               # NEW — config-driven instantiation
│   └── voice_store.py           # NEW — cloned voice persistence
├── core/
│   └── settings.py              # MODIFIED — add Qwen3Settings
├── routers/
│   ├── tts_router.py            # MODIFIED — backend-agnostic handlers
│   └── voice_router.py          # NEW — clone/delete voice endpoints
└── app.py                       # MODIFIED — factory in lifespan

pyproject.toml                   # MODIFIED — optional deps [qwen]
data/config/config.yaml          # MODIFIED — qwen3 section
docker/Dockerfile.gpu            # NEW — GPU image with torch
compose.yml                      # MODIFIED — profiles
tests/unit/tts/                  # NEW — backend tests
tests/integration/tts/           # NEW — e2e tests
```

Estimated: **~8 new files**, **~5 modified files**, **0 deleted files**.
