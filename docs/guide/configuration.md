# Configuration

`evoice.toml` in the working directory (or `--config path`), then `EVOICE_<SECTION>__<KEY>` environment
overrides, e.g. `EVOICE_STT__PIPELINE__ASR__CHUNK=560ms`.

The configuration is a contract:

- every closed choice is an enum — an unknown backend fails to load;
- every table rejects unknown keys;
- values are range-checked (threads ≥ 1, thresholds in range, deadlines positive, backend-specific keys
  only with their backend);
- a configuration that cannot run never starts.

## Sections

| Section | Governs |
|---|---|
| `[server]` | bind address, upload cap, emotion exposure, logging |
| `[stt]` | default language |
| `[stt.pipeline]` | backpressure, stall timeout, tick, pre-roll, concurrent jobs, gate, gain |
| `[stt.pipeline.denoise]` | speech enhancement before gain and VAD |
| `[stt.pipeline.ww]` | wake word gate |
| `[stt.pipeline.vad]` | segmentation |
| `[stt.pipeline.lid]` | per-segment language identification |
| `[stt.pipeline.asr]` | live and file ASR |
| `[stt.pipeline.ser]` | emotion recognition |
| `[stt.ops]` | data root, model manifest, verification depth |
| `[tts]` | reserved |

## Every key with its default

The template below is `evoice.example.toml`; a test asserts it equals the built-in defaults.

```toml
--8<-- "evoice.example.toml"
```

## Notes per node

`asr.backend`
:   Streaming backends (`nemotron`, `kroko`) emit partials while speech lasts; batch backends
    (`parakeet`, `canary`, `cohere`, `whisper`) transcribe each segment when it ends.
    `asr.offline.backend` picks the engine for uploaded files; `live` reuses the live backend.
    `asr.offline.choices` loads further engines that a request selects by its model field (see
    [choosing the engine](../api/index.md#choosing-the-engine)).

`lid`
:   Whisper tiny identifies each segment's language before batch decoding. Canary and Cohere then decode
    in that language, and the final reports it. A detection outside es/en, or a segment shorter than
    `min`, keeps the requested language. Streaming backends fix their language at speech onset.

`denoise`
:   GTCRN runs on every frame before gain and VAD; it delays audio by one model frame.

`ser.backend`
:   `emotion2vec-large` needs the local export (`make export`). `off` removes the cost; the per-request
    `emotion=off` only hides the field.
