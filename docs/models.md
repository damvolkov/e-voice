# Models

`evoice/stt/models.toml` and `evoice/tts/models.toml` pin every artifact by URL (Hugging Face by commit, GitHub
releases, or `file://` for local exports) and sha256; archives declare `unpack` and `strip`. The store
is `core`'s, shared by both services.

```bash
e-voice pull                  # the configured models (make setup)
e-voice pull canary-180m-flash-int8 whisper-tiny
e-voice pull --all            # the whole manifest
e-voice verify --full         # re-hash everything
```

How `pull` installs:

- it stages, hashes while downloading, unpacks, and moves into place atomically under a lock;
- it writes a per-file digest stamp;
- `serve` never downloads, and refuses to start on a missing or stale model (`verify = "stamp"` checks
  the stamp, `full` re-hashes).

| Scope | Directory | Contents |
|---|---|---|
| `pipeline` (default) | `data/<service>/models` | what a configured service loads; the Docker volume |
| `ops` | `data/<service>/ops/models` | tool and test models (Piper voices for wake calibration, the speaker model of the TTS bench) |

## Local exports

Some models have no published ONNX. `make export` builds them from a pinned checkpoint into
`data/stt/ops/exports`, then installs them. Their manifest entries use `file://` sources.

emotion2vec+ large (`emotion2vec/emotion2vec_plus_large@6c303ba`) is exported with FunASR's exporter:
the backbone, with waveform normalization folded in, plus the `proj` head as JSON. The export runs as a
uv script (`eval/export/emotion2vec.py`) with a CPU-only torch, outside the eval package.

!!! note
    The digests pin this exporter's output; another torch/FunASR build may trace different bytes, which
    `pull` refuses. Publishing the export would make it reproducible.

Pocket TTS has no ONNX for its current weights either. `make pocket` downloads the pinned checkpoint
(`kyutai/pocket-tts@3e82814`, gated — below) into `data/tts/ops/checkpoints`, exports every variant
with `lomotron/pocket-tts-onnx-export@8f52199` (five graphs, fp32 + dynamic int8) into
`data/tts/ops/exports`, writes a golden of the reference loop next to each bundle, and prints the
`evoice/tts/models.toml` entries. Downloads never touch the default Hugging Face cache: `HF_HOME` is
`data/ops/hf`.

| Model id | Checkpoint | Graphs |
|---|---|---|
| `pocket-es` | `languages/spanish` (6 layers) | 562 MB fp32 + int8 |
| `pocket-es-24l` | `languages/spanish_24l` | 24 layers |
| `pocket-en` | `languages/english_2026-09` (6 layers) | as `pocket-es` |

### Gated checkpoints

Pocket TTS (`kyutai/pocket-tts`) is gated on Hugging Face: its terms (CC-BY-4.0 weights; no voice
cloning without the speaker's consent) must be accepted once per account on the model page. The export
then needs a token of that same account, in `data/ops/hf/token` (`HF_HOME=data/ops/hf hf auth login`).
`make pocket` unsets `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN`, which would override that file. A token of an account that has not accepted the terms
fails with `Access denied ... requires approval`. Only the export touches the gated repo: the service
loads the exported bundle and never needs a token.
