# Models

`stt/models.toml` pins every artifact by URL (Hugging Face by commit, GitHub releases) and sha256; archives
declare `unpack` and `strip`.

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
| `pipeline` (default) | `data/stt/models` | what a configured pipeline loads; the Docker volume |
| `ops` | `data/stt/ops/models` | tool and test models (Piper voices for wake calibration) |

## Local exports

Some models have no published ONNX. `make export` builds them from a pinned checkpoint into
`data/stt/ops/exports`, then installs them. Their manifest entries use `file://` sources.

emotion2vec+ large (`emotion2vec/emotion2vec_plus_large@6c303ba`) is exported with FunASR's exporter:
the backbone, with waveform normalization folded in, plus the `proj` head as JSON. The export runs as a
uv script (`eval/export/emotion2vec.py`) with a CPU-only torch, outside the eval package.

!!! note
    The digests pin this exporter's output; another torch/FunASR build may trace different bytes, which
    `pull` refuses. Publishing the export would make it reproducible.
