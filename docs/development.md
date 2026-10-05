# Development

## Layout

```
evoice/                 the services and their shared core (Rust crates)
├── core/               crate e-voice-core: settings loader, model store, runtime, audio, logger, probe
│   ├── link.rs         build script of every binary (rpath to data/ops/sherpa/lib, version)
│   └── ops/sherpa.sh   vendors the pinned sherpa-onnx + onnxruntime libraries into data/ops/sherpa
├── stt/                speech to text (crate e-voice-stt, binary e-voice)
│   ├── src/            api · config · core · schema · workflow (session, runner, ww/ vad/ asr/ ser/) · ops
│   ├── tests/          workflow, api, ops
│   ├── ops/            bench.sh, bench/*.toml
│   └── models.toml
└── tts/                text to speech (crate e-voice-tts, binary e-voice-tts)
    ├── src/            api · config · core (voices, encode) · schema · workflow (session, runner, synth/) · ops
    ├── tests/          workflow (conformance, backends), api, ops
    ├── ops/            bench.sh, bench/*.toml
    └── models.toml
cli/                    ecli: microphone → STT, TTS → speaker
docker/                 Dockerfile.{full,stt,tts} (+ .dockerignore each), compose.{full,stt,tts}.yml
eval/                   Python (uv): dataset fetch, scoring, reports; export/ (model side only)
docs/                   this site (Zensical); docs/history: dated decisions and measurements
data/                   ignored: ops/{target,sherpa,hf}, stt/{models,ops}, tts/{models,voices,ops}
```

## Gates

| Command | Runs |
|---|---|
| `make check` | rustfmt, clippy (pedantic, `-D warnings`), layer bounds, tests without models |
| `make backends` | model-backed tests: every backend in isolation, the pipeline, the gateway, the tools |
| `make coverage` | every test with line coverage ≥ 90 % |
| `make hooks` | every prek hook — what CI runs (pre-push adds coverage) |
| `make docs` | the OpenAPI document, then this site |

The rules:

- unsafe code is forbidden;
- `unwrap`, `expect`, `panic`, indexing and unchecked arithmetic are denied outside tests;
- `if`/`else` is used only as an expression; `match` and typed dispatch carry decisions.

## Releases

Every image is versioned on its own: `e-voice-stt`, `e-voice-tts` and `e-voice-full` carry their own
`X.Y.Z`, tagged `stt-vX.Y.Z`, `tts-vX.Y.Z` and `full-vX.Y.Z`, each with its GitHub release. A release
names the service that changed; `full`, which ships both, is bumped with either:

```bash
gh workflow run release.yml -f service=stt -f bump=patch   # e-voice-stt and e-voice-full
gh workflow run release.yml -f service=tts -f bump=patch   # e-voice-tts and e-voice-full
gh workflow run release.yml -f service=both -f bump=minor  # all three
```

Images are pushed as `:X.Y.Z`, `:X.Y` and `:latest` when the `PUBLISH_IMAGE` gate is on; each binary
reports its version (`--version`, `/health`). v0.2.0 is the common baseline every service starts from.

## Docs

```bash
make docs        # e-voice openapi > docs/api/openapi.json, then zensical build → data/ops/site
make docs-serve  # live preview on :8000
```

The site is published to GitHub Pages by `.github/workflows/docs.yml` on every push to `main`.
