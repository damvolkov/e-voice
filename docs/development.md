# Development

## Layout

```
stt/                    the service (crate e-voice-stt, binary e-voice)
├── src/                api · config · core · schema · workflow · ops · main.rs
├── tests/              workflow · api · ops
├── ops/                sherpa.sh, bench.sh, bench/*.toml
└── models.toml
cli/                    ecli, the terminal tester
eval/                   Python (uv): dataset fetch, scoring, report; export/ scripts
docker/                 Dockerfile.{full,stt,tts} (+ .dockerignore each), compose.{full,stt,tts}.yml
docs/                   this site (Zensical); docs/history: dated decisions and measurements
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

## Docs

```bash
make docs        # e-voice openapi > docs/api/openapi.json, then zensical build → data/ops/site
make docs-serve  # live preview on :8000
```

The site is published to GitHub Pages by `.github/workflows/docs.yml` on every push to `main`.
