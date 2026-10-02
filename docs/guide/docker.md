# Docker

Everything container-related lives in `docker/`: one two-stage Dockerfile and one compose file per
deployment.

| Mode | Dockerfile | Compose | Image | Port |
|---|---|---|---|---|
| `full` (default) | `Dockerfile.full` | `compose.full.yml` | `e-voice:full` | 5500 |
| `stt` | `Dockerfile.stt` | `compose.stt.yml` | `e-voice:stt` | 5500 |
| `tts` (template) | `Dockerfile.tts` | `compose.tts.yml` | `e-voice:tts` | 5600 |

```bash
make up                  # MODE=full: every service in one container
make up MODE=stt         # STT alone
make up MODE=split       # stt + tts side by side (tts is a template until its crate exists)
make image MODE=…        # build only
make down MODE=…
```

## Build

The builder stage (cargo-chef on Debian trixie) runs the same Makefile targets as a workstation —
`make system`, `make sherpa`, `make dist` — so there is one definition of every step. The runtime stage
is `debian:trixie-slim` with `ca-certificates` and `tini` only, a non-root `evoice` user (uid 10001), and
a healthcheck on `/health`. Each Dockerfile has its own `.dockerignore` beside it.

## Models

A one-shot `pull` service installs exactly the configured models into the `e-voice-stt` volume before the
service starts; `serve` never downloads. Use a host directory instead with an absolute path:

```bash
EVOICE_STT_DATA=/srv/e-voice/stt make up
```

## Settings

Environment variables override any key (`EVOICE_<SECTION>__<KEY>`); compose passes `EVOICE_LANG` as
`EVOICE_STT__LANG`. A full file can be mounted at `/app/evoice.toml`.

```bash
EVOICE_PORT=5501 EVOICE_LANG=en make up
```
