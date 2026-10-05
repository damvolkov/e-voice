# Docker

Everything container-related lives in `docker/`: one two-stage Dockerfile and one compose file per
deployment.

| Mode | Dockerfile | Compose | Image | Port |
|---|---|---|---|---|
| `full` (default) | `Dockerfile.full` | `compose.full.yml` | `e-voice:full` | 5500 + 5600 |
| `stt` | `Dockerfile.stt` | `compose.stt.yml` | `e-voice:stt` | 5500 |
| `tts` | `Dockerfile.tts` | `compose.tts.yml` | `e-voice:tts` | 5600 |

```bash
make up                  # MODE=full: STT and TTS in one container
make up MODE=stt         # STT alone
make up MODE=tts         # TTS alone (its Pocket bundles come from `make pocket`)
make up MODE=split       # stt + tts side by side
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

## TTS

Images run TTS on **Qwen3** (`EVOICE_TTS__SYNTH__BACKEND=qwen3`, 2 workers × 4 threads): its weights are
public, so `pull` installs them like any STT model. Pocket's bundles are local exports of a gated
checkpoint and are never downloaded inside a container: run `make pocket` on the host and set
`EVOICE_TTS_BACKEND=pocket`; the pull service mounts `data/tts/ops/exports` read-only and installs them.
Learned voices live in the `e-voice-tts` volume (`/data/tts/voices`), learned through `POST /v1/voices`;
`EVOICE_TTS_VOICE=<id>` sets the default voice, `EVOICE_TTS_PORT` the port.

`full` runs both binaries under tini (`docker/full.sh`): `pull` installs both model sets, `serve` starts
both and stops the container when either exits; the healthcheck probes both ports.

## Published images

The release workflow publishes `ghcr.io/damvolkov/e-voice-stt`, `-tts` and `-full` (`:X.Y.Z`, `:X.Y`,
`:latest`) when the `PUBLISH_IMAGE` gate is on.
