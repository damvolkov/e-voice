#!/usr/bin/env bash
# Both services in one container. `pull` installs both model sets; `serve` runs STT (:5500) and TTS
# (:5600) and exits with the first one that stops, so the container restarts as a whole. tini -g
# forwards signals to both.
set -euo pipefail

case "${1:-serve}" in
    pull) e-voice pull && e-voice-tts pull ;;
    serve)
        e-voice serve &
        e-voice-tts serve &
        wait -n
        ;;
    *) exec "$@" ;;
esac
