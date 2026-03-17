#!/bin/sh
set -e

# Start nginx in background
nginx -g "daemon on;"

# Start e-voice (API + Gradio)
exec python -m e_voice
