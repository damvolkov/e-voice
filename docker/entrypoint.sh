#!/bin/sh
set -e

# Seed default config if volume is empty or config missing
if [ ! -f /app/data/config/config.yaml ]; then
    cp /defaults/config.yaml /app/data/config/config.yaml
fi

nginx -g "daemon on;"
exec python -m e_voice.main
