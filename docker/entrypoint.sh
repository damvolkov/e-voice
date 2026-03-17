#!/bin/sh
set -e

nginx -g "daemon on;"
exec python -m e_voice
