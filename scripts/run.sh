#!/usr/bin/env bash
# Convenience launcher. Loads .env if present, then runs the API.
set -e
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

exec python -m backend.api.app
