#!/bin/sh
set -e
DATA="${GPTLAB_DATA_DIR:-/data}"
mkdir -p "$DATA/runs"

# Seed demo checkpoints, but never overwrite a run that already exists.
if [ -d /app/demo/runs ]; then
  for d in /app/demo/runs/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    [ -e "$DATA/runs/$name" ] || cp -r "$d" "$DATA/runs/$name"
  done
fi

exec "$@"