#!/bin/sh
set -eu

cleanup() {
  jobs -p | xargs -r kill 2>/dev/null || true
  wait || true
}

trap 'cleanup; exit 0' INT TERM

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/app/llama-server}"
if [ ! -x "${LLAMA_SERVER_BIN}" ]; then
  if command -v llama-server >/dev/null 2>&1; then
    LLAMA_SERVER_BIN="$(command -v llama-server)"
  else
    echo "localaik: llama-server binary not found at ${LLAMA_SERVER_BIN} and not present in PATH" >&2
    exit 1
  fi
fi

"${LLAMA_SERVER_BIN}" \
  --model /models/model.gguf \
  --mmproj /models/mmproj-model-f16.gguf \
  --port 8080 \
  --host 127.0.0.1 \
  --ctx-size 8192 &

echo "localaik: loading model..."
tries=0
until curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; do
  tries=$((tries + 1))
  if [ "${tries}" -ge 120 ]; then
    echo "localaik: model failed to load after 120s" >&2
    cleanup
    exit 1
  fi
  sleep 1
done

echo "localaik: model ready"
echo "localaik: listening on port ${PORT:-8090}"
exec localaik \
  --port "${PORT:-8090}" \
  --upstream "http://127.0.0.1:8080/v1"
