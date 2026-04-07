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

SERVER_ARGS="--model /models/model.gguf"
SERVER_ARGS="${SERVER_ARGS} --mmproj /models/mmproj-model-f16.gguf"
SERVER_ARGS="${SERVER_ARGS} --port 8080"
SERVER_ARGS="${SERVER_ARGS} --host 127.0.0.1"
SERVER_ARGS="${SERVER_ARGS} --ctx-size ${LK_CTX_SIZE:-8192}"

[ -n "${LK_THREADS:-}" ]       && SERVER_ARGS="${SERVER_ARGS} --threads ${LK_THREADS}"
[ -n "${LK_THREADS_BATCH:-}" ] && SERVER_ARGS="${SERVER_ARGS} --threads-batch ${LK_THREADS_BATCH}"
[ -n "${LK_BATCH_SIZE:-}" ]    && SERVER_ARGS="${SERVER_ARGS} --batch-size ${LK_BATCH_SIZE}"
[ -n "${LK_UBATCH_SIZE:-}" ]   && SERVER_ARGS="${SERVER_ARGS} --ubatch-size ${LK_UBATCH_SIZE}"
[ -n "${LK_GPU_LAYERS:-}" ]    && SERVER_ARGS="${SERVER_ARGS} --n-gpu-layers ${LK_GPU_LAYERS}"
[ -n "${LK_PARALLEL:-}" ]      && SERVER_ARGS="${SERVER_ARGS} --parallel ${LK_PARALLEL}"

[ "${LK_FLASH_ATTN:-0}" = "1" ]    && SERVER_ARGS="${SERVER_ARGS} --flash-attn"
[ "${LK_CONT_BATCHING:-0}" = "1" ] && SERVER_ARGS="${SERVER_ARGS} --cont-batching"
[ "${LK_MLOCK:-0}" = "1" ]         && SERVER_ARGS="${SERVER_ARGS} --mlock"

# shellcheck disable=SC2086
"${LLAMA_SERVER_BIN}" ${SERVER_ARGS} &

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
