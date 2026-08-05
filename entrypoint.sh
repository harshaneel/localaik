#!/bin/sh
set -eu

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

LLAMA_PID=""
PROXY_PID=""

# Armed before either child exists, and iterating so an unset PID cannot make
# kill swallow the other one.
trap 'for p in ${LLAMA_PID} ${PROXY_PID}; do kill "${p}" 2>/dev/null || true; done; exit 0' INT TERM

# shellcheck disable=SC2086
"${LLAMA_SERVER_BIN}" ${SERVER_ARGS} &
LLAMA_PID=$!

localaik --port "${PORT:-8090}" --upstream "http://127.0.0.1:8080/v1" &
PROXY_PID=$!

echo "localaik: supervising; /health reports 503 until the model is ready"

while kill -0 "${LLAMA_PID}" 2>/dev/null && kill -0 "${PROXY_PID}" 2>/dev/null; do
  sleep 1
done

if kill -0 "${LLAMA_PID}" 2>/dev/null; then
  died="localaik"
  dead_pid="${PROXY_PID}"
  survivor="${LLAMA_PID}"
else
  died="llama-server"
  dead_pid="${LLAMA_PID}"
  survivor="${PROXY_PID}"
fi

kill "${survivor}" 2>/dev/null || true
# Waited so its last log lines reach docker logs before the container tears down.
wait "${survivor}" 2>/dev/null || true

status=0
wait "${dead_pid}" || status=$?
if [ "${status}" -eq 0 ]; then
  status=1
fi

echo "localaik: ${died} exited with status ${status}, stopping the container" >&2
exit "${status}"
