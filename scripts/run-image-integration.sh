#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${IMAGE:-gokhalh/localaik:test}"
PORT="${PORT:-18090}"
CONTAINER_NAME="${CONTAINER_NAME:-localaik-image-integration}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-180}"
GO_TEST_FLAGS="${GO_TEST_FLAGS:--count=1}"

cleanup() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to run image integration tests" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to run image integration tests" >&2
  exit 1
fi

if [[ "${BUILD_IMAGE}" == "1" ]]; then
  docker build -t "${IMAGE}" "${ROOT_DIR}"
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run -d --name "${CONTAINER_NAME}" -p "${PORT}:8090" "${IMAGE}" >/dev/null

deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "localaik image did not become healthy within ${STARTUP_TIMEOUT_SECONDS}s" >&2
    docker logs "${CONTAINER_NAME}" >&2 || true
    exit 1
  fi
  sleep 2
done

cd "${ROOT_DIR}"
LOCALAIK_BASE_URL="http://127.0.0.1:${PORT}" GOCACHE="${GOCACHE:-/tmp/localaik-go-build-cache}" \
  go test ${GO_TEST_FLAGS} -tags=docker_integration ./integration -run '^TestImage'
