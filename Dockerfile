FROM --platform=$BUILDPLATFORM golang:1.25-alpine AS proxy-builder
ARG TARGETOS TARGETARCH
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH go build -ldflags="-s -w" -o /out/localaik ./cmd/localaik

FROM alpine:3@sha256:28bd5fe8b56d1bd048e5babf5b10710ebe0bae67db86916198a6eec434943f8b AS proxy
RUN apk add --no-cache ca-certificates poppler-utils tini
COPY --from=proxy-builder /out/localaik /usr/local/bin/localaik
ENV PORT=8090
# No inference engine here, so the loopback default could never work.
ENV LK_REQUIRE_UPSTREAM=1
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s \
  CMD wget -q -O - "http://127.0.0.1:${PORT:-8090}/health" >/dev/null 2>&1 || exit 1
EXPOSE 8090
ENTRYPOINT ["tini", "--", "localaik"]

# Upstream does not ship semver for the server image; pin by digest for reproducible multi-arch builds.
# Logical tag at pin time: server (includes llama-server --mmproj for Gemma 3 vision). Bump digest to upgrade.
FROM ghcr.io/ggml-org/llama.cpp@sha256:80910e898e5d9a6b46ca9d1b4674d3e15faf6d32b9692eb6011ccd34b2cb8a06

ARG MODEL_URL="https://huggingface.co/lmstudio-community/gemma-3-4b-it-GGUF/resolve/c536c4707e747055eecad7da65d46b6fb0ebaa79/gemma-3-4b-it-Q4_K_M.gguf"
ARG MODEL_SHA256="be49949e48422e4547b00af14179a193d3777eea7fbbd7d6e1b0861304628a01"
ARG MMPROJ_URL="https://huggingface.co/lmstudio-community/gemma-3-4b-it-GGUF/resolve/d400f8ba80bfa661d94a756ea3b663db8b00da85/mmproj-model-f16.gguf"
ARG MMPROJ_SHA256="8c0fb064b019a6972856aaae2c7e4792858af3ca4561be2dbf649123ba6c40cb"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    poppler-utils \
    tini && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /models
RUN curl -L --fail --output /models/model.gguf "${MODEL_URL}" && \
    echo "${MODEL_SHA256}  /models/model.gguf" | sha256sum -c -
RUN curl -L --fail --output /models/mmproj-model-f16.gguf "${MMPROJ_URL}" && \
    echo "${MMPROJ_SHA256}  /models/mmproj-model-f16.gguf" | sha256sum -c -

COPY --from=proxy-builder /out/localaik /usr/local/bin/localaik
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PORT=8090

HEALTHCHECK --interval=5s --timeout=3s --start-period=180s \
  CMD curl -sf http://127.0.0.1:${PORT:-8090}/health || exit 1

EXPOSE 8090
ENTRYPOINT ["tini", "--", "/entrypoint.sh"]
