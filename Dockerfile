FROM --platform=$BUILDPLATFORM golang:1.25-alpine AS proxy-builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /out/localaik ./cmd/localaik

FROM ghcr.io/ggml-org/llama.cpp:server

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

HEALTHCHECK --interval=5s --timeout=3s --start-period=60s \
  CMD curl -sf http://127.0.0.1:${PORT:-8090}/health || exit 1

EXPOSE 8090
ENTRYPOINT ["tini", "--", "/entrypoint.sh"]
