# localaik

[![CI](https://github.com/harshaneel/localaik/actions/workflows/release.yml/badge.svg)](https://github.com/harshaneel/localaik/actions/workflows/release.yml)
[![Docker Hub](https://img.shields.io/docker/v/gokhalh/localaik?sort=semver&label=Docker%20Hub)](https://hub.docker.com/r/gokhalh/localaik)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A local compatibility server for the Gemini and OpenAI APIs. Run one container, point your SDK at `http://localhost:8090`, and get both protocol shapes on the same port for tests and development.

## How it works

```
┌────────────────────────────────────────────────────────┐
│  localaik container                                    │
│                                                        │
│  ┌──────────────────────────┐    ┌──────────────────┐  │
│  │  localaik proxy (:8090)  │    │ llama.cpp (:8080)│  │
│  │                          │    │                  │  │
│  │  /v1beta/* (Gemini)  ────┼──▶ │  Gemma 3 model   │  │
│  │  /v1/*     (OpenAI)  ────┼──▶ │                  │  │
│  │                          │    └──────────────────┘  │
│  │                          │                          │
│  │                          │    ┌──────────────────┐  │
│  │  PDF uploads ────────────┼──▶ │    pdftoppm      │  │
│  │                          │    │  PDF ─▶ images   │  │
│  └──────────────────────────┘    └──────────────────┘  │
└────────────────────────────────────────────────────────┘
```

## Quick start

```bash
docker run -d -p 8090:8090 gokhalh/localaik
```

Or with Docker Compose:

```yaml
services:
  localaik:
    image: gokhalh/localaik
    ports:
      - "8090:8090"
```

localaik is a plain HTTP server, so any language or SDK that can set a base URL will work.

### Gemini SDK

**Go:**
```go
client, err := genai.NewClient(ctx, &genai.ClientConfig{
    APIKey:      "test",
    HTTPOptions: genai.HTTPOptions{BaseURL: "http://localhost:8090"},
})
```

**Python:**
```python
from google import genai

client = genai.Client(
    api_key="test",
    http_options=genai.types.HttpOptions(api_version="v1beta", base_url="http://localhost:8090"),
)
```

Or set the environment variable for any language:

```bash
export GOOGLE_GEMINI_BASE_URL=http://localhost:8090
```

### OpenAI SDK

**Python:**
```python
from openai import OpenAI

client = OpenAI(api_key="test", base_url="http://localhost:8090/v1")
```

**Go:**
```go
client := openai.NewClient(
    option.WithAPIKey("test"),
    option.WithBaseURL("http://localhost:8090/v1"),
)
```

## Docker tags


| Tag                   | Model              | Image size |
| --------------------- | ------------------ | ---------- |
| `latest`, `gemma3-4b` | Gemma 3 4B Q4_K_M  | ~3 GB      |
| `gemma3-12b`          | Gemma 3 12B Q4_K_M | ~7 GB      |


Version-pinned tags follow the pattern `v0.1.1-gemma3-4b`, `v0.1.1-gemma3-12b`.

## Implemented routes


| Route                                               | Used by                        | Notes                                   |
| --------------------------------------------------- | ------------------------------ | --------------------------------------- |
| `POST /v1beta/models/{model}:generateContent`       | Gemini `GenerateContent`       | Translated to upstream chat completions |
| `POST /v1beta/models/{model}:streamGenerateContent` | Gemini `GenerateContentStream` | Gemini-style SSE (typically `?alt=sse`) |
| `POST /v1/chat/completions`                         | OpenAI chat completions        | Forwarded to upstream                   |
| `GET /health`                                       | Health checks                  | Custom route                            |


All other API routes return `404`.

## Tested SDKs

Automated contract tests validate against:

- `google.golang.org/genai` v1.51.0
- `github.com/openai/openai-go/v3` v3.30.0

Other SDK versions and languages may work if they emit the same HTTP shapes.

## Gemini compatibility

**Supported features:**

- Text, image (`inlineData`), and PDF input (auto-converted to page images)
- `fileData` for image URLs and local/`data:`-URI PDF/text files
- `systemInstruction`
- `generationConfig`: temperature, topP, topK, candidateCount, maxOutputTokens, stopSequences, responseLogprobs, logprobs, presencePenalty, frequencyPenalty, seed
- Structured output via `responseMimeType`, `responseSchema`, `responseJsonSchema`
- Function declarations via `tools`, function calling config via `toolConfig`
- `functionCall` and `functionResponse` parts
- Streaming SSE responses
- Usage metadata and finish reasons

**Partial support:**

- `top_k`, `n`, logprobs, and tool choice behavior depends on the upstream runtime
- `executableCode`, `codeExecutionResult`, `toolCall`, `toolResponse` parts preserved as text context

**Not supported:**

- SDK methods outside `GenerateContent` / `GenerateContentStream`
- Non-function tools (Google Search, Maps, URL context, code execution)
- Embeddings, token counting, cached content, live/bidi sessions, uploads

## OpenAI compatibility

**Supported:** text chat completions, structured output, vision inputs, tool-related fields (all passed through to upstream).

**Not supported:** Responses API, Assistants, Embeddings, Images, Audio, Files, Vector stores.

## Development

```bash
# Run the proxy locally (requires a running llama.cpp server)
go run ./cmd/localaik --port 8090 --upstream http://127.0.0.1:8080/v1

# Common commands
make help              # Show all targets
make lint              # Format check + go vet
make test-unit         # Unit tests
make test-integration  # Integration tests (requires docker-up)
make test              # All of the above
make docker-up         # Build and start container
make docker-down       # Stop container
```

### Building the image

```bash
# Default (Gemma 3 4B)
docker build -t gokhalh/localaik .

# Custom model
docker build \
  --build-arg MODEL_URL=... \
  --build-arg MODEL_SHA256=... \
  --build-arg MMPROJ_URL=... \
  --build-arg MMPROJ_SHA256=... \
  -t gokhalh/localaik:custom .
```

## Limitations

- Intended for tests and development, not production
- Image size is dominated by model weights
- Cold starts can take tens of seconds while the model loads
- PDF rendering adds latency per page

