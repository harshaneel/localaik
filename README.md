# localaik

[CI](https://github.com/harshaneel/localaik/actions/workflows/release.yml)
[Docker Hub](https://hub.docker.com/r/gokhalh/localaik)
[License: MIT](LICENSE)
[Go Report Card](https://goreportcard.com/report/github.com/harshaneel/localaik)
[Go Version](https://github.com/harshaneel/localaik/blob/main/go.mod)
[Go Reference](https://pkg.go.dev/github.com/harshaneel/localaik)

A local compatibility server for the Gemini and OpenAI APIs. Run one container, point your SDK at `http://localhost:8090`, and get both protocol shapes on the same port for tests and development.

## Motivation

Testing code that calls Gemini or OpenAI is painful: real API calls are slow, cost money, and need network access. localaik gives you a single Docker container that speaks both protocols backed by a local model — no API key, no internet, deterministic enough for CI.

## Architecture

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

SDK requests hit the localaik proxy, which translates Gemini or OpenAI wire format and forwards to the local llama.cpp server running a Gemma 3 model.

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

More runnable samples (curl, Go, Python, JavaScript, Java) live under **[examples/](examples/README.md)**.

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

## Tuning (v0.1.3 onwards)

Pass environment variables to tune the underlying model server:

```bash
docker run -d -p 8090:8090 \
  -e LK_THREADS=8 \
  -e LK_CTX_SIZE=4096 \
  -e LK_FLASH_ATTN=1 \
  -e LK_CONT_BATCHING=1 \
  -e LK_PARALLEL=2 \
  gokhalh/localaik
```

Or with Docker Compose:

```yaml
services:
  localaik:
    image: gokhalh/localaik
    ports:
      - "8090:8090"
    environment:
      LK_THREADS: 8
      LK_CTX_SIZE: 4096
      LK_FLASH_ATTN: 1
      LK_CONT_BATCHING: 1
      LK_PARALLEL: 2
```


| Variable           | Default         | Description                         |
| ------------------ | --------------- | ----------------------------------- |
| `LK_CTX_SIZE`      | 8192            | Context window in tokens            |
| `LK_THREADS`       | auto            | CPU threads for inference           |
| `LK_THREADS_BATCH` | same as threads | CPU threads for prompt processing   |
| `LK_BATCH_SIZE`    | 2048            | Prompt processing batch size        |
| `LK_UBATCH_SIZE`   | 512             | Micro-batch size                    |
| `LK_GPU_LAYERS`    | 0               | Layers offloaded to GPU (99 = all)  |
| `LK_PARALLEL`      | 1               | Max concurrent request slots        |
| `LK_FLASH_ATTN`    | 0 (off)         | Flash attention (`1` to enable)     |
| `LK_CONT_BATCHING` | 0 (off)         | Continuous batching (`1` to enable) |
| `LK_MLOCK`         | 0 (off)         | Lock model in RAM (`1` to enable)   |


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

## Use in CI

Run localaik as a GitHub Actions service container so your tests hit a real local model instead of mocks:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      localaik:
        image: gokhalh/localaik
        ports:
          - 8090:8090
        options: >-
          --health-cmd "curl -f http://localhost:8090/health"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 30
    steps:
      - uses: actions/checkout@v4
      - run: go test ./...
        env:
          GOOGLE_GEMINI_BASE_URL: http://localhost:8090
          OPENAI_BASE_URL: http://localhost:8090/v1
```

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

> **Tip:** Run `make docker-up` to build and start the localaik container, which includes a local llama.cpp server with a bundled model. This is the easiest way to get a working upstream for development.

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

