# localaik

[![CI](https://github.com/harshaneel/localaik/actions/workflows/release.yml/badge.svg)](https://github.com/harshaneel/localaik/actions/workflows/release.yml)
[![Docker Hub](https://img.shields.io/docker/v/gokhalh/localaik?sort=semver&label=Docker%20Hub)](https://hub.docker.com/r/gokhalh/localaik)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/harshaneel/localaik)](https://goreportcard.com/report/github.com/harshaneel/localaik)
[![Go Version](https://img.shields.io/github/go-mod/go-version/harshaneel/localaik)](https://github.com/harshaneel/localaik/blob/main/go.mod)
[![Go Reference](https://pkg.go.dev/badge/github.com/harshaneel/localaik.svg)](https://pkg.go.dev/github.com/harshaneel/localaik)

A local compatibility server for the Gemini, OpenAI, and Anthropic APIs. Run one container, point your SDK at `http://localhost:8090`, and get all three protocol shapes on the same port for tests and development.

## Motivation

Testing code that calls Gemini, OpenAI, or Anthropic is painful: real API calls are slow, cost money, and need network access. localaik gives you a single Docker container that speaks all three protocols backed by a local model, or the `proxy` tag if you already run your own model server. The model-bundled tags need no API key and no internet, and are deterministic enough for CI.

## Architecture

```
┌────────────────────────────────────────────────────────┐
│  localaik container                                    │
│                                                        │
│  ┌──────────────────────────┐    ┌──────────────────┐  │
│  │  localaik proxy (:8090)  │    │ llama.cpp (:8080)│  │
│  │                          │    │                  │  │
│  │  /v1beta/*    (Gemini)  ─┼──▶ │  Gemma 3 model   │  │
│  │  /v1/*        (OpenAI)  ─┼──▶ │                  │  │
│  │  /v1/messages (Anthropic)┼──▶ │                  │  │
│  │                          │    └──────────────────┘  │
│  │                          │                          │
│  │                          │    ┌──────────────────┐  │
│  │  PDF uploads ────────────┼──▶ │    pdftoppm      │  │
│  │                          │    │  PDF ─▶ images   │  │
│  └──────────────────────────┘    └──────────────────┘  │
└────────────────────────────────────────────────────────┘
```

SDK requests hit the localaik proxy, which translates Gemini, OpenAI, or Anthropic wire format and forwards to the local llama.cpp server running a Gemma 3 model.

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

### Anthropic SDK

The Anthropic SDKs append `v1/` themselves, so the base URL is the bare host.

**Python:**

```python
from anthropic import Anthropic

client = Anthropic(api_key="test", base_url="http://localhost:8090")
```

**Go:**

```go
client := anthropic.NewClient(
    option.WithAPIKey("test"),
    option.WithBaseURL("http://localhost:8090/"),
)
```

`max_tokens` is required by the Messages API, so localaik rejects requests without it exactly as the real endpoint does.

## Docker tags


| Tag                   | Model              | Image size |
| --------------------- | ------------------ | ---------- |
| `latest`, `gemma3-4b` | Gemma 3 4B Q4_K_M  | ~3 GB      |
| `gemma3-12b`          | Gemma 3 12B Q4_K_M | ~7 GB      |
| `proxy`               | none (you supply)  | ~41 MB     |


Version-pinned tags follow the pattern `v0.1.1-gemma3-4b`, `v0.1.1-gemma3-12b`,
`v0.1.1-proxy`. The `proxy` tag is never published as `latest`.

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


## Bring your own model server (`:proxy`)

If you already run llama.cpp, vLLM, or anything else that speaks the OpenAI
chat-completions API, the `proxy` tag gives you the translation layer alone. It
contains no model and no inference engine.

```bash
docker run -d -p 127.0.0.1:8090:8090 \
  -e LK_UPSTREAM=http://llama.internal:8080/v1 \
  gokhalh/localaik:proxy
```

| Env var | Default | Description |
| --- | --- | --- |
| `LK_UPSTREAM` | `http://127.0.0.1:8080/v1` | Base URL of your model server |
| `LK_UPSTREAM_AUTH_HEADER` | unset | A full header line sent to your server, for example `Authorization: Bearer abc123` |
| `PORT` | `8090` | Port localaik listens on |

`LK_UPSTREAM_AUTH_HEADER` is sent only to your upstream. Credentials that
clients send to localaik are still discarded and never forwarded. It is attached
only to requests whose host matches `LK_UPSTREAM`, and while it is set a
redirect from your upstream is returned to the caller rather than followed.

`/health` returns 503 until your upstream answers, so existing healthchecks and
CI wait loops work unchanged.

### Security

`:proxy` has a different risk profile from the model-bundled tags. Those keep
llama.cpp bound to localhost inside the container, so the only thing reachable
is a disposable local model. `:proxy` forwards into infrastructure you care
about, and localaik does not authenticate its callers by design.

**Anyone who can reach port 8090 can use your model server without
credentials.** Bind to localhost and do not publish the port on a shared
network. localaik is a testing tool, not a gateway.

## Implemented routes


| Route                                               | Used by                        | Notes                                   |
| --------------------------------------------------- | ------------------------------ | --------------------------------------- |
| `POST /v1beta/models/{model}:generateContent`       | Gemini `GenerateContent`       | Translated to upstream chat completions |
| `POST /v1beta/models/{model}:streamGenerateContent` | Gemini `GenerateContentStream` | Gemini-style SSE (typically `?alt=sse`) |
| `POST /v1beta/models/{model}:countTokens`           | Gemini `CountTokens`           | Translated to upstream `/tokenize`      |
| `GET /v1beta/models`                                | Gemini `Models.List`           | Translated from upstream `/v1/models`   |
| `GET /v1beta/models/{model}`                        | Gemini `Models.Get`            | Translated from upstream `/v1/models`   |
| `POST /v1/chat/completions`                         | OpenAI chat completions        | Forwarded to upstream                   |
| `POST /v1/completions`                              | OpenAI legacy completions      | Forwarded to upstream                   |
| `GET /v1/models`                                    | OpenAI `Models.List`           | Forwarded to upstream                   |
| `GET /v1/models/{model}`                            | OpenAI `Models.Retrieve`       | Forwarded to upstream                   |
| `POST /v1/messages`                                 | Anthropic `Messages.New`       | Translated to upstream chat completions |
| `POST /v1/messages/count_tokens`                    | Anthropic `Messages.CountTokens` | Translated to upstream `/tokenize`    |
| `GET /health`                                       | Health checks                  | Custom route                            |

`POST /v1/messages` streams when the request body sets `"stream": true`, matching
how the Anthropic SDKs signal streaming (there is no separate route).


All other API routes return `404`, in the error shape of whichever protocol owns
the path prefix.

Note that `/v1/` is shared: `GET /v1/models` is served as the OpenAI models list,
so an Anthropic client calling `client.models.list()` gets OpenAI-shaped data
rather than a `404`. Only `/v1/messages` and `/v1/messages/count_tokens` are
handled as Anthropic.

## Tested SDKs

Automated contract tests validate against:

- `google.golang.org/genai` v1.57.0
- `github.com/openai/openai-go/v3` v3.36.0
- `github.com/anthropics/anthropic-sdk-go` v1.61.0

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
          ANTHROPIC_BASE_URL: http://localhost:8090
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

- SDK methods outside `GenerateContent` / `GenerateContentStream` / `CountTokens` / `Models.List` / `Models.Get`
- Non-function tools (Google Search, Maps, URL context, code execution)
- Embeddings, cached content, live/bidi sessions, uploads
- `ComputeTokens` is Vertex-only in the SDK and not exposed on `BackendGeminiAPI`

## OpenAI compatibility

**Supported:** text chat completions, legacy `/v1/completions`, `Models.List` / `Models.Retrieve`, structured output, vision inputs, tool-related fields (all passed through to upstream).

**Not supported:** Responses API, Assistants, Embeddings, Images, Audio, Files, Vector stores.

## Anthropic compatibility

**Supported features:**

- `Messages.New` and `Messages.CountTokens`
- `content` as a bare string or an array of blocks, in messages and in `system`
- `text`, `image`, and `document` blocks; base64 PDFs are auto-converted to page images
- `image` blocks with a `url` source (the URL is passed to upstream as-is)
- `tool_use` and `tool_result` blocks, including `is_error`
- Tool definitions via `tools`, constrained via `tool_choice` (`auto`, `any`, `tool`, `none`)
- `max_tokens` (required, and rejected below 1), `temperature`, `top_p`, `top_k`, `stop_sequences`
- Streaming via `"stream": true`, emitted as the full Anthropic event sequence
  (`message_start`, `content_block_start` / `content_block_delta` / `content_block_stop`,
  `message_delta`, `message_stop`) so the SDK's own accumulator works unmodified
- `x-api-key` and `anthropic-version` headers accepted and ignored
- Anthropic-shaped error bodies (`{"type":"error","error":{...}}`) with the status-appropriate error type

**Known differences from the real API:**

| Behavior | localaik |
| --- | --- |
| `stop_reason: "stop_sequence"` | Never emitted. The upstream runtime reports plain `stop` for both a natural end of turn and a stop-sequence hit, so those come back as `end_turn`. |
| `count_tokens` on multimodal input | Counts text only, including text-source `document` blocks. Images, base64 documents, and tool blocks are skipped, so counts run lower than the real API. |
| Streamed tool arguments | Text streams token by token, but a tool call's arguments arrive in a single `input_json_delta` rather than several. Tool calls are accumulated per OpenAI's `tool_calls[].index` and emitted whole at the end of the stream. SDK accumulators build the same result either way. |
| `tool_use` ids when upstream sends none | Synthesized from the tool name, with a numeric suffix when that would collide. Two parallel calls to one tool would otherwise share an id, and the client's `tool_result` blocks key on it. |
| A tool call upstream never named | Dropped, since the Messages API never emits a `tool_use` without a name and no client can invoke one. If that leaves no tool calls at all, `stop_reason` is `end_turn` rather than `tool_use`. |
| A tool call upstream restates verbatim | Ignored once its arguments are a whole object, since gateways resend the entire `tool_calls` array on a later chunk. A repeat of a partial fragment is treated as a continuation, because that is indistinguishable from one. |
| Text interleaved into a tool call's arguments | Not something OpenAI's contract produces. A call whose arguments are mid-object is held back, so nothing is lost, but the blocks come out in a different order than upstream sent them. A call whose first chunk carried `"arguments": ""` is not distinguishable from a finished no-argument call, so text arriving there ends it early and any arguments that follow start a second block. |
| Two different tool calls sharing one `tool_calls[].index` | Merged, because the index is what identifies a call. OpenAI's contract does not allow this, and every attempt to infer a split from ids, names or argument shapes broke a well-formed stream instead. Their concatenated arguments stop parsing, so the block ends up with `{}`. |
| Streamed `usage` | Only as good as what upstream reports; the counts are zero when the runtime omits usage from streamed responses. |
| A stream that ends without a `finish_reason` | Closed out as `end_turn` rather than reported as an error, so a runtime that omits the field still produces a usable response. A truncated upstream stream therefore looks complete. A `200` carrying no stream frames at all is reported as an error rather than an empty message. |
| Anthropic's built-in tools in `tools` (web search, code execution, computer use, text editor, bash) | Skipped, along with `tool_choice` if nothing else remains. They are declared by a versioned `type` with no `input_schema`, so there is nothing to describe to the local model, and offering them would invite calls that go nowhere. |
| `tool_use` / `tool_result` with the ids omitted | Not re-paired. Both ids are required by the Messages API; when they are absent there is nothing to match on, so each side gets a placeholder instead. |
| Multiple upstream choices while streaming | The stream follows the lowest choice index; the rest are dropped. |
| `thinking` / `redacted_thinking` blocks in request history | Accepted and dropped rather than replayed into the prompt. |
| Multiple candidates | Only the first upstream choice becomes the message; the Messages API returns one message. |
| `document` blocks with a `url` source | Summarised as text; localaik does not fetch remote documents. |

**Not supported:**

- Message Batches, Files, Models list/get on the Anthropic paths
- Extended thinking (`thinking` is accepted and ignored), citations, prompt caching
- Server tools (web search, web fetch, code execution, computer use, text editor)
- Beta endpoints under `/v1/beta/`

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

# Proxy only, no model or inference engine (make docker-build-proxy)
docker build --target proxy -t gokhalh/localaik:proxy .
```

## Limitations

- Intended for tests and development, not production
- Image size is dominated by model weights (not applicable to the `proxy` tag, which ships none)
- Cold starts can take tens of seconds while the model loads
- PDF rendering adds latency per page

