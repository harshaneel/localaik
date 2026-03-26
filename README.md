# localaik

`localaik` is a local compatibility server for a subset of the Gemini and OpenAI APIs. Run one container, point your SDK at `http://localhost:8090`, and the proxy serves both protocol shapes on the same port for tests and development.

Inside the container:

- `localaik` exposes Gemini-compatible and OpenAI-compatible routes on one port.
- `localaik` translates Gemini requests to OpenAI-compatible chat completions.
- `localaik` forwards OpenAI chat-completions requests upstream with auth headers stripped.
- `llama.cpp` serves a bundled Gemma model on an internal port.
- `pdftoppm` renders Gemini PDF uploads into page images before forwarding them to the model.

## Quick start

```yaml
services:
  localaik:
    image: gokhalh/localaik
    ports:
      - "8090:8090"
```

Gemini GenAI clients can usually work without code changes:

```bash
export GOOGLE_GEMINI_BASE_URL=http://localhost:8090
```

```go
client, err := genai.NewClient(ctx, &genai.ClientConfig{
    // `APIKey` is only present to match the SDK request shape.
    APIKey: "test",
    HTTPOptions: genai.HTTPOptions{
        BaseURL: "http://localhost:8090",
    },
})
```

OpenAI-compatible clients can usually work by pointing their `base_url` at `/v1`. Example:

```python
from openai import OpenAI

client = OpenAI(
  # `api_key` is only present to match the SDK request shape.
  api_key="test",
  base_url="http://localhost:8090/v1",
)
```

The snippets above are configuration examples. The pinned SDK versions covered by automated tests are listed below.

## Implemented routes

`localaik` does not implement the full Gemini or OpenAI API surface. It implements the routes below and returns `404` for other API routes.

| Route | Used by | Notes |
| --- | --- | --- |
| `POST /v1beta/models/{model}:generateContent` | Gemini GenAI `Models.GenerateContent` | Gemini request/response shape translated through upstream chat completions |
| `POST /v1beta/models/{model}:streamGenerateContent` | Gemini GenAI `Models.GenerateContentStream` | Typically called with `?alt=sse`; response is Gemini-style SSE |
| `POST /v1/chat/completions` | OpenAI chat-completions clients | Forwarded to upstream chat completions |
| `GET /health` | `localaik` health checks | Custom route; not part of Gemini or OpenAI |

## Tested SDKs

The automated SDK contract tests in this repo currently validate against these Go SDKs:

- `google.golang.org/genai` `v1.51.0`
- `github.com/openai/openai-go/v3` `v3.30.0`

Other SDK versions and languages may work if they emit the same HTTP shapes, but only the versions above are part of this repo's automated coverage.

## Gemini compatibility

Supported SDK methods:

- `Models.GenerateContent`
- `Models.GenerateContentStream`

Supported request and response features:

- Text input
- Image input via `inlineData`
- PDF input via `inlineData` with automatic PDF-to-image conversion
- `fileData` for image URLs plus local or `data:`-URI PDF/text files
- `systemInstruction`
- `generationConfig.temperature`
- `generationConfig.topP`
- `generationConfig.topK`
- `generationConfig.candidateCount`
- `generationConfig.maxOutputTokens`
- `generationConfig.stopSequences`
- `generationConfig.responseLogprobs`
- `generationConfig.logprobs`
- `generationConfig.presencePenalty`
- `generationConfig.frequencyPenalty`
- `generationConfig.seed`
- Structured output via `responseMimeType`, `responseSchema`, and `responseJsonSchema`
- Function declarations via `tools`
- Function calling config via `toolConfig.functionCallingConfig`
- `functionCall` and `functionResponse` parts
- `x-goog-api-key` accepted and ignored
- Gemini-style streaming SSE responses
- Usage metadata and finish reasons translated from upstream chat completions

Partial support:

- OpenAI-compatible upstream runtimes decide whether forwarded controls like `top_k`, `n`, logprobs, or tool choice are honored
- `executableCode`, `codeExecutionResult`, `toolCall`, and `toolResponse` parts are preserved as text context rather than executed with native Gemini semantics
- Gemini response translation focuses on text, function calls, finish reasons, and usage; richer Gemini metadata such as grounding, citations, safety details, and prompt feedback are not surfaced

Not supported:

- Other GenAI SDK methods and endpoints outside `GenerateContent` and `GenerateContentStream`
- Non-function Gemini tools such as Google Search, Google Maps, URL context, and Gemini server-side code execution
- Embeddings, token counting, cached content APIs, live/bidi sessions, uploads, and other Gemini endpoints outside the routes listed above

## OpenAI compatibility

Supported SDK methods:

- `client.Chat.Completions.New`
- Chat completions streaming

Supported request and response features:

- Text chat completions
- Structured output fields passed through to upstream chat completions
- Vision inputs passed through to upstream chat completions
- Tool-related chat-completions fields passed through to upstream chat completions
- `Authorization: Bearer ...` accepted and ignored

Not supported:

- Other OpenAI SDK methods and endpoints outside chat completions
- Responses API
- Assistants
- Embeddings
- Images
- Audio
- Files or uploads
- Vector stores

## Development

Run the proxy locally:

```bash
go run ./cmd/localaik --port 8090 --upstream http://127.0.0.1:8080/v1
```

Common development commands:

```bash
make lint
make test-unit
make test-integration
make test
make docker-up
make docker-down
```

`make test-integration` assumes the container is already running, for example via `make docker-up`.

`make docker-up` waits for `/health` before returning.

Run image smoke tests against a locally built container:

```bash
IMAGE=gokhalh/localaik:test PORT=18090 CONTAINER_NAME=localaik-image-integration ./scripts/run-image-integration.sh
```

Reuse an existing image instead of rebuilding:

```bash
BUILD_IMAGE=0 IMAGE=gokhalh/localaik:latest PORT=18090 CONTAINER_NAME=localaik-image-integration ./scripts/run-image-integration.sh
```

Run a single image-backed test manually without cache:

```bash
go test -count=1 -tags=docker_integration ./integration -run '^TestImageHealth$$'
```

Run the container locally through Make:

```bash
make docker-up
```

Use a different port or image:

```bash
PORT=8090 IMAGE=gokhalh/localaik:latest BUILD_IMAGE=0 make docker-up
```

Stop it again:

```bash
make docker-down
```

## Building the image

The default image bakes in the Gemma 3 4B Q4 model plus its multimodal projection weights:

```bash
docker build -t gokhalh/localaik .
```

The Dockerfile also supports alternate pinned model artifacts at build time:

```bash
docker build \
  --build-arg MODEL_URL=... \
  --build-arg MODEL_SHA256=... \
  --build-arg MMPROJ_URL=... \
  --build-arg MMPROJ_SHA256=... \
  -t gokhalh/localaik:custom .
```

## Tags

- `latest`, `gemma3-4b`: Gemma 3 4B Q4
- `gemma3-12b`: Gemma 3 12B Q4

## Limitations

- Intended for tests and development, not production.
- Image size is dominated by model weights.
- Cold starts can take tens of seconds while the model loads.
- PDF rendering adds latency for each page.
- Function calling is implemented through the chat-completions compatibility layer, but non-function Gemini tools and embeddings are still not implemented.
