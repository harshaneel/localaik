# Examples

Small, runnable samples that talk to **localaik** on `http://localhost:8090`. Use them to sanity-check your setup or as templates for your own code.

## Prerequisites

1. Start localaik (pick one):

   ```bash
   docker run -d -p 8090:8090 gokhalh/localaik
   ```

   Or from the repo root: `make docker-up` (defaults to port `18090` — set `PORT=8090` if you want the examples unchanged).

2. Wait until the model is loaded (`GET /health` returns 200). The first start can take a while.

3. Run an example from the directory listed below (each folder has its own dependencies).

**Python 3**

- The **[python/](python/)** samples need **Python 3** and the listed SDK packages (`google-genai`, OpenAI client, etc.).
- The **[curl/](curl/)** scripts are shell + `curl`, but they pipe the response through **`python3 -m json.tool`** so the JSON is pretty-printed. Install **Python 3** on your PATH, or remove the `| python3 -m json.tool` suffix and read raw JSON instead.

## Layout

| Language    | Gemini | OpenAI | Structured output (Gemini) |
| ----------- | ------ | ------ | --------------------------- |
| **curl**    | [curl/gemini.sh](curl/gemini.sh) | [curl/openai.sh](curl/openai.sh) | [curl/gemini-structured.sh](curl/gemini-structured.sh) |
| **Go**      | [go/gemini](go/gemini/main.go) | [go/openai](go/openai/main.go) | [go/gemini-structured](go/gemini-structured/main.go) |
| **Python**  | [python/gemini](python/gemini/main.py) | [python/openai](python/openai/main.py) | [python/gemini-structured](python/gemini-structured/main.py) |
| **JavaScript** | [javascript/gemini](javascript/gemini/index.mjs) | [javascript/openai](javascript/openai/index.mjs) | [javascript/gemini-structured](javascript/gemini-structured/index.mjs) |
| **Java**    | [java/gemini](java/gemini/Gemini.java) | [java/openai](java/openai/OpenAI.java) | [java/gemini-structured](java/gemini-structured/GeminiStructured.java) |

## Conventions

- **Base URL:** `http://localhost:8090` for Gemini-style calls; OpenAI clients use `http://localhost:8090/v1`.
- **API key:** Examples use a placeholder such as `test` where the SDK requires one; localaik does not validate keys.
- **Model name:** Samples use `localaik` as the model id where applicable; the proxy forwards to the bundled upstream model.

## Dependencies

Examples do not share a single lockfile. Install what you need per language (e.g. `google-genai` for Python, `google.golang.org/genai` for Go, official OpenAI packages for OpenAI examples). **curl-only:** aside from optional `python3` for JSON formatting (see above), no extra installs.
