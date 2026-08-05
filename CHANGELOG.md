# Changelog

Releases are cut by pushing a `v*` tag, which publishes the images to Docker Hub.
Entries before v0.4.0 were reconstructed from git history.

## v0.4.0 (2026-08-04)

### Added

- **`:proxy` image variant, 41 MB.** The translation layer alone, with no model and no inference engine, for pointing at an OpenAI-compatible server you already run. Published as `proxy` and `vX.Y.Z-proxy`. It is never published as `latest`, because `latest` has always been a self-contained container and quietly turning that into one requiring an external server would break existing users.
- **`LK_UPSTREAM`** sets the upstream base URL, defaulting to `http://127.0.0.1:8080/v1`. An explicit `--upstream` flag still wins over the environment.
- **`LK_UPSTREAM_AUTH_HEADER`** sends a credential to your upstream, given as a complete header line such as `Authorization: Bearer abc123`. It is attached only to requests whose host matches `LK_UPSTREAM`, and while it is set an upstream redirect is returned to the caller rather than followed. Credentials that clients send to localaik are still discarded and never forwarded.

### Security

- **`:proxy` does not authenticate its callers, and it forwards into your infrastructure.** The model-bundled tags keep llama.cpp bound to localhost inside the container, so the only thing reachable is a disposable local model. `:proxy` makes the container a network hop to a server you care about, while still accepting and ignoring client credentials by design. Anyone who can reach port 8090 can use your model server without credentials. Bind to localhost and do not publish the port on a shared network.

### Changed

- **The container no longer waits for the model before starting the proxy.** The port accepts connections immediately and `/health` returns 503 until the model is ready.

  This removes the invariant that an open port implies a loaded model. A TCP liveness check, `nc -z`, or a Docker Compose `depends_on` without `condition: service_healthy` will now let requests through early, and those requests fail with a 502. Wait for `GET /health` to return 200 instead.

- **There is no longer a model-load timeout.** The old 120s bound is gone rather than raised. If the model never becomes ready, the container keeps running and `/health` keeps reporting 503 instead of exiting. Use a restart policy or an orchestrator liveness probe if you want automatic recovery.
- `HEALTHCHECK --start-period` for the model-bundled images is now 180s, up from 60s, matching the readiness bounds the Makefile, CI and integration helpers already use. The `:proxy` image keeps its own 5s start period. Docker still reports healthy on the first passing check, so this does not delay readiness.

### Fixed

- **A container whose model took longer than 120s to load would wedge instead of failing.** The entrypoint's cleanup ran `jobs -p | xargs kill` followed by a bare `wait`. `jobs` lists nothing in a non-interactive shell, so nothing was killed and the wait blocked forever on a live child. The script never reached its exit, leaving a container that stayed running with a healthy engine on `127.0.0.1:8080` and nothing listening on 8090. `docker stop` hung on the same code path.
- **`llama-server` dying after startup now stops the container.** The entrypoint previously handed off with `exec` and stopped watching, so a post-startup crash silently broke every request instead of surfacing.

## v0.3.0 (2026-08-03)

### Added

- **Anthropic Messages API compatibility layer:** `POST /v1/messages` including streaming, and `POST /v1/messages/count_tokens`. Covers tool use and PDF documents.

### Fixed

- Streamed tool calls are accumulated per OpenAI's `index` contract rather than by inferring call boundaries from identity, which had produced mismatched and duplicated calls.
- Streaming `stop_reason` aligned with the non-streaming path.
- Duplicate and invented tool calls from certain streaming shapes, including echoed deltas and merged argument-less calls, are no longer emitted.
- Tool results are ordered before other content in non-assistant turns, which affects both the Gemini and Anthropic translations.
- Anthropic's server tools (web search, code execution, computer use) are skipped rather than forwarded as callable no-ops, since localaik cannot execute them.
- `count_tokens` counts document and system blocks and `tool_result` placeholders, matching what is actually sent to the model.
- `max_tokens` below 1 is rejected, and a 200 response carrying no stream frames is treated as an error.

## v0.2.1 (2026-05-18)

### Added

- Gemini and OpenAI `Models.List` and `Models.Get`, `CountTokens`, and the legacy OpenAI completions route.

## v0.1.4 (2026-05-18)

### Changed

- Bumped `google.golang.org/genai` to v1.57.0 and `openai-go` to v3.36.0.

## v0.1.3 (2026-04-06)

### Added

- `LK_*` environment variables for tuning the underlying model server: context size, thread counts, batch sizes, GPU layers, parallel slots, flash attention, continuous batching and mlock.
- Runnable examples under `examples/`.

## v0.1.2 (2026-03-30)

### Changed

- README improvements and reduced cyclomatic complexity.

## v0.1.1 (2026-03-25)

### Fixed

- Corrected a 404 on the mmproj download URL for the `gemma3-12b` variant.

## v0.1.0 (2026-03-25)

Initial release. A single container serving the Gemini and OpenAI wire formats on one port, backed by a local llama.cpp server running Gemma 3 4B. Anthropic support arrived in v0.3.0.
