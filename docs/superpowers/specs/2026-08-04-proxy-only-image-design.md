# Proxy-only image (`:proxy`)

**Date:** 2026-08-04
**Status:** Approved, not yet implemented
**Scope:** One new published image variant plus the configuration needed to use it.

## Problem

localaik ships one kind of image: llama.cpp, a Gemma model, and the translating
proxy, welded together. The smallest published tag is 3.16 GB compressed. Almost
all of that is model data: the weights are 2.49 GB and the vision projector
another 0.85 GB, against ~130 MB for llama.cpp and ~8 MB for the proxy itself.

Some users already run an inference server. It may be llama.cpp on their laptop,
or a shared internal deployment. Those users want only the translation layer:
something that accepts Gemini, OpenAI and Anthropic shaped requests and forwards
them to a server they already operate. Today they must pull 3.16 GB and run a
second model they will never call.

The Go binary already supports this. `cmd/localaik/main.go` takes `--upstream`
for any OpenAI-compatible base URL and assumes nothing about llama.cpp or
locality. What is missing is a container image that omits the inference stack,
and the configuration to authenticate against a remote upstream.

## Non-goals

- Authenticating localaik's own callers. It remains a test double that accepts
  and ignores client credentials.
- Any model download. That is the separate `:no-model` variant, specified later.
- Any change to `gemma3-4b`, `gemma3-12b` or `latest`. Those keep working
  byte-identically.

## Design

### The image

`Dockerfile` gains a third stage. Stage order matters: the llama.cpp stage stays
last so that a bare `docker build .` and the existing `make docker-build` keep
producing the full image.

```
FROM golang:1.25-alpine AS proxy-builder      # exists, unchanged
FROM alpine AS proxy                          # new
FROM ghcr.io/ggml-org/llama.cpp@sha256:...    # exists, stays last
```

The `proxy` stage installs `poppler-utils` and `ca-certificates`, copies the
binary, and runs it under `tini`. No entrypoint script: with configuration read
from the environment there is nothing for a shell to decide.

`poppler-utils` is required, not optional. `main.go` constructs
`pdf.NewExecRenderer("pdftoppm")`, and without it every PDF request fails at
render time. Dropping it would save roughly 50 MB and silently remove a
documented feature, which is a bad trade against an image that is already about
50x smaller than today's.

Expected size: 50-90 MB. To be measured during implementation, not asserted here.

### Configuration

`main.go` grows environment fallbacks for its two flags, following the pattern
`PORT` already sets for `--port`.

| Variable | Flag | Default | Purpose |
| --- | --- | --- | --- |
| `LK_UPSTREAM` | `--upstream` | `http://127.0.0.1:8080/v1` | Base URL of the model server |
| `LK_UPSTREAM_AUTH_HEADER` | none | unset | Credential sent to upstream only |
| `PORT` | `--port` | `8090` | Listen port, unchanged |

Precedence is flag over environment over default, so the full image's entrypoint
keeps working unchanged: it passes `--upstream` explicitly.

`LK_UPSTREAM_AUTH_HEADER` holds a complete header line, for example
`Authorization: Bearer abc123`, rather than a bare token. This covers `Bearer`,
llama.cpp's `--api-key`, and any custom scheme without the proxy needing to know
which is in use.

### Upstream authentication

The proxy currently sends no credential upstream, deliberately. Three separate
places enforce that:

- `cloneHeaders` strips `Authorization`, `X-Api-Key` and `X-Goog-Api-Key` from
  passthrough requests.
- The Gemini and Anthropic handlers build fresh requests carrying only
  `Content-Type` and `Accept`.
- `fetchUpstreamJSON` forwards no headers at all, and documents why.

That is correct when upstream is `127.0.0.1:8080` inside the same container.
Against a remote server that requires a key, every request would 401.

The credential is therefore injected in the HTTP client's transport rather than
at each call site. All upstream traffic already flows through `s.client.Do`, so a
`RoundTripper` wrapper applies the header to every request and cannot be
forgotten when a sixth upstream path is added later.

Two properties must hold simultaneously, and both are tested:

1. Credentials the caller sent are still stripped and never reach upstream.
2. The proxy's own credential is added to every upstream request.

### Health and readiness

`handleHealth` already probes upstream on every call and returns 503 when it is
unreachable, so it works unchanged against a remote server. `HEALTHCHECK
--start-period` drops from 60s to 5s in the `proxy` stage, since no model loads.

### Publishing

`release.yml` gains a matrix entry carrying a build target. Existing entries
default to the full image. Tags follow the current scheme minus `latest`:
`proxy` and `vX.Y.Z-proxy`.

`:proxy` must not become `latest`. Anyone pulling `latest` today gets a
self-contained container, and silently turning that into one requiring an
external server would break them.

## Security

`:proxy` has a materially different risk profile from every existing tag, and
the README must say so.

In the baked images llama.cpp binds `127.0.0.1` inside the container. The only
reachable service is the proxy, and behind it a disposable local model. In
`:proxy` the container becomes a network hop into infrastructure the operator
cares about. Because localaik accepts and ignores client credentials by design,
anyone who can reach port 8090 can drive the upstream server unauthenticated.

The mitigation is documentation, not code: bind to localhost, and do not publish
the port on a shared network. Adding client authentication is explicitly
rejected, because it would invite treating a test double as production
infrastructure.

`LK_UPSTREAM_AUTH_HEADER` is a secret in an environment variable. That is
acceptable here: the container legitimately needs it, it travels as a request
header rather than a process argument, and it is never echoed to logs or
forwarded to callers. Implementation must not log its value, and must not enable
shell tracing anywhere it is in scope.

## Testing

Existing tests need no changes. They already stub upstream through an
`http.RoundTripper`, which is exactly the seam this feature uses.

New unit tests:

- Flag beats environment beats default, for both `--upstream` and `--port`.
- `LK_UPSTREAM_AUTH_HEADER` reaches all four upstream paths: chat completions,
  `/tokenize`, models list, and the Anthropic messages route.
- Client `Authorization`, `X-Api-Key` and `X-Goog-Api-Key` are still stripped
  when the proxy's own credential is configured, verified together in one test so
  the two behaviours cannot silently merge.
- No credential is sent when `LK_UPSTREAM_AUTH_HEADER` is unset.

Image test, behind the existing `docker_integration` tag: build `--target
proxy`, run it against a stub OpenAI-compatible server, and confirm a Gemini, an
OpenAI and an Anthropic request each round-trip. This is fast, since no model is
involved.

## Verification before merge

Two claims in this spec are estimates and must be measured:

1. Final image size. Recorded in the README once known.
2. That `pdftoppm` from alpine's `poppler-utils` behaves the same as the
   Debian-based full image for the PDF-to-PNG path. The existing PDF tests
   should be run inside the built `:proxy` image, not only on the host.

## Follow-up

`:no-model`, a variant keeping llama.cpp but fetching the model from
`LK_MODEL_URL` at startup, is a separate change. Investigation during this design
established that the pinned llama.cpp build already provides `--model-url`,
`--mmproj-url`, `--hf-token`, `LLAMA_CACHE` and `--offline`, so that work is
mostly packaging and documentation rather than download logic. It also needs a
startup-order change so the proxy answers `/health` during a long download, which
`:proxy` does not require.
