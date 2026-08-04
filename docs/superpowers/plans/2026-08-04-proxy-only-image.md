# Proxy-only image (`:proxy`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a `:proxy` image containing only the translating proxy and `pdftoppm`, for users who already run an OpenAI-compatible model server.

**Architecture:** Three changes, each independently testable. The Go binary learns two environment fallbacks and gains an optional upstream credential injected at the HTTP transport layer, so no upstream call site changes. The `Dockerfile` gains a third build stage. CI publishes it as a new matrix entry.

**Tech Stack:** Go 1.25, standard library only. Docker multi-stage build, alpine base. GitHub Actions with `docker/build-push-action@v6`.

## Global Constraints

- Go version: `1.25` (from `go.mod`). Standard library only; add no dependencies.
- Run `gofmt -w` on every Go file touched. `make lint` runs `gofmt -l` and `go vet ./...` and must stay clean.
- No em-dashes in any file, including comments, docs and commit messages.
- Comments: default to zero. One line only when the WHY is non-obvious. Never restate the code. Never explain rejected alternatives or review feedback.
- The llama.cpp stage must remain the LAST stage in `Dockerfile`, so `docker build .` and `make docker-build` keep producing the full image.
- `:proxy` must never be tagged `latest`.
- `LK_UPSTREAM_AUTH_HEADER` is a secret. Never log its value. Never add `set -x` anywhere it is in scope.
- Existing behaviour must not change: client credentials (`Authorization`, `X-Api-Key`, `X-Goog-Api-Key`) are still stripped and never forwarded upstream.

## File Structure

| File | Responsibility |
| --- | --- |
| `internal/server/upstreamauth.go` (create) | The `RoundTripper` that adds the proxy's own credential to upstream requests. Isolated so the security-sensitive logic is one small readable unit. |
| `internal/server/upstreamauth_test.go` (create) | Tests for that transport in isolation. |
| `internal/server/server.go` (modify) | `Config` gains `UpstreamAuthHeader`; `New` wraps the client transport when it is set. |
| `internal/server/auth_integration_test.go` (create) | Proves the header reaches all four upstream paths while client credentials are still stripped. |
| `cmd/localaik/main.go` (modify) | Environment fallbacks for `--upstream`, plus reading `LK_UPSTREAM_AUTH_HEADER`. |
| `cmd/localaik/main_test.go` (create) | Flag over environment over default precedence. |
| `Dockerfile` (modify) | New `proxy` stage inserted before the llama.cpp stage. |
| `.github/workflows/release.yml` (modify) | Matrix entry with a build target. |
| `Makefile` (modify) | `docker-build-proxy` target for local verification. |
| `integration/proxy_image_test.go` (create) | Behind `docker_integration`: builds and exercises the image. |
| `README.md` (modify) | Tag table row, configuration, and the security warning. |

---

### Task 1: Upstream auth transport

Adds the credential-injecting `RoundTripper` and wires it into `server.New`. Nothing consumes it yet.

**Files:**
- Create: `internal/server/upstreamauth.go`
- Create: `internal/server/upstreamauth_test.go`
- Modify: `internal/server/server.go:18-22` (Config), `internal/server/server.go:46-49` (client setup)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `server.Config.UpstreamAuthHeader string` (new field, optional)
  - `func newUpstreamAuthTransport(base http.RoundTripper, header string) http.RoundTripper`
  - Header format is a full header line, `"Name: value"`, split on the first colon.

- [ ] **Step 1: Write the failing test**

Create `internal/server/upstreamauth_test.go`:

```go
package server

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

type capturingTransport struct {
	seen http.Header
}

func (c *capturingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	c.seen = req.Header.Clone()
	recorder := httptest.NewRecorder()
	recorder.WriteHeader(http.StatusOK)
	return recorder.Result(), nil
}

func TestUpstreamAuthTransportAddsHeader(t *testing.T) {
	capture := &capturingTransport{}
	transport := newUpstreamAuthTransport(capture, "Authorization: Bearer secret")

	req := httptest.NewRequest(http.MethodGet, "http://upstream.test/v1/models", nil)
	if _, err := transport.RoundTrip(req); err != nil {
		t.Fatalf("RoundTrip returned error: %v", err)
	}

	if got := capture.seen.Get("Authorization"); got != "Bearer secret" {
		t.Fatalf("Authorization = %q, want %q", got, "Bearer secret")
	}
}

func TestUpstreamAuthTransportTrimsWhitespace(t *testing.T) {
	capture := &capturingTransport{}
	transport := newUpstreamAuthTransport(capture, "  X-Api-Key :   abc123  ")

	req := httptest.NewRequest(http.MethodGet, "http://upstream.test/v1/models", nil)
	if _, err := transport.RoundTrip(req); err != nil {
		t.Fatalf("RoundTrip returned error: %v", err)
	}

	if got := capture.seen.Get("X-Api-Key"); got != "abc123" {
		t.Fatalf("X-Api-Key = %q, want %q", got, "abc123")
	}
}

func TestUpstreamAuthTransportIgnoresMalformedHeader(t *testing.T) {
	for _, header := range []string{"", "   ", "NoColonHere", ": novalue", "Name:"} {
		capture := &capturingTransport{}
		transport := newUpstreamAuthTransport(capture, header)

		req := httptest.NewRequest(http.MethodGet, "http://upstream.test/v1/models", nil)
		if _, err := transport.RoundTrip(req); err != nil {
			t.Fatalf("RoundTrip returned error for %q: %v", header, err)
		}
		if len(capture.seen) != 0 {
			t.Fatalf("header %q produced %v, want none", header, capture.seen)
		}
	}
}

func TestUpstreamAuthTransportDoesNotMutateCallerRequest(t *testing.T) {
	capture := &capturingTransport{}
	transport := newUpstreamAuthTransport(capture, "Authorization: Bearer secret")

	req := httptest.NewRequest(http.MethodGet, "http://upstream.test/v1/models", nil)
	if _, err := transport.RoundTrip(req); err != nil {
		t.Fatalf("RoundTrip returned error: %v", err)
	}

	if req.Header.Get("Authorization") != "" {
		t.Fatal("RoundTrip mutated the caller's request")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/server/ -run UpstreamAuthTransport -v`
Expected: FAIL, `undefined: newUpstreamAuthTransport`

- [ ] **Step 3: Write minimal implementation**

Create `internal/server/upstreamauth.go`:

```go
package server

import (
	"net/http"
	"strings"
)

type upstreamAuthTransport struct {
	base  http.RoundTripper
	name  string
	value string
}

// newUpstreamAuthTransport returns base unchanged when header is not a usable
// "Name: value" line, so a misconfigured value cannot silently drop requests.
func newUpstreamAuthTransport(base http.RoundTripper, header string) http.RoundTripper {
	if base == nil {
		base = http.DefaultTransport
	}

	name, value, found := strings.Cut(header, ":")
	name = strings.TrimSpace(name)
	value = strings.TrimSpace(value)
	if !found || name == "" || value == "" {
		return base
	}

	return &upstreamAuthTransport{base: base, name: name, value: value}
}

func (t *upstreamAuthTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req.Clone(req.Context())
	clone.Header.Set(t.name, t.value)
	return t.base.RoundTrip(clone)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./internal/server/ -run UpstreamAuthTransport -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Wire it into Config**

In `internal/server/server.go`, add the field to `Config`:

```go
type Config struct {
	UpstreamBaseURL    string
	UpstreamAuthHeader string
	HTTPClient         *http.Client
	PDFRenderer        pdf.Renderer
}
```

Then in `New`, replace the existing client setup block:

```go
	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
```

with:

```go
	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	if cfg.UpstreamAuthHeader != "" {
		clone := *client
		clone.Transport = newUpstreamAuthTransport(clone.Transport, cfg.UpstreamAuthHeader)
		client = &clone
	}
```

- [ ] **Step 6: Run the full server suite**

Run: `go test ./internal/server/ -count=1`
Expected: PASS. Copying the client rather than mutating it means every existing test that passes its own `HTTPClient` is unaffected.

- [ ] **Step 7: Format, lint, commit**

```bash
gofmt -w internal/server/upstreamauth.go internal/server/upstreamauth_test.go internal/server/server.go
make lint
git add internal/server/upstreamauth.go internal/server/upstreamauth_test.go internal/server/server.go
git commit -m "feat: Add optional upstream auth header to the proxy

Injected at the transport layer so every upstream request carries it without
each call site opting in."
```

---

### Task 2: Prove the header reaches every upstream path

Task 1 tested the transport alone. This proves the wiring covers all four upstream endpoints and that client credentials are still stripped.

**Files:**
- Create: `internal/server/auth_integration_test.go`

**Interfaces:**
- Consumes: `server.Config.UpstreamAuthHeader` from Task 1; `roundTripHandler` from `internal/server/server_test.go:147`; `newTestServer` from `internal/server/meta_test.go:17`.
- Produces: nothing.

- [ ] **Step 1: Write the failing test**

Create `internal/server/auth_integration_test.go`:

```go
package server

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/harshaneel/localaik/internal/pdf"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

// Every upstream route must carry the proxy's credential and none of the
// caller's.
func TestUpstreamAuthHeaderReachesEveryUpstreamPath(t *testing.T) {
	cases := []struct {
		name   string
		method string
		path   string
		body   string
	}{
		{"openai_chat", http.MethodPost, "/v1/chat/completions", `{"model":"m","messages":[]}`},
		{"openai_models", http.MethodGet, "/v1/models", ""},
		{"gemini_generate", http.MethodPost, "/v1beta/models/m:generateContent", `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`},
		{"gemini_count_tokens", http.MethodPost, "/v1beta/models/m:countTokens", `{"contents":[{"parts":[{"text":"hi"}]}]}`},
		{"anthropic_messages", http.MethodPost, "/v1/messages", `{"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`},
		{"anthropic_count_tokens", http.MethodPost, "/v1/messages/count_tokens", `{"messages":[{"role":"user","content":"hi"}]}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var called bool
			var seenAuth, seenClientAuth, seenAPIKey, seenGoogKey string

			upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				called = true
				seenAuth = r.Header.Get("X-Proxy-Token")
				seenClientAuth = r.Header.Get("Authorization")
				seenAPIKey = r.Header.Get("X-Api-Key")
				seenGoogKey = r.Header.Get("X-Goog-Api-Key")

				switch r.URL.Path {
				case "/tokenize":
					writeJSON(w, http.StatusOK, map[string]any{"tokens": []int{1, 2}})
				case "/v1/models":
					writeJSON(w, http.StatusOK, openaip.ModelList{Object: "list", Data: []openaip.Model{{ID: "m"}}})
				default:
					writeJSON(w, http.StatusOK, openaip.ChatCompletionResponse{
						Choices: []openaip.Choice{{Message: openaip.Message{Content: "ok"}, FinishReason: "stop"}},
					})
				}
			})

			srv, err := New(Config{
				UpstreamBaseURL:    "http://upstream.test/v1",
				UpstreamAuthHeader: "X-Proxy-Token: upstream-secret",
				HTTPClient:         &http.Client{Transport: roundTripHandler{handler: upstream}},
				PDFRenderer:        pdf.RendererFunc(func(context.Context, []byte) ([][]byte, error) { return nil, nil }),
			})
			if err != nil {
				t.Fatalf("New returned error: %v", err)
			}

			var reader *bytes.Buffer
			if tc.body != "" {
				reader = bytes.NewBufferString(tc.body)
			} else {
				reader = bytes.NewBuffer(nil)
			}
			req := httptest.NewRequest(tc.method, tc.path, reader)
			req.Header.Set("Authorization", "Bearer client-secret")
			req.Header.Set("X-Api-Key", "client-anthropic-key")
			req.Header.Set("X-Goog-Api-Key", "client-google-key")

			rec := httptest.NewRecorder()
			srv.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
			}
			if !called {
				t.Fatal("upstream was never called, so the header check proves nothing")
			}
			if seenAuth != "upstream-secret" {
				t.Fatalf("X-Proxy-Token = %q, want the proxy credential", seenAuth)
			}
			if seenClientAuth != "" || seenAPIKey != "" || seenGoogKey != "" {
				t.Fatalf("client credentials leaked upstream: auth=%q apikey=%q googkey=%q", seenClientAuth, seenAPIKey, seenGoogKey)
			}
		})
	}
}

func TestNoUpstreamAuthHeaderWhenUnset(t *testing.T) {
	var seen http.Header

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Clone()
		writeJSON(w, http.StatusOK, openaip.ChatCompletionResponse{
			Choices: []openaip.Choice{{Message: openaip.Message{Content: "ok"}, FinishReason: "stop"}},
		})
	})

	srv := newTestServer(t, upstream)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if got := seen.Get("Authorization"); got != "" {
		t.Fatalf("Authorization = %q, want none when no credential is configured", got)
	}
}

// The Gemini streaming route builds its own request; confirm the credential is
// present there too.
func TestUpstreamAuthHeaderOnStreamingRoute(t *testing.T) {
	var seen string

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Get("X-Proxy-Token")
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	})

	srv, err := New(Config{
		UpstreamBaseURL:    "http://upstream.test/v1",
		UpstreamAuthHeader: "X-Proxy-Token: upstream-secret",
		HTTPClient:         &http.Client{Transport: roundTripHandler{handler: upstream}},
		PDFRenderer:        pdf.RendererFunc(func(context.Context, []byte) ([][]byte, error) { return nil, nil }),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	body := `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/m:streamGenerateContent", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if seen != "upstream-secret" {
		t.Fatalf("X-Proxy-Token = %q on the streaming route, want the proxy credential", seen)
	}
}
```

- [ ] **Step 2: Run the tests**

Run: `go test ./internal/server/ -run 'UpstreamAuthHeader|NoUpstreamAuthHeader' -v`
Expected: PASS. Task 1 already made this work; these tests exist to lock the behaviour against future call sites.

If any subtest fails with the credential missing, the cause is an upstream request that bypasses `s.client`. Find it and route it through `s.client` rather than weakening the test.

- [ ] **Step 3: Format, lint, commit**

```bash
gofmt -w internal/server/auth_integration_test.go
make lint
go test ./internal/server/ -count=1
git add internal/server/auth_integration_test.go
git commit -m "test: Cover upstream auth on every upstream route

Locks both halves at once: the proxy credential is added, the caller's is not
forwarded."
```

---

### Task 3: Environment fallbacks in main.go

**Files:**
- Modify: `cmd/localaik/main.go:14-32`
- Create: `cmd/localaik/main_test.go`

**Interfaces:**
- Consumes: `server.Config.UpstreamAuthHeader` from Task 1.
- Produces:
  - `func resolveFlagDefault(envName, fallback string) string`
  - Environment names: `LK_UPSTREAM`, `LK_UPSTREAM_AUTH_HEADER`, existing `PORT`.

- [ ] **Step 1: Write the failing test**

Create `cmd/localaik/main_test.go`:

```go
package main

import "testing"

func TestResolveFlagDefaultPrefersEnv(t *testing.T) {
	t.Setenv("LK_TEST_VALUE", "from-env")

	if got := resolveFlagDefault("LK_TEST_VALUE", "fallback"); got != "from-env" {
		t.Fatalf("resolveFlagDefault = %q, want from-env", got)
	}
}

func TestResolveFlagDefaultFallsBack(t *testing.T) {
	t.Setenv("LK_TEST_VALUE", "")

	if got := resolveFlagDefault("LK_TEST_VALUE", "fallback"); got != "fallback" {
		t.Fatalf("resolveFlagDefault = %q, want fallback", got)
	}
}

func TestResolveFlagDefaultUnsetFallsBack(t *testing.T) {
	if got := resolveFlagDefault("LK_DEFINITELY_UNSET_VALUE", "fallback"); got != "fallback" {
		t.Fatalf("resolveFlagDefault = %q, want fallback", got)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./cmd/localaik/ -v`
Expected: FAIL, `undefined: resolveFlagDefault`

- [ ] **Step 3: Write minimal implementation**

Replace the body of `main.go` up to and including the `server.New` call:

```go
func resolveFlagDefault(envName, fallback string) string {
	if value := os.Getenv(envName); value != "" {
		return value
	}
	return fallback
}

func main() {
	port := flag.String("port", resolveFlagDefault("PORT", "8090"), "port to listen on")
	upstream := flag.String("upstream", resolveFlagDefault("LK_UPSTREAM", "http://127.0.0.1:8080/v1"), "upstream OpenAI-compatible base URL")
	flag.Parse()

	handler, err := server.New(server.Config{
		UpstreamBaseURL:    *upstream,
		UpstreamAuthHeader: os.Getenv("LK_UPSTREAM_AUTH_HEADER"),
		HTTPClient:         &http.Client{},
		PDFRenderer:        pdf.NewExecRenderer("pdftoppm"),
	})
	if err != nil {
		log.Fatalf("localaik: %v", err)
	}
```

Leave the rest of `main` unchanged. Delete the old `defaultPort` block that this replaces.

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./cmd/localaik/ -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Verify flag still beats environment**

Run:

```bash
LK_UPSTREAM=http://from-env:9999/v1 go run ./cmd/localaik --upstream http://from-flag:1111/v1 --port 18099 &
sleep 2
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18099/health
kill %1
```

Expected: `503`. A 503 proves it started and tried to reach an upstream. Then confirm the flag won by checking the process was pointed at `from-flag`: rerun without the flag and confirm it still starts. There is no endpoint that echoes the upstream, so this step verifies startup only; precedence itself is covered by the unit tests.

- [ ] **Step 6: Format, lint, commit**

```bash
gofmt -w cmd/localaik/main.go cmd/localaik/main_test.go
make lint
go test ./cmd/... ./internal/... -count=1
git add cmd/localaik/main.go cmd/localaik/main_test.go
git commit -m "feat: Read upstream and auth header from the environment

Follows the pattern PORT already set, so the container needs no shell wrapper."
```

---

### Task 4: The `proxy` Dockerfile stage

**Files:**
- Modify: `Dockerfile` (insert a stage between line 5 and line 9)
- Modify: `Makefile:20-27` area (add a target)

**Interfaces:**
- Consumes: the `proxy-builder` stage that already exists at `Dockerfile:1`.
- Produces: build target named `proxy`; image entrypoint runs `localaik` under `tini`.

- [ ] **Step 1: Add the stage**

In `Dockerfile`, immediately after the `proxy-builder` stage (after line 5) and before the `FROM ghcr.io/ggml-org/llama.cpp@sha256:...` line, insert:

```dockerfile
FROM alpine:3 AS proxy
RUN apk add --no-cache ca-certificates poppler-utils tini
COPY --from=proxy-builder /out/localaik /usr/local/bin/localaik
ENV PORT=8090
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s \
  CMD wget -q -O - "http://127.0.0.1:${PORT:-8090}/health" >/dev/null 2>&1 || exit 1
EXPOSE 8090
ENTRYPOINT ["tini", "--", "localaik"]
```

Two notes, both verified against a real build of this stage:

`wget` rather than `curl`, because alpine's busybox already provides `wget` and this avoids installing curl solely for the healthcheck. The llama.cpp stage keeps using `curl`, which it already installs.

`alpine:3` is a moving tag. The repo pins the llama.cpp base by digest, so pinning this one by digest is more consistent. Resolve it during implementation with `docker inspect alpine:3 --format '{{index .RepoDigests 0}}'` and use that. A moving tag is acceptable if you prefer, since nothing here depends on a specific alpine version.

Confirmed present in `alpine:3` at time of writing: `pdftoppm` 25.12.0 from `poppler-utils`, `tini`, and busybox `wget`. Base plus these three packages measures 35 MB before the binary is copied in.

- [ ] **Step 2: Confirm the llama.cpp stage is still last**

Run: `grep -n '^FROM' Dockerfile`
Expected: three lines, with `ghcr.io/ggml-org/llama.cpp` on the last one.

- [ ] **Step 3: Confirm the default build is unchanged**

Run: `docker build -t localaik:default-check . && docker image inspect localaik:default-check --format '{{.Size}}'`
Expected: a size over 3000000000, proving the default target is still the full image.

- [ ] **Step 4: Build the proxy image and record its size**

Run:

```bash
docker build --target proxy -t localaik:proxy-check .
docker image inspect localaik:proxy-check --format '{{.Size}}' | awk '{printf "%.0f MB\n", $1/1000000}'
```

Expected: roughly 43 MB. The base plus packages measures 35 MB and the static binary adds about 8 MB. Anything above 60 MB means something unintended got copied in. Write the measured number down; Task 7 puts it in the README.

- [ ] **Step 5: Verify pdftoppm is present and executable**

Run: `docker run --rm --entrypoint pdftoppm localaik:proxy-check -v`
Expected: `pdftoppm version 25.12.0` or later, printed to stderr. `pdftoppm -v` exits non-zero on some builds while still printing the version, so treat printed output as success.

- [ ] **Step 6: Verify the binary starts and reports not-ready**

Run:

```bash
docker run -d --name proxy-smoke -p 18098:8090 localaik:proxy-check
sleep 3
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18098/health
docker rm -f proxy-smoke
```

Expected: `503`. There is no upstream, so 503 is correct and proves the server is listening.

- [ ] **Step 7: Add the Makefile target**

In `Makefile`, add `docker-build-proxy` to the `.PHONY` list on line 12, add a help line in the `help` target after the `docker-build` line:

```
		'make docker-build-proxy      Build the proxy-only image' \
```

and add the target after `docker-build`:

```makefile
docker-build-proxy:
	@docker build --target proxy -t "$(IMAGE)-proxy" .
```

- [ ] **Step 8: Verify the target works**

Run: `make docker-build-proxy && docker images --format '{{.Repository}}:{{.Tag}}' | grep proxy`
Expected: the tagged image is listed.

- [ ] **Step 9: Commit**

```bash
git add Dockerfile Makefile
git commit -m "feat: Add a proxy-only Dockerfile stage

alpine plus poppler-utils and the binary, no inference stack. The llama.cpp
stage stays last so the default build is unchanged."
```

---

### Task 5: Image integration test

**Files:**
- Create: `integration/proxy_image_test.go`

**Interfaces:**
- Consumes: the `proxy` build target from Task 4.
- Produces: nothing.

- [ ] **Step 1: Write the test**

Create `integration/proxy_image_test.go`:

```go
//go:build docker_integration

package integration

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// Exercises the built proxy image against a stub upstream, proving all three
// protocol surfaces round-trip without an inference stack in the container.
func TestProxyImageRoundTripsAllProtocols(t *testing.T) {
	image := "localaik:proxy-integration"

	build := exec.Command("docker", "build", "--target", "proxy", "-t", image, "..")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("docker build failed: %v\n%s", err, out)
	}

	var seenAuth string
	stub := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenAuth = r.Header.Get("X-Proxy-Token")
		w.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/tokenize") {
			_, _ = w.Write([]byte(`{"tokens":[1,2,3]}`))
			return
		}
		_, _ = w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"stubbed"},"finish_reason":"stop"}]}`))
	}))

	listener, err := net.Listen("tcp", "0.0.0.0:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	stub.Listener = listener
	stub.Start()
	defer stub.Close()

	stubPort := listener.Addr().(*net.TCPAddr).Port
	upstream := fmt.Sprintf("http://host.docker.internal:%d/v1", stubPort)

	run := exec.Command("docker", "run", "-d", "--name", "proxy-integration",
		"--add-host", "host.docker.internal:host-gateway",
		"-p", "18097:8090",
		"-e", "LK_UPSTREAM="+upstream,
		"-e", "LK_UPSTREAM_AUTH_HEADER=X-Proxy-Token: integration-secret",
		image)
	if out, err := run.CombinedOutput(); err != nil {
		t.Fatalf("docker run failed: %v\n%s", err, out)
	}
	defer exec.Command("docker", "rm", "-f", "proxy-integration").Run()

	waitForHealth(t, "http://127.0.0.1:18097/health")

	cases := []struct {
		name string
		path string
		body string
	}{
		{"openai", "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`},
		{"gemini", "/v1beta/models/m:generateContent", `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`},
		{"anthropic", "/v1/messages", `{"max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp, err := http.Post("http://127.0.0.1:18097"+tc.path, "application/json", strings.NewReader(tc.body))
			if err != nil {
				t.Fatalf("post: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				t.Fatalf("status = %d, want 200", resp.StatusCode)
			}
			var decoded map[string]any
			if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if len(decoded) == 0 {
				t.Fatal("empty response body")
			}
		})
	}

	if seenAuth != "integration-secret" {
		t.Fatalf("upstream saw X-Proxy-Token = %q, want integration-secret", seenAuth)
	}
}

func waitForHealth(t *testing.T, url string) {
	t.Helper()
	for i := 0; i < 60; i++ {
		resp, err := http.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return
			}
		}
		time.Sleep(time.Second)
	}
	t.Fatalf("never became healthy: %s", url)
}
```

- [ ] **Step 2: Run it**

Run: `go test -count=1 -tags=docker_integration ./integration -run ProxyImage -v`
Expected: PASS, 3 subtests. Requires a running Docker daemon.

If `host.docker.internal` does not resolve on Linux, the `--add-host ...:host-gateway` flag is what makes it work; confirm the Docker version supports it with `docker --version` (needs 20.10 or newer).

- [ ] **Step 3: Confirm it does not run in the default suite**

Run: `go test ./integration/ -count=1`
Expected: PASS without building any image, since the file is behind the `docker_integration` tag.

- [ ] **Step 4: Commit**

```bash
gofmt -w integration/proxy_image_test.go
make lint
git add integration/proxy_image_test.go
git commit -m "test: Exercise the built proxy image on all three protocols"
```

---

### Task 6: Publish the tag

**Files:**
- Modify: `.github/workflows/release.yml:52-66` (matrix), `:96-107` (build step)

**Interfaces:**
- Consumes: the `proxy` build target from Task 4.
- Produces: DockerHub tags `proxy` and `<ref>-proxy`.

- [ ] **Step 1: Add a target to the existing matrix entries**

In the `docker` job's matrix, add `target: ""` to both existing entries, so all entries carry the same keys:

```yaml
          - variant: gemma3-4b
            latest: true
            target: ""
            model_url: https://huggingface.co/lmstudio-community/gemma-3-4b-it-GGUF/resolve/c536c4707e747055eecad7da65d46b6fb0ebaa79/gemma-3-4b-it-Q4_K_M.gguf
```

Do the same for `gemma3-12b`. Leave every existing URL and checksum untouched.

- [ ] **Step 2: Add the proxy entry**

Append to the matrix, after the `gemma3-12b` entry:

```yaml
          - variant: proxy
            latest: false
            target: proxy
            model_url: ""
            model_sha256: ""
            mmproj_url: ""
            mmproj_sha256: ""
```

The empty model values are required because every matrix entry must define the same keys for the build-args block to render.

- [ ] **Step 3: Pass the target to the build**

In the `Build and push image` step, add a `target` key above `platforms`:

```yaml
          context: .
          file: ./Dockerfile
          target: ${{ matrix.target }}
          platforms: linux/amd64,linux/arm64
```

An empty `target` means the default final stage, which is the llama.cpp one. That preserves the existing images exactly.

- [ ] **Step 4: Verify the workflow parses**

Run: `python3 -c "import sys,yaml;yaml.safe_load(open('.github/workflows/release.yml'))" 2>/dev/null || docker run --rm -v "$PWD:/w" -w /w mikefarah/yq:4 '.jobs.docker.strategy.matrix.include | length' .github/workflows/release.yml`
Expected: `3`, or no output and exit 0 from the Python check. If neither tool is available, run `gh workflow view ci-release` after pushing and confirm it is not reporting a parse error.

- [ ] **Step 5: Confirm latest is not applied to proxy**

Run: `grep -A3 'variant: proxy' .github/workflows/release.yml | grep latest`
Expected: `latest: false`

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: Publish the proxy image variant

Adds a build target to the matrix. Empty target keeps the existing entries on
the default final stage."
```

---

### Task 7: Document it

**Files:**
- Modify: `README.md` at the Docker tags table (line 129 area), and a new configuration section after it

**Interfaces:**
- Consumes: the measured image size from Task 4 Step 4.
- Produces: nothing.

- [ ] **Step 1: Add the tag table row**

In the `## Docker tags` table, add after the `gemma3-12b` row. Replace `<MEASURED>` with the number from Task 4 Step 4:

```
| `proxy`               | none (you supply)  | ~<MEASURED> MB |
```

Then extend the sentence below the table:

```
Version-pinned tags follow the pattern `v0.1.1-gemma3-4b`, `v0.1.1-gemma3-12b`,
`v0.1.1-proxy`. The `proxy` tag is never published as `latest`.
```

- [ ] **Step 2: Add the usage section**

Insert a new section immediately before `## Implemented routes`:

```markdown
## Bring your own model server (`:proxy`)

If you already run llama.cpp, vLLM, or anything else that speaks the OpenAI
chat-completions API, the `proxy` tag gives you the translation layer alone. It
contains no model and no inference engine.

```bash
docker run -d -p 8090:8090 \
  -e LK_UPSTREAM=http://llama.internal:8080/v1 \
  gokhalh/localaik:proxy
```

| Env var | Default | Description |
| --- | --- | --- |
| `LK_UPSTREAM` | `http://127.0.0.1:8080/v1` | Base URL of your model server |
| `LK_UPSTREAM_AUTH_HEADER` | unset | A full header line sent to your server, for example `Authorization: Bearer abc123` |
| `PORT` | `8090` | Port localaik listens on |

`LK_UPSTREAM_AUTH_HEADER` is sent only to your upstream. Credentials that
clients send to localaik are still discarded and never forwarded.

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
```

- [ ] **Step 3: Update the tested-SDKs and limitations text if it claims self-containment**

Run: `grep -n 'one container\|self-contained\|no internet\|No API key' README.md`

For each hit, confirm the claim is still true or scope it to the model-bundled tags. The `## Motivation` paragraph says "a single Docker container that speaks all three protocols backed by a local model", which remains accurate for the default tags; add ", or the `proxy` tag if you already run your own model server." to the end of that sentence.

- [ ] **Step 4: Check the rendered result**

Run: `grep -n '^## ' README.md`
Expected: the new `## Bring your own model server (:proxy)` heading appears between `## Tuning` and `## Implemented routes`.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: Document the proxy tag and its security profile"
```

---

### Task 8: Full verification and review

**Files:** none modified.

- [ ] **Step 1: Run everything**

```bash
make lint
go test -count=1 ./cmd/... ./internal/... ./integration/
go test -count=1 -tags=docker_integration ./integration -run ProxyImage
```

Expected: all pass.

- [ ] **Step 2: Confirm the existing images are untouched**

```bash
git diff main --stat -- Dockerfile
docker build -t localaik:regression-check .
docker run -d --name regression-check -p 18096:8090 localaik:regression-check
sleep 90
curl -s http://127.0.0.1:18096/health
docker rm -f regression-check
```

Expected: `{"status":"ok"}`. The only `Dockerfile` change should be the inserted stage.

- [ ] **Step 3: Confirm no secret is logged**

```bash
docker run --rm -e LK_UPSTREAM_AUTH_HEADER="Authorization: Bearer super-secret" \
  -e LK_UPSTREAM=http://127.0.0.1:9/v1 localaik:proxy-check 2>&1 | head -20 | grep -c super-secret
```

Expected: `0`. The container will fail to reach its upstream, which is fine; the check is that the credential never appears in output.

- [ ] **Step 4: Run the three required reviews**

Per the repo's PR workflow, before opening the PR:

1. The `everything-claude-code:code-reviewer` agent on the pending diff.
2. The `superpowers:requesting-code-review` skill against `main..HEAD`.
3. The `codex:review` command, falling back to `codex:codex-rescue` with a saved diff.

Fix anything actionable and re-review. Do not open the PR while findings are outstanding.

- [ ] **Step 5: Open the PR**

Use the `newpr` skill to generate the description.

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
| --- | --- |
| Third Dockerfile stage, alpine plus poppler-utils | 4 |
| llama.cpp stage stays last | 4 (Step 2), 8 (Step 2) |
| poppler required, not optional | 4 (Step 5) |
| `LK_UPSTREAM` env fallback | 3 |
| `LK_UPSTREAM_AUTH_HEADER`, full header line | 1, 3 |
| Flag over env over default | 3 |
| Credential injected in the transport, one place | 1 |
| Client credentials still stripped | 2 |
| Both properties tested together | 2 (Step 1) |
| `/health` unchanged against remote upstream | 4 (Step 6), 5 |
| `HEALTHCHECK` start-period reduced to 5s | 4 (Step 1) |
| Matrix entry, `proxy` and `vX.Y.Z-proxy` | 6 |
| Never `latest` | 6 (Steps 2, 5) |
| README security warning | 7 (Step 2) |
| Measure final image size | 4 (Step 4), 7 (Step 1) |
| Verify alpine pdftoppm matches | 4 (Step 5), 5 |
| Never log the credential | 8 (Step 3) |
| No model download | not implemented, correctly out of scope |
| No client authentication | not implemented, correctly out of scope |

**Placeholder scan:** `<MEASURED>` in Task 7 Step 1 is an intentional handoff from Task 4 Step 4, which produces the number. No other placeholders.

**Type consistency:** `newUpstreamAuthTransport(base http.RoundTripper, header string) http.RoundTripper` is defined in Task 1 Step 3 and referenced in Task 1 Steps 1 and 5 only. `resolveFlagDefault(envName, fallback string) string` is defined in Task 3 Step 3 and used in the same step. `Config.UpstreamAuthHeader` is added in Task 1 Step 5 and consumed in Tasks 2 and 3. `roundTripHandler` and `newTestServer` are pre-existing and referenced with their file locations. Consistent.
