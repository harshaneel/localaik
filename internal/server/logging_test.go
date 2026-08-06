package server

import (
	"bytes"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newTestLogger() (*log.Logger, *bytes.Buffer) {
	var buf bytes.Buffer
	return log.New(&buf, "", 0), &buf
}

func TestWithRequestLogLogsAccessLine(t *testing.T) {
	logger, buf := newTestLogger()
	h := WithRequestLog(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}), logger)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil))

	line := buf.String()
	for _, want := range []string{"POST", "/v1/chat/completions", "200"} {
		if !strings.Contains(line, want) {
			t.Fatalf("access line %q missing %q", line, want)
		}
	}
	if strings.TrimSpace(line) == "" {
		t.Fatal("expected an access line")
	}
}

func TestWithRequestLogCapturesErrorStatusAndText(t *testing.T) {
	logger, buf := newTestLogger()
	h := WithRequestLog(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	}), logger)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", nil))

	line := buf.String()
	if !strings.Contains(line, "502") || !strings.Contains(line, "Bad Gateway") {
		t.Fatalf("error line %q missing status or text", line)
	}
}

func TestWithRequestLogDefaultsToStatusOK(t *testing.T) {
	logger, buf := newTestLogger()
	h := WithRequestLog(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "body without an explicit WriteHeader")
	}), logger)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	if !strings.Contains(buf.String(), "200") {
		t.Fatalf("expected status 200 in %q", buf.String())
	}
}

// The wrapper must stay an http.Flusher, or SSE streaming through the proxy
// would buffer instead of flushing.
func TestWithRequestLogPreservesFlusher(t *testing.T) {
	logger, _ := newTestLogger()
	flushed := false
	h := WithRequestLog(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("the wrapped writer is not an http.Flusher; streaming would break")
		}
		_, _ = io.WriteString(w, "data: chunk\n\n")
		f.Flush()
		flushed = true
	}), logger)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1beta/models/m:streamGenerateContent", nil))

	if !flushed {
		t.Fatal("handler never reached the flush path")
	}
	if !rec.Flushed {
		t.Fatal("Flush did not reach the underlying ResponseWriter")
	}
}

func TestWithRequestLogNilLoggerDisables(t *testing.T) {
	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { called = true })

	h := WithRequestLog(next, nil)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/health", nil))

	if !called {
		t.Fatal("nil logger must still pass the request through to next")
	}
}

func TestWithRequestLogSkipsSuccessfulHealthChecks(t *testing.T) {
	logger, buf := newTestLogger()
	h := WithRequestLog(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}), logger)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/health", nil))

	if strings.TrimSpace(buf.String()) != "" {
		t.Fatalf("a healthy /health probe should not log, got %q", buf.String())
	}
}

func TestWithRequestLogLogsFailingHealthChecks(t *testing.T) {
	logger, buf := newTestLogger()
	h := WithRequestLog(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}), logger)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/health", nil))

	if !strings.Contains(buf.String(), "503") {
		t.Fatalf("a failing /health probe should log, got %q", buf.String())
	}
}

// Nothing about the request headers, which can carry credentials, may reach the
// access line.
func TestWithRequestLogNeverLogsHeaders(t *testing.T) {
	logger, buf := newTestLogger()
	h := WithRequestLog(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}), logger)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	req.Header.Set("Authorization", "Bearer super-secret-token")
	req.Header.Set("X-Api-Key", "another-secret")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if strings.Contains(buf.String(), "super-secret-token") || strings.Contains(buf.String(), "another-secret") {
		t.Fatalf("access line leaked a header credential: %q", buf.String())
	}
}
