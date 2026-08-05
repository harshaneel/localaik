package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/harshaneel/localaik/internal/pdf"
)

func TestRedactUpstream(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{"no userinfo is returned unchanged", "http://llama.internal:8080/v1", "http://llama.internal:8080/v1"},
		{"password is removed", "http://user:secret@llama.internal:8080/v1", "http://redacted@llama.internal:8080/v1"},
		{"username alone is removed", "http://user@llama.internal:8080/v1", "http://redacted@llama.internal:8080/v1"},
		{"unparseable input does not echo back", "http://[::1", "invalid"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := RedactUpstream(tc.raw); got != tc.want {
				t.Errorf("RedactUpstream(%q) = %q, want %q", tc.raw, got, tc.want)
			}
		})
	}
}

// An unreachable upstream is the case an operator has to debug, so /health names
// it rather than only reporting that something is wrong.
func TestHealthReportsTheUpstreamWhenUnreachable(t *testing.T) {
	srv, err := New(Config{
		UpstreamBaseURL: "http://127.0.0.1:9/v1",
		HTTPClient:      &http.Client{},
		PDFRenderer:     pdf.RendererFunc(nil),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/health", nil))

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", rec.Code)
	}

	var body map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if body["status"] != "unhealthy" {
		t.Errorf("status = %q, want unhealthy", body["status"])
	}
	if body["upstream"] != "http://127.0.0.1:9/v1" {
		t.Errorf("upstream = %q, want the configured URL", body["upstream"])
	}
}

func TestHealthDoesNotLeakUpstreamCredentials(t *testing.T) {
	srv, err := New(Config{
		UpstreamBaseURL: "http://user:supersecret@127.0.0.1:9/v1",
		HTTPClient:      &http.Client{},
		PDFRenderer:     pdf.RendererFunc(nil),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/health", nil))

	if strings.Contains(rec.Body.String(), "supersecret") {
		t.Fatalf("/health leaked the upstream password: %s", rec.Body.String())
	}
}
