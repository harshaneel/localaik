package server

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/harshaneel/localaik/internal/pdf"
)

// redirectingTransport answers upstream.test with a 302 to location and every
// other host with a marker body, recording the headers each host received.
type redirectingTransport struct {
	location string

	mu   sync.Mutex
	seen map[string]http.Header
}

func (r *redirectingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	r.mu.Lock()
	if r.seen == nil {
		r.seen = make(map[string]http.Header)
	}
	r.seen[req.URL.Hostname()] = req.Header.Clone()
	r.mu.Unlock()

	recorder := httptest.NewRecorder()
	if req.URL.Hostname() == "upstream.test" {
		recorder.Header().Set("Location", r.location)
		recorder.WriteHeader(http.StatusFound)
		return recorder.Result(), nil
	}

	recorder.Header().Set("Content-Type", "application/json")
	recorder.WriteHeader(http.StatusOK)
	_, _ = recorder.WriteString(`{"leaked":true}`)
	return recorder.Result(), nil
}

func (r *redirectingTransport) headers(host string) (http.Header, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	header, ok := r.seen[host]
	return header, ok
}

// A 3xx re-enters the transport with the target's URL, so a transport that sets
// the credential unconditionally hands it to whatever host the redirect names.
func TestUpstreamAuthTransportWithholdsCredentialFromOtherHosts(t *testing.T) {
	for _, name := range []string{"Authorization", "X-Proxy-Token"} {
		t.Run(name, func(t *testing.T) {
			base := &redirectingTransport{location: "http://redirect.test/v1/models"}
			client := &http.Client{
				Transport: newUpstreamAuthTransport(base, name+": upstream-secret", "upstream.test"),
			}

			resp, err := client.Get("http://upstream.test/v1/models")
			if err != nil {
				t.Fatalf("Get returned error: %v", err)
			}
			defer resp.Body.Close()

			configured, ok := base.headers("upstream.test")
			if !ok {
				t.Fatal("configured upstream was never called")
			}
			if got := configured.Get(name); got != "upstream-secret" {
				t.Fatalf("configured upstream saw %s = %q, want the credential", name, got)
			}

			target, ok := base.headers("redirect.test")
			if !ok {
				t.Fatal("redirect target was never reached, so this test proves nothing")
			}
			if got := target.Get(name); got != "" {
				t.Fatalf("redirect target received the credential: %s = %q", name, got)
			}
		})
	}
}

func TestCredentialedClientDoesNotFollowUpstreamRedirects(t *testing.T) {
	base := &redirectingTransport{location: "http://redirect.test/v1/chat/completions"}

	srv, err := New(Config{
		UpstreamBaseURL:    "http://upstream.test/v1",
		UpstreamAuthHeader: "X-Proxy-Token: upstream-secret",
		HTTPClient:         &http.Client{Transport: base},
		PDFRenderer:        pdf.RendererFunc(func(context.Context, []byte) ([][]byte, error) { return nil, nil }),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"m","messages":[]}`))
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusFound {
		t.Fatalf("status = %d, want the 302 passed through to the caller", rec.Code)
	}
	if strings.Contains(rec.Body.String(), "leaked") {
		t.Fatalf("caller received the redirect target's body: %s", rec.Body.String())
	}
	if _, ok := base.headers("redirect.test"); ok {
		t.Fatal("credentialed client followed the redirect")
	}
}

// The model-bundled images configure no credential and must keep the stdlib's
// redirect handling.
func TestClientWithoutCredentialStillFollowsRedirects(t *testing.T) {
	base := &redirectingTransport{location: "http://redirect.test/v1/chat/completions"}

	srv, err := New(Config{
		UpstreamBaseURL: "http://upstream.test/v1",
		HTTPClient:      &http.Client{Transport: base},
		PDFRenderer:     pdf.RendererFunc(func(context.Context, []byte) ([][]byte, error) { return nil, nil }),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"m","messages":[]}`))
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 after following the redirect", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "leaked") {
		t.Fatalf("redirect was not followed; body = %s", rec.Body.String())
	}
}
