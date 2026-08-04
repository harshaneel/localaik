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
