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

// The single definition of a usable credential line. A capturing transport
// bypasses net/http's wire-level field checks, so these run at the predicate.
func TestValidUpstreamAuthHeader(t *testing.T) {
	cases := []struct {
		name   string
		header string
		want   bool
	}{
		{"typical", "Authorization: Bearer token123", true},
		{"value keeps later colons", "Authorization: Bearer a:b", true},
		{"untrimmed", "  X-Api-Key :   abc123  ", true},
		{"empty", "", false},
		{"whitespace only", "  :  ", false},
		{"no colon", "InvalidHeader NoColon", false},
		{"no name", ": value", false},
		{"no value", "Name:", false},
		{"space in name", "Bad Name: secret", false},
		{"newline in value", "Authorization: Bearer a\nX-Evil: b", false},
		{"carriage return in value", "Authorization: Bearer a\rb", false},
		{"null in value", "Authorization: Bearer a\x00b", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ValidUpstreamAuthHeader(tc.header); got != tc.want {
				t.Fatalf("ValidUpstreamAuthHeader(%q) = %v, want %v", tc.header, got, tc.want)
			}
		})
	}
}

// A header the predicate rejects must never reach the wire, where net/http
// would fail every request with an error that does not name the env var.
func TestRejectedHeaderIsNeverSentUpstream(t *testing.T) {
	for _, header := range []string{"Bad Name: secret", "Authorization: Bearer a\nX-Evil: b"} {
		capture := &capturingTransport{}
		transport := newUpstreamAuthTransport(capture, header, "upstream.test")

		req := httptest.NewRequest(http.MethodGet, "http://upstream.test/v1/models", nil)
		if _, err := transport.RoundTrip(req); err != nil {
			t.Fatalf("RoundTrip returned error for %q: %v", header, err)
		}
		if len(capture.seen) != 0 {
			t.Fatalf("header %q produced %v, want none", header, capture.seen)
		}
	}
}

func TestUpstreamAuthTransportAddsHeader(t *testing.T) {
	capture := &capturingTransport{}
	transport := newUpstreamAuthTransport(capture, "Authorization: Bearer secret", "upstream.test")

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
	transport := newUpstreamAuthTransport(capture, "  X-Api-Key :   abc123  ", "upstream.test")

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
		transport := newUpstreamAuthTransport(capture, header, "upstream.test")

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
	transport := newUpstreamAuthTransport(capture, "Authorization: Bearer secret", "upstream.test")

	req := httptest.NewRequest(http.MethodGet, "http://upstream.test/v1/models", nil)
	if _, err := transport.RoundTrip(req); err != nil {
		t.Fatalf("RoundTrip returned error: %v", err)
	}

	if req.Header.Get("Authorization") != "" {
		t.Fatal("RoundTrip mutated the caller's request")
	}
}
