package server

import (
	"net/http"
	"strings"

	"golang.org/x/net/http/httpguts"
)

type upstreamAuthTransport struct {
	base  http.RoundTripper
	host  string
	name  string
	value string
}

// ValidUpstreamAuthHeader reports whether header is a line the transport will
// actually send. Startup warnings must use this, not a second predicate.
func ValidUpstreamAuthHeader(header string) bool {
	_, _, ok := parseUpstreamAuthHeader(header)
	return ok
}

// Only the first colon separates the pair, so values may contain further ones.
func parseUpstreamAuthHeader(header string) (string, string, bool) {
	name, value, found := strings.Cut(header, ":")
	name = strings.TrimSpace(name)
	value = strings.TrimSpace(value)
	if !found || name == "" || value == "" {
		return "", "", false
	}
	// net/http rejects these at the wire on every request, with an error that
	// never mentions the env var that caused it.
	if !httpguts.ValidHeaderFieldName(name) || !httpguts.ValidHeaderFieldValue(value) {
		return "", "", false
	}
	return name, value, true
}

// newUpstreamAuthTransport returns base unchanged when header is not a usable
// "Name: value" line, so a misconfigured value cannot silently drop requests.
func newUpstreamAuthTransport(base http.RoundTripper, header, host string) http.RoundTripper {
	if base == nil {
		base = http.DefaultTransport
	}

	name, value, ok := parseUpstreamAuthHeader(header)
	if !ok {
		return base
	}

	return &upstreamAuthTransport{base: base, host: host, name: name, value: value}
}

func (t *upstreamAuthTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// A 3xx re-enters here with the target's URL, so setting the header for
	// every host would hand the credential to whatever the redirect names.
	if !strings.EqualFold(req.URL.Hostname(), t.host) {
		return t.base.RoundTrip(req)
	}

	clone := req.Clone(req.Context())
	clone.Header.Set(t.name, t.value)
	return t.base.RoundTrip(clone)
}
