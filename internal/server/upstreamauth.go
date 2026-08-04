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
