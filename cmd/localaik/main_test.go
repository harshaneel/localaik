package main

import (
	"testing"

	"github.com/harshaneel/localaik/internal/server"
)

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

func TestResolveUpstream(t *testing.T) {
	tests := []struct {
		name       string
		flagValue  string
		flagSet    bool
		env        string
		wantURL    string
		wantSource string
	}{
		{"flag beats env", "http://flag:1/v1", true, "http://env:2/v1", "http://flag:1/v1", "flag"},
		{"env beats default", defaultUpstream, false, "http://env:2/v1", "http://env:2/v1", "LK_UPSTREAM"},
		{"default when neither", defaultUpstream, false, "", defaultUpstream, "default"},
		{"empty env is not a value", defaultUpstream, false, "", defaultUpstream, "default"},
		// Passing the default explicitly is how an operator opts into this
		// container's own loopback, so it must not read as "default".
		{"flag matching the default still counts as explicit", defaultUpstream, true, "", defaultUpstream, "flag"},
		// An explicit empty flag must surface as empty so main rejects it,
		// rather than falling through to the loopback default.
		{"explicit empty flag stays empty", "", true, "http://env:2/v1", "", "flag"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotURL, gotSource := resolveUpstream(tc.flagValue, tc.flagSet, tc.env)
			if gotURL != tc.wantURL {
				t.Errorf("url = %q, want %q", gotURL, tc.wantURL)
			}
			if gotSource != tc.wantSource {
				t.Errorf("source = %q, want %q", gotSource, tc.wantSource)
			}
		})
	}
}

func TestUpstreamRequiredButUnset(t *testing.T) {
	tests := []struct {
		name       string
		source     string
		requireEnv string
		want       bool
	}{
		{"proxy image with nothing configured", "default", "1", true},
		{"proxy image with LK_UPSTREAM set", "LK_UPSTREAM", "1", false},
		{"proxy image with an explicit flag", "flag", "1", false},
		// The bundled images pass --upstream, so they are safe even if someone
		// sets the marker by hand.
		{"bundled image on the default", "default", "", false},
		{"bundled image with a flag", "flag", "", false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := upstreamRequiredButUnset(tc.source, tc.requireEnv); got != tc.want {
				t.Errorf("upstreamRequiredButUnset(%q, %q) = %v, want %v", tc.source, tc.requireEnv, got, tc.want)
			}
		})
	}
}

func TestRequestLoggingEnabled(t *testing.T) {
	tests := []struct {
		env  string
		want bool
	}{
		{"", true},
		{"off", false},
		{"OFF", false},
		{"Off", false},
		{"on", true},
		{"0", true},
		{"false", true},
	}
	for _, tc := range tests {
		if got := requestLoggingEnabled(tc.env); got != tc.want {
			t.Errorf("requestLoggingEnabled(%q) = %v, want %v", tc.env, got, tc.want)
		}
	}
}

// The startup warning must be driven by the same predicate the transport uses;
// server.ValidUpstreamAuthHeader owns the table of cases.
func TestStartupWarningUsesTheServerPredicate(t *testing.T) {
	if server.ValidUpstreamAuthHeader("Authorization: Bearer token123") != true {
		t.Fatal("a valid header line was rejected")
	}
	if server.ValidUpstreamAuthHeader("InvalidHeader NoColon") != false {
		t.Fatal("a header line with no colon was accepted")
	}
}
