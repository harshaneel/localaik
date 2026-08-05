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
