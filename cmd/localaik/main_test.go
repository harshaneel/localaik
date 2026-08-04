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

func TestIsValidAuthHeaderUnset(t *testing.T) {
	if got := isValidAuthHeader(""); got != false {
		t.Fatalf("isValidAuthHeader(\"\") = %v, want false", got)
	}
}

func TestIsValidAuthHeaderValid(t *testing.T) {
	if got := isValidAuthHeader("Authorization: Bearer token123"); got != true {
		t.Fatalf("isValidAuthHeader(\"Authorization: Bearer token123\") = %v, want true", got)
	}
}

func TestIsValidAuthHeaderNoColon(t *testing.T) {
	if got := isValidAuthHeader("InvalidHeader NoColon"); got != false {
		t.Fatalf("isValidAuthHeader(\"InvalidHeader NoColon\") = %v, want false", got)
	}
}

func TestIsValidAuthHeaderEmptyName(t *testing.T) {
	if got := isValidAuthHeader(": value"); got != false {
		t.Fatalf("isValidAuthHeader(\": value\") = %v, want false", got)
	}
}

func TestIsValidAuthHeaderEmptyValue(t *testing.T) {
	if got := isValidAuthHeader("Name:"); got != false {
		t.Fatalf("isValidAuthHeader(\"Name:\") = %v, want false", got)
	}
}

func TestIsValidAuthHeaderWhitespaceOnly(t *testing.T) {
	if got := isValidAuthHeader("  :  "); got != false {
		t.Fatalf("isValidAuthHeader(\"  :  \") = %v, want false", got)
	}
}
