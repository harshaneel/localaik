//go:build docker_integration

package integration

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"
)

// Exercises the built proxy image against a stub upstream, proving all three
// protocol surfaces round-trip without an inference stack in the container.
func TestProxyImageRoundTripsAllProtocols(t *testing.T) {
	image := "localaik:proxy-integration"

	build := exec.Command("docker", "build", "--target", "proxy", "-t", image, "..")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("docker build failed: %v\n%s", err, out)
	}

	var authMu sync.Mutex
	var seenAuth string
	stub := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authMu.Lock()
		seenAuth = r.Header.Get("X-Proxy-Token")
		authMu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/tokenize") {
			_, _ = w.Write([]byte(`{"tokens":[1,2,3]}`))
			return
		}
		_, _ = w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"stubbed"},"finish_reason":"stop"}]}`))
	}))

	// NewUnstartedServer already bound a loopback-only listener; close it and
	// swap in one bound to 0.0.0.0 so the container can reach it.
	stub.Listener.Close()
	listener, err := net.Listen("tcp", "0.0.0.0:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	stub.Listener = listener
	stub.Start()
	defer stub.Close()

	stubPort := listener.Addr().(*net.TCPAddr).Port
	upstream := fmt.Sprintf("http://host.docker.internal:%d/v1", stubPort)

	// Best-effort: clear any container left behind by a prior run that
	// crashed before its own cleanup ran, so the name doesn't collide.
	_ = exec.Command("docker", "rm", "-f", "proxy-integration").Run()

	run := exec.Command("docker", "run", "-d", "--name", "proxy-integration",
		"--add-host", "host.docker.internal:host-gateway",
		"-p", "18097:8090",
		"-e", "LK_UPSTREAM="+upstream,
		"-e", "LK_UPSTREAM_AUTH_HEADER=X-Proxy-Token: integration-secret",
		image)
	if out, err := run.CombinedOutput(); err != nil {
		t.Fatalf("docker run failed: %v\n%s", err, out)
	}
	defer exec.Command("docker", "rm", "-f", "proxy-integration").Run()

	waitForHealth(t, "http://127.0.0.1:18097/health")

	cases := []struct {
		name        string
		path        string
		body        string
		expectedKey string
	}{
		{"openai", "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, "choices"},
		{"gemini", "/v1beta/models/m:generateContent", `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`, "candidates"},
		{"anthropic", "/v1/messages", `{"max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`, "content"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp, err := http.Post("http://127.0.0.1:18097"+tc.path, "application/json", strings.NewReader(tc.body))
			if err != nil {
				t.Fatalf("post: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				t.Fatalf("status = %d, want 200", resp.StatusCode)
			}
			var decoded map[string]any
			if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
				t.Fatalf("decode: %v", err)
			}
			// The expected key catches a response shaped by the wrong handler.
			if _, ok := decoded[tc.expectedKey]; !ok {
				t.Fatalf("response missing %q key, got keys %v", tc.expectedKey, mapKeys(decoded))
			}
			if tc.name == "anthropic" {
				if role, _ := decoded["role"].(string); role != "assistant" {
					t.Fatalf("role = %q, want assistant", role)
				}
			}
		})
	}

	authMu.Lock()
	got := seenAuth
	authMu.Unlock()
	if got != "integration-secret" {
		t.Fatalf("upstream saw X-Proxy-Token = %q, want integration-secret", got)
	}
}

func mapKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

func waitForHealth(t *testing.T, url string) {
	t.Helper()
	for i := 0; i < 60; i++ {
		resp, err := http.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return
			}
		}
		time.Sleep(time.Second)
	}
	t.Fatalf("never became healthy: %s", url)
}
