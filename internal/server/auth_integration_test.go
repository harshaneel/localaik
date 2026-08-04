package server

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/harshaneel/localaik/internal/pdf"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

// Every upstream route must carry the proxy's credential and none of the
// caller's.
func TestUpstreamAuthHeaderReachesEveryUpstreamPath(t *testing.T) {
	cases := []struct {
		name   string
		method string
		path   string
		body   string
	}{
		{"openai_chat", http.MethodPost, "/v1/chat/completions", `{"model":"m","messages":[]}`},
		{"openai_models", http.MethodGet, "/v1/models", ""},
		{"openai_completions", http.MethodPost, "/v1/completions", `{"prompt":"hello"}`},
		{"openai_model_get", http.MethodGet, "/v1/models/gpt-4", ""},
		{"gemini_generate", http.MethodPost, "/v1beta/models/m:generateContent", `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`},
		{"gemini_count_tokens", http.MethodPost, "/v1beta/models/m:countTokens", `{"contents":[{"parts":[{"text":"hi"}]}]}`},
		{"gemini_models_list", http.MethodGet, "/v1beta/models", ""},
		{"gemini_model_get", http.MethodGet, "/v1beta/models/gemini-2.5-pro", ""},
		{"anthropic_messages", http.MethodPost, "/v1/messages", `{"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`},
		{"anthropic_count_tokens", http.MethodPost, "/v1/messages/count_tokens", `{"messages":[{"role":"user","content":"hi"}]}`},
		{"health", http.MethodGet, "/health", ""},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var called bool
			var seenAuth, seenClientAuth, seenAPIKey, seenGoogKey string

			upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				called = true
				seenAuth = r.Header.Get("X-Proxy-Token")
				seenClientAuth = r.Header.Get("Authorization")
				seenAPIKey = r.Header.Get("X-Api-Key")
				seenGoogKey = r.Header.Get("X-Goog-Api-Key")

				switch r.URL.Path {
				case "/tokenize":
					writeJSON(w, http.StatusOK, map[string]any{"tokens": []int{1, 2}})
				case "/v1/models":
					writeJSON(w, http.StatusOK, openaip.ModelList{Object: "list", Data: []openaip.Model{{ID: "m"}}})
				case "/v1/models/gpt-4":
					writeJSON(w, http.StatusOK, openaip.Model{ID: "gpt-4"})
				case "/health":
					w.WriteHeader(http.StatusOK)
				default:
					writeJSON(w, http.StatusOK, openaip.ChatCompletionResponse{
						Choices: []openaip.Choice{{Message: openaip.Message{Content: "ok"}, FinishReason: "stop"}},
					})
				}
			})

			srv, err := New(Config{
				UpstreamBaseURL:    "http://upstream.test/v1",
				UpstreamAuthHeader: "X-Proxy-Token: upstream-secret",
				HTTPClient:         &http.Client{Transport: roundTripHandler{handler: upstream}},
				PDFRenderer:        pdf.RendererFunc(func(context.Context, []byte) ([][]byte, error) { return nil, nil }),
			})
			if err != nil {
				t.Fatalf("New returned error: %v", err)
			}

			var reader *bytes.Buffer
			if tc.body != "" {
				reader = bytes.NewBufferString(tc.body)
			} else {
				reader = bytes.NewBuffer(nil)
			}
			req := httptest.NewRequest(tc.method, tc.path, reader)
			req.Header.Set("Authorization", "Bearer client-secret")
			req.Header.Set("X-Api-Key", "client-anthropic-key")
			req.Header.Set("X-Goog-Api-Key", "client-google-key")

			rec := httptest.NewRecorder()
			srv.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
			}
			if !called {
				t.Fatal("upstream was never called, so the header check proves nothing")
			}
			if seenAuth != "upstream-secret" {
				t.Fatalf("X-Proxy-Token = %q, want the proxy credential", seenAuth)
			}
			if seenClientAuth != "" || seenAPIKey != "" || seenGoogKey != "" {
				t.Fatalf("client credentials leaked upstream: auth=%q apikey=%q googkey=%q", seenClientAuth, seenAPIKey, seenGoogKey)
			}
		})
	}
}

func TestNoUpstreamAuthHeaderWhenUnset(t *testing.T) {
	var seen http.Header

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Clone()
		writeJSON(w, http.StatusOK, openaip.ChatCompletionResponse{
			Choices: []openaip.Choice{{Message: openaip.Message{Content: "ok"}, FinishReason: "stop"}},
		})
	})

	srv := newTestServer(t, upstream)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if got := seen.Get("Authorization"); got != "" {
		t.Fatalf("Authorization = %q, want none when no credential is configured", got)
	}
}

// The Gemini streaming route builds its own request; confirm the credential is
// present there too.
func TestUpstreamAuthHeaderOnStreamingRoute(t *testing.T) {
	var seen string

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Get("X-Proxy-Token")
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	})

	srv, err := New(Config{
		UpstreamBaseURL:    "http://upstream.test/v1",
		UpstreamAuthHeader: "X-Proxy-Token: upstream-secret",
		HTTPClient:         &http.Client{Transport: roundTripHandler{handler: upstream}},
		PDFRenderer:        pdf.RendererFunc(func(context.Context, []byte) ([][]byte, error) { return nil, nil }),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	body := `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/m:streamGenerateContent", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if seen != "upstream-secret" {
		t.Fatalf("X-Proxy-Token = %q on the streaming route, want the proxy credential", seen)
	}
	if !bytes.Contains(rec.Body.Bytes(), []byte("data:")) {
		t.Fatalf("response body missing data: frame; got %s", rec.Body.String())
	}
}

func TestNewDoesNotMutateCallerClient(t *testing.T) {
	sentinelTransport := &http.Transport{DisableCompression: true}
	clientToPass := &http.Client{Transport: sentinelTransport}

	_, err := New(Config{
		UpstreamBaseURL:    "http://upstream.test/v1",
		UpstreamAuthHeader: "X-Proxy-Token: upstream-secret",
		HTTPClient:         clientToPass,
		PDFRenderer:        pdf.RendererFunc(func(context.Context, []byte) ([][]byte, error) { return nil, nil }),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	if clientToPass.Transport != sentinelTransport {
		t.Fatal("New mutated the caller's http.Client; it should have made a copy")
	}
}
