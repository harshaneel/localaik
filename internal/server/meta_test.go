package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/harshaneel/localaik/internal/pdf"
	"github.com/harshaneel/localaik/internal/protocol/gemini"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

func newTestServer(t *testing.T, upstream http.Handler) *Server {
	t.Helper()
	srv, err := New(Config{
		UpstreamBaseURL: "http://upstream.test/v1",
		HTTPClient: &http.Client{
			Transport: roundTripHandler{handler: upstream},
		},
		PDFRenderer: pdf.RendererFunc(func(_ context.Context, _ []byte) ([][]byte, error) { return nil, nil }),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	return srv
}

func TestServerOpenAILegacyCompletionsPassthrough(t *testing.T) {
	var seenPath, seenMethod string

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenPath, seenMethod = r.URL.Path, r.Method
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"cmpl_1","object":"text_completion"}`))
	})

	srv := newTestServer(t, upstream)

	req := httptest.NewRequest(http.MethodPost, "/v1/completions", bytes.NewBufferString(`{"prompt":"hi"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if seenPath != "/v1/completions" || seenMethod != http.MethodPost {
		t.Fatalf("upstream got %s %s, want POST /v1/completions", seenMethod, seenPath)
	}
}

func TestServerOpenAIModelsList(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" || r.Method != http.MethodGet {
			t.Fatalf("upstream got %s %s, want GET /v1/models", r.Method, r.URL.Path)
		}
		writeJSON(w, http.StatusOK, openaip.ModelList{
			Object: "list",
			Data:   []openaip.Model{{ID: "gemma-3-4b", Object: "model"}},
		})
	})

	srv := newTestServer(t, upstream)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var got openaip.ModelList
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(got.Data) != 1 || got.Data[0].ID != "gemma-3-4b" {
		t.Fatalf("response = %#v", got)
	}
}

func TestServerOpenAIModelGet(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models/gemma-3-4b" || r.Method != http.MethodGet {
			t.Fatalf("upstream got %s %s, want GET /v1/models/gemma-3-4b", r.Method, r.URL.Path)
		}
		writeJSON(w, http.StatusOK, openaip.Model{ID: "gemma-3-4b", Object: "model"})
	})

	srv := newTestServer(t, upstream)

	req := httptest.NewRequest(http.MethodGet, "/v1/models/gemma-3-4b", nil)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
}

func TestServerGeminiModelsList(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" || r.Method != http.MethodGet {
			t.Fatalf("upstream got %s %s, want GET /v1/models", r.Method, r.URL.Path)
		}
		writeJSON(w, http.StatusOK, openaip.ModelList{
			Object: "list",
			Data:   []openaip.Model{{ID: "gemma-3-4b", Object: "model"}},
		})
	})

	srv := newTestServer(t, upstream)

	req := httptest.NewRequest(http.MethodGet, "/v1beta/models", nil)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var got gemini.ListModelsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(got.Models) != 1 || got.Models[0].Name != "models/gemma-3-4b" {
		t.Fatalf("response = %#v", got)
	}
}

func TestServerGeminiModelGet(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models/gemma-3-4b" {
			t.Fatalf("upstream path = %q, want /v1/models/gemma-3-4b", r.URL.Path)
		}
		writeJSON(w, http.StatusOK, openaip.Model{ID: "gemma-3-4b"})
	})

	srv := newTestServer(t, upstream)

	req := httptest.NewRequest(http.MethodGet, "/v1beta/models/gemma-3-4b", nil)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var got gemini.Model
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got.Name != "models/gemma-3-4b" {
		t.Fatalf("response = %#v", got)
	}
}

func TestServerGeminiCountTokens(t *testing.T) {
	var upstreamBody map[string]any

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenize" || r.Method != http.MethodPost {
			t.Fatalf("upstream got %s %s, want POST /tokenize", r.Method, r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&upstreamBody); err != nil {
			t.Fatalf("decode upstream body: %v", err)
		}
		writeJSON(w, http.StatusOK, map[string]any{"tokens": []int{1, 2, 3, 4, 5}})
	})

	srv := newTestServer(t, upstream)

	body := `{"contents":[{"role":"user","parts":[{"text":"hello"},{"text":"world"}]}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemma-3-4b:countTokens", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if upstreamBody["content"] != "hello\nworld" {
		t.Fatalf("upstream content = %#v, want hello\\nworld", upstreamBody["content"])
	}
	var got gemini.CountTokensResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got.TotalTokens != 5 {
		t.Fatalf("totalTokens = %d, want 5", got.TotalTokens)
	}
}

func TestServerGeminiModelGetMalformedPath(t *testing.T) {
	srv := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("upstream should not be reached, got %s", r.URL.Path)
	}))

	cases := []string{
		"/v1beta/models/foo/bar",  // path traversal
		"/v1beta/models/foo:bar",  // action-verb collision
		"/v1beta/models/foo:list", // action-verb collision
	}
	for _, path := range cases {
		t.Run(path, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, path, nil)
			rec := httptest.NewRecorder()
			srv.ServeHTTP(rec, req)
			if rec.Code != http.StatusNotFound {
				t.Fatalf("status = %d, want 404", rec.Code)
			}
		})
	}
}

func TestServerOpenAIModelGetMalformedPath(t *testing.T) {
	srv := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("upstream should not be reached, got %s", r.URL.Path)
	}))

	req := httptest.NewRequest(http.MethodGet, "/v1/models/foo/bar", nil)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", rec.Code)
	}
}

func TestServerMethodConfusion(t *testing.T) {
	srv := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("upstream should not be reached, got %s %s", r.Method, r.URL.Path)
	}))

	cases := []struct {
		method string
		path   string
	}{
		{http.MethodPost, "/v1/models"},
		{http.MethodPost, "/v1/models/gemma-3-4b"},
		{http.MethodPut, "/v1beta/models"},
		{http.MethodPut, "/v1beta/models/gemma-3-4b"},
		{http.MethodGet, "/v1/chat/completions"},
		{http.MethodGet, "/v1/completions"},
		{http.MethodGet, "/v1beta/models/gemma-3-4b:countTokens"},
	}
	for _, tc := range cases {
		t.Run(tc.method+" "+tc.path, func(t *testing.T) {
			req := httptest.NewRequest(tc.method, tc.path, nil)
			rec := httptest.NewRecorder()
			srv.ServeHTTP(rec, req)
			if rec.Code != http.StatusNotFound {
				t.Fatalf("status = %d, want 404", rec.Code)
			}
		})
	}
}

func TestServerGeminiModelsListUpstreamError(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"upstream blew up","type":"server_error"}}`))
	})

	srv := newTestServer(t, upstream)

	req := httptest.NewRequest(http.MethodGet, "/v1beta/models", nil)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}
	var errResp gemini.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode response: %v; body=%s", err, rec.Body.String())
	}
	if errResp.Error.Message == "" {
		t.Fatalf("error message missing; body=%s", rec.Body.String())
	}
}

func TestServerGeminiCountTokensUpstreamError(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"bad tokens","type":"invalid_request_error"}}`))
	})

	srv := newTestServer(t, upstream)

	body := `{"contents":[{"parts":[{"text":"hi"}]}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemma-3-4b:countTokens", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", rec.Code, rec.Body.String())
	}
	var errResp gemini.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if errResp.Error.Message == "" {
		t.Fatalf("error message missing; body=%s", rec.Body.String())
	}
}

func TestServerPassthroughStripsAuthHeaders(t *testing.T) {
	var seenAuth, seenGoogKey string

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenAuth = r.Header.Get("Authorization")
		seenGoogKey = r.Header.Get("X-Goog-Api-Key")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	})

	cases := []struct {
		name   string
		method string
		path   string
	}{
		{"legacy_completions", http.MethodPost, "/v1/completions"},
		{"openai_models_list", http.MethodGet, "/v1/models"},
		{"openai_model_get", http.MethodGet, "/v1/models/gemma-3-4b"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			seenAuth, seenGoogKey = "", ""
			srv := newTestServer(t, upstream)

			req := httptest.NewRequest(tc.method, tc.path, bytes.NewBufferString(`{}`))
			req.Header.Set("Authorization", "Bearer secret")
			req.Header.Set("X-Goog-Api-Key", "secret-key")
			rec := httptest.NewRecorder()
			srv.ServeHTTP(rec, req)

			if seenAuth != "" {
				t.Fatalf("Authorization leaked upstream: %q", seenAuth)
			}
			if seenGoogKey != "" {
				t.Fatalf("X-Goog-Api-Key leaked upstream: %q", seenGoogKey)
			}
		})
	}
}
