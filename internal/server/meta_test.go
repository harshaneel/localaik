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

	req := httptest.NewRequest(http.MethodGet, "/v1beta/models/foo/bar", nil)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", rec.Code)
	}
}
