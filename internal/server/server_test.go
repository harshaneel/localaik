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

func TestServerGeminiGenerateContent(t *testing.T) {
	var upstreamReq openaip.ChatCompletionRequest

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("upstream path = %q, want /v1/chat/completions", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&upstreamReq); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		writeJSON(w, http.StatusOK, openaip.ChatCompletionResponse{
			Choices: []openaip.Choice{{
				Index: 0,
				Message: openaip.Message{
					Role:    "assistant",
					Content: "hello from upstream",
				},
				FinishReason: "stop",
			}},
			Usage: &openaip.Usage{PromptTokens: 3, CompletionTokens: 4, TotalTokens: 7},
		})
	})

	server, err := New(Config{
		UpstreamBaseURL: "http://upstream.test/v1",
		HTTPClient: &http.Client{
			Transport: roundTripHandler{handler: upstream},
		},
		PDFRenderer: pdf.RendererFunc(func(_ context.Context, _ []byte) ([][]byte, error) { return nil, nil }),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	body := `{"contents":[{"role":"user","parts":[{"text":"hello"}]}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-pro:generateContent", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	server.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	if len(upstreamReq.Messages) != 1 || upstreamReq.Messages[0].Content != "hello" {
		t.Fatalf("translated upstream request = %#v", upstreamReq)
	}

	var got gemini.GenerateContentResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got.Candidates[0].Content.Parts[0].Text != "hello from upstream" {
		t.Fatalf("response = %#v", got)
	}
}

func TestServerOpenAIPassthrough(t *testing.T) {
	var authorization string

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("upstream path = %q, want /v1/chat/completions", r.URL.Path)
		}
		authorization = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"ok":true}`))
	})

	server, err := New(Config{
		UpstreamBaseURL: "http://upstream.test/v1",
		HTTPClient: &http.Client{
			Transport: roundTripHandler{handler: upstream},
		},
		PDFRenderer: pdf.RendererFunc(func(_ context.Context, _ []byte) ([][]byte, error) { return nil, nil }),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"x","messages":[]}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test")
	recorder := httptest.NewRecorder()

	server.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201", recorder.Code)
	}
	if recorder.Body.String() != `{"ok":true}` {
		t.Fatalf("body = %q, want %q", recorder.Body.String(), `{"ok":true}`)
	}
	if authorization != "" {
		t.Fatalf("authorization header leaked upstream: %q", authorization)
	}
}

func TestServerHealth(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Fatalf("upstream path = %q, want /health", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
	})

	server, err := New(Config{
		UpstreamBaseURL: "http://upstream.test/v1",
		HTTPClient: &http.Client{
			Transport: roundTripHandler{handler: upstream},
		},
		PDFRenderer: pdf.RendererFunc(func(_ context.Context, _ []byte) ([][]byte, error) { return nil, nil }),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	recorder := httptest.NewRecorder()
	server.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", recorder.Code)
	}
}

type roundTripHandler struct {
	handler http.Handler
}

func (t roundTripHandler) RoundTrip(req *http.Request) (*http.Response, error) {
	recorder := httptest.NewRecorder()
	t.handler.ServeHTTP(recorder, req)
	return recorder.Result(), nil
}
