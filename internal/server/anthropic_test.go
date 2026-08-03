package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/harshaneel/localaik/internal/pdf"
	"github.com/harshaneel/localaik/internal/protocol/anthropic"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

func TestServerAnthropicMessages(t *testing.T) {
	var upstreamReq openaip.ChatCompletionRequest
	var upstreamPath string

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamPath = r.URL.Path
		if err := json.NewDecoder(r.Body).Decode(&upstreamReq); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		writeJSON(w, http.StatusOK, openaip.ChatCompletionResponse{
			ID: "chatcmpl-xyz",
			Choices: []openaip.Choice{{
				Message:      openaip.Message{Role: "assistant", Content: "hello from upstream"},
				FinishReason: "stop",
			}},
			Usage: &openaip.Usage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
		})
	})

	srv := newTestServer(t, upstream)

	body := `{"model":"claude-sonnet-4-5","max_tokens":256,"system":"Be brief.","messages":[{"role":"user","content":"hello"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if upstreamPath != "/v1/chat/completions" {
		t.Fatalf("upstream path = %q, want /v1/chat/completions", upstreamPath)
	}
	if len(upstreamReq.Messages) != 2 {
		t.Fatalf("upstream messages = %#v, want system + user", upstreamReq.Messages)
	}
	if upstreamReq.MaxTokens == nil || *upstreamReq.MaxTokens != 256 {
		t.Fatalf("upstream max_tokens = %v, want 256", upstreamReq.MaxTokens)
	}
	if upstreamReq.Stream {
		t.Fatal("upstream stream = true, want false for a non-streaming request")
	}

	var got anthropic.MessagesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got.ID != "msg_xyz" {
		t.Fatalf("id = %q, want msg_xyz", got.ID)
	}
	if got.Model != "claude-sonnet-4-5" {
		t.Fatalf("model = %q, want the requested model", got.Model)
	}
	if len(got.Content) != 1 || got.Content[0].Text != "hello from upstream" {
		t.Fatalf("content = %#v", got.Content)
	}
	if got.StopReason == nil || *got.StopReason != anthropic.StopReasonEndTurn {
		t.Fatalf("stop_reason = %v, want end_turn", got.StopReason)
	}
	if got.Usage != (anthropic.Usage{InputTokens: 5, OutputTokens: 3}) {
		t.Fatalf("usage = %#v", got.Usage)
	}
}

func TestServerAnthropicMessagesStreaming(t *testing.T) {
	var upstreamReq openaip.ChatCompletionRequest
	var seenAccept string

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenAccept = r.Header.Get("Accept")
		if err := json.NewDecoder(r.Body).Decode(&upstreamReq); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n" +
			"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
			"data: [DONE]\n\n"))
	})

	srv := newTestServer(t, upstream)

	body := `{"model":"claude-sonnet-4-5","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if !upstreamReq.Stream {
		t.Fatal("upstream stream = false, want true")
	}
	if seenAccept != "text/event-stream" {
		t.Fatalf("upstream Accept = %q, want text/event-stream", seenAccept)
	}
	if got := rec.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", got)
	}

	out := rec.Body.String()
	for _, want := range []string{
		"event: " + anthropic.EventMessageStart,
		"event: " + anthropic.EventContentBlockStart,
		"event: " + anthropic.EventContentBlockDelta,
		"event: " + anthropic.EventContentBlockStop,
		"event: " + anthropic.EventMessageDelta,
		"event: " + anthropic.EventMessageStop,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("stream missing %q; body=%s", want, out)
		}
	}
	if !strings.Contains(out, `"text":"hi"`) {
		t.Fatalf("stream missing the text delta; body=%s", out)
	}
}

func TestServerAnthropicMessagesRendersPDFDocuments(t *testing.T) {
	var renderCalls int
	var upstreamReq openaip.ChatCompletionRequest

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&upstreamReq); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		writeJSON(w, http.StatusOK, openaip.ChatCompletionResponse{
			Choices: []openaip.Choice{{Message: openaip.Message{Content: "summary"}, FinishReason: "stop"}},
		})
	})

	srv, err := New(Config{
		UpstreamBaseURL: "http://upstream.test/v1",
		HTTPClient:      &http.Client{Transport: roundTripHandler{handler: upstream}},
		PDFRenderer: pdf.RendererFunc(func(context.Context, []byte) ([][]byte, error) {
			renderCalls++
			return [][]byte{[]byte("page")}, nil
		}),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	document := base64.StdEncoding.EncodeToString([]byte("%PDF-1.4"))
	body := `{"max_tokens":128,"messages":[{"role":"user","content":[` +
		`{"type":"text","text":"summarise"},` +
		`{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"` + document + `"}}` +
		`]}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if renderCalls != 1 {
		t.Fatalf("PDF renderer called %d times, want 1", renderCalls)
	}

	parts, ok := upstreamReq.Messages[0].Content.([]any)
	if !ok {
		t.Fatalf("upstream content = %#v, want a multimodal array", upstreamReq.Messages[0].Content)
	}
	if len(parts) != 2 {
		t.Fatalf("upstream content parts = %#v, want text + rendered page", parts)
	}
}

func TestServerAnthropicMessagesRejectsMissingMaxTokens(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("upstream should not be called; got %s %s", r.Method, r.URL.Path)
	})

	srv := newTestServer(t, upstream)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{"messages":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()

	srv.ServeHTTP(rec, req)

	assertAnthropicError(t, rec, http.StatusBadRequest, anthropic.ErrorTypeInvalidRequest)
}

func TestServerAnthropicMessagesRejectsInvalidJSON(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("upstream should not be called; got %s %s", r.Method, r.URL.Path)
	})

	srv := newTestServer(t, upstream)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{"messages":`))
	rec := httptest.NewRecorder()

	srv.ServeHTTP(rec, req)

	assertAnthropicError(t, rec, http.StatusBadRequest, anthropic.ErrorTypeInvalidRequest)
}

func TestServerAnthropicMessagesUpstreamError(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"model exploded","type":"server_error"}}`))
	})

	srv := newTestServer(t, upstream)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{"max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()

	srv.ServeHTTP(rec, req)

	got := assertAnthropicError(t, rec, http.StatusInternalServerError, anthropic.ErrorTypeAPI)
	if got.Error.Message != "model exploded" {
		t.Fatalf("error message = %q, want the upstream message", got.Error.Message)
	}
}

func TestServerAnthropicCountTokens(t *testing.T) {
	var upstreamPath, upstreamContent string

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamPath = r.URL.Path
		var payload struct {
			Content string `json:"content"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		upstreamContent = payload.Content
		writeJSON(w, http.StatusOK, map[string]any{"tokens": []int{1, 2, 3, 4}})
	})

	srv := newTestServer(t, upstream)

	body := `{"max_tokens":16,"system":"sys","messages":[{"role":"user","content":"hello"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if upstreamPath != "/tokenize" {
		t.Fatalf("upstream path = %q, want /tokenize", upstreamPath)
	}
	if upstreamContent != "sys\nhello" {
		t.Fatalf("tokenized content = %q, want %q", upstreamContent, "sys\nhello")
	}

	var got anthropic.CountTokensResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got.InputTokens != 4 {
		t.Fatalf("input_tokens = %d, want 4", got.InputTokens)
	}
}

// count_tokens has no max_tokens requirement — it measures a prompt rather than
// running one.
func TestServerAnthropicCountTokensWithoutMaxTokens(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"tokens": []int{1, 2}})
	})

	srv := newTestServer(t, upstream)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", bytes.NewBufferString(`{"messages":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()

	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
}

func TestServerAnthropicCountTokensUpstreamError(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"bad tokens","type":"invalid_request_error"}}`))
	})

	srv := newTestServer(t, upstream)

	req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", bytes.NewBufferString(`{"messages":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()

	srv.ServeHTTP(rec, req)

	got := assertAnthropicError(t, rec, http.StatusBadRequest, anthropic.ErrorTypeInvalidRequest)
	if got.Error.Message != "bad tokens" {
		t.Fatalf("error message = %q, want the upstream message", got.Error.Message)
	}
}

// Unroutable requests under /v1/messages must come back in the Anthropic error
// shape, not the OpenAI one that the rest of /v1/ uses.
func TestServerAnthropicRoutingErrors(t *testing.T) {
	cases := []struct {
		name   string
		method string
		path   string
	}{
		{"get_messages", http.MethodGet, "/v1/messages"},
		{"unknown_subpath", http.MethodPost, "/v1/messages/batches"},
		{"get_count_tokens", http.MethodGet, "/v1/messages/count_tokens"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				t.Fatalf("upstream should not be called; got %s %s", r.Method, r.URL.Path)
			})
			srv := newTestServer(t, upstream)

			req := httptest.NewRequest(tc.method, tc.path, nil)
			rec := httptest.NewRecorder()
			srv.ServeHTTP(rec, req)

			assertAnthropicError(t, rec, http.StatusNotFound, anthropic.ErrorTypeNotFound)
		})
	}
}

// Both Anthropic routes must reach upstream without carrying the client's
// credentials. Each case asserts a 200 so the check cannot pass just because
// upstream was never called.
func TestServerAnthropicDoesNotForwardCredentials(t *testing.T) {
	cases := []struct {
		name string
		path string
		body string
	}{
		{"messages", "/v1/messages", `{"max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`},
		{"count_tokens", "/v1/messages/count_tokens", `{"messages":[{"role":"user","content":"hi"}]}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var called bool
			var seenAPIKey, seenAuth, seenGoogKey string

			upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				called = true
				seenAPIKey = r.Header.Get("X-Api-Key")
				seenAuth = r.Header.Get("Authorization")
				seenGoogKey = r.Header.Get("X-Goog-Api-Key")

				if r.URL.Path == "/tokenize" {
					writeJSON(w, http.StatusOK, map[string]any{"tokens": []int{1}})
					return
				}
				writeJSON(w, http.StatusOK, openaip.ChatCompletionResponse{
					Choices: []openaip.Choice{{Message: openaip.Message{Content: "ok"}, FinishReason: "stop"}},
				})
			})

			srv := newTestServer(t, upstream)

			req := httptest.NewRequest(http.MethodPost, tc.path, bytes.NewBufferString(tc.body))
			req.Header.Set("X-Api-Key", "sk-ant-secret")
			req.Header.Set("Authorization", "Bearer secret")
			req.Header.Set("X-Goog-Api-Key", "goog-secret")
			rec := httptest.NewRecorder()

			srv.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
			}
			if !called {
				t.Fatal("upstream was never called, so the header check proves nothing")
			}
			if seenAPIKey != "" {
				t.Fatalf("x-api-key leaked upstream: %q", seenAPIKey)
			}
			if seenAuth != "" {
				t.Fatalf("Authorization leaked upstream: %q", seenAuth)
			}
			if seenGoogKey != "" {
				t.Fatalf("X-Goog-Api-Key leaked upstream: %q", seenGoogKey)
			}
		})
	}
}

func assertAnthropicError(t *testing.T, rec *httptest.ResponseRecorder, wantStatus int, wantType string) anthropic.ErrorResponse {
	t.Helper()

	if rec.Code != wantStatus {
		t.Fatalf("status = %d, want %d; body=%s", rec.Code, wantStatus, rec.Body.String())
	}

	var got anthropic.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode error response: %v; body=%s", err, rec.Body.String())
	}
	if got.Type != "error" {
		t.Fatalf("envelope type = %q, want error; body=%s", got.Type, rec.Body.String())
	}
	if got.Error.Type != wantType {
		t.Fatalf("error type = %q, want %q; body=%s", got.Error.Type, wantType, rec.Body.String())
	}
	if got.Error.Message == "" {
		t.Fatalf("error message missing; body=%s", rec.Body.String())
	}
	return got
}
