package integration

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/harshaneel/localaik/internal/pdf"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
	"github.com/harshaneel/localaik/internal/server"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	genaisdk "google.golang.org/genai"
)

func TestSDKGenAIModelsList(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Fatalf("upstream path = %q, want /v1/models", r.URL.Path)
		}
		writeJSON(w, http.StatusOK, openaip.ModelList{
			Object: "list",
			Data: []openaip.Model{
				{ID: "gemma-3-4b", Object: "model"},
				{ID: "gemma-3-12b", Object: "model"},
			},
		})
	})

	proxy := newCapturedProxyHandlerForUpstream(t, upstream)

	client, err := genaisdk.NewClient(context.Background(), &genaisdk.ClientConfig{
		APIKey:  "test",
		Backend: genaisdk.BackendGeminiAPI,
		HTTPClient: &http.Client{
			Transport: newHandlerTransport(proxy),
		},
		HTTPOptions: genaisdk.HTTPOptions{
			BaseURL: "http://localaik.test",
		},
	})
	if err != nil {
		t.Fatalf("genai.NewClient returned error: %v", err)
	}

	page, err := client.Models.List(context.Background(), nil)
	if err != nil {
		t.Fatalf("Models.List returned error: %v", err)
	}

	if len(page.Items) != 2 {
		t.Fatalf("page.Items = %#v, want 2 models", page.Items)
	}
	if page.Items[0].Name != "models/gemma-3-4b" {
		t.Fatalf("page.Items[0].Name = %q, want models/gemma-3-4b", page.Items[0].Name)
	}
}

func TestSDKGenAIModelsGet(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models/gemma-3-4b" {
			t.Fatalf("upstream path = %q, want /v1/models/gemma-3-4b", r.URL.Path)
		}
		writeJSON(w, http.StatusOK, openaip.Model{ID: "gemma-3-4b", Object: "model"})
	})

	proxy := newCapturedProxyHandlerForUpstream(t, upstream)

	client, err := genaisdk.NewClient(context.Background(), &genaisdk.ClientConfig{
		APIKey:  "test",
		Backend: genaisdk.BackendGeminiAPI,
		HTTPClient: &http.Client{
			Transport: newHandlerTransport(proxy),
		},
		HTTPOptions: genaisdk.HTTPOptions{
			BaseURL: "http://localaik.test",
		},
	})
	if err != nil {
		t.Fatalf("genai.NewClient returned error: %v", err)
	}

	model, err := client.Models.Get(context.Background(), "gemma-3-4b", nil)
	if err != nil {
		t.Fatalf("Models.Get returned error: %v", err)
	}

	if model.Name != "models/gemma-3-4b" {
		t.Fatalf("model.Name = %q, want models/gemma-3-4b", model.Name)
	}
}

func TestSDKGenAICountTokens(t *testing.T) {
	var upstreamBody map[string]any

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenize" {
			t.Fatalf("upstream path = %q, want /tokenize", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&upstreamBody); err != nil {
			t.Fatalf("decode upstream body: %v", err)
		}
		writeJSON(w, http.StatusOK, map[string]any{"tokens": []int{1, 2, 3, 4, 5, 6, 7}})
	})

	proxy := newCapturedProxyHandlerForUpstream(t, upstream)

	client, err := genaisdk.NewClient(context.Background(), &genaisdk.ClientConfig{
		APIKey:  "test",
		Backend: genaisdk.BackendGeminiAPI,
		HTTPClient: &http.Client{
			Transport: newHandlerTransport(proxy),
		},
		HTTPOptions: genaisdk.HTTPOptions{
			BaseURL: "http://localaik.test",
		},
	})
	if err != nil {
		t.Fatalf("genai.NewClient returned error: %v", err)
	}

	resp, err := client.Models.CountTokens(context.Background(), "gemma-3-4b", genaisdk.Text("hello world"), nil)
	if err != nil {
		t.Fatalf("Models.CountTokens returned error: %v", err)
	}

	if resp.TotalTokens != 7 {
		t.Fatalf("TotalTokens = %d, want 7", resp.TotalTokens)
	}
	if upstreamBody["content"] != "hello world" {
		t.Fatalf("upstream content = %#v, want hello world", upstreamBody["content"])
	}
}

func TestSDKOpenAIModelsList(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Fatalf("upstream path = %q, want /v1/models", r.URL.Path)
		}
		writeJSON(w, http.StatusOK, openaip.ModelList{
			Object: "list",
			Data: []openaip.Model{
				{ID: "gemma-3-4b", Object: "model"},
			},
		})
	})

	proxy := newCapturedProxyHandlerForUpstream(t, upstream)

	client := openaisdk.NewClient(
		option.WithBaseURL("http://localaik.test/v1/"),
		option.WithAPIKey("test"),
		option.WithHTTPClient(&http.Client{Transport: newHandlerTransport(proxy)}),
	)

	page, err := client.Models.List(context.Background())
	if err != nil {
		t.Fatalf("Models.List returned error: %v", err)
	}
	if len(page.Data) != 1 || page.Data[0].ID != "gemma-3-4b" {
		t.Fatalf("page.Data = %#v", page.Data)
	}
}

func TestSDKOpenAIModelsGet(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models/gemma-3-4b" {
			t.Fatalf("upstream path = %q, want /v1/models/gemma-3-4b", r.URL.Path)
		}
		writeJSON(w, http.StatusOK, openaip.Model{ID: "gemma-3-4b", Object: "model"})
	})

	proxy := newCapturedProxyHandlerForUpstream(t, upstream)

	client := openaisdk.NewClient(
		option.WithBaseURL("http://localaik.test/v1/"),
		option.WithAPIKey("test"),
		option.WithHTTPClient(&http.Client{Transport: newHandlerTransport(proxy)}),
	)

	model, err := client.Models.Get(context.Background(), "gemma-3-4b")
	if err != nil {
		t.Fatalf("Models.Get returned error: %v", err)
	}
	if model.ID != "gemma-3-4b" {
		t.Fatalf("model.ID = %q, want gemma-3-4b", model.ID)
	}
}

func TestSDKOpenAILegacyCompletions(t *testing.T) {
	var upstreamBody map[string]any

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/completions" {
			t.Fatalf("upstream path = %q, want /v1/completions", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&upstreamBody); err != nil {
			t.Fatalf("decode upstream body: %v", err)
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"id":      "cmpl_1",
			"object":  "text_completion",
			"created": 1,
			"model":   "localaik",
			"choices": []map[string]any{
				{"index": 0, "text": "pong", "finish_reason": "stop"},
			},
		})
	})

	proxy := newCapturedProxyHandlerForUpstream(t, upstream)

	client := openaisdk.NewClient(
		option.WithBaseURL("http://localaik.test/v1/"),
		option.WithAPIKey("test"),
		option.WithHTTPClient(&http.Client{Transport: newHandlerTransport(proxy)}),
	)

	resp, err := client.Completions.New(context.Background(), openaisdk.CompletionNewParams{
		Model:  "localaik",
		Prompt: openaisdk.CompletionNewParamsPromptUnion{OfString: openaisdk.String("ping")},
	})
	if err != nil {
		t.Fatalf("Completions.New returned error: %v", err)
	}
	if resp.Choices[0].Text != "pong" {
		t.Fatalf("response = %#v", resp.Choices)
	}
	if upstreamBody["prompt"] != "ping" {
		t.Fatalf("upstream prompt = %#v, want ping", upstreamBody["prompt"])
	}
}

// newCapturedProxyHandlerForUpstream wires up the localaik proxy with an
// arbitrary upstream handler so individual tests can stub /v1/models,
// /tokenize, /v1/completions, etc. without sharing routing logic with the
// chat-completions tests in sdk_test.go.
func newCapturedProxyHandlerForUpstream(t *testing.T, upstream http.Handler) http.Handler {
	t.Helper()

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	// Route everything else to the test-supplied upstream.
	mux.Handle("/", upstream)

	proxyServer, err := server.New(server.Config{
		UpstreamBaseURL: "http://upstream.test/v1",
		HTTPClient: &http.Client{
			Transport: newHandlerTransport(mux),
		},
		PDFRenderer: pdf.RendererFunc(func(_ context.Context, _ []byte) ([][]byte, error) { return nil, nil }),
	})
	if err != nil {
		t.Fatalf("server.New returned error: %v", err)
	}

	return proxyServer
}
