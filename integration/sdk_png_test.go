package integration

import (
	"context"
	"encoding/base64"
	"net/http"
	"strings"
	"testing"

	"github.com/harshaneel/localaik/internal/pdf"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	genaisdk "google.golang.org/genai"
)

// pngFixture is a minimal byte sequence that starts with the PNG magic header.
// We do not need a decodable image; the proxy only base64-encodes the bytes.
var pngFixture = []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 'l', 'o', 'c', 'a', 'l', 'a', 'i', 'k'}

func TestSDKGenAIPNGInlineData(t *testing.T) {
	var upstreamRequest capturedRequest

	proxyHandler := newCapturedProxyHandler(
		t,
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			upstreamRequest.capture(t, r)
			writeJSON(w, http.StatusOK, openaip.ChatCompletionResponse{
				Choices: []openaip.Choice{{
					Index:        0,
					Message:      openaip.Message{Role: "assistant", Content: "ok"},
					FinishReason: "stop",
				}},
			})
		}),
		pdf.RendererFunc(func(_ context.Context, _ []byte) ([][]byte, error) { return nil, nil }),
		nil,
	)

	client, err := genaisdk.NewClient(context.Background(), &genaisdk.ClientConfig{
		APIKey:  "test",
		Backend: genaisdk.BackendGeminiAPI,
		HTTPClient: &http.Client{
			Transport: newHandlerTransport(proxyHandler),
		},
		HTTPOptions: genaisdk.HTTPOptions{BaseURL: "http://localaik.test"},
	})
	if err != nil {
		t.Fatalf("genai.NewClient returned error: %v", err)
	}

	_, err = client.Models.GenerateContent(
		context.Background(),
		"gemini-test",
		[]*genaisdk.Content{{
			Parts: []*genaisdk.Part{
				{Text: "What is in this image?"},
				genaisdk.NewPartFromBytes(pngFixture, "image/png"),
			},
		}},
		nil,
	)
	if err != nil {
		t.Fatalf("GenerateContent returned error: %v", err)
	}

	want := "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngFixture)
	assertUpstreamHasImageURL(t, upstreamRequest, want)
}

func TestSDKGenAIPNGFileDataDataURI(t *testing.T) {
	var upstreamRequest capturedRequest

	proxyHandler := newCapturedProxyHandler(
		t,
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			upstreamRequest.capture(t, r)
			writeJSON(w, http.StatusOK, openaip.ChatCompletionResponse{
				Choices: []openaip.Choice{{
					Index:        0,
					Message:      openaip.Message{Role: "assistant", Content: "ok"},
					FinishReason: "stop",
				}},
			})
		}),
		pdf.RendererFunc(func(_ context.Context, _ []byte) ([][]byte, error) { return nil, nil }),
		nil,
	)

	client, err := genaisdk.NewClient(context.Background(), &genaisdk.ClientConfig{
		APIKey:  "test",
		Backend: genaisdk.BackendGeminiAPI,
		HTTPClient: &http.Client{
			Transport: newHandlerTransport(proxyHandler),
		},
		HTTPOptions: genaisdk.HTTPOptions{BaseURL: "http://localaik.test"},
	})
	if err != nil {
		t.Fatalf("genai.NewClient returned error: %v", err)
	}

	encoded := base64.StdEncoding.EncodeToString(pngFixture)
	dataURI := "data:image/png;base64," + encoded

	_, err = client.Models.GenerateContent(
		context.Background(),
		"gemini-test",
		[]*genaisdk.Content{{
			Parts: []*genaisdk.Part{
				{Text: "Read this image"},
				{FileData: &genaisdk.FileData{FileURI: dataURI, MIMEType: "image/png"}},
			},
		}},
		nil,
	)
	if err != nil {
		t.Fatalf("GenerateContent returned error: %v", err)
	}

	assertUpstreamHasImageURL(t, upstreamRequest, dataURI)
}

func TestSDKGenAIPNGFileDataHTTPURL(t *testing.T) {
	var upstreamRequest capturedRequest

	proxyHandler := newCapturedProxyHandler(
		t,
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			upstreamRequest.capture(t, r)
			writeJSON(w, http.StatusOK, openaip.ChatCompletionResponse{
				Choices: []openaip.Choice{{
					Index:        0,
					Message:      openaip.Message{Role: "assistant", Content: "ok"},
					FinishReason: "stop",
				}},
			})
		}),
		pdf.RendererFunc(func(_ context.Context, _ []byte) ([][]byte, error) { return nil, nil }),
		nil,
	)

	client, err := genaisdk.NewClient(context.Background(), &genaisdk.ClientConfig{
		APIKey:  "test",
		Backend: genaisdk.BackendGeminiAPI,
		HTTPClient: &http.Client{
			Transport: newHandlerTransport(proxyHandler),
		},
		HTTPOptions: genaisdk.HTTPOptions{BaseURL: "http://localaik.test"},
	})
	if err != nil {
		t.Fatalf("genai.NewClient returned error: %v", err)
	}

	const remoteURL = "https://example.test/cat.png"
	_, err = client.Models.GenerateContent(
		context.Background(),
		"gemini-test",
		[]*genaisdk.Content{{
			Parts: []*genaisdk.Part{
				{Text: "Read this image"},
				{FileData: &genaisdk.FileData{FileURI: remoteURL, MIMEType: "image/png"}},
			},
		}},
		nil,
	)
	if err != nil {
		t.Fatalf("GenerateContent returned error: %v", err)
	}

	// External HTTP image URIs should pass through unmodified so the upstream
	// (or downstream provider) can fetch them directly.
	assertUpstreamHasImageURL(t, upstreamRequest, remoteURL)
}

func TestSDKOpenAIPNGImagePassthrough(t *testing.T) {
	var upstreamRequest capturedRequest

	proxyHandler := newCapturedProxyHandler(
		t,
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			upstreamRequest.capture(t, r)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"localaik","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
		}),
		pdf.RendererFunc(func(_ context.Context, _ []byte) ([][]byte, error) { return nil, nil }),
		nil,
	)

	client := openaisdk.NewClient(
		option.WithBaseURL("http://localaik.test/v1/"),
		option.WithAPIKey("test"),
		option.WithHTTPClient(&http.Client{Transport: newHandlerTransport(proxyHandler)}),
	)

	dataURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngFixture)

	_, err := client.Chat.Completions.New(context.Background(), openaisdk.ChatCompletionNewParams{
		Model: openaisdk.ChatModelGPT4o,
		Messages: []openaisdk.ChatCompletionMessageParamUnion{
			openaisdk.UserMessage([]openaisdk.ChatCompletionContentPartUnionParam{
				openaisdk.TextContentPart("What is in this image?"),
				openaisdk.ImageContentPart(openaisdk.ChatCompletionContentPartImageImageURLParam{URL: dataURI}),
			}),
		},
	})
	if err != nil {
		t.Fatalf("Chat.Completions.New returned error: %v", err)
	}

	// OpenAI route is a byte-for-byte passthrough, so the data URI shows up
	// verbatim in the upstream body. We do not use assertUpstreamHasImageURL
	// here because the passthrough handler never decodes the body into the
	// proxy's openaip types, so there is no structured shape worth walking.
	if !strings.Contains(string(upstreamRequest.Body), dataURI) {
		t.Fatalf("upstream body missing data URI; body=%s", string(upstreamRequest.Body))
	}
}

func assertUpstreamHasImageURL(t *testing.T, upstreamRequest capturedRequest, wantURL string) {
	t.Helper()

	body := upstreamRequest.mustJSONMap(t)
	messages, ok := body["messages"].([]any)
	if !ok || len(messages) == 0 {
		t.Fatalf("upstream body has no messages: %#v", body)
	}

	for _, m := range messages {
		msg, ok := m.(map[string]any)
		if !ok {
			continue
		}
		parts, ok := msg["content"].([]any)
		if !ok {
			continue
		}
		for _, p := range parts {
			part, ok := p.(map[string]any)
			if !ok {
				continue
			}
			if part["type"] != "image_url" {
				continue
			}
			imageURL, ok := part["image_url"].(map[string]any)
			if !ok {
				continue
			}
			if imageURL["url"] == wantURL {
				return
			}
		}
	}
	t.Fatalf("upstream messages missing image_url=%q; body=%s", wantURL, string(upstreamRequest.Body))
}
