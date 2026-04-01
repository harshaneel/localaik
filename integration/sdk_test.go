package integration

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/harshaneel/localaik/internal/pdf"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
	"github.com/harshaneel/localaik/internal/server"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	genaisdk "google.golang.org/genai"
)

func TestSDKGenAIGenerateContent(t *testing.T) {
	var sdkRequest capturedRequest
	var upstreamRequest capturedRequest

	proxyHandler := newCapturedProxyHandler(
		t,
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			upstreamRequest.capture(t, r)
			writeJSON(w, http.StatusOK, openaip.ChatCompletionResponse{
				Choices: []openaip.Choice{{
					Index: 0,
					Message: openaip.Message{
						Role:    "assistant",
						Content: `{"answer":"done"}`,
					},
					FinishReason: "stop",
				}},
				Usage: &openaip.Usage{
					PromptTokens:     10,
					CompletionTokens: 6,
					TotalTokens:      16,
				},
			})
		}),
		pdf.RendererFunc(func(_ context.Context, pdfBytes []byte) ([][]byte, error) {
			if string(pdfBytes) != "%PDF-1.4 fake" {
				t.Fatalf("renderer got PDF %q, want test payload", string(pdfBytes))
			}
			return [][]byte{[]byte("page-1"), []byte("page-2")}, nil
		}),
		func(r *http.Request, body []byte) {
			sdkRequest = newCapturedRequest(t, r, body)
		},
	)

	client, err := genaisdk.NewClient(context.Background(), &genaisdk.ClientConfig{
		APIKey:  "test",
		Backend: genaisdk.BackendGeminiAPI,
		HTTPClient: &http.Client{
			Transport: newHandlerTransport(proxyHandler),
		},
		HTTPOptions: genaisdk.HTTPOptions{
			BaseURL: "http://localaik.test",
		},
	})
	if err != nil {
		t.Fatalf("genai.NewClient returned error: %v", err)
	}

	config := &genaisdk.GenerateContentConfig{
		SystemInstruction: &genaisdk.Content{
			Parts: []*genaisdk.Part{{Text: "Return strict JSON."}},
		},
		Temperature:      genaisdk.Ptr[float32](0.4),
		MaxOutputTokens:  128,
		ResponseMIMEType: "application/json",
		ResponseSchema: &genaisdk.Schema{
			Type: genaisdk.TypeObject,
			Properties: map[string]*genaisdk.Schema{
				"answer": {Type: genaisdk.TypeString},
			},
		},
	}

	parts := []*genaisdk.Part{
		{Text: "Extract the answer"},
		genaisdk.NewPartFromBytes([]byte("%PDF-1.4 fake"), "application/pdf"),
		genaisdk.NewPartFromBytes([]byte("img"), "image/png"),
	}

	resp, err := client.Models.GenerateContent(
		context.Background(),
		"gemini-test",
		[]*genaisdk.Content{{Parts: parts}},
		config,
	)
	if err != nil {
		t.Fatalf("GenerateContent returned error: %v", err)
	}

	t.Run("response", func(t *testing.T) {
		if got := resp.Text(); got != `{"answer":"done"}` {
			t.Fatalf("response text = %q, want JSON string", got)
		}
		if resp.UsageMetadata == nil || resp.UsageMetadata.TotalTokenCount != 16 {
			t.Fatalf("usage metadata = %#v", resp.UsageMetadata)
		}
	})

	t.Run("sdk request", func(t *testing.T) {
		if sdkRequest.Path != "/v1beta/models/gemini-test:generateContent" {
			t.Fatalf("sdk request path = %q, want /v1beta/models/gemini-test:generateContent", sdkRequest.Path)
		}
		if sdkRequest.Headers.Get("X-Goog-Api-Key") != "test" {
			t.Fatalf("sdk request x-goog-api-key = %q, want test", sdkRequest.Headers.Get("X-Goog-Api-Key"))
		}

		sdkBody := sdkRequest.mustJSONMap(t)
		if _, ok := sdkBody["systemInstruction"]; !ok {
			t.Fatalf("sdk request body missing systemInstruction: %#v", sdkBody)
		}
		if _, ok := sdkBody["generationConfig"]; !ok {
			t.Fatalf("sdk request body missing generationConfig: %#v", sdkBody)
		}
		if _, ok := sdkBody["contents"]; !ok {
			t.Fatalf("sdk request body missing contents: %#v", sdkBody)
		}
	})

	t.Run("upstream request", func(t *testing.T) {
		assertGenAIUpstreamRequest(t, upstreamRequest)
	})
}

func TestSDKGenAIStreamGenerateContent(t *testing.T) {
	var sdkRequest capturedRequest
	var upstreamRequest capturedRequest

	proxyHandler := newCapturedProxyHandler(
		t,
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			upstreamRequest.capture(t, r)
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n")
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"}}]}\n\n")
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
		}),
		pdf.RendererFunc(func(_ context.Context, _ []byte) ([][]byte, error) {
			return nil, nil
		}),
		func(r *http.Request, body []byte) {
			sdkRequest = newCapturedRequest(t, r, body)
		},
	)

	client, err := genaisdk.NewClient(context.Background(), &genaisdk.ClientConfig{
		APIKey:  "test",
		Backend: genaisdk.BackendGeminiAPI,
		HTTPClient: &http.Client{
			Transport: newHandlerTransport(proxyHandler),
		},
		HTTPOptions: genaisdk.HTTPOptions{
			BaseURL: "http://localaik.test",
		},
	})
	if err != nil {
		t.Fatalf("genai.NewClient returned error: %v", err)
	}

	var texts []string
	for chunk, err := range client.Models.GenerateContentStream(
		context.Background(),
		"gemini-test",
		genaisdk.Text("stream please"),
		nil,
	) {
		if err != nil {
			t.Fatalf("GenerateContentStream yielded error: %v", err)
		}
		texts = append(texts, chunk.Text())
	}

	if strings.Join(texts, "") != "hello world" {
		t.Fatalf("stream texts = %#v, want hello world", texts)
	}
	if sdkRequest.Path != "/v1beta/models/gemini-test:streamGenerateContent" {
		t.Fatalf("sdk request path = %q, want streamGenerateContent path", sdkRequest.Path)
	}
	if sdkRequest.RawQuery != "alt=sse" {
		t.Fatalf("sdk request query = %q, want alt=sse", sdkRequest.RawQuery)
	}
	if upstreamRequest.Path != "/v1/chat/completions" {
		t.Fatalf("upstream path = %q, want /v1/chat/completions", upstreamRequest.Path)
	}

	upstreamBody := upstreamRequest.mustJSONMap(t)
	if upstreamBody["stream"] != true {
		t.Fatalf("upstream stream flag = %v, want true", upstreamBody["stream"])
	}
}

func TestSDKGenAIGenerateContentWithFilesAndTools(t *testing.T) {
	var upstreamRequest capturedRequest

	proxyHandler := newCapturedProxyHandler(
		t,
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			upstreamRequest.capture(t, r)
			writeJSON(w, http.StatusOK, openaip.ChatCompletionResponse{
				Choices: []openaip.Choice{{
					Index: 0,
					Message: openaip.Message{
						Role: "assistant",
						ToolCalls: []openaip.ToolCall{{
							ID:   "call_123",
							Type: "function",
							Function: &openaip.ToolCallFunction{
								Name:      "lookup_weather",
								Arguments: `{"city":"Boston"}`,
							},
						}},
					},
					FinishReason: "tool_calls",
				}},
			})
		}),
		pdf.RendererFunc(func(_ context.Context, pdfBytes []byte) ([][]byte, error) {
			if string(pdfBytes) != "%PDF-tools" {
				t.Fatalf("renderer got PDF %q, want test payload", string(pdfBytes))
			}
			return [][]byte{[]byte("page-1")}, nil
		}),
		nil,
	)

	client, err := genaisdk.NewClient(context.Background(), &genaisdk.ClientConfig{
		APIKey:  "test",
		Backend: genaisdk.BackendGeminiAPI,
		HTTPClient: &http.Client{
			Transport: newHandlerTransport(proxyHandler),
		},
		HTTPOptions: genaisdk.HTTPOptions{
			BaseURL: "http://localaik.test",
		},
	})
	if err != nil {
		t.Fatalf("genai.NewClient returned error: %v", err)
	}

	topP := float32(0.9)
	topK := float32(40)
	maxOutputTokens := int32(64)
	logprobs := int32(2)
	presencePenalty := float32(0.1)
	frequencyPenalty := float32(0.2)
	seed := int32(9)

	resp, err := client.Models.GenerateContent(
		context.Background(),
		"gemini-test",
		[]*genaisdk.Content{{
			Parts: []*genaisdk.Part{
				{Text: "Use the tool if needed."},
				{
					FileData: &genaisdk.FileData{
						FileURI:  "data:application/pdf;base64," + base64.StdEncoding.EncodeToString([]byte("%PDF-tools")),
						MIMEType: "application/pdf",
					},
				},
			},
		}},
		&genaisdk.GenerateContentConfig{
			TopP:             &topP,
			TopK:             &topK,
			CandidateCount:   2,
			MaxOutputTokens:  maxOutputTokens,
			StopSequences:    []string{"DONE"},
			ResponseLogprobs: true,
			Logprobs:         &logprobs,
			PresencePenalty:  &presencePenalty,
			FrequencyPenalty: &frequencyPenalty,
			Seed:             &seed,
			Tools: []*genaisdk.Tool{{
				FunctionDeclarations: []*genaisdk.FunctionDeclaration{{
					Name:        "lookup_weather",
					Description: "Look up the weather by city.",
					Parameters: &genaisdk.Schema{
						Type: genaisdk.TypeObject,
						Properties: map[string]*genaisdk.Schema{
							"city": {Type: genaisdk.TypeString},
						},
						Required: []string{"city"},
					},
				}},
			}},
			ToolConfig: &genaisdk.ToolConfig{
				FunctionCallingConfig: &genaisdk.FunctionCallingConfig{
					Mode:                 genaisdk.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"lookup_weather"},
				},
			},
		},
	)
	if err != nil {
		t.Fatalf("GenerateContent returned error: %v", err)
	}

	upstreamBody := upstreamRequest.mustJSONMap(t)

	t.Run("response", func(t *testing.T) {
		assertToolCallResponse(t, resp)
	})

	t.Run("upstream config", func(t *testing.T) {
		assertToolsUpstreamConfig(t, upstreamBody)
	})

	t.Run("upstream tools", func(t *testing.T) {
		tools, ok := upstreamBody["tools"].([]any)
		if !ok || len(tools) != 1 {
			t.Fatalf("upstream tools = %#v", upstreamBody["tools"])
		}
		toolChoice, ok := upstreamBody["tool_choice"].(map[string]any)
		if !ok {
			t.Fatalf("upstream tool_choice = %#v", upstreamBody["tool_choice"])
		}
		function, ok := toolChoice["function"].(map[string]any)
		if !ok || function["name"] != "lookup_weather" {
			t.Fatalf("upstream tool_choice function = %#v", upstreamBody["tool_choice"])
		}
	})
}

func TestSDKOpenAIChatCompletions(t *testing.T) {
	var upstreamRequest capturedRequest

	proxyHandler := newCapturedProxyHandler(
		t,
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			upstreamRequest.capture(t, r)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"localaik","choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}]}`))
		}),
		pdf.RendererFunc(func(_ context.Context, _ []byte) ([][]byte, error) {
			return nil, nil
		}),
		nil,
	)

	client := openaisdk.NewClient(
		option.WithBaseURL("http://localaik.test/v1/"),
		option.WithAPIKey("test"),
		option.WithHTTPClient(&http.Client{
			Transport: newHandlerTransport(proxyHandler),
		}),
	)

	resp, err := client.Chat.Completions.New(context.Background(), openaisdk.ChatCompletionNewParams{
		Messages: []openaisdk.ChatCompletionMessageParamUnion{
			openaisdk.UserMessage("ping"),
		},
		Model: openaisdk.ChatModelGPT4o,
	})
	if err != nil {
		t.Fatalf("Chat.Completions.New returned error: %v", err)
	}

	if got := resp.Choices[0].Message.Content; got != "pong" {
		t.Fatalf("response content = %q, want pong", got)
	}
	if upstreamRequest.Headers.Get("Authorization") != "" {
		t.Fatalf("authorization leaked upstream: %q", upstreamRequest.Headers.Get("Authorization"))
	}
}

func assertGenAIUpstreamRequest(t *testing.T, upstreamRequest capturedRequest) {
	t.Helper()
	if upstreamRequest.Path != "/v1/chat/completions" {
		t.Fatalf("upstream path = %q, want /v1/chat/completions", upstreamRequest.Path)
	}
	if upstreamRequest.Headers.Get("X-Goog-Api-Key") != "" {
		t.Fatalf("x-goog-api-key leaked upstream: %q", upstreamRequest.Headers.Get("X-Goog-Api-Key"))
	}

	upstreamBody := upstreamRequest.mustJSONMap(t)
	messages, ok := upstreamBody["messages"].([]any)
	if !ok || len(messages) != 2 {
		t.Fatalf("upstream messages = %#v", upstreamBody["messages"])
	}
	responseFormat, ok := upstreamBody["response_format"].(map[string]any)
	if !ok || responseFormat["type"] != "json_schema" {
		t.Fatalf("response format = %#v", upstreamBody["response_format"])
	}
}

func assertToolCallResponse(t *testing.T, resp *genaisdk.GenerateContentResponse) {
	t.Helper()
	if len(resp.Candidates) != 1 || resp.Candidates[0].Content == nil || len(resp.Candidates[0].Content.Parts) != 1 {
		t.Fatalf("response candidates = %#v", resp.Candidates)
	}
	functionCall := resp.Candidates[0].Content.Parts[0].FunctionCall
	if functionCall == nil || functionCall.Name != "lookup_weather" || functionCall.ID != "call_123" {
		t.Fatalf("response function call = %#v", resp.Candidates[0].Content.Parts[0])
	}
	if functionCall.Args["city"] != "Boston" {
		t.Fatalf("response function args = %#v", functionCall.Args)
	}
}

func assertToolsUpstreamConfig(t *testing.T, upstreamBody map[string]any) {
	t.Helper()
	gotTopP, ok := upstreamBody["top_p"].(float64)
	if !ok || math.Abs(gotTopP-0.9) > 1e-6 {
		t.Fatalf("upstream top_p = %#v, want about 0.9", upstreamBody["top_p"])
	}
	if upstreamBody["top_k"] != float64(40) {
		t.Fatalf("upstream top_k = %#v, want 40", upstreamBody["top_k"])
	}
	if upstreamBody["n"] != float64(2) {
		t.Fatalf("upstream n = %#v, want 2", upstreamBody["n"])
	}
	if upstreamBody["seed"] != float64(9) {
		t.Fatalf("upstream seed = %#v, want 9", upstreamBody["seed"])
	}
	if upstreamBody["logprobs"] != true {
		t.Fatalf("upstream logprobs = %#v, want true", upstreamBody["logprobs"])
	}
	if upstreamBody["top_logprobs"] != float64(2) {
		t.Fatalf("upstream top_logprobs = %#v, want 2", upstreamBody["top_logprobs"])
	}
}

type capturedRequest struct {
	Path     string
	RawQuery string
	Headers  http.Header
	Body     []byte
}

func newCapturedRequest(t *testing.T, r *http.Request, body []byte) capturedRequest {
	t.Helper()
	return capturedRequest{
		Path:     r.URL.Path,
		RawQuery: r.URL.RawQuery,
		Headers:  r.Header.Clone(),
		Body:     append([]byte(nil), body...),
	}
}

func (c *capturedRequest) capture(t *testing.T, r *http.Request) {
	t.Helper()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("read captured request body: %v", err)
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	*c = newCapturedRequest(t, r, body)
}

func (c capturedRequest) mustJSONMap(t *testing.T) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal(c.Body, &out); err != nil {
		t.Fatalf("unmarshal captured request JSON: %v\nbody=%s", err, string(c.Body))
	}
	return out
}

type handlerTransport struct {
	handler http.Handler
}

func newHandlerTransport(handler http.Handler) http.RoundTripper {
	return handlerTransport{handler: handler}
}

func (t handlerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	recorder := httptest.NewRecorder()
	t.handler.ServeHTTP(recorder, req)
	return recorder.Result(), nil
}

func newCapturedProxyHandler(t *testing.T, upstreamChatHandler http.Handler, renderer pdf.Renderer, capture func(*http.Request, []byte)) http.Handler {
	t.Helper()

	upstreamMux := http.NewServeMux()
	upstreamMux.Handle("/v1/chat/completions", upstreamChatHandler)
	upstreamMux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	proxyServer, err := server.New(server.Config{
		UpstreamBaseURL: "http://upstream.test/v1",
		HTTPClient: &http.Client{
			Transport: newHandlerTransport(upstreamMux),
		},
		PDFRenderer: renderer,
	})
	if err != nil {
		t.Fatalf("server.New returned error: %v", err)
	}

	if capture == nil {
		return proxyServer
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read sdk request body: %v", err)
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		capture(r, body)
		r.Body = io.NopCloser(bytes.NewReader(body))
		proxyServer.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
