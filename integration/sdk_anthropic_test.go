package integration

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"

	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

// newAnthropicSDKClient points the official Anthropic Go SDK at the localaik
// proxy, which in turn talks to the supplied llama.cpp stand-in. Nothing here
// stubs the SDK itself — the request and response go over the real wire format.
func newAnthropicSDKClient(t *testing.T, upstream http.Handler) anthropicsdk.Client {
	t.Helper()

	proxy := newCapturedProxyHandlerForUpstream(t, upstream)

	return anthropicsdk.NewClient(
		anthropicoption.WithAPIKey("test"),
		anthropicoption.WithBaseURL("http://localaik.test/"),
		anthropicoption.WithHTTPClient(&http.Client{Transport: newHandlerTransport(proxy)}),
		anthropicoption.WithMaxRetries(0),
	)
}

func TestSDKAnthropicMessages(t *testing.T) {
	var upstreamReq openaip.ChatCompletionRequest

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("upstream path = %q, want /v1/chat/completions", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&upstreamReq); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		writeJSON(w, http.StatusOK, openaip.ChatCompletionResponse{
			ID: "chatcmpl-sdk1",
			Choices: []openaip.Choice{{
				Message:      openaip.Message{Role: "assistant", Content: "Paris is the capital."},
				FinishReason: "stop",
			}},
			Usage: &openaip.Usage{PromptTokens: 12, CompletionTokens: 6, TotalTokens: 18},
		})
	})

	client := newAnthropicSDKClient(t, upstream)

	message, err := client.Messages.New(context.Background(), anthropicsdk.MessageNewParams{
		Model:     anthropicsdk.ModelClaudeSonnet4_5,
		MaxTokens: 256,
		System: []anthropicsdk.TextBlockParam{
			{Text: "Answer in one sentence."},
		},
		Messages: []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("What is the capital of France?")),
		},
	})
	if err != nil {
		t.Fatalf("Messages.New returned error: %v", err)
	}

	if len(upstreamReq.Messages) != 2 {
		t.Fatalf("upstream messages = %#v, want system + user", upstreamReq.Messages)
	}
	if upstreamReq.Messages[0].Role != "system" || upstreamReq.Messages[0].Content != "Answer in one sentence." {
		t.Fatalf("upstream system message = %#v", upstreamReq.Messages[0])
	}
	if upstreamReq.MaxTokens == nil || *upstreamReq.MaxTokens != 256 {
		t.Fatalf("upstream max_tokens = %v, want 256", upstreamReq.MaxTokens)
	}

	if message.ID != "msg_sdk1" {
		t.Fatalf("id = %q, want msg_sdk1", message.ID)
	}
	if message.Role != "assistant" {
		t.Fatalf("role = %q, want assistant", message.Role)
	}
	if string(message.Model) != string(anthropicsdk.ModelClaudeSonnet4_5) {
		t.Fatalf("model = %q, want the requested model echoed back", message.Model)
	}
	if len(message.Content) != 1 || message.Content[0].Type != "text" {
		t.Fatalf("content = %#v, want a single text block", message.Content)
	}
	if message.Content[0].Text != "Paris is the capital." {
		t.Fatalf("text = %q", message.Content[0].Text)
	}
	if message.StopReason != anthropicsdk.StopReasonEndTurn {
		t.Fatalf("stop_reason = %q, want end_turn", message.StopReason)
	}
	if message.Usage.InputTokens != 12 || message.Usage.OutputTokens != 6 {
		t.Fatalf("usage = %#v, want input 12 / output 6", message.Usage)
	}
}

func TestSDKAnthropicMessagesStreaming(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, chunk := range []string{
			`{"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`,
			`{"choices":[{"index":0,"delta":{"content":"Hello"}}]}`,
			`{"choices":[{"index":0,"delta":{"content":", world"}}]}`,
			`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			`{"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}`,
			`[DONE]`,
		} {
			_, _ = w.Write([]byte("data: " + chunk + "\n\n"))
		}
	})

	client := newAnthropicSDKClient(t, upstream)

	stream := client.Messages.NewStreaming(context.Background(), anthropicsdk.MessageNewParams{
		Model:     anthropicsdk.ModelClaudeSonnet4_5,
		MaxTokens: 64,
		Messages: []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("greet me")),
		},
	})

	// Accumulate is the SDK's own state machine: it rejects out-of-order or
	// gap-indexed content blocks, so a clean accumulation proves the event
	// sequence localaik emits is well-formed.
	accumulated := anthropicsdk.Message{}
	var deltas []string
	for stream.Next() {
		event := stream.Current()
		if err := accumulated.Accumulate(event); err != nil {
			t.Fatalf("Accumulate returned error: %v", err)
		}
		if delta := event.Delta.Text; delta != "" {
			deltas = append(deltas, delta)
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	if len(deltas) != 2 || deltas[0] != "Hello" || deltas[1] != ", world" {
		t.Fatalf("text deltas = %#v, want [Hello , world]", deltas)
	}
	if len(accumulated.Content) != 1 || accumulated.Content[0].Text != "Hello, world" {
		t.Fatalf("accumulated content = %#v, want the joined text", accumulated.Content)
	}
	if accumulated.StopReason != anthropicsdk.StopReasonEndTurn {
		t.Fatalf("stop_reason = %q, want end_turn", accumulated.StopReason)
	}
	if accumulated.Usage.OutputTokens != 3 {
		t.Fatalf("output tokens = %d, want 3", accumulated.Usage.OutputTokens)
	}
}

// A stream where upstream starts a second tool call before the first one's
// arguments finish has to accumulate cleanly in the SDK: its accumulator rejects
// gap-indexed content blocks and errors when a block's input is not valid JSON at
// the point it is marshalled.
func TestSDKAnthropicMessagesStreamingInterleavedToolCalls(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, chunk := range []string{
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t0","type":"function","function":{"name":"first","arguments":"{\"a\":"}}]}}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"t1","type":"function","function":{"name":"second","arguments":"{\"b\":2}"}}]}}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}`,
			`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
			`[DONE]`,
		} {
			_, _ = w.Write([]byte("data: " + chunk + "\n\n"))
		}
	})

	client := newAnthropicSDKClient(t, upstream)

	stream := client.Messages.NewStreaming(context.Background(), anthropicsdk.MessageNewParams{
		Model:     anthropicsdk.ModelClaudeSonnet4_5,
		MaxTokens: 128,
		Messages: []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("call both tools")),
		},
	})

	accumulated := anthropicsdk.Message{}
	for stream.Next() {
		if err := accumulated.Accumulate(stream.Current()); err != nil {
			t.Fatalf("Accumulate returned error: %v", err)
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	if len(accumulated.Content) != 2 {
		t.Fatalf("content = %#v, want two tool_use blocks", accumulated.Content)
	}

	wantInputs := []map[string]any{
		{"a": float64(1)},
		{"b": float64(2)},
	}
	for i, wantInput := range wantInputs {
		block, ok := accumulated.Content[i].AsAny().(anthropicsdk.ToolUseBlock)
		if !ok {
			t.Fatalf("block %d = %#v, want a ToolUseBlock", i, accumulated.Content[i].AsAny())
		}
		var input map[string]any
		if err := json.Unmarshal(block.Input, &input); err != nil {
			t.Fatalf("block %d input %s does not decode: %v", i, block.Input, err)
		}
		if len(input) != len(wantInput) {
			t.Fatalf("block %d input = %#v, want %#v", i, input, wantInput)
		}
		for key, want := range wantInput {
			if input[key] != want {
				t.Fatalf("block %d input = %#v, want %#v", i, input, wantInput)
			}
		}
	}

	if accumulated.StopReason != anthropicsdk.StopReasonToolUse {
		t.Fatalf("stop_reason = %q, want tool_use", accumulated.StopReason)
	}
}

func TestSDKAnthropicMessagesToolUse(t *testing.T) {
	var upstreamReq openaip.ChatCompletionRequest

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&upstreamReq); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		writeJSON(w, http.StatusOK, openaip.ChatCompletionResponse{
			Choices: []openaip.Choice{{
				Message: openaip.Message{
					Role: "assistant",
					ToolCalls: []openaip.ToolCall{{
						ID:   "toolu_sdk",
						Type: "function",
						Function: &openaip.ToolCallFunction{
							Name:      "get_weather",
							Arguments: `{"city":"Paris"}`,
						},
					}},
				},
				FinishReason: "tool_calls",
			}},
		})
	})

	client := newAnthropicSDKClient(t, upstream)

	message, err := client.Messages.New(context.Background(), anthropicsdk.MessageNewParams{
		Model:     anthropicsdk.ModelClaudeSonnet4_5,
		MaxTokens: 128,
		Messages: []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("weather in Paris?")),
		},
		Tools: []anthropicsdk.ToolUnionParam{{
			OfTool: &anthropicsdk.ToolParam{
				Name:        "get_weather",
				Description: anthropicsdk.String("Look up the weather for a city"),
				InputSchema: anthropicsdk.ToolInputSchemaParam{
					Properties: map[string]any{
						"city": map[string]any{"type": "string"},
					},
					Required: []string{"city"},
				},
			},
		}},
		ToolChoice: anthropicsdk.ToolChoiceUnionParam{
			OfTool: &anthropicsdk.ToolChoiceToolParam{Name: "get_weather"},
		},
	})
	if err != nil {
		t.Fatalf("Messages.New returned error: %v", err)
	}

	if len(upstreamReq.Tools) != 1 || upstreamReq.Tools[0].Function == nil {
		t.Fatalf("upstream tools = %#v, want one function tool", upstreamReq.Tools)
	}
	if upstreamReq.Tools[0].Function.Name != "get_weather" {
		t.Fatalf("upstream tool name = %q", upstreamReq.Tools[0].Function.Name)
	}
	wantChoice := map[string]any{
		"type":     "function",
		"function": map[string]any{"name": "get_weather"},
	}
	gotChoice, _ := json.Marshal(upstreamReq.ToolChoice)
	wantChoiceJSON, _ := json.Marshal(wantChoice)
	if string(gotChoice) != string(wantChoiceJSON) {
		t.Fatalf("upstream tool_choice = %s, want %s", gotChoice, wantChoiceJSON)
	}

	if message.StopReason != anthropicsdk.StopReasonToolUse {
		t.Fatalf("stop_reason = %q, want tool_use", message.StopReason)
	}
	if len(message.Content) != 1 {
		t.Fatalf("content = %#v, want one tool_use block", message.Content)
	}

	toolUse, ok := message.Content[0].AsAny().(anthropicsdk.ToolUseBlock)
	if !ok {
		t.Fatalf("content block = %#v, want a ToolUseBlock", message.Content[0].AsAny())
	}
	if toolUse.ID != "toolu_sdk" || toolUse.Name != "get_weather" {
		t.Fatalf("tool_use block = %#v", toolUse)
	}

	var input map[string]any
	if err := json.Unmarshal(toolUse.Input, &input); err != nil {
		t.Fatalf("decode tool input: %v", err)
	}
	if input["city"] != "Paris" {
		t.Fatalf("tool input = %#v, want city=Paris", input)
	}
}

func TestSDKAnthropicCountTokens(t *testing.T) {
	var upstreamPath string

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamPath = r.URL.Path
		writeJSON(w, http.StatusOK, map[string]any{"tokens": []int{1, 2, 3}})
	})

	client := newAnthropicSDKClient(t, upstream)

	count, err := client.Messages.CountTokens(context.Background(), anthropicsdk.MessageCountTokensParams{
		Model: anthropicsdk.ModelClaudeSonnet4_5,
		Messages: []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hello world")),
		},
	})
	if err != nil {
		t.Fatalf("Messages.CountTokens returned error: %v", err)
	}

	if upstreamPath != "/tokenize" {
		t.Fatalf("upstream path = %q, want /tokenize", upstreamPath)
	}
	if count.InputTokens != 3 {
		t.Fatalf("input_tokens = %d, want 3", count.InputTokens)
	}
}

func TestSDKAnthropicUpstreamErrorSurfacesAsAPIError(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"context too long","type":"invalid_request_error"}}`))
	})

	client := newAnthropicSDKClient(t, upstream)

	_, err := client.Messages.New(context.Background(), anthropicsdk.MessageNewParams{
		Model:     anthropicsdk.ModelClaudeSonnet4_5,
		MaxTokens: 16,
		Messages: []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
		},
	})
	if err == nil {
		t.Fatal("expected an error from the SDK")
	}

	var apiErr *anthropicsdk.Error
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %#v, want an *anthropic.Error the SDK can classify", err)
	}
	if apiErr.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", apiErr.StatusCode)
	}
	if apiErr.Error() == "" {
		t.Fatal("SDK error message is empty; the error body did not decode")
	}
}
