package translate

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/harshaneel/localaik/internal/pdf"
	"github.com/harshaneel/localaik/internal/protocol/anthropic"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

func nopRenderer() pdf.Renderer {
	return pdf.RendererFunc(func(context.Context, []byte) ([][]byte, error) { return nil, nil })
}

// decodeAnthropicRequest parses wire JSON so tests exercise the custom Content
// unmarshalling alongside the translation itself.
func decodeAnthropicRequest(t *testing.T, body string) anthropic.MessagesRequest {
	t.Helper()
	var req anthropic.MessagesRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("decode Anthropic request: %v", err)
	}
	return req
}

func TestAnthropicRequestToOpenAIStringContent(t *testing.T) {
	req := decodeAnthropicRequest(t, `{
		"model": "claude-sonnet-4-5",
		"max_tokens": 512,
		"system": "Answer briefly.",
		"temperature": 0.3,
		"top_p": 0.9,
		"top_k": 40,
		"messages": [{"role": "user", "content": "hello"}]
	}`)

	got, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
	if err != nil {
		t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
	}

	if len(got.Messages) != 2 {
		t.Fatalf("messages = %#v, want system + user", got.Messages)
	}
	if got.Messages[0].Role != "system" || got.Messages[0].Content != "Answer briefly." {
		t.Fatalf("system message = %#v", got.Messages[0])
	}
	if got.Messages[1].Role != "user" || got.Messages[1].Content != "hello" {
		t.Fatalf("user message = %#v", got.Messages[1])
	}
	if got.MaxTokens == nil || *got.MaxTokens != 512 {
		t.Fatalf("max_tokens = %v, want 512", got.MaxTokens)
	}
	if got.Temperature == nil || *got.Temperature != 0.3 {
		t.Fatalf("temperature = %v, want 0.3", got.Temperature)
	}
	if got.TopP == nil || *got.TopP != 0.9 {
		t.Fatalf("top_p = %v, want 0.9", got.TopP)
	}
	if got.TopK == nil || *got.TopK != 40 {
		t.Fatalf("top_k = %v, want 40", got.TopK)
	}
	if got.Model != DefaultOpenAIModel {
		t.Fatalf("model = %q, want %q (the requested model is ignored upstream)", got.Model, DefaultOpenAIModel)
	}
}

func TestAnthropicRequestToOpenAIBlockContent(t *testing.T) {
	imageData := base64.StdEncoding.EncodeToString([]byte("png-bytes"))

	req := decodeAnthropicRequest(t, `{
		"max_tokens": 64,
		"system": [{"type": "text", "text": "Be terse."}],
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "what is this"},
				{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "`+imageData+`"}}
			]
		}]
	}`)

	got, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
	if err != nil {
		t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
	}

	if len(got.Messages) != 2 {
		t.Fatalf("messages = %#v, want system + user", got.Messages)
	}
	if got.Messages[0].Content != "Be terse." {
		t.Fatalf("system content = %#v, want flattened text", got.Messages[0].Content)
	}

	parts, ok := got.Messages[1].Content.([]openaip.ContentPart)
	if !ok {
		t.Fatalf("user content = %#v, want []ContentPart for multimodal input", got.Messages[1].Content)
	}
	if len(parts) != 2 {
		t.Fatalf("content parts = %#v, want text + image", parts)
	}
	if parts[0].Type != "text" || parts[0].Text != "what is this" {
		t.Fatalf("first part = %#v", parts[0])
	}
	wantURL := "data:image/png;base64," + imageData
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil || parts[1].ImageURL.URL != wantURL {
		t.Fatalf("second part = %#v, want image data URI", parts[1])
	}
}

func TestAnthropicRequestToOpenAIDocumentPDFRendersPages(t *testing.T) {
	var renderedPDF []byte
	renderer := pdf.RendererFunc(func(_ context.Context, data []byte) ([][]byte, error) {
		renderedPDF = data
		return [][]byte{[]byte("page-1"), []byte("page-2")}, nil
	})

	req := decodeAnthropicRequest(t, `{
		"max_tokens": 64,
		"messages": [{
			"role": "user",
			"content": [{
				"type": "document",
				"source": {"type": "base64", "media_type": "application/pdf", "data": "`+base64.StdEncoding.EncodeToString([]byte("%PDF-1.4"))+`"}
			}]
		}]
	}`)

	got, err := AnthropicRequestToOpenAI(context.Background(), req, renderer)
	if err != nil {
		t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
	}

	if string(renderedPDF) != "%PDF-1.4" {
		t.Fatalf("renderer received %q, want the decoded PDF bytes", renderedPDF)
	}

	parts, ok := got.Messages[0].Content.([]openaip.ContentPart)
	if !ok {
		t.Fatalf("content = %#v, want []ContentPart", got.Messages[0].Content)
	}
	if len(parts) != 2 {
		t.Fatalf("content parts = %#v, want one per rendered page", parts)
	}
	for i, want := range []string{"page-1", "page-2"} {
		wantURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte(want))
		if parts[i].Type != "image_url" || parts[i].ImageURL == nil || parts[i].ImageURL.URL != wantURL {
			t.Fatalf("part %d = %#v, want PNG data URI for %q", i, parts[i], want)
		}
	}
}

func TestAnthropicRequestToOpenAIDocumentTextSource(t *testing.T) {
	req := decodeAnthropicRequest(t, `{
		"max_tokens": 64,
		"messages": [{
			"role": "user",
			"content": [{
				"type": "document",
				"source": {"type": "text", "media_type": "text/plain", "data": "plain contract text"}
			}]
		}]
	}`)

	got, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
	if err != nil {
		t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
	}
	if got.Messages[0].Content != "plain contract text" {
		t.Fatalf("content = %#v, want inlined text", got.Messages[0].Content)
	}
}

func TestAnthropicRequestToOpenAIImageURLSource(t *testing.T) {
	req := decodeAnthropicRequest(t, `{
		"max_tokens": 64,
		"messages": [{
			"role": "user",
			"content": [{
				"type": "image",
				"source": {"type": "url", "url": "https://example.test/cat.png"}
			}]
		}]
	}`)

	got, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
	if err != nil {
		t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
	}

	parts, ok := got.Messages[0].Content.([]openaip.ContentPart)
	if !ok {
		t.Fatalf("content = %#v, want []ContentPart", got.Messages[0].Content)
	}
	if len(parts) != 1 || parts[0].ImageURL == nil || parts[0].ImageURL.URL != "https://example.test/cat.png" {
		t.Fatalf("parts = %#v, want the URL passed through", parts)
	}
}

func TestAnthropicRequestToOpenAIRejectsMissingMaxTokens(t *testing.T) {
	req := decodeAnthropicRequest(t, `{"messages":[{"role":"user","content":"hi"}]}`)

	_, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
	if err == nil {
		t.Fatal("expected an error when max_tokens is absent")
	}
	if got := err.Error(); got != "max_tokens: field required" {
		t.Fatalf("error = %q", got)
	}
}

func TestAnthropicRequestToOpenAIRejectsNonPositiveMaxTokens(t *testing.T) {
	for _, maxTokens := range []string{"0", "-1"} {
		t.Run("max_tokens="+maxTokens, func(t *testing.T) {
			req := decodeAnthropicRequest(t, `{"max_tokens":`+maxTokens+`,"messages":[{"role":"user","content":"hi"}]}`)

			_, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
			if err == nil {
				t.Fatalf("expected an error for max_tokens %s", maxTokens)
			}
			if !strings.Contains(err.Error(), "max_tokens") {
				t.Fatalf("error = %q, want it to name max_tokens", err)
			}
		})
	}
}

func TestAnthropicRequestToOpenAIRejectsOutOfRangeSampling(t *testing.T) {
	cases := map[string]string{
		"temperature_above_one": `"temperature":1.5`,
		"temperature_negative":  `"temperature":-0.1`,
		"top_p_above_one":       `"top_p":2`,
		"top_k_negative":        `"top_k":-1`,
	}

	for name, field := range cases {
		t.Run(name, func(t *testing.T) {
			req := decodeAnthropicRequest(t, `{"max_tokens":16,`+field+`,"messages":[{"role":"user","content":"hi"}]}`)

			if _, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer()); err == nil {
				t.Fatalf("expected an error for %s", field)
			}
		})
	}
}

func TestAnthropicRequestToOpenAIAcceptsRangeBoundaries(t *testing.T) {
	req := decodeAnthropicRequest(t, `{
		"max_tokens": 16,
		"temperature": 1,
		"top_p": 0,
		"top_k": 0,
		"messages": [{"role": "user", "content": "hi"}]
	}`)

	if _, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer()); err != nil {
		t.Fatalf("AnthropicRequestToOpenAI returned error at the range boundaries: %v", err)
	}
}

// A tool_result answers the assistant message that requested it, so it must come
// before any other content in the same turn. The SDK's tool-use loop emits a user
// turn shaped exactly like this.
func TestAnthropicRequestToOpenAIToolResultPrecedesFollowUpText(t *testing.T) {
	req := decodeAnthropicRequest(t, `{
		"max_tokens": 64,
		"messages": [
			{"role": "user", "content": "weather in Paris?"},
			{"role": "assistant", "content": [
				{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"},
				{"type": "text", "text": "and tomorrow?"}
			]}
		]
	}`)

	got, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
	if err != nil {
		t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
	}

	var roles []string
	for _, message := range got.Messages {
		roles = append(roles, message.Role)
	}
	want := []string{"user", "assistant", "tool", "user"}
	if !reflect.DeepEqual(roles, want) {
		t.Fatalf("message roles = %v, want %v (the tool result must follow the assistant that asked for it)", roles, want)
	}

	if got.Messages[2].ToolCallID != "toolu_1" || got.Messages[2].Content != "18C" {
		t.Fatalf("tool message = %#v", got.Messages[2])
	}
	if got.Messages[3].Content != "and tomorrow?" {
		t.Fatalf("follow-up message = %#v", got.Messages[3])
	}
}

func TestAnthropicRequestToOpenAIRejectsEmptyMessages(t *testing.T) {
	req := decodeAnthropicRequest(t, `{"max_tokens":16,"messages":[]}`)

	_, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
	if err == nil {
		t.Fatal("expected an error when messages is empty")
	}
}

func TestAnthropicRequestToOpenAIToolRoundTrip(t *testing.T) {
	req := decodeAnthropicRequest(t, `{
		"max_tokens": 128,
		"tools": [{
			"name": "get_weather",
			"description": "Look up weather",
			"input_schema": {"type": "OBJECT", "properties": {"city": {"type": "STRING"}}}
		}],
		"tool_choice": {"type": "tool", "name": "get_weather"},
		"messages": [
			{"role": "user", "content": "weather in Paris?"},
			{"role": "assistant", "content": [
				{"type": "text", "text": "Checking."},
				{"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "toolu_01", "content": "18C and clear"}
			]}
		]
	}`)

	got, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
	if err != nil {
		t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
	}

	if len(got.Messages) != 3 {
		t.Fatalf("messages = %#v, want user + assistant + tool", got.Messages)
	}

	assistant := got.Messages[1]
	if assistant.Role != "assistant" || assistant.Content != "Checking." {
		t.Fatalf("assistant message = %#v", assistant)
	}
	if len(assistant.ToolCalls) != 1 {
		t.Fatalf("assistant tool calls = %#v, want one", assistant.ToolCalls)
	}
	call := assistant.ToolCalls[0]
	if call.ID != "toolu_01" || call.Type != "function" || call.Function == nil {
		t.Fatalf("tool call = %#v", call)
	}
	if call.Function.Name != "get_weather" || call.Function.Arguments != `{"city": "Paris"}` {
		t.Fatalf("tool call function = %#v", call.Function)
	}

	toolMessage := got.Messages[2]
	if toolMessage.Role != "tool" || toolMessage.ToolCallID != "toolu_01" || toolMessage.Content != "18C and clear" {
		t.Fatalf("tool message = %#v", toolMessage)
	}

	if len(got.Tools) != 1 || got.Tools[0].Function == nil {
		t.Fatalf("tools = %#v, want one function tool", got.Tools)
	}
	// Anthropic input_schema becomes OpenAI parameters, with Gemini-style
	// uppercase type names normalised on the way through.
	wantSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{"type": "string"},
		},
	}
	if !reflect.DeepEqual(got.Tools[0].Function.Parameters, wantSchema) {
		t.Fatalf("tool parameters = %#v, want %#v", got.Tools[0].Function.Parameters, wantSchema)
	}

	wantChoice := map[string]any{
		"type":     "function",
		"function": map[string]any{"name": "get_weather"},
	}
	if !reflect.DeepEqual(got.ToolChoice, wantChoice) {
		t.Fatalf("tool_choice = %#v, want %#v", got.ToolChoice, wantChoice)
	}
}

func TestAnthropicRequestToOpenAIToolResultErrorFlag(t *testing.T) {
	req := decodeAnthropicRequest(t, `{
		"max_tokens": 32,
		"messages": [{"role": "user", "content": [
			{"type": "tool_result", "tool_use_id": "toolu_9", "is_error": true, "content": "city not found"}
		]}]
	}`)

	got, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
	if err != nil {
		t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
	}
	if len(got.Messages) != 1 {
		t.Fatalf("messages = %#v, want one tool message", got.Messages)
	}
	if got.Messages[0].Content != "Error: city not found" {
		t.Fatalf("tool content = %#v, want the failure marked", got.Messages[0].Content)
	}
}

// A tool_use block in a user-role message cannot become an OpenAI tool call, so
// it is preserved as text context instead of being dropped.
func TestAnthropicRequestToOpenAIUserRoleToolUseBecomesText(t *testing.T) {
	req := decodeAnthropicRequest(t, `{
		"max_tokens": 32,
		"messages": [{"role": "user", "content": [
			{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
		]}]
	}`)

	got, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
	if err != nil {
		t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
	}
	if len(got.Messages) != 1 {
		t.Fatalf("messages = %#v, want one user message", got.Messages)
	}
	if len(got.Messages[0].ToolCalls) != 0 {
		t.Fatalf("user message has tool calls %#v, want none", got.Messages[0].ToolCalls)
	}
	want := `Tool use get_weather: {"city": "Paris"}`
	if got.Messages[0].Content != want {
		t.Fatalf("content = %#v, want %q", got.Messages[0].Content, want)
	}
}

// An OpenAI tool message carries a plain string, so non-text tool_result blocks
// are summarised rather than silently dropped.
func TestAnthropicRequestToOpenAIToolResultNonTextBlocks(t *testing.T) {
	req := decodeAnthropicRequest(t, `{
		"max_tokens": 32,
		"messages": [{"role": "user", "content": [{
			"type": "tool_result",
			"tool_use_id": "toolu_1",
			"content": [
				{"type": "text", "text": "here is the chart"},
				{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}},
				{"type": "document", "source": {"type": "base64"}}
			]
		}]}]
	}`)

	got, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
	if err != nil {
		t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
	}

	want := "here is the chart\n[image image/png]\n[document]"
	if got.Messages[0].Content != want {
		t.Fatalf("tool content = %#v, want %q", got.Messages[0].Content, want)
	}
}

func TestAnthropicRequestToOpenAISourceErrors(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{
			name: "undecodable_base64",
			body: `{"max_tokens":16,"messages":[{"role":"user","content":[
				{"type":"image","source":{"type":"base64","media_type":"image/png","data":"!!!not base64!!!"}}
			]}]}`,
		},
		{
			name: "unsupported_source_type",
			body: `{"max_tokens":16,"messages":[{"role":"user","content":[
				{"type":"image","source":{"type":"file_id","media_type":"image/png"}}
			]}]}`,
		},
		{
			// Without a media type an image cannot be told from a PDF, and the
			// fallback would forward a description of the bytes as prompt text.
			name: "base64_without_media_type",
			body: `{"max_tokens":16,"messages":[{"role":"user","content":[
				{"type":"image","source":{"type":"base64","data":"aGk="}}
			]}]}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := decodeAnthropicRequest(t, tc.body)

			_, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
			if err == nil {
				t.Fatal("expected an error for an unusable source")
			}
		})
	}
}

// A source-less image or document block carries nothing translatable, so it is
// skipped rather than erroring.
func TestAnthropicRequestToOpenAISourcelessBlockIsSkipped(t *testing.T) {
	req := decodeAnthropicRequest(t, `{
		"max_tokens": 16,
		"messages": [{"role": "user", "content": [
			{"type": "image"},
			{"type": "text", "text": "still here"}
		]}]
	}`)

	got, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
	if err != nil {
		t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
	}
	if got.Messages[0].Content != "still here" {
		t.Fatalf("content = %#v, want only the text", got.Messages[0].Content)
	}
}

// Anthropic server tools carry a versioned type and no input_schema. localaik
// cannot run them, so forwarding them as callable functions would invite the
// model to call something that does not exist.
func TestAnthropicRequestToOpenAISkipsServerTools(t *testing.T) {
	req := decodeAnthropicRequest(t, `{
		"max_tokens": 32,
		"tools": [
			{"type": "web_search_20250305", "name": "web_search"},
			{"type": "computer_20241022", "name": "computer"},
			{"name": "get_weather", "input_schema": {"type": "object"}},
			{"type": "custom", "name": "explicitly_custom", "input_schema": {"type": "object"}}
		],
		"messages": [{"role": "user", "content": "hi"}]
	}`)

	got, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
	if err != nil {
		t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
	}

	var names []string
	for _, tool := range got.Tools {
		if tool.Function == nil {
			t.Fatalf("tool without a function: %#v", tool)
		}
		names = append(names, tool.Function.Name)
	}
	want := []string{"get_weather", "explicitly_custom"}
	if !reflect.DeepEqual(names, want) {
		t.Fatalf("forwarded tools = %v, want %v", names, want)
	}
}

func TestAnthropicToolChoiceMapping(t *testing.T) {
	cases := []struct {
		name string
		json string
		want any
	}{
		{"auto", `{"type":"auto"}`, "auto"},
		{"any", `{"type":"any"}`, "required"},
		{"none", `{"type":"none"}`, "none"},
		{"tool_without_name", `{"type":"tool"}`, "required"},
		{"unknown", `{"type":"future_mode"}`, nil},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// A tool has to survive translation for tool_choice to be forwarded at
			// all; see TestAnthropicRequestToOpenAIDropsToolChoiceWithNoTools.
			req := decodeAnthropicRequest(t, `{
				"max_tokens": 8,
				"tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
				"tool_choice": `+tc.json+`,
				"messages": [{"role": "user", "content": "hi"}]
			}`)

			got, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
			if err != nil {
				t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
			}
			if !reflect.DeepEqual(got.ToolChoice, tc.want) {
				t.Fatalf("tool_choice = %#v, want %#v", got.ToolChoice, tc.want)
			}
		})
	}
}

func TestAnthropicRequestToOpenAIDropsThinkingBlocks(t *testing.T) {
	req := decodeAnthropicRequest(t, `{
		"max_tokens": 64,
		"messages": [
			{"role": "assistant", "content": [
				{"type": "thinking", "thinking": "internal scratch", "signature": "sig"},
				{"type": "redacted_thinking", "data": "opaque"},
				{"type": "text", "text": "visible answer"}
			]}
		]
	}`)

	got, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
	if err != nil {
		t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
	}
	if len(got.Messages) != 1 || got.Messages[0].Content != "visible answer" {
		t.Fatalf("messages = %#v, want only the visible text", got.Messages)
	}
}

func TestAnthropicRequestToOpenAIStopSequences(t *testing.T) {
	cases := []struct {
		name string
		json string
		want any
	}{
		{"single_collapses_to_string", `["STOP"]`, "STOP"},
		{"multiple_stay_a_list", `["A","B"]`, []string{"A", "B"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := decodeAnthropicRequest(t, `{"max_tokens":8,"stop_sequences":`+tc.json+`,"messages":[{"role":"user","content":"hi"}]}`)

			got, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
			if err != nil {
				t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
			}
			if !reflect.DeepEqual(got.Stop, tc.want) {
				t.Fatalf("stop = %#v, want %#v", got.Stop, tc.want)
			}
		})
	}
}

func TestOpenAIResponseToAnthropicText(t *testing.T) {
	resp := openaip.ChatCompletionResponse{
		ID:    "chatcmpl-abc123",
		Model: "gemma-3-4b",
		Choices: []openaip.Choice{{
			Message:      openaip.Message{Role: "assistant", Content: "hi there"},
			FinishReason: "stop",
		}},
		Usage: &openaip.Usage{PromptTokens: 11, CompletionTokens: 4, TotalTokens: 15},
	}

	got := OpenAIResponseToAnthropic(resp, "claude-sonnet-4-5")

	if got.ID != "msg_abc123" {
		t.Fatalf("id = %q, want the upstream id under the msg_ prefix", got.ID)
	}
	if got.Type != "message" || got.Role != "assistant" {
		t.Fatalf("envelope = %#v", got)
	}
	if got.Model != "claude-sonnet-4-5" {
		t.Fatalf("model = %q, want the requested model echoed back", got.Model)
	}
	want := []anthropic.ContentBlock{{Type: "text", Text: "hi there"}}
	if !reflect.DeepEqual(got.Content, want) {
		t.Fatalf("content = %#v, want %#v", got.Content, want)
	}
	if got.StopReason == nil || *got.StopReason != anthropic.StopReasonEndTurn {
		t.Fatalf("stop_reason = %v, want end_turn", got.StopReason)
	}
	if got.StopSequence != nil {
		t.Fatalf("stop_sequence = %v, want null", *got.StopSequence)
	}
	if got.Usage != (anthropic.Usage{InputTokens: 11, OutputTokens: 4}) {
		t.Fatalf("usage = %#v", got.Usage)
	}
}

func TestOpenAIResponseToAnthropicToolCalls(t *testing.T) {
	resp := openaip.ChatCompletionResponse{
		Choices: []openaip.Choice{{
			Message: openaip.Message{
				Role: "assistant",
				ToolCalls: []openaip.ToolCall{{
					ID:   "toolu_77",
					Type: "function",
					Function: &openaip.ToolCallFunction{
						Name:      "get_weather",
						Arguments: `{"city":"Paris"}`,
					},
				}},
			},
			FinishReason: "tool_calls",
		}},
	}

	got := OpenAIResponseToAnthropic(resp, "")

	if len(got.Content) != 1 {
		t.Fatalf("content = %#v, want one tool_use block", got.Content)
	}
	block := got.Content[0]
	if block.Type != anthropic.BlockTypeToolUse || block.ID != "toolu_77" || block.Name != "get_weather" {
		t.Fatalf("block = %#v", block)
	}
	if string(block.Input) != `{"city":"Paris"}` {
		t.Fatalf("input = %s, want the arguments as a JSON object", block.Input)
	}
	if got.StopReason == nil || *got.StopReason != anthropic.StopReasonToolUse {
		t.Fatalf("stop_reason = %v, want tool_use", got.StopReason)
	}
	if got.Model != DefaultOpenAIModel {
		t.Fatalf("model = %q, want the fallback model", got.Model)
	}
}

// Some runtimes report finish_reason "stop", or nothing, alongside tool calls.
// Agent loops branch on stop_reason to decide whether to run tools, so a message
// carrying tool_use has to say tool_use. A "length" finish is the exception: the
// arguments were cut off, and reporting max_tokens is both what the real API does
// and what stops an agent executing a truncated call.
func TestOpenAIResponseToAnthropicStopReasonWithToolCalls(t *testing.T) {
	cases := map[string]string{
		"stop":           anthropic.StopReasonToolUse,
		"":               anthropic.StopReasonToolUse,
		"content_filter": anthropic.StopReasonToolUse,
		"tool_calls":     anthropic.StopReasonToolUse,
		"length":         anthropic.StopReasonMaxTokens,
	}

	for finishReason, want := range cases {
		t.Run("finish_reason="+finishReason, func(t *testing.T) {
			resp := openaip.ChatCompletionResponse{
				Choices: []openaip.Choice{{
					Message: openaip.Message{
						Content: "let me check",
						ToolCalls: []openaip.ToolCall{{
							ID:       "toolu_1",
							Function: &openaip.ToolCallFunction{Name: "get_weather", Arguments: `{}`},
						}},
					},
					FinishReason: finishReason,
				}},
			}

			got := OpenAIResponseToAnthropic(resp, "")

			if got.StopReason == nil {
				t.Fatal("stop_reason is null on a message carrying tool_use")
			}
			if *got.StopReason != want {
				t.Fatalf("stop_reason = %q, want %q", *got.StopReason, want)
			}
		})
	}
}

// The streaming path has to resolve stop_reason the same way, since that is the
// path agent SDKs actually use.
func TestWriteAnthropicStreamStopReasonMatchesNonStreaming(t *testing.T) {
	cases := map[string]string{
		"stop":       anthropic.StopReasonToolUse,
		"tool_calls": anthropic.StopReasonToolUse,
		"length":     anthropic.StopReasonMaxTokens,
	}

	for finishReason, want := range cases {
		t.Run("finish_reason="+finishReason, func(t *testing.T) {
			upstream := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]}}]}` + "\n\n" +
				`data: {"choices":[{"delta":{},"finish_reason":"` + finishReason + `"}]}` + "\n\n" +
				"data: [DONE]\n\n"

			events, _ := writeAnthropicStream(t, upstream, "")

			var got string
			for _, event := range events {
				if event.name != anthropic.EventMessageDelta {
					continue
				}
				decoded := decodeEvent(t, event)
				if decoded.Delta == nil || decoded.Delta.StopReason == nil {
					t.Fatalf("message_delta has no stop_reason: %s", event.data)
				}
				got = *decoded.Delta.StopReason
			}

			if got != want {
				t.Fatalf("stop_reason = %q, want %q", got, want)
			}
		})
	}
}

// A streamed message with no tool calls keeps the plain mapping.
func TestWriteAnthropicStreamStopReasonWithoutToolCalls(t *testing.T) {
	upstream := `data: {"choices":[{"delta":{"content":"hi"}}]}` + "\n\n" +
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
		"data: [DONE]\n\n"

	events, _ := writeAnthropicStream(t, upstream, "")

	for _, event := range events {
		if event.name != anthropic.EventMessageDelta {
			continue
		}
		decoded := decodeEvent(t, event)
		if *decoded.Delta.StopReason != anthropic.StopReasonEndTurn {
			t.Fatalf("stop_reason = %q, want end_turn", *decoded.Delta.StopReason)
		}
	}
}

// An unnamed tool call cannot be invoked and the Messages API never emits one.
func TestOpenAIResponseToAnthropicDropsUnnamedToolCalls(t *testing.T) {
	resp := openaip.ChatCompletionResponse{
		Choices: []openaip.Choice{{
			Message: openaip.Message{
				ToolCalls: []openaip.ToolCall{
					{ID: "toolu_1", Function: &openaip.ToolCallFunction{Name: "", Arguments: `{"a":1}`}},
					{ID: "toolu_2", Function: &openaip.ToolCallFunction{Name: "ok", Arguments: `{}`}},
				},
			},
			FinishReason: "tool_calls",
		}},
	}

	got := OpenAIResponseToAnthropic(resp, "")

	if len(got.Content) != 1 || got.Content[0].Name != "ok" {
		t.Fatalf("content = %#v, want only the named tool call", got.Content)
	}
}

// Parallel calls to one tool from an upstream that sends no ids must not share an
// id, and the disambiguating suffix must not itself collide with a real id.
func TestOpenAIResponseToAnthropicToolUseIDsAreUnique(t *testing.T) {
	cases := []struct {
		name  string
		calls []openaip.ToolCall
	}{
		{
			name: "same_tool_twice_without_ids",
			calls: []openaip.ToolCall{
				{Function: &openaip.ToolCallFunction{Name: "read_file", Arguments: `{"p":"a.txt"}`}},
				{Function: &openaip.ToolCallFunction{Name: "read_file", Arguments: `{"p":"b.txt"}`}},
				{Function: &openaip.ToolCallFunction{Name: "read_file", Arguments: `{"p":"c.txt"}`}},
			},
		},
		{
			// A real id shaped like the suffix this would synthesize.
			name: "real_id_collides_with_a_suffix",
			calls: []openaip.ToolCall{
				{ID: "toolu_f_2", Function: &openaip.ToolCallFunction{Name: "g", Arguments: `{}`}},
				{Function: &openaip.ToolCallFunction{Name: "f", Arguments: `{}`}},
				{Function: &openaip.ToolCallFunction{Name: "f", Arguments: `{}`}},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp := openaip.ChatCompletionResponse{
				Choices: []openaip.Choice{{Message: openaip.Message{ToolCalls: tc.calls}}},
			}

			got := OpenAIResponseToAnthropic(resp, "")

			if len(got.Content) != len(tc.calls) {
				t.Fatalf("content = %#v, want %d blocks", got.Content, len(tc.calls))
			}
			seen := map[string]bool{}
			for _, block := range got.Content {
				if block.ID == "" {
					t.Fatalf("block %#v has no id", block)
				}
				if seen[block.ID] {
					t.Fatalf("id %q used twice; content = %#v", block.ID, got.Content)
				}
				seen[block.ID] = true
			}
		})
	}
}

func TestOpenAIResponseToAnthropicMalformedToolArguments(t *testing.T) {
	resp := openaip.ChatCompletionResponse{
		Choices: []openaip.Choice{{
			Message: openaip.Message{
				ToolCalls: []openaip.ToolCall{{
					ID:       "toolu_1",
					Function: &openaip.ToolCallFunction{Name: "f", Arguments: `{"city":`},
				}},
			},
		}},
	}

	got := OpenAIResponseToAnthropic(resp, "")

	if len(got.Content) != 1 || string(got.Content[0].Input) != `{}` {
		t.Fatalf("content = %#v, want truncated arguments replaced with an empty object", got.Content)
	}
}

// tool_use.input is always a JSON object. Anything else upstream produces has to
// be replaced, not passed through, or clients decoding input into a struct fail.
func TestOpenAIResponseToAnthropicNonObjectToolArguments(t *testing.T) {
	cases := []struct {
		name      string
		arguments string
		want      string
	}{
		{"truncated", `{"city":`, `{}`},
		{"empty", ``, `{}`},
		{"whitespace", `   `, `{}`},
		{"null_literal", `null`, `{}`},
		{"array", `[1,2]`, `{}`},
		{"bare_string", `"text"`, `{}`},
		{"number", `42`, `{}`},
		{"valid_object", `{"city":"Paris"}`, `{"city":"Paris"}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp := openaip.ChatCompletionResponse{
				Choices: []openaip.Choice{{
					Message: openaip.Message{
						ToolCalls: []openaip.ToolCall{{
							ID:       "toolu_1",
							Function: &openaip.ToolCallFunction{Name: "f", Arguments: tc.arguments},
						}},
					},
				}},
			}

			got := OpenAIResponseToAnthropic(resp, "")
			if len(got.Content) != 1 {
				t.Fatalf("content = %#v, want one tool_use block", got.Content)
			}
			if string(got.Content[0].Input) != tc.want {
				t.Fatalf("input = %s, want %s", got.Content[0].Input, tc.want)
			}

			// Whatever we emit must decode as an object for the client.
			var decoded map[string]any
			if err := json.Unmarshal(got.Content[0].Input, &decoded); err != nil {
				t.Fatalf("input %s does not decode as an object: %v", got.Content[0].Input, err)
			}
		})
	}
}

func TestOpenAIResponseToAnthropicEmptyChoices(t *testing.T) {
	got := OpenAIResponseToAnthropic(openaip.ChatCompletionResponse{}, "")

	if got.Content == nil {
		t.Fatal("content is nil; the Messages API always sends an array")
	}
	if len(got.Content) != 0 {
		t.Fatalf("content = %#v, want empty", got.Content)
	}
	if got.StopReason != nil {
		t.Fatalf("stop_reason = %v, want null", *got.StopReason)
	}

	encoded, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal response: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if content, ok := decoded["content"].([]any); !ok || len(content) != 0 {
		t.Fatalf("content encoded as %v, want []", decoded["content"])
	}
	if _, ok := decoded["stop_reason"]; !ok {
		t.Fatalf("stop_reason missing from %s; it must be present as null", encoded)
	}
}

func TestOpenAIFinishReasonToAnthropic(t *testing.T) {
	cases := map[string]string{
		"":               "",
		"stop":           anthropic.StopReasonEndTurn,
		"length":         anthropic.StopReasonMaxTokens,
		"tool_calls":     anthropic.StopReasonToolUse,
		"function_call":  anthropic.StopReasonToolUse,
		"content_filter": anthropic.StopReasonEndTurn,
		"surprise":       anthropic.StopReasonEndTurn,
	}

	for reason, want := range cases {
		if got := OpenAIFinishReasonToAnthropic(reason); got != want {
			t.Fatalf("OpenAIFinishReasonToAnthropic(%q) = %q, want %q", reason, got, want)
		}
	}
}

// Ids supplied by the client must survive unchanged on both sides, which is the
// only case Anthropic actually permits: id and tool_use_id are both required.
func TestAnthropicToolIDsPassThroughOnBothSides(t *testing.T) {
	req := decodeAnthropicRequest(t, `{
		"max_tokens": 32,
		"messages": [
			{"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_7", "name": "get_weather", "input": {}}]},
			{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_7", "content": "done"}]}
		]
	}`)

	got, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
	if err != nil {
		t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
	}
	if len(got.Messages) != 2 {
		t.Fatalf("messages = %#v, want assistant + tool", got.Messages)
	}
	if len(got.Messages[0].ToolCalls) != 1 || got.Messages[0].ToolCalls[0].ID != "toolu_7" {
		t.Fatalf("assistant tool calls = %#v, want the client id preserved", got.Messages[0].ToolCalls)
	}
	if got.Messages[1].ToolCallID != "toolu_7" {
		t.Fatalf("tool_call_id = %q, want toolu_7", got.Messages[1].ToolCallID)
	}
}

// A malformed request that omits ids cannot be re-paired: a tool_result has no
// name to derive one from. The only guarantee is that neither side goes upstream
// with an empty tool_call_id, which OpenAI would reject outright.
func TestAnthropicMissingToolIDsStillProduceNonEmptyIDs(t *testing.T) {
	req := decodeAnthropicRequest(t, `{
		"max_tokens": 32,
		"messages": [
			{"role": "assistant", "content": [{"type": "tool_use", "name": "get_weather", "input": {}}]},
			{"role": "user", "content": [{"type": "tool_result", "content": "done"}]}
		]
	}`)

	got, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
	if err != nil {
		t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
	}
	if got.Messages[0].ToolCalls[0].ID == "" {
		t.Fatal("tool call id is empty")
	}
	if got.Messages[1].ToolCallID == "" {
		t.Fatal("tool_call_id is empty")
	}
}

// Skipping every declared tool must also drop tool_choice: pointing it at a tool
// that is no longer in the request is rejected by the upstream runtime.
func TestAnthropicRequestToOpenAIDropsToolChoiceWithNoTools(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{
			name: "all_tools_skipped",
			body: `{"max_tokens":32,
				"tools":[{"type":"web_search_20250305","name":"web_search"}],
				"tool_choice":{"type":"tool","name":"web_search"},
				"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name: "no_tools_declared",
			body: `{"max_tokens":32,"tool_choice":{"type":"any"},"messages":[{"role":"user","content":"hi"}]}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := decodeAnthropicRequest(t, tc.body)

			got, err := AnthropicRequestToOpenAI(context.Background(), req, nopRenderer())
			if err != nil {
				t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
			}
			if len(got.Tools) != 0 {
				t.Fatalf("tools = %#v, want none", got.Tools)
			}
			if got.ToolChoice != nil {
				t.Fatalf("tool_choice = %#v, want nil when there are no tools", got.ToolChoice)
			}
		})
	}
}

func TestAnthropicToolUseIDFallbacks(t *testing.T) {
	cases := []struct {
		name string
		id   string
		tool string
		want string
	}{
		{"uses_given_id", "toolu_abc", "get_weather", "toolu_abc"},
		{"derives_from_name", "", "Get Weather", "toolu_get_weather"},
		{"falls_back_when_bare", "", "", "toolu_localaik"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := anthropicToolUseID(tc.id, tc.tool); got != tc.want {
				t.Fatalf("anthropicToolUseID(%q, %q) = %q, want %q", tc.id, tc.tool, got, tc.want)
			}
		})
	}
}

func TestCountTokensTextFromAnthropic(t *testing.T) {
	imageData := base64.StdEncoding.EncodeToString([]byte("png"))

	req := decodeAnthropicRequest(t, `{
		"max_tokens": 16,
		"system": "system prompt",
		"messages": [
			{"role": "user", "content": [
				{"type": "text", "text": "first"},
				{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "`+imageData+`"}}
			]},
			{"role": "assistant", "content": "second"},
			{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "third"}]}
		]
	}`)

	got := CountTokensTextFromAnthropic(req)
	want := "system prompt\nfirst\nsecond\nthird"
	if got != want {
		t.Fatalf("CountTokensTextFromAnthropic = %q, want %q", got, want)
	}
}

// A text-source document block is inlined into the prompt by /v1/messages, so it
// has to be counted too, or count_tokens disagrees with what actually gets sent.
func TestCountTokensTextFromAnthropicIncludesTextDocuments(t *testing.T) {
	body := `{
		"max_tokens": 16,
		"messages": [{"role": "user", "content": [
			{"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "contract body"}},
			{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}},
			{"type": "text", "text": "summarise"}
		]}]
	}`

	req := decodeAnthropicRequest(t, body)

	got := CountTokensTextFromAnthropic(req)
	want := "contract body\nsummarise"
	if got != want {
		t.Fatalf("CountTokensTextFromAnthropic = %q, want %q (base64 PDFs stay uncounted)", got, want)
	}

	// The same document must reach the model, so the two paths agree on it.
	translated, err := AnthropicRequestToOpenAI(context.Background(), decodeAnthropicRequest(t, body), nopRenderer())
	if err != nil {
		t.Fatalf("AnthropicRequestToOpenAI returned error: %v", err)
	}
	if !strings.Contains(fmt.Sprint(translated.Messages[0].Content), "contract body") {
		t.Fatalf("translated content = %#v, want the document text inlined", translated.Messages[0].Content)
	}
}

func TestOpenAIErrorToAnthropic(t *testing.T) {
	cases := []struct {
		name     string
		status   int
		body     string
		wantType string
		wantMsg  string
	}{
		{
			name:     "uses_upstream_message",
			status:   http.StatusBadRequest,
			body:     `{"error":{"message":"bad prompt","type":"invalid_request_error"}}`,
			wantType: anthropic.ErrorTypeInvalidRequest,
			wantMsg:  "bad prompt",
		},
		{
			name:     "falls_back_to_status_text",
			status:   http.StatusInternalServerError,
			body:     `not json`,
			wantType: anthropic.ErrorTypeAPI,
			wantMsg:  http.StatusText(http.StatusInternalServerError),
		},
		{
			name:     "rate_limit",
			status:   http.StatusTooManyRequests,
			body:     `{}`,
			wantType: anthropic.ErrorTypeRateLimit,
			wantMsg:  http.StatusText(http.StatusTooManyRequests),
		},
		{
			name:     "overloaded",
			status:   529,
			body:     `{}`,
			wantType: anthropic.ErrorTypeOverloaded,
			wantMsg:  "upstream error",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := OpenAIErrorToAnthropic(tc.status, []byte(tc.body))

			if got.Type != "error" {
				t.Fatalf("envelope type = %q, want error", got.Type)
			}
			if got.Error.Type != tc.wantType {
				t.Fatalf("error type = %q, want %q", got.Error.Type, tc.wantType)
			}
			if got.Error.Message != tc.wantMsg {
				t.Fatalf("error message = %q, want %q", got.Error.Message, tc.wantMsg)
			}
		})
	}
}

func TestAnthropicContentJSONRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		json string
	}{
		{"string_stays_a_string", `"hello"`},
		{"blocks_stay_an_array", `[{"type":"text","text":"hello"}]`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var content anthropic.Content
			if err := json.Unmarshal([]byte(tc.json), &content); err != nil {
				t.Fatalf("decode content: %v", err)
			}
			if content.Text() != "hello" {
				t.Fatalf("Text() = %q, want hello", content.Text())
			}

			encoded, err := json.Marshal(content)
			if err != nil {
				t.Fatalf("encode content: %v", err)
			}
			if string(encoded) != tc.json {
				t.Fatalf("re-encoded as %s, want %s", encoded, tc.json)
			}
		})
	}
}

func TestAnthropicContentNullDecodesEmpty(t *testing.T) {
	var content anthropic.Content
	if err := json.Unmarshal([]byte("null"), &content); err != nil {
		t.Fatalf("decode null content: %v", err)
	}
	if len(content.Blocks) != 0 || content.IsString {
		t.Fatalf("content = %#v, want zero value", content)
	}
}
