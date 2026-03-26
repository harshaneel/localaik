package translate

import (
	"context"
	"encoding/base64"
	"reflect"
	"testing"

	"github.com/harshaneel/localaik/internal/pdf"
	"github.com/harshaneel/localaik/internal/protocol/gemini"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

func TestGeminiRequestToOpenAI(t *testing.T) {
	temp := 0.2
	maxTokens := 321

	req := gemini.GenerateContentRequest{
		SystemInstruction: &gemini.Content{
			Parts: []gemini.Part{{Text: "Return JSON only."}},
		},
		Contents: []gemini.Content{
			{
				Role: "user",
				Parts: []gemini.Part{
					{Text: "Describe this document"},
					{
						InlineData: &gemini.Blob{
							MimeType: "image/png",
							Data:     base64.StdEncoding.EncodeToString([]byte("image-bytes")),
						},
					},
					{
						InlineData: &gemini.Blob{
							MimeType: "application/pdf",
							Data:     base64.StdEncoding.EncodeToString([]byte("%PDF-1.4")),
						},
					},
				},
			},
		},
		GenerationConfig: &gemini.GenerationConfig{
			Temperature:      &temp,
			MaxOutputTokens:  &maxTokens,
			ResponseMimeType: "application/json",
			ResponseSchema: map[string]any{
				"type": "OBJECT",
				"properties": map[string]any{
					"summary": map[string]any{"type": "STRING"},
					"scores": map[string]any{
						"type":  "ARRAY",
						"items": map[string]any{"type": "INTEGER"},
					},
				},
			},
		},
	}

	renderer := pdf.RendererFunc(func(ctx context.Context, pdf []byte) ([][]byte, error) {
		if string(pdf) != "%PDF-1.4" {
			t.Fatalf("renderer received wrong PDF payload: %q", string(pdf))
		}
		return [][]byte{[]byte("page-1"), []byte("page-2")}, nil
	})

	got, err := GeminiRequestToOpenAI(context.Background(), req, renderer)
	if err != nil {
		t.Fatalf("GeminiRequestToOpenAI returned error: %v", err)
	}

	if got.Model != DefaultOpenAIModel {
		t.Fatalf("model = %q, want %q", got.Model, DefaultOpenAIModel)
	}
	if got.Temperature == nil || *got.Temperature != temp {
		t.Fatalf("temperature = %#v, want %v", got.Temperature, temp)
	}
	if got.MaxTokens == nil || *got.MaxTokens != maxTokens {
		t.Fatalf("maxTokens = %#v, want %d", got.MaxTokens, maxTokens)
	}
	if len(got.Messages) != 2 {
		t.Fatalf("message count = %d, want 2", len(got.Messages))
	}
	if got.Messages[0].Role != "system" || got.Messages[0].Content != "Return JSON only." {
		t.Fatalf("system message = %#v", got.Messages[0])
	}

	userParts, ok := got.Messages[1].Content.([]openaip.ContentPart)
	if !ok {
		t.Fatalf("user content type = %T, want []ContentPart", got.Messages[1].Content)
	}
	if len(userParts) != 4 {
		t.Fatalf("user content parts = %d, want 4", len(userParts))
	}
	if userParts[0].Type != "text" || userParts[0].Text != "Describe this document" {
		t.Fatalf("first part = %#v", userParts[0])
	}
	if userParts[1].Type != "image_url" || userParts[1].ImageURL == nil || userParts[1].ImageURL.URL == "" {
		t.Fatalf("second part = %#v", userParts[1])
	}
	if userParts[2].ImageURL == nil || userParts[3].ImageURL == nil {
		t.Fatalf("pdf parts missing image URLs: %#v", userParts)
	}
	if got.ResponseFormat == nil || got.ResponseFormat.Type != "json_schema" {
		t.Fatalf("response format = %#v, want json_schema", got.ResponseFormat)
	}
	if got.ResponseFormat.JSONSchema == nil || got.ResponseFormat.JSONSchema.Name != "response" {
		t.Fatalf("json schema config = %#v", got.ResponseFormat.JSONSchema)
	}
	wantSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"summary": map[string]any{"type": "string"},
			"scores": map[string]any{
				"type":  "array",
				"items": map[string]any{"type": "integer"},
			},
		},
	}
	if !reflect.DeepEqual(got.ResponseFormat.JSONSchema.Schema, wantSchema) {
		t.Fatalf("normalized schema = %#v, want %#v", got.ResponseFormat.JSONSchema.Schema, wantSchema)
	}
}

func TestGeminiRequestToOpenAIJSONModeWithoutSchema(t *testing.T) {
	req := gemini.GenerateContentRequest{
		Contents: []gemini.Content{{
			Role:  "user",
			Parts: []gemini.Part{{Text: "hi"}},
		}},
		GenerationConfig: &gemini.GenerationConfig{
			ResponseMimeType: "application/json",
		},
	}

	got, err := GeminiRequestToOpenAI(context.Background(), req, pdf.RendererFunc(func(context.Context, []byte) ([][]byte, error) {
		return nil, nil
	}))
	if err != nil {
		t.Fatalf("GeminiRequestToOpenAI returned error: %v", err)
	}
	if got.ResponseFormat == nil || got.ResponseFormat.Type != "json_object" {
		t.Fatalf("response format = %#v, want json_object", got.ResponseFormat)
	}
}

func TestGeminiRequestToOpenAIWithFileDataAndTools(t *testing.T) {
	topP := 0.9
	topK := 40.0
	candidateCount := 2
	maxTokens := 128
	responseLogprobs := true
	logprobs := 3
	presencePenalty := 0.1
	frequencyPenalty := 0.2
	seed := 7

	req := gemini.GenerateContentRequest{
		Contents: []gemini.Content{
			{
				Role: "user",
				Parts: []gemini.Part{
					{Text: "Look up this document"},
					{
						FileData: &gemini.FileData{
							FileURI:  dataURI("application/pdf", []byte("%PDF-filedata")),
							MimeType: "application/pdf",
						},
					},
				},
			},
			{
				Role: "model",
				Parts: []gemini.Part{
					{
						FunctionCall: &gemini.FunctionCall{
							ID:   "call_lookup",
							Name: "lookup_weather",
							Args: map[string]any{"city": "Boston"},
						},
					},
				},
			},
			{
				Role: "user",
				Parts: []gemini.Part{
					{
						FunctionResponse: &gemini.FunctionResponse{
							ID:       "call_lookup",
							Name:     "lookup_weather",
							Response: map[string]any{"output": "sunny"},
						},
					},
				},
			},
		},
		Tools: []gemini.Tool{{
			FunctionDeclarations: []gemini.FunctionDeclaration{{
				Name:        "lookup_weather",
				Description: "Look up the weather by city.",
				Parameters: map[string]any{
					"type": "OBJECT",
					"properties": map[string]any{
						"city": map[string]any{"type": "STRING"},
					},
					"required": []any{"city"},
				},
			}},
		}},
		ToolConfig: &gemini.ToolConfig{
			FunctionCallingConfig: &gemini.FunctionCallingConfig{
				Mode:                 "ANY",
				AllowedFunctionNames: []string{"lookup_weather"},
			},
		},
		GenerationConfig: &gemini.GenerationConfig{
			TopP:             &topP,
			TopK:             &topK,
			CandidateCount:   &candidateCount,
			MaxOutputTokens:  &maxTokens,
			StopSequences:    []string{"END", "DONE"},
			ResponseLogprobs: &responseLogprobs,
			Logprobs:         &logprobs,
			PresencePenalty:  &presencePenalty,
			FrequencyPenalty: &frequencyPenalty,
			Seed:             &seed,
		},
	}

	renderer := pdf.RendererFunc(func(ctx context.Context, pdfBytes []byte) ([][]byte, error) {
		if string(pdfBytes) != "%PDF-filedata" {
			t.Fatalf("renderer received wrong fileData PDF payload: %q", string(pdfBytes))
		}
		return [][]byte{[]byte("page-1")}, nil
	})

	got, err := GeminiRequestToOpenAI(context.Background(), req, renderer)
	if err != nil {
		t.Fatalf("GeminiRequestToOpenAI returned error: %v", err)
	}

	if got.TopP == nil || *got.TopP != topP {
		t.Fatalf("top_p = %#v, want %v", got.TopP, topP)
	}
	if got.TopK == nil || *got.TopK != int(topK) {
		t.Fatalf("top_k = %#v, want %d", got.TopK, int(topK))
	}
	if got.N == nil || *got.N != candidateCount {
		t.Fatalf("n = %#v, want %d", got.N, candidateCount)
	}
	if got.Stop == nil {
		t.Fatalf("stop = nil, want stop sequences")
	}
	if got.Logprobs == nil || !*got.Logprobs {
		t.Fatalf("logprobs = %#v, want true", got.Logprobs)
	}
	if got.TopLogprobs == nil || *got.TopLogprobs != logprobs {
		t.Fatalf("top_logprobs = %#v, want %d", got.TopLogprobs, logprobs)
	}
	if got.PresencePenalty == nil || *got.PresencePenalty != presencePenalty {
		t.Fatalf("presence_penalty = %#v, want %v", got.PresencePenalty, presencePenalty)
	}
	if got.FrequencyPenalty == nil || *got.FrequencyPenalty != frequencyPenalty {
		t.Fatalf("frequency_penalty = %#v, want %v", got.FrequencyPenalty, frequencyPenalty)
	}
	if got.Seed == nil || *got.Seed != seed {
		t.Fatalf("seed = %#v, want %d", got.Seed, seed)
	}
	if len(got.Messages) != 3 {
		t.Fatalf("message count = %d, want 3", len(got.Messages))
	}

	userParts, ok := got.Messages[0].Content.([]openaip.ContentPart)
	if !ok || len(userParts) != 2 {
		t.Fatalf("user content = %#v, want text+image parts", got.Messages[0].Content)
	}
	if userParts[1].ImageURL == nil || userParts[1].ImageURL.URL == "" {
		t.Fatalf("translated fileData part = %#v", userParts[1])
	}

	if got.Messages[1].Role != "assistant" || len(got.Messages[1].ToolCalls) != 1 {
		t.Fatalf("assistant tool call message = %#v", got.Messages[1])
	}
	if got.Messages[1].ToolCalls[0].Function == nil || got.Messages[1].ToolCalls[0].Function.Name != "lookup_weather" {
		t.Fatalf("assistant tool call = %#v", got.Messages[1].ToolCalls[0])
	}

	if got.Messages[2].Role != "tool" || got.Messages[2].ToolCallID != "call_lookup" {
		t.Fatalf("tool message = %#v", got.Messages[2])
	}

	if len(got.Tools) != 1 || got.Tools[0].Function == nil {
		t.Fatalf("tools = %#v, want one function tool", got.Tools)
	}
	wantParameters := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{"type": "string"},
		},
		"required": []any{"city"},
	}
	if !reflect.DeepEqual(got.Tools[0].Function.Parameters, wantParameters) {
		t.Fatalf("tool parameters = %#v, want %#v", got.Tools[0].Function.Parameters, wantParameters)
	}

	toolChoice, ok := got.ToolChoice.(map[string]any)
	if !ok {
		t.Fatalf("tool_choice = %#v, want function selector", got.ToolChoice)
	}
	function, ok := toolChoice["function"].(map[string]any)
	if !ok || function["name"] != "lookup_weather" {
		t.Fatalf("tool_choice function = %#v", got.ToolChoice)
	}
}

func TestOpenAIResponseToGemini(t *testing.T) {
	resp := openaip.ChatCompletionResponse{
		Choices: []openaip.Choice{
			{
				Index: 0,
				Message: openaip.Message{
					Role: "assistant",
					Content: []any{
						map[string]any{"type": "text", "text": "chunk-1"},
						map[string]any{"type": "text", "text": " chunk-2"},
					},
				},
				FinishReason: "stop",
			},
		},
		Usage: &openaip.Usage{
			PromptTokens:     11,
			CompletionTokens: 7,
			TotalTokens:      18,
		},
	}

	got := OpenAIResponseToGemini(resp)
	if len(got.Candidates) != 1 {
		t.Fatalf("candidate count = %d, want 1", len(got.Candidates))
	}
	if got.Candidates[0].Content == nil || len(got.Candidates[0].Content.Parts) != 2 {
		t.Fatalf("candidate content = %#v", got.Candidates[0].Content)
	}
	if got.Candidates[0].Content.Role != "model" {
		t.Fatalf("candidate role = %q, want model", got.Candidates[0].Content.Role)
	}
	if got.Candidates[0].Content.Parts[0].Text != "chunk-1" || got.Candidates[0].Content.Parts[1].Text != " chunk-2" {
		t.Fatalf("candidate parts = %#v", got.Candidates[0].Content.Parts)
	}
	if got.Candidates[0].FinishReason != "STOP" {
		t.Fatalf("finish reason = %q, want STOP", got.Candidates[0].FinishReason)
	}
	if got.UsageMetadata == nil || got.UsageMetadata.TotalTokenCount != 18 {
		t.Fatalf("usage metadata = %#v", got.UsageMetadata)
	}
}

func TestOpenAIResponseToGeminiWithToolCalls(t *testing.T) {
	resp := openaip.ChatCompletionResponse{
		Choices: []openaip.Choice{{
			Index: 0,
			Message: openaip.Message{
				Role:    "assistant",
				Content: "calling a function",
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
	}

	got := OpenAIResponseToGemini(resp)
	if got.Candidates[0].FinishReason != "STOP" {
		t.Fatalf("finish reason = %q, want STOP", got.Candidates[0].FinishReason)
	}
	if len(got.Candidates[0].Content.Parts) != 2 {
		t.Fatalf("parts = %#v, want text + functionCall", got.Candidates[0].Content.Parts)
	}
	functionCall := got.Candidates[0].Content.Parts[1].FunctionCall
	if functionCall == nil || functionCall.Name != "lookup_weather" || functionCall.ID != "call_123" {
		t.Fatalf("function call part = %#v", got.Candidates[0].Content.Parts[1])
	}
	if functionCall.Args["city"] != "Boston" {
		t.Fatalf("function call args = %#v", functionCall.Args)
	}
}

func TestOpenAIStreamChunkToGemini(t *testing.T) {
	chunk := openaip.StreamChunk{
		Choices: []openaip.StreamChoice{
			{
				Index:        0,
				Delta:        openaip.Delta{Content: "partial"},
				FinishReason: "",
			},
			{
				Index:        1,
				Delta:        openaip.Delta{},
				FinishReason: "length",
			},
		},
	}

	got := OpenAIStreamChunkToGemini(chunk)
	if len(got.Candidates) != 2 {
		t.Fatalf("candidate count = %d, want 2", len(got.Candidates))
	}
	if got.Candidates[0].Content == nil || got.Candidates[0].Content.Parts[0].Text != "partial" {
		t.Fatalf("first candidate = %#v", got.Candidates[0])
	}
	if got.Candidates[1].FinishReason != "MAX_TOKENS" {
		t.Fatalf("second finish reason = %q, want MAX_TOKENS", got.Candidates[1].FinishReason)
	}
}

func TestOpenAIStreamChunkToGeminiWithToolCalls(t *testing.T) {
	chunk := openaip.StreamChunk{
		Choices: []openaip.StreamChoice{{
			Index: 0,
			Delta: openaip.Delta{
				ToolCalls: []openaip.ToolCallDelta{{
					Index: 0,
					ID:    "call_123",
					Type:  "function",
					Function: &openaip.ToolCallFunction{
						Name:      "lookup_weather",
						Arguments: `{"city":"Boston"}`,
					},
				}},
			},
			FinishReason: "tool_calls",
		}},
	}

	got := OpenAIStreamChunkToGemini(chunk)
	if len(got.Candidates) != 1 {
		t.Fatalf("candidate count = %d, want 1", len(got.Candidates))
	}
	if got.Candidates[0].FinishReason != "STOP" {
		t.Fatalf("finish reason = %q, want STOP", got.Candidates[0].FinishReason)
	}
	if got.Candidates[0].Content == nil || len(got.Candidates[0].Content.Parts) != 1 {
		t.Fatalf("candidate content = %#v", got.Candidates[0].Content)
	}
	functionCall := got.Candidates[0].Content.Parts[0].FunctionCall
	if functionCall == nil || functionCall.Name != "lookup_weather" || functionCall.ID != "call_123" {
		t.Fatalf("function call delta = %#v", got.Candidates[0].Content.Parts[0])
	}
}
