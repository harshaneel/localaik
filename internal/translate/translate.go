package translate

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"strings"

	"github.com/harshaneel/localaik/internal/pdf"
	"github.com/harshaneel/localaik/internal/protocol/gemini"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

const DefaultOpenAIModel = "localaik"

func GeminiRequestToOpenAI(ctx context.Context, req gemini.GenerateContentRequest, renderer pdf.Renderer) (openaip.ChatCompletionRequest, error) {
	result := openaip.ChatCompletionRequest{
		Model:    DefaultOpenAIModel,
		Messages: make([]openaip.Message, 0, len(req.Contents)+1),
	}

	if req.SystemInstruction != nil {
		content, err := geminiPartsToOpenAIContent(ctx, req.SystemInstruction.Parts, renderer)
		if err != nil {
			return openaip.ChatCompletionRequest{}, fmt.Errorf("translate systemInstruction: %w", err)
		}
		if !isEmptyOpenAIContent(content) {
			result.Messages = append(result.Messages, openaip.Message{
				Role:    "system",
				Content: content,
			})
		}
	}

	for _, content := range req.Contents {
		translated, err := geminiContentToOpenAIMessages(ctx, content, renderer)
		if err != nil {
			return openaip.ChatCompletionRequest{}, err
		}
		result.Messages = append(result.Messages, translated...)
	}

	if len(result.Messages) == 0 {
		return openaip.ChatCompletionRequest{}, fmt.Errorf("Gemini request must include at least one message")
	}

	applyGenerationConfig(req.GenerationConfig, &result)

	result.Tools = geminiToolsToOpenAITools(req.Tools)
	result.ToolChoice = geminiToolChoiceToOpenAI(req.ToolConfig)

	return result, nil
}

func applyGenerationConfig(cfg *gemini.GenerationConfig, result *openaip.ChatCompletionRequest) {
	if cfg == nil {
		return
	}
	result.TopP = cfg.TopP
	if cfg.TopK != nil {
		topK := int(*cfg.TopK)
		result.TopK = &topK
	}
	result.N = cfg.CandidateCount
	result.Temperature = cfg.Temperature
	result.MaxTokens = cfg.MaxOutputTokens
	result.PresencePenalty = cfg.PresencePenalty
	result.FrequencyPenalty = cfg.FrequencyPenalty
	result.Seed = cfg.Seed
	if len(cfg.StopSequences) > 0 {
		if len(cfg.StopSequences) == 1 {
			result.Stop = cfg.StopSequences[0]
		} else {
			result.Stop = cfg.StopSequences
		}
	}
	if cfg.ResponseLogprobs != nil && *cfg.ResponseLogprobs {
		value := true
		result.Logprobs = &value
	}
	if cfg.Logprobs != nil {
		value := true
		result.Logprobs = &value
		result.TopLogprobs = cfg.Logprobs
	}

	schema := cfg.ResponseSchema
	if schema == nil {
		schema = cfg.ResponseJSONSchema
	}

	if schema != nil {
		result.ResponseFormat = &openaip.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &openaip.JSONSchemaConfig{
				Name:   "response",
				Schema: normalizeJSONSchema(schema),
				Strict: true,
			},
		}
	} else if strings.EqualFold(cfg.ResponseMimeType, "application/json") {
		result.ResponseFormat = &openaip.ResponseFormat{
			Type: "json_object",
		}
	}
}

func OpenAIResponseToGemini(resp openaip.ChatCompletionResponse) gemini.GenerateContentResponse {
	out := gemini.GenerateContentResponse{
		Candidates: make([]gemini.Candidate, 0, len(resp.Choices)),
	}

	for _, choice := range resp.Choices {
		parts := openAIContentToGeminiParts(choice.Message.Content)
		parts = append(parts, openAIToolCallsToGeminiParts(choice.Message.ToolCalls)...)
		out.Candidates = append(out.Candidates, gemini.Candidate{
			Index:        choice.Index,
			FinishReason: openAIFinishReasonToGemini(choice.FinishReason),
			Content: &gemini.Content{
				Role:  "model",
				Parts: parts,
			},
		})
	}

	if resp.Usage != nil {
		out.UsageMetadata = &gemini.UsageMetadata{
			PromptTokenCount:     resp.Usage.PromptTokens,
			CandidatesTokenCount: resp.Usage.CompletionTokens,
			TotalTokenCount:      resp.Usage.TotalTokens,
		}
	}

	return out
}

func OpenAIStreamChunkToGemini(chunk openaip.StreamChunk) gemini.GenerateContentResponse {
	out := gemini.GenerateContentResponse{
		Candidates: make([]gemini.Candidate, 0, len(chunk.Choices)),
	}

	for _, choice := range chunk.Choices {
		parts := openAIContentToGeminiParts(choice.Delta.Content)
		parts = append(parts, openAIToolCallDeltasToGeminiParts(choice.Delta.ToolCalls)...)
		candidate := gemini.Candidate{
			Index:        choice.Index,
			FinishReason: openAIFinishReasonToGemini(choice.FinishReason),
		}
		if len(parts) > 0 {
			candidate.Content = &gemini.Content{
				Role:  "model",
				Parts: parts,
			}
		}
		if candidate.Content != nil || candidate.FinishReason != "" {
			out.Candidates = append(out.Candidates, candidate)
		}
	}

	if chunk.Usage != nil {
		out.UsageMetadata = &gemini.UsageMetadata{
			PromptTokenCount:     chunk.Usage.PromptTokens,
			CandidatesTokenCount: chunk.Usage.CompletionTokens,
			TotalTokenCount:      chunk.Usage.TotalTokens,
		}
	}

	return out
}

func geminiContentToOpenAIMessages(ctx context.Context, content gemini.Content, renderer pdf.Renderer) ([]openaip.Message, error) {
	role := geminiRoleToOpenAI(content.Role)

	var contentParts []openaip.ContentPart
	contentTextOnly := true
	var assistantToolCalls []openaip.ToolCall
	var toolMessages []openaip.Message

	appendContentParts := func(parts []openaip.ContentPart) {
		contentParts = append(contentParts, parts...)
		if !contentPartSliceIsTextOnly(parts) {
			contentTextOnly = false
		}
	}

	for _, part := range content.Parts {
		switch {
		case part.FunctionResponse != nil:
			toolMessages = append(toolMessages, geminiFunctionResponseToOpenAIMessages(*part.FunctionResponse)...)
		case part.FunctionCall != nil:
			if role == "assistant" {
				assistantToolCalls = append(assistantToolCalls, geminiFunctionCallToOpenAIToolCall(*part.FunctionCall))
				continue
			}
			appendContentParts(textContentParts(formatFunctionCallAsText(*part.FunctionCall)))
		case part.ToolCall != nil:
			appendContentParts(textContentParts(formatToolCallAsText(*part.ToolCall)))
		case part.ToolResponse != nil:
			appendContentParts(textContentParts(formatToolResponseAsText(*part.ToolResponse)))
		case part.ExecutableCode != nil:
			appendContentParts(textContentParts(formatExecutableCodeAsText(*part.ExecutableCode)))
		case part.CodeExecutionResult != nil:
			appendContentParts(textContentParts(formatCodeExecutionResultAsText(*part.CodeExecutionResult)))
		default:
			parts, err := geminiPartToOpenAIContentParts(ctx, part, renderer)
			if err != nil {
				return nil, err
			}
			appendContentParts(parts)
		}
	}

	return assembleOpenAIMessages(geminiRoleToOpenAI(content.Role), contentParts, contentTextOnly, assistantToolCalls, toolMessages), nil
}

func assembleOpenAIMessages(role string, contentParts []openaip.ContentPart, contentTextOnly bool, assistantToolCalls []openaip.ToolCall, toolMessages []openaip.Message) []openaip.Message {
	messages := make([]openaip.Message, 0, 1+len(toolMessages))
	if len(contentParts) > 0 || len(assistantToolCalls) > 0 {
		message := openaip.Message{Role: role}
		if len(contentParts) > 0 {
			message.Content = openAIContentFromParts(contentParts, contentTextOnly)
		} else if len(assistantToolCalls) > 0 {
			message.Content = ""
		}
		if len(assistantToolCalls) > 0 {
			message.ToolCalls = assistantToolCalls
		}
		messages = append(messages, message)
	}
	messages = append(messages, toolMessages...)
	return messages
}

func geminiPartsToOpenAIContent(ctx context.Context, parts []gemini.Part, renderer pdf.Renderer) (any, error) {
	contentParts := make([]openaip.ContentPart, 0, len(parts))
	textOnly := true

	for _, part := range parts {
		translatedParts, err := geminiPartToOpenAIContentParts(ctx, part, renderer)
		if err != nil {
			return nil, err
		}
		contentParts = append(contentParts, translatedParts...)
		if !contentPartSliceIsTextOnly(translatedParts) {
			textOnly = false
		}
	}

	if len(contentParts) == 0 {
		return "", nil
	}

	return openAIContentFromParts(contentParts, textOnly), nil
}

func geminiRoleToOpenAI(role string) string {
	switch strings.ToLower(role) {
	case "model", "assistant":
		return "assistant"
	case "system":
		return "system"
	default:
		return "user"
	}
}

func openAIFinishReasonToGemini(reason string) string {
	switch strings.ToLower(reason) {
	case "":
		return ""
	case "stop":
		return "STOP"
	case "length":
		return "MAX_TOKENS"
	case "content_filter":
		return "SAFETY"
	case "tool_calls", "function_call":
		return "STOP"
	default:
		return "OTHER"
	}
}

func geminiPartToOpenAIContentParts(ctx context.Context, part gemini.Part, renderer pdf.Renderer) ([]openaip.ContentPart, error) {
	switch {
	case part.Text != "":
		return textContentParts(part.Text), nil
	case part.InlineData != nil:
		return blobToOpenAIContentParts(ctx, part.InlineData.MimeType, part.InlineData.Data, renderer)
	case part.FileData != nil:
		return fileDataToOpenAIContentParts(ctx, *part.FileData, renderer)
	default:
		return nil, nil
	}
}

func blobToOpenAIContentParts(ctx context.Context, mimeType, encodedData string, renderer pdf.Renderer) ([]openaip.ContentPart, error) {
	data, err := base64.StdEncoding.DecodeString(encodedData)
	if err != nil {
		return nil, fmt.Errorf("decode inlineData: %w", err)
	}
	return bytesToOpenAIContentParts(ctx, mimeType, data, renderer)
}

func fileDataToOpenAIContentParts(ctx context.Context, fileData gemini.FileData, renderer pdf.Renderer) ([]openaip.ContentPart, error) {
	mimeType := strings.ToLower(fileData.MimeType)
	switch {
	case strings.HasPrefix(mimeType, "image/"):
		if data, ok, err := maybeReadFileURI(fileData.FileURI); err != nil {
			return nil, fmt.Errorf("load fileData image: %w", err)
		} else if ok {
			return []openaip.ContentPart{{
				Type: "image_url",
				ImageURL: &openaip.ImageURL{
					URL: dataURI(mimeType, data),
				},
			}}, nil
		}
		return []openaip.ContentPart{{
			Type: "image_url",
			ImageURL: &openaip.ImageURL{
				URL: fileData.FileURI,
			},
		}}, nil
	case mimeType == "application/pdf":
		data, err := readFileURI(fileData.FileURI)
		if err != nil {
			return nil, fmt.Errorf("load fileData PDF: %w", err)
		}
		return bytesToOpenAIContentParts(ctx, mimeType, data, renderer)
	case strings.HasPrefix(mimeType, "text/") || mimeType == "application/json":
		data, err := readFileURI(fileData.FileURI)
		if err != nil {
			return nil, fmt.Errorf("load fileData text: %w", err)
		}
		return textContentParts(string(data)), nil
	default:
		return textContentParts(fmt.Sprintf("Attached file: %s (%s)", fileData.FileURI, fileData.MimeType)), nil
	}
}

func bytesToOpenAIContentParts(ctx context.Context, mimeType string, data []byte, renderer pdf.Renderer) ([]openaip.ContentPart, error) {
	mimeType = strings.ToLower(mimeType)
	switch {
	case mimeType == "application/pdf":
		renderedPages, err := renderer.RenderPDF(ctx, data)
		if err != nil {
			return nil, fmt.Errorf("render PDF: %w", err)
		}
		parts := make([]openaip.ContentPart, 0, len(renderedPages))
		for _, page := range renderedPages {
			parts = append(parts, openaip.ContentPart{
				Type: "image_url",
				ImageURL: &openaip.ImageURL{
					URL: dataURI("image/png", page),
				},
			})
		}
		return parts, nil
	case strings.HasPrefix(mimeType, "text/") || mimeType == "application/json":
		return textContentParts(string(data)), nil
	case strings.HasPrefix(mimeType, "image/"):
		return []openaip.ContentPart{{
			Type: "image_url",
			ImageURL: &openaip.ImageURL{
				URL: dataURI(mimeType, data),
			},
		}}, nil
	default:
		return textContentParts(fmt.Sprintf("Embedded file (%s, %d bytes)", mimeType, len(data))), nil
	}
}

func geminiToolsToOpenAITools(tools []gemini.Tool) []openaip.Tool {
	var out []openaip.Tool
	for _, tool := range tools {
		for _, decl := range tool.FunctionDeclarations {
			parameters := decl.ParametersJSONSchema
			if parameters == nil {
				parameters = decl.Parameters
			}
			out = append(out, openaip.Tool{
				Type: "function",
				Function: &openaip.FunctionDefinition{
					Name:        decl.Name,
					Description: decl.Description,
					Parameters:  normalizeJSONSchema(parameters),
					Strict:      parameters != nil,
				},
			})
		}
	}
	return out
}

func geminiToolChoiceToOpenAI(cfg *gemini.ToolConfig) any {
	if cfg == nil || cfg.FunctionCallingConfig == nil {
		return nil
	}

	switch strings.ToUpper(cfg.FunctionCallingConfig.Mode) {
	case "", "MODE_UNSPECIFIED", "AUTO":
		return "auto"
	case "NONE":
		return "none"
	case "ANY", "VALIDATED":
		if len(cfg.FunctionCallingConfig.AllowedFunctionNames) == 1 {
			return map[string]any{
				"type": "function",
				"function": map[string]any{
					"name": cfg.FunctionCallingConfig.AllowedFunctionNames[0],
				},
			}
		}
		return "required"
	default:
		return nil
	}
}

func geminiFunctionCallToOpenAIToolCall(call gemini.FunctionCall) openaip.ToolCall {
	return openaip.ToolCall{
		ID:   toolCallID(call.ID, call.Name),
		Type: "function",
		Function: &openaip.ToolCallFunction{
			Name:      call.Name,
			Arguments: marshalJSONText(call.Args),
		},
	}
}

func geminiFunctionResponseToOpenAIMessages(response gemini.FunctionResponse) []openaip.Message {
	payload := map[string]any{
		"response": response.Response,
	}
	if len(response.Parts) > 0 {
		payload["parts"] = simplifyFunctionResponseParts(response.Parts)
	}

	return []openaip.Message{{
		Role:       "tool",
		Name:       response.Name,
		ToolCallID: toolCallID(response.ID, response.Name),
		Content:    marshalJSONText(payload),
	}}
}

func simplifyFunctionResponseParts(parts []gemini.FunctionResponsePart) []map[string]any {
	out := make([]map[string]any, 0, len(parts))
	for _, part := range parts {
		switch {
		case part.InlineData != nil:
			out = append(out, map[string]any{
				"inlineData": map[string]any{
					"mimeType": part.InlineData.MimeType,
				},
			})
		case part.FileData != nil:
			out = append(out, map[string]any{
				"fileData": map[string]any{
					"fileUri":  part.FileData.FileURI,
					"mimeType": part.FileData.MimeType,
				},
			})
		}
	}
	return out
}

func collectTextFromAnySlice(items []any) []string {
	texts := make([]string, 0, len(items))
	for _, item := range items {
		switch piece := item.(type) {
		case map[string]any:
			if text, ok := piece["text"].(string); ok {
				texts = append(texts, text)
			}
		case openaip.ContentPart:
			if piece.Type == "text" && piece.Text != "" {
				texts = append(texts, piece.Text)
			}
		}
	}
	return texts
}

func textFromMap(m map[string]any) (string, bool) {
	if text, ok := m["text"].(string); ok {
		return text, true
	}
	if content, ok := m["content"].(string); ok {
		return content, true
	}
	return "", false
}

func openAIContentToGeminiParts(content any) []gemini.Part {
	switch value := content.(type) {
	case nil:
		return nil
	case string:
		return textToGeminiParts(value)
	case []openaip.ContentPart:
		parts := make([]gemini.Part, 0, len(value))
		for _, part := range value {
			if part.Type == "text" && part.Text != "" {
				parts = append(parts, gemini.Part{Text: part.Text})
			}
		}
		return parts
	case []any:
		texts := collectTextFromAnySlice(value)
		parts := make([]gemini.Part, 0, len(texts))
		for _, text := range texts {
			parts = append(parts, gemini.Part{Text: text})
		}
		return parts
	case map[string]any:
		if text, ok := textFromMap(value); ok {
			return textToGeminiParts(text)
		}
	}
	return textToGeminiParts(extractTextFromOpenAIContent(content))
}

func openAIToolCallsToGeminiParts(calls []openaip.ToolCall) []gemini.Part {
	parts := make([]gemini.Part, 0, len(calls))
	for _, call := range calls {
		if call.Type != "" && call.Type != "function" {
			continue
		}
		function := call.Function
		if function == nil {
			continue
		}
		parts = append(parts, gemini.Part{
			FunctionCall: &gemini.FunctionCall{
				ID:   call.ID,
				Name: function.Name,
				Args: parseJSONObject(function.Arguments),
			},
		})
	}
	return parts
}

func openAIToolCallDeltasToGeminiParts(calls []openaip.ToolCallDelta) []gemini.Part {
	parts := make([]gemini.Part, 0, len(calls))
	for _, call := range calls {
		if call.Type != "" && call.Type != "function" {
			continue
		}
		if call.Function == nil && call.ID == "" {
			continue
		}
		functionCall := &gemini.FunctionCall{
			ID: call.ID,
		}
		if call.Function != nil {
			functionCall.Name = call.Function.Name
			functionCall.Args = parseJSONObject(call.Function.Arguments)
		}
		parts = append(parts, gemini.Part{FunctionCall: functionCall})
	}
	return parts
}

func normalizeJSONSchema(value any) any {
	return normalizeJSONSchemaValue("", value)
}

func normalizeJSONSchemaValue(key string, value any) any {
	switch typed := value.(type) {
	case map[string]any:
		out := make(map[string]any, len(typed))
		for nestedKey, nestedValue := range typed {
			out[nestedKey] = normalizeJSONSchemaValue(nestedKey, nestedValue)
		}
		return out
	case []any:
		out := make([]any, len(typed))
		for i, item := range typed {
			out[i] = normalizeJSONSchemaValue(key, item)
		}
		return out
	case string:
		if key == "type" {
			if normalized, ok := normalizeGeminiSchemaType(typed); ok {
				return normalized
			}
		}
		return typed
	default:
		return value
	}
}

func normalizeGeminiSchemaType(value string) (string, bool) {
	switch strings.ToUpper(value) {
	case "STRING":
		return "string", true
	case "NUMBER":
		return "number", true
	case "INTEGER":
		return "integer", true
	case "BOOLEAN":
		return "boolean", true
	case "ARRAY":
		return "array", true
	case "OBJECT":
		return "object", true
	case "NULL":
		return "null", true
	default:
		return "", false
	}
}

func extractTextFromOpenAIContent(content any) string {
	switch value := content.(type) {
	case nil:
		return ""
	case string:
		return value
	case []openaip.ContentPart:
		parts := make([]string, 0, len(value))
		for _, part := range value {
			if part.Type == "text" && part.Text != "" {
				parts = append(parts, part.Text)
			}
		}
		return strings.Join(parts, "")
	case []any:
		return strings.Join(collectTextFromAnySlice(value), "")
	case map[string]any:
		if text, ok := textFromMap(value); ok {
			return text
		}
	}

	marshaled, err := json.Marshal(content)
	if err != nil {
		return ""
	}
	return string(marshaled)
}

func textToGeminiParts(text string) []gemini.Part {
	if text == "" {
		return nil
	}
	return []gemini.Part{{Text: text}}
}

func dataURI(mimeType string, data []byte) string {
	if mimeType == "" {
		mimeType = "application/octet-stream"
	}
	return "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(data)
}

func openAIContentFromParts(parts []openaip.ContentPart, textOnly bool) any {
	if textOnly {
		if len(parts) == 1 {
			return parts[0].Text
		}

		texts := make([]string, 0, len(parts))
		for _, part := range parts {
			texts = append(texts, part.Text)
		}
		return strings.Join(texts, "\n\n")
	}

	return parts
}

func contentPartSliceIsTextOnly(parts []openaip.ContentPart) bool {
	for _, part := range parts {
		if part.Type != "text" {
			return false
		}
	}
	return true
}

func textContentParts(text string) []openaip.ContentPart {
	if text == "" {
		return nil
	}
	return []openaip.ContentPart{{
		Type: "text",
		Text: text,
	}}
}

func formatFunctionCallAsText(call gemini.FunctionCall) string {
	return fmt.Sprintf("Function call %s: %s", call.Name, marshalJSONText(call.Args))
}

func formatToolCallAsText(call gemini.ToolCall) string {
	return fmt.Sprintf("Tool call %s: %s", call.ToolType, marshalJSONText(call.Args))
}

func formatToolResponseAsText(response gemini.ToolResponse) string {
	return fmt.Sprintf("Tool response %s: %s", response.ToolType, marshalJSONText(response.Response))
}

func formatExecutableCodeAsText(code gemini.ExecutableCode) string {
	return fmt.Sprintf("Executable code (%s):\n%s", code.Language, code.Code)
}

func formatCodeExecutionResultAsText(result gemini.CodeExecutionResult) string {
	if result.Output == "" {
		return fmt.Sprintf("Code execution result: %s", result.Outcome)
	}
	return fmt.Sprintf("Code execution result (%s): %s", result.Outcome, result.Output)
}

func marshalJSONText(value any) string {
	if value == nil {
		return "{}"
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return "{}"
	}
	return string(encoded)
}

func parseJSONObject(value string) map[string]any {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(value), &out); err != nil {
		return nil
	}
	return out
}

func toolCallID(id, name string) string {
	if id != "" {
		return id
	}
	if name == "" {
		return "call"
	}
	return "call_" + strings.ReplaceAll(strings.ToLower(name), " ", "_")
}

func readFileURI(fileURI string) ([]byte, error) {
	data, ok, err := maybeReadFileURI(fileURI)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("unsupported file URI: %s", fileURI)
	}
	return data, nil
}

func maybeReadFileURI(fileURI string) ([]byte, bool, error) {
	switch {
	case strings.HasPrefix(fileURI, "data:"):
		data, err := decodeDataURI(fileURI)
		return data, true, err
	case strings.HasPrefix(fileURI, "file://"):
		parsed, err := url.Parse(fileURI)
		if err != nil {
			return nil, false, err
		}
		data, err := os.ReadFile(parsed.Path)
		return data, true, err
	case hasURIProtocol(fileURI):
		return nil, false, nil
	default:
		data, err := os.ReadFile(fileURI)
		return data, true, err
	}
}

func decodeDataURI(value string) ([]byte, error) {
	prefix, payload, found := strings.Cut(value, ",")
	if !found {
		return nil, fmt.Errorf("invalid data URI")
	}
	if strings.HasSuffix(prefix, ";base64") {
		return base64.StdEncoding.DecodeString(payload)
	}
	decoded, err := url.QueryUnescape(payload)
	if err != nil {
		return nil, err
	}
	return []byte(decoded), nil
}

func hasURIProtocol(value string) bool {
	if len(value) < 2 {
		return false
	}
	if strings.Contains(value[:2], "/") {
		return false
	}
	return strings.Contains(value, "://")
}

func isEmptyOpenAIContent(content any) bool {
	switch value := content.(type) {
	case nil:
		return true
	case string:
		return value == ""
	case []openaip.ContentPart:
		return len(value) == 0
	default:
		return false
	}
}
