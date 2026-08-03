package translate

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/harshaneel/localaik/internal/pdf"
	"github.com/harshaneel/localaik/internal/protocol/anthropic"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

// AnthropicRequestToOpenAI converts an Anthropic Messages request into the
// OpenAI chat completion request llama.cpp serves.
//
// max_tokens is required by the Messages API, so a missing value is rejected
// here rather than silently defaulted — matching what a client would get from
// the real endpoint.
func AnthropicRequestToOpenAI(ctx context.Context, req anthropic.MessagesRequest, renderer pdf.Renderer) (openaip.ChatCompletionRequest, error) {
	if req.MaxTokens == nil {
		return openaip.ChatCompletionRequest{}, fmt.Errorf("max_tokens: field required")
	}

	result := openaip.ChatCompletionRequest{
		Model:    DefaultOpenAIModel,
		Messages: make([]openaip.Message, 0, len(req.Messages)+1),
	}

	if req.System != nil {
		content, err := anthropicBlocksToOpenAIContent(ctx, req.System.Blocks, renderer)
		if err != nil {
			return openaip.ChatCompletionRequest{}, fmt.Errorf("translate system: %w", err)
		}
		if !isEmptyOpenAIContent(content) {
			result.Messages = append(result.Messages, openaip.Message{
				Role:    "system",
				Content: content,
			})
		}
	}

	for _, message := range req.Messages {
		translated, err := anthropicMessageToOpenAIMessages(ctx, message, renderer)
		if err != nil {
			return openaip.ChatCompletionRequest{}, err
		}
		result.Messages = append(result.Messages, translated...)
	}

	if len(result.Messages) == 0 {
		return openaip.ChatCompletionRequest{}, fmt.Errorf("messages: at least one message is required")
	}

	result.MaxTokens = req.MaxTokens
	result.Temperature = req.Temperature
	result.TopP = req.TopP
	result.TopK = req.TopK

	if len(req.StopSequences) == 1 {
		result.Stop = req.StopSequences[0]
	} else if len(req.StopSequences) > 1 {
		result.Stop = req.StopSequences
	}

	result.Tools = anthropicToolsToOpenAITools(req.Tools)
	// A tool_choice with no tools to choose from is rejected by the upstream
	// runtime, so it is dropped rather than left pointing at a tool that was
	// skipped above.
	if len(result.Tools) > 0 {
		result.ToolChoice = anthropicToolChoiceToOpenAI(req.ToolChoice)
	}

	return result, nil
}

// OpenAIResponseToAnthropic converts an upstream chat completion into a Messages
// API reply. model is the model the client asked for, echoed back the way the
// real API does.
//
// The Messages API returns a single message, so only the first choice is used;
// any additional choices are dropped.
func OpenAIResponseToAnthropic(resp openaip.ChatCompletionResponse, model string) anthropic.MessagesResponse {
	out := anthropic.MessagesResponse{
		ID:      anthropicMessageID(resp.ID),
		Type:    "message",
		Role:    "assistant",
		Model:   anthropicResponseModel(model, resp.Model),
		Content: []anthropic.ContentBlock{},
	}

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		if text := extractTextFromOpenAIContent(choice.Message.Content); text != "" {
			out.Content = append(out.Content, anthropic.ContentBlock{
				Type: anthropic.BlockTypeText,
				Text: text,
			})
		}
		out.Content = append(out.Content, openAIToolCallsToAnthropicBlocks(choice.Message.ToolCalls)...)

		if reason := OpenAIFinishReasonToAnthropic(choice.FinishReason); reason != "" {
			out.StopReason = &reason
		}
	}

	if resp.Usage != nil {
		out.Usage = anthropic.Usage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
		}
	}

	return out
}

// OpenAIErrorToAnthropic converts an upstream error body into the Anthropic
// error envelope.
func OpenAIErrorToAnthropic(statusCode int, body []byte) anthropic.ErrorResponse {
	message := http.StatusText(statusCode)
	if message == "" {
		message = "upstream error"
	}

	var upstream openaip.ErrorResponse
	if err := json.Unmarshal(body, &upstream); err == nil && upstream.Error.Message != "" {
		message = upstream.Error.Message
	}

	return anthropic.ErrorResponse{
		Type: anthropic.EventError,
		Error: anthropic.Error{
			Type:    anthropic.ErrorTypeForHTTP(statusCode),
			Message: message,
		},
	}
}

// CountTokensTextFromAnthropic flattens a Messages request into a single text
// payload for llama.cpp's /tokenize endpoint.
//
// The walk covers whatever /v1/messages turns into prompt text, so the two paths
// agree on what a request contains: text blocks, text-source documents, and
// tool_result bodies including the placeholders non-text results become. What
// the text tokenizer genuinely cannot measure is skipped (images, base64
// documents, tool call inputs, tool definitions), so counts for those requests
// come out lower than the real API reports.
func CountTokensTextFromAnthropic(req anthropic.MessagesRequest) string {
	var b strings.Builder

	appendText := func(text string) {
		if text == "" {
			return
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(text)
	}

	appendBlocks := func(blocks []anthropic.ContentBlock) {
		for _, block := range blocks {
			switch block.Type {
			case anthropic.BlockTypeText:
				appendText(block.Text)
			case anthropic.BlockTypeToolResult:
				if block.Content != nil {
					appendText(anthropicToolResultText(*block.Content))
				}
			case anthropic.BlockTypeDocument:
				if block.Source != nil && block.Source.Type == anthropic.SourceTypeText {
					appendText(block.Source.Data)
				}
			}
		}
	}

	// system takes the same walk as a message body: it accepts the same block
	// shapes, and /v1/messages inlines them the same way.
	if req.System != nil {
		appendBlocks(req.System.Blocks)
	}

	for _, message := range req.Messages {
		appendBlocks(message.Content.Blocks)
	}

	return b.String()
}

// OpenAIFinishReasonToAnthropic maps an OpenAI finish_reason to an Anthropic
// stop_reason.
//
// llama.cpp reports "stop" both for a natural end of turn and for a
// stop_sequences hit, so "stop_sequence" is never produced — such requests come
// back as "end_turn".
func OpenAIFinishReasonToAnthropic(reason string) string {
	switch strings.ToLower(reason) {
	case "":
		return ""
	case "length":
		return anthropic.StopReasonMaxTokens
	case "tool_calls", "function_call":
		return anthropic.StopReasonToolUse
	default:
		return anthropic.StopReasonEndTurn
	}
}

func anthropicMessageToOpenAIMessages(ctx context.Context, message anthropic.Message, renderer pdf.Renderer) ([]openaip.Message, error) {
	role := anthropicRoleToOpenAI(message.Role)

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

	for _, block := range message.Content.Blocks {
		switch block.Type {
		case anthropic.BlockTypeToolResult:
			toolMessages = append(toolMessages, anthropicToolResultToOpenAIMessage(block))
		case anthropic.BlockTypeToolUse:
			if role == "assistant" {
				assistantToolCalls = append(assistantToolCalls, anthropicToolUseToOpenAIToolCall(block))
				continue
			}
			appendContentParts(textContentParts(formatAnthropicToolUseAsText(block)))
		case anthropic.BlockTypeThinking, anthropic.BlockTypeRedactedThinking:
			// Thinking blocks are the model's own scratch space, replayed back by
			// clients that keep full history. Feeding them to Gemma as prompt text
			// would pollute the context, so they are dropped.
			continue
		default:
			parts, err := anthropicBlockToOpenAIContentParts(ctx, block, renderer)
			if err != nil {
				return nil, err
			}
			appendContentParts(parts)
		}
	}

	return assembleOpenAIMessages(role, contentParts, contentTextOnly, assistantToolCalls, toolMessages), nil
}

func anthropicBlocksToOpenAIContent(ctx context.Context, blocks []anthropic.ContentBlock, renderer pdf.Renderer) (any, error) {
	contentParts := make([]openaip.ContentPart, 0, len(blocks))
	textOnly := true

	for _, block := range blocks {
		parts, err := anthropicBlockToOpenAIContentParts(ctx, block, renderer)
		if err != nil {
			return nil, err
		}
		contentParts = append(contentParts, parts...)
		if !contentPartSliceIsTextOnly(parts) {
			textOnly = false
		}
	}

	if len(contentParts) == 0 {
		return "", nil
	}

	return openAIContentFromParts(contentParts, textOnly), nil
}

func anthropicBlockToOpenAIContentParts(ctx context.Context, block anthropic.ContentBlock, renderer pdf.Renderer) ([]openaip.ContentPart, error) {
	switch block.Type {
	case anthropic.BlockTypeText, "":
		return textContentParts(block.Text), nil
	case anthropic.BlockTypeImage, anthropic.BlockTypeDocument:
		return anthropicSourceToOpenAIContentParts(ctx, block, renderer)
	default:
		return nil, nil
	}
}

func anthropicSourceToOpenAIContentParts(ctx context.Context, block anthropic.ContentBlock, renderer pdf.Renderer) ([]openaip.ContentPart, error) {
	if block.Source == nil {
		return nil, nil
	}
	source := *block.Source
	mediaType := strings.ToLower(source.MediaType)

	switch source.Type {
	case anthropic.SourceTypeBase64, "":
		data, err := base64.StdEncoding.DecodeString(source.Data)
		if err != nil {
			return nil, fmt.Errorf("decode %s source: %w", block.Type, err)
		}
		return bytesToOpenAIContentParts(ctx, mediaType, data, renderer)
	case anthropic.SourceTypeText:
		return textContentParts(source.Data), nil
	case anthropic.SourceTypeURL:
		if block.Type == anthropic.BlockTypeImage {
			return []openaip.ContentPart{{
				Type:     "image_url",
				ImageURL: &openaip.ImageURL{URL: source.URL},
			}}, nil
		}
		return textContentParts(fmt.Sprintf("Attached document: %s", source.URL)), nil
	default:
		return nil, fmt.Errorf("unsupported %s source type: %s", block.Type, source.Type)
	}
}

func anthropicToolResultToOpenAIMessage(block anthropic.ContentBlock) openaip.Message {
	text := ""
	if block.Content != nil {
		text = anthropicToolResultText(*block.Content)
	}
	if block.IsError {
		text = strings.TrimSpace("Error: " + text)
	}

	return openaip.Message{
		Role: "tool",
		// A tool_result carries only tool_use_id, with no name to fall back on, so
		// an absent id can only become a placeholder. That will not match the
		// name-derived id the tool_use side synthesizes: a request omitting ids on
		// both sides is malformed Anthropic (both fields are required) and cannot
		// be re-paired here. The placeholder exists only so the message does not go
		// upstream with an empty tool_call_id.
		ToolCallID: toolCallID(block.ToolUseID, ""),
		Content:    text,
	}
}

// anthropicToolResultText renders tool_result content as the plain string an
// OpenAI tool message carries. Non-text blocks cannot survive that shape, so
// they are summarised rather than dropped silently.
func anthropicToolResultText(content anthropic.Content) string {
	pieces := make([]string, 0, len(content.Blocks))
	for _, block := range content.Blocks {
		switch block.Type {
		case anthropic.BlockTypeText, "":
			if block.Text != "" {
				pieces = append(pieces, block.Text)
			}
		default:
			mediaType := ""
			if block.Source != nil {
				mediaType = block.Source.MediaType
			}
			if mediaType == "" {
				pieces = append(pieces, fmt.Sprintf("[%s]", block.Type))
				continue
			}
			pieces = append(pieces, fmt.Sprintf("[%s %s]", block.Type, mediaType))
		}
	}
	return strings.Join(pieces, "\n")
}

func anthropicToolUseToOpenAIToolCall(block anthropic.ContentBlock) openaip.ToolCall {
	return openaip.ToolCall{
		ID:   toolCallID(block.ID, block.Name),
		Type: "function",
		Function: &openaip.ToolCallFunction{
			Name:      block.Name,
			Arguments: string(anthropicToolInput(string(block.Input))),
		},
	}
}

func anthropicToolsToOpenAITools(tools []anthropic.Tool) []openaip.Tool {
	var out []openaip.Tool
	for _, tool := range tools {
		if tool.Name == "" {
			continue
		}
		// Anthropic's built-in tools (web search, code execution, computer use,
		// text editor, bash) are declared by a versioned `type` and carry no
		// input_schema, because their shape is implied by that type rather than
		// described. There is nothing to hand llama.cpp, so they are skipped
		// instead of forwarded as a parameterless function the model might call.
		// Keying on the absent schema rather than on the type string also means a
		// future tool type that does ship a schema still gets forwarded.
		if tool.InputSchema == nil {
			continue
		}
		out = append(out, openaip.Tool{
			Type: "function",
			Function: &openaip.FunctionDefinition{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  normalizeJSONSchema(tool.InputSchema),
				Strict:      true,
			},
		})
	}
	return out
}

func anthropicToolChoiceToOpenAI(choice *anthropic.ToolChoice) any {
	if choice == nil {
		return nil
	}

	switch strings.ToLower(choice.Type) {
	case "", "auto":
		return "auto"
	case "none":
		return "none"
	case "any":
		return "required"
	case "tool":
		if choice.Name != "" {
			return map[string]any{
				"type": "function",
				"function": map[string]any{
					"name": choice.Name,
				},
			}
		}
		return "required"
	default:
		return nil
	}
}

func openAIToolCallsToAnthropicBlocks(calls []openaip.ToolCall) []anthropic.ContentBlock {
	blocks := make([]anthropic.ContentBlock, 0, len(calls))
	ids := newToolUseIDs()
	for _, call := range calls {
		if call.Type != "" && call.Type != "function" {
			continue
		}
		if call.Function == nil {
			continue
		}
		blocks = append(blocks, anthropic.ContentBlock{
			Type:  anthropic.BlockTypeToolUse,
			ID:    ids.assign(call.ID, call.Function.Name),
			Name:  call.Function.Name,
			Input: anthropicToolInput(call.Function.Arguments),
		})
	}
	return blocks
}

// toolUseIDs hands out tool_use ids that are unique within one message.
//
// An upstream that sends no tool-call ids leaves nothing to derive them from but
// the tool name, so two parallel calls to the same tool would otherwise share an
// id. The client's tool_result blocks would then carry duplicate tool_use_ids,
// which the real API rejects on the next turn, breaking an agent loop a turn
// after the mistake was made.
type toolUseIDs struct {
	used map[string]int
}

func newToolUseIDs() *toolUseIDs {
	return &toolUseIDs{used: make(map[string]int)}
}

func (t *toolUseIDs) assign(id, name string) string {
	candidate := anthropicToolUseID(id, name)

	seen := t.used[candidate]
	t.used[candidate] = seen + 1
	if seen == 0 {
		return candidate
	}
	return fmt.Sprintf("%s_%d", candidate, seen+1)
}

// anthropicToolInput normalises OpenAI's stringified arguments into the JSON
// object Anthropic's tool_use.input expects.
//
// tool_use.input is always an object, so anything that is empty, malformed, or
// valid-but-not-an-object (a bare "null", array, or string) becomes an empty
// object rather than being passed through. Clients decode input into a struct
// and would fail on the other shapes.
func anthropicToolInput(arguments string) json.RawMessage {
	if !isJSONObject(arguments) {
		return json.RawMessage(`{}`)
	}
	return json.RawMessage(strings.TrimSpace(arguments))
}

// isJSONObject reports whether value is a complete JSON object. Note that
// json.Valid on its own is not enough: it also accepts bare scalars and arrays.
func isJSONObject(value string) bool {
	trimmed := strings.TrimSpace(value)
	return strings.HasPrefix(trimmed, "{") && json.Valid([]byte(trimmed))
}

func anthropicRoleToOpenAI(role string) string {
	switch strings.ToLower(role) {
	case "assistant", "model":
		return "assistant"
	case "system":
		return "system"
	default:
		return "user"
	}
}

func anthropicMessageID(openAIID string) string {
	trimmed := strings.TrimSpace(strings.TrimPrefix(openAIID, "chatcmpl-"))
	if trimmed == "" {
		return "msg_localaik"
	}
	return "msg_" + trimmed
}

func anthropicToolUseID(id, name string) string {
	if id != "" {
		return id
	}
	if name == "" {
		return "toolu_localaik"
	}
	return "toolu_" + strings.ReplaceAll(strings.ToLower(name), " ", "_")
}

func anthropicResponseModel(requested, upstream string) string {
	if requested != "" {
		return requested
	}
	if upstream != "" {
		return upstream
	}
	return DefaultOpenAIModel
}

func formatAnthropicToolUseAsText(block anthropic.ContentBlock) string {
	return fmt.Sprintf("Tool use %s: %s", block.Name, string(anthropicToolInput(string(block.Input))))
}
