// Package anthropic models the wire format of the Anthropic Messages API
// (https://docs.anthropic.com/en/api/messages) as served by localaik. Only the
// fields localaik reads or emits are represented; unknown fields decode away
// silently so newer SDK versions keep working.
package anthropic

import (
	"bytes"
	"encoding/json"
	"strings"
)

// MessagesRequest is the body of POST /v1/messages.
type MessagesRequest struct {
	Model         string      `json:"model,omitempty"`
	Messages      []Message   `json:"messages"`
	System        *Content    `json:"system,omitempty"`
	MaxTokens     *int        `json:"max_tokens,omitempty"`
	Temperature   *float64    `json:"temperature,omitempty"`
	TopP          *float64    `json:"top_p,omitempty"`
	TopK          *int        `json:"top_k,omitempty"`
	StopSequences []string    `json:"stop_sequences,omitempty"`
	Stream        bool        `json:"stream,omitempty"`
	Tools         []Tool      `json:"tools,omitempty"`
	ToolChoice    *ToolChoice `json:"tool_choice,omitempty"`
	Metadata      *Metadata   `json:"metadata,omitempty"`
	Thinking      *Thinking   `json:"thinking,omitempty"`
}

// Message is a single turn in the conversation.
type Message struct {
	Role    string  `json:"role"`
	Content Content `json:"content"`
}

// Content holds a message body or system prompt. The wire format allows either
// a bare string or an array of typed blocks; both decode into Blocks. IsString
// records which form arrived so a value re-marshals in the shape it came in.
type Content struct {
	Blocks   []ContentBlock
	IsString bool
}

// StringContent builds a Content that marshals back as a bare JSON string.
func StringContent(text string) Content {
	c := Content{IsString: true}
	if text != "" {
		c.Blocks = []ContentBlock{{Type: BlockTypeText, Text: text}}
	}
	return c
}

func (c *Content) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		*c = Content{}
		return nil
	}

	if trimmed[0] == '"' {
		var text string
		if err := json.Unmarshal(trimmed, &text); err != nil {
			return err
		}
		*c = StringContent(text)
		return nil
	}

	var blocks []ContentBlock
	if err := json.Unmarshal(trimmed, &blocks); err != nil {
		return err
	}
	*c = Content{Blocks: blocks}
	return nil
}

func (c Content) MarshalJSON() ([]byte, error) {
	if c.IsString {
		return json.Marshal(c.Text())
	}
	if c.Blocks == nil {
		return []byte("[]"), nil
	}
	return json.Marshal(c.Blocks)
}

// Text concatenates every text block, ignoring non-text blocks.
func (c Content) Text() string {
	var b strings.Builder
	for _, block := range c.Blocks {
		if block.Type == BlockTypeText && block.Text != "" {
			b.WriteString(block.Text)
		}
	}
	return b.String()
}

// Content block type discriminators.
const (
	BlockTypeText             = "text"
	BlockTypeImage            = "image"
	BlockTypeDocument         = "document"
	BlockTypeToolUse          = "tool_use"
	BlockTypeToolResult       = "tool_result"
	BlockTypeThinking         = "thinking"
	BlockTypeRedactedThinking = "redacted_thinking"
)

// ContentBlock is the union of every block shape the Messages API defines. The
// Type field selects which of the remaining fields are meaningful.
type ContentBlock struct {
	Type string `json:"type"`

	// text, and the human-readable half of thinking blocks.
	Text string `json:"text,omitempty"`

	// image and document blocks.
	Source *Source `json:"source,omitempty"`

	// tool_use blocks.
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// tool_result blocks.
	ToolUseID string   `json:"tool_use_id,omitempty"`
	Content   *Content `json:"content,omitempty"`
	IsError   bool     `json:"is_error,omitempty"`

	// thinking and redacted_thinking blocks.
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"`
}

// Source describes the payload of an image or document block.
type Source struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

// Source type discriminators.
const (
	SourceTypeBase64 = "base64"
	SourceTypeURL    = "url"
	SourceTypeText   = "text"
)

// Tool is a client-side tool definition.
type Tool struct {
	Type        string `json:"type,omitempty"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"input_schema,omitempty"`
}

// ToolChoice constrains how the model may call tools.
type ToolChoice struct {
	Type                   string `json:"type"`
	Name                   string `json:"name,omitempty"`
	DisableParallelToolUse *bool  `json:"disable_parallel_tool_use,omitempty"`
}

// Metadata carries opaque request metadata; localaik accepts and ignores it.
type Metadata struct {
	UserID string `json:"user_id,omitempty"`
}

// Thinking configures extended thinking; localaik accepts and ignores it.
type Thinking struct {
	Type         string `json:"type,omitempty"`
	BudgetTokens *int   `json:"budget_tokens,omitempty"`
}

// MessagesResponse is the body of a non-streaming POST /v1/messages reply.
type MessagesResponse struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Model        string         `json:"model,omitempty"`
	Content      []ContentBlock `json:"content"`
	StopReason   *string        `json:"stop_reason"`
	StopSequence *string        `json:"stop_sequence"`
	Usage        Usage          `json:"usage"`
}

// Usage reports token counts. Anthropic names these differently from OpenAI and
// omits a total.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// CountTokensResponse is the body of POST /v1/messages/count_tokens.
type CountTokensResponse struct {
	InputTokens int `json:"input_tokens"`
}

// Stop reason values.
const (
	StopReasonEndTurn      = "end_turn"
	StopReasonMaxTokens    = "max_tokens"
	StopReasonStopSequence = "stop_sequence"
	StopReasonToolUse      = "tool_use"
)

// Streaming event type discriminators.
const (
	EventMessageStart      = "message_start"
	EventMessageDelta      = "message_delta"
	EventMessageStop       = "message_stop"
	EventContentBlockStart = "content_block_start"
	EventContentBlockDelta = "content_block_delta"
	EventContentBlockStop  = "content_block_stop"
	EventError             = "error"
)

// The types below are the write side of the event stream. They are kept separate
// from StreamEvent because the Messages API is specific about which fields are
// present-but-null versus omitted, and that differs per event: a
// content_block_delta must not carry stop fields, while a message_delta must
// carry them even when null. StreamEvent stays the permissive decode-side union.

// MessageStartEvent opens the stream.
type MessageStartEvent struct {
	Type    string           `json:"type"`
	Message MessagesResponse `json:"message"`
}

// ContentBlockStartEvent opens one content block.
type ContentBlockStartEvent struct {
	Type         string             `json:"type"`
	Index        int                `json:"index"`
	ContentBlock StreamContentBlock `json:"content_block"`
}

// StreamContentBlock is the content_block payload of a content_block_start
// event. Text and Name are pointers so each block type emits exactly the fields
// the Messages API defines for it, including ones that are required but empty: a
// text block sends `"text": ""` and a tool_use block sends `"name"` even when
// upstream never supplied one. Strict clients reject a block missing a required
// field, so omitting them is not a safe simplification.
type StreamContentBlock struct {
	Type  string          `json:"type"`
	Text  *string         `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  *string         `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

// TextStreamBlock builds the content_block_start payload for a text block.
func TextStreamBlock() StreamContentBlock {
	empty := ""
	return StreamContentBlock{Type: BlockTypeText, Text: &empty}
}

// ToolUseStreamBlock builds the content_block_start payload for a tool_use
// block. input starts as an empty object; the first input_json_delta replaces it.
func ToolUseStreamBlock(id, name string) StreamContentBlock {
	return StreamContentBlock{
		Type:  BlockTypeToolUse,
		ID:    id,
		Name:  &name,
		Input: json.RawMessage(`{}`),
	}
}

// ContentBlockDeltaEvent carries an incremental update to one content block.
type ContentBlockDeltaEvent struct {
	Type  string     `json:"type"`
	Index int        `json:"index"`
	Delta BlockDelta `json:"delta"`
}

// BlockDelta is the payload of a content_block_delta. Exactly one of Text or
// PartialJSON is meaningful, selected by Type.
type BlockDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
}

// ContentBlockStopEvent closes one content block.
type ContentBlockStopEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
}

// MessageDeltaEvent carries the terminal stop metadata and output token count.
type MessageDeltaEvent struct {
	Type  string       `json:"type"`
	Delta MessageDelta `json:"delta"`
	Usage Usage        `json:"usage"`
}

// MessageDelta holds the terminal stop metadata. Both fields are always present,
// null when unset, matching the Messages API.
type MessageDelta struct {
	StopReason   *string `json:"stop_reason"`
	StopSequence *string `json:"stop_sequence"`
}

// MessageStopEvent ends the stream.
type MessageStopEvent struct {
	Type string `json:"type"`
}

// ErrorEvent reports a failure that happened after the stream headers were sent.
type ErrorEvent struct {
	Type  string `json:"type"`
	Error Error  `json:"error"`
}

// StreamEvent is the union of every server-sent event the Messages API emits.
// Which fields are populated depends on Type. This is the decode-side type; the
// writer emits the per-event types above.
type StreamEvent struct {
	Type string `json:"type"`

	// message_start.
	Message *MessagesResponse `json:"message,omitempty"`

	// content_block_start, content_block_delta, content_block_stop.
	Index        *int          `json:"index,omitempty"`
	ContentBlock *ContentBlock `json:"content_block,omitempty"`

	// content_block_delta and message_delta.
	Delta *Delta `json:"delta,omitempty"`

	// message_delta.
	Usage *Usage `json:"usage,omitempty"`

	// error.
	Error *Error `json:"error,omitempty"`
}

// Delta carries the incremental payload of a content_block_delta or the
// terminal stop metadata of a message_delta.
type Delta struct {
	Type         string  `json:"type,omitempty"`
	Text         string  `json:"text,omitempty"`
	PartialJSON  string  `json:"partial_json,omitempty"`
	StopReason   *string `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence,omitempty"`
}

// Delta type discriminators.
const (
	DeltaTypeText      = "text_delta"
	DeltaTypeInputJSON = "input_json_delta"
)

// ErrorResponse is the body of any non-2xx Messages API reply.
type ErrorResponse struct {
	Type  string `json:"type"`
	Error Error  `json:"error"`
}

// Error is the inner payload of an ErrorResponse.
type Error struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}
