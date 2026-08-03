package anthropic

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestContentUnmarshalJSON(t *testing.T) {
	cases := []struct {
		name       string
		json       string
		wantText   string
		wantBlocks int
		wantString bool
	}{
		{"bare_string", `"hello"`, "hello", 1, true},
		{"empty_string", `""`, "", 0, true},
		{"block_array", `[{"type":"text","text":"hello"}]`, "hello", 1, false},
		{"multiple_text_blocks", `[{"type":"text","text":"a"},{"type":"text","text":"b"}]`, "ab", 2, false},
		{"empty_array", `[]`, "", 0, false},
		{"null", `null`, "", 0, false},
		{"mixed_blocks_text_only", `[{"type":"text","text":"a"},{"type":"image","source":{"type":"base64"}}]`, "a", 2, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var content Content
			if err := json.Unmarshal([]byte(tc.json), &content); err != nil {
				t.Fatalf("decode %s: %v", tc.json, err)
			}
			if content.Text() != tc.wantText {
				t.Fatalf("Text() = %q, want %q", content.Text(), tc.wantText)
			}
			if len(content.Blocks) != tc.wantBlocks {
				t.Fatalf("blocks = %d, want %d", len(content.Blocks), tc.wantBlocks)
			}
			if content.IsString != tc.wantString {
				t.Fatalf("IsString = %v, want %v", content.IsString, tc.wantString)
			}
		})
	}
}

func TestContentUnmarshalJSONRejectsWrongShapes(t *testing.T) {
	for _, body := range []string{`42`, `true`, `{"type":"text"}`, `[`} {
		var content Content
		if err := json.Unmarshal([]byte(body), &content); err == nil {
			t.Fatalf("decoding %s succeeded, want an error", body)
		}
	}
}

// Content must re-marshal in the shape it arrived in: a client that sent a bare
// string should not get an array back.
func TestContentMarshalJSONPreservesShape(t *testing.T) {
	cases := map[string]string{
		"string":       `"hello"`,
		"block_array":  `[{"type":"text","text":"hello"}]`,
		"empty_array":  `[]`,
		"empty_string": `""`,
	}

	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			var content Content
			if err := json.Unmarshal([]byte(body), &content); err != nil {
				t.Fatalf("decode: %v", err)
			}
			encoded, err := json.Marshal(content)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			if string(encoded) != body {
				t.Fatalf("round-tripped %s to %s", body, encoded)
			}
		})
	}
}

// A zero-value Content marshals as [], never null: the field is an array in every
// place the Messages API uses it.
func TestContentZeroValueMarshalsAsArray(t *testing.T) {
	encoded, err := json.Marshal(Content{})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if string(encoded) != `[]` {
		t.Fatalf("zero Content marshalled to %s, want []", encoded)
	}
}

// Nested tool_result content is itself a Content, so the recursion has to decode.
func TestContentNestedToolResultContent(t *testing.T) {
	var content Content
	body := `[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"inner"}]}]`
	if err := json.Unmarshal([]byte(body), &content); err != nil {
		t.Fatalf("decode: %v", err)
	}

	if len(content.Blocks) != 1 {
		t.Fatalf("blocks = %#v, want one tool_result", content.Blocks)
	}
	nested := content.Blocks[0].Content
	if nested == nil {
		t.Fatal("nested content is nil")
	}
	if nested.Text() != "inner" {
		t.Fatalf("nested text = %q, want inner", nested.Text())
	}
}

func TestErrorTypeForHTTP(t *testing.T) {
	cases := map[int]string{
		http.StatusBadRequest:            ErrorTypeInvalidRequest,
		http.StatusUnauthorized:          ErrorTypeAuthentication,
		http.StatusForbidden:             ErrorTypePermission,
		http.StatusNotFound:              ErrorTypeNotFound,
		http.StatusRequestEntityTooLarge: ErrorTypeRequestTooLarge,
		http.StatusTooManyRequests:       ErrorTypeRateLimit,
		529:                              ErrorTypeOverloaded,
		http.StatusInternalServerError:   ErrorTypeAPI,
		http.StatusBadGateway:            ErrorTypeAPI,
		http.StatusServiceUnavailable:    ErrorTypeAPI,
		http.StatusConflict:              ErrorTypeInvalidRequest,
	}

	for status, want := range cases {
		if got := ErrorTypeForHTTP(status); got != want {
			t.Fatalf("ErrorTypeForHTTP(%d) = %q, want %q", status, got, want)
		}
	}
}

func TestWriteError(t *testing.T) {
	recorder := httptest.NewRecorder()
	WriteError(recorder, http.StatusTooManyRequests, "slow down")

	if recorder.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429", recorder.Code)
	}
	if got := recorder.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}

	var got ErrorResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Type != "error" {
		t.Fatalf("envelope type = %q, want error", got.Type)
	}
	if got.Error.Type != ErrorTypeRateLimit || got.Error.Message != "slow down" {
		t.Fatalf("error = %#v", got.Error)
	}
}

// stop_reason and stop_sequence are always present on a message, null when unset.
func TestMessagesResponseAlwaysCarriesStopFields(t *testing.T) {
	encoded, err := json.Marshal(MessagesResponse{Content: []ContentBlock{}})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("decode: %v", err)
	}

	for _, key := range []string{"stop_reason", "stop_sequence", "content", "usage"} {
		if _, ok := decoded[key]; !ok {
			t.Fatalf("%q missing from %s", key, encoded)
		}
	}
	if decoded["stop_reason"] != nil || decoded["stop_sequence"] != nil {
		t.Fatalf("stop fields should be null when unset: %s", encoded)
	}
}

// A text content_block_start sends "text": "", a tool_use one omits it.
func TestStreamContentBlockTextPresence(t *testing.T) {
	encoded, err := json.Marshal(TextStreamBlock())
	if err != nil {
		t.Fatalf("encode text block: %v", err)
	}
	if string(encoded) != `{"type":"text","text":""}` {
		t.Fatalf("text block = %s, want an explicit empty text field", encoded)
	}

	encoded, err = json.Marshal(ToolUseStreamBlock("toolu_1", "get_weather"))
	if err != nil {
		t.Fatalf("encode tool block: %v", err)
	}
	if string(encoded) != `{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}` {
		t.Fatalf("tool block = %s", encoded)
	}
}

// message_delta carries both stop fields even when only one is set;
// content_block_delta must carry neither.
func TestDeltaEventFieldPresence(t *testing.T) {
	reason := StopReasonEndTurn
	encoded, err := json.Marshal(MessageDeltaEvent{
		Type:  EventMessageDelta,
		Delta: MessageDelta{StopReason: &reason},
	})
	if err != nil {
		t.Fatalf("encode message_delta: %v", err)
	}
	if want := `"stop_sequence":null`; !strings.Contains(string(encoded), want) {
		t.Fatalf("message_delta = %s, want %s", encoded, want)
	}

	encoded, err = json.Marshal(ContentBlockDeltaEvent{
		Type:  EventContentBlockDelta,
		Delta: BlockDelta{Type: DeltaTypeText, Text: "hi"},
	})
	if err != nil {
		t.Fatalf("encode content_block_delta: %v", err)
	}
	if strings.Contains(string(encoded), "stop_reason") || strings.Contains(string(encoded), "stop_sequence") {
		t.Fatalf("content_block_delta = %s, want no stop fields", encoded)
	}
}
