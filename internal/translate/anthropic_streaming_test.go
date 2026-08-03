package translate

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/harshaneel/localaik/internal/protocol/anthropic"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

type sseEvent struct {
	name string
	data string
}

// parseAnthropicSSE splits a written Anthropic event stream into ordered
// (event name, data payload) pairs.
func parseAnthropicSSE(t *testing.T, body string) []sseEvent {
	t.Helper()

	var events []sseEvent
	for _, frame := range strings.Split(strings.TrimSpace(body), "\n\n") {
		if strings.TrimSpace(frame) == "" {
			continue
		}
		event := sseEvent{}
		for _, line := range strings.Split(frame, "\n") {
			switch {
			case strings.HasPrefix(line, "event: "):
				event.name = strings.TrimPrefix(line, "event: ")
			case strings.HasPrefix(line, "data: "):
				event.data = strings.TrimPrefix(line, "data: ")
			default:
				t.Fatalf("unexpected SSE line %q in frame %q", line, frame)
			}
		}
		if event.name == "" {
			t.Fatalf("frame %q has no event name", frame)
		}
		events = append(events, event)
	}
	return events
}

func eventNames(events []sseEvent) []string {
	names := make([]string, 0, len(events))
	for _, event := range events {
		names = append(names, event.name)
	}
	return names
}

func decodeEvent(t *testing.T, event sseEvent) anthropic.StreamEvent {
	t.Helper()
	var decoded anthropic.StreamEvent
	if err := json.Unmarshal([]byte(event.data), &decoded); err != nil {
		t.Fatalf("decode %s payload %q: %v", event.name, event.data, err)
	}
	if decoded.Type != event.name {
		t.Fatalf("event name %q does not match payload type %q", event.name, decoded.Type)
	}
	return decoded
}

// mustWriteStream translates upstream and returns the parsed events, failing on
// any translation error.
func mustWriteStream(t *testing.T, upstream string) []sseEvent {
	t.Helper()
	events, _ := writeAnthropicStream(t, upstream, "")
	return events
}

func writeAnthropicStream(t *testing.T, upstream, model string) ([]sseEvent, *httptest.ResponseRecorder) {
	t.Helper()

	recorder := httptest.NewRecorder()
	if err := WriteAnthropicStreamFromOpenAISSE(recorder, strings.NewReader(upstream), model); err != nil {
		t.Fatalf("WriteAnthropicStreamFromOpenAISSE returned error: %v", err)
	}
	if got := recorder.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", got)
	}
	return parseAnthropicSSE(t, recorder.Body.String()), recorder
}

func TestWriteAnthropicStreamTextOnly(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"content":" world"}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	events, _ := writeAnthropicStream(t, upstream, "claude-sonnet-4-5")

	want := []string{
		anthropic.EventMessageStart,
		anthropic.EventContentBlockStart,
		anthropic.EventContentBlockDelta,
		anthropic.EventContentBlockDelta,
		anthropic.EventContentBlockStop,
		anthropic.EventMessageDelta,
		anthropic.EventMessageStop,
	}
	if got := eventNames(events); !equalStrings(got, want) {
		t.Fatalf("event sequence = %v, want %v", got, want)
	}

	start := decodeEvent(t, events[0])
	if start.Message == nil {
		t.Fatalf("message_start has no message: %s", events[0].data)
	}
	if start.Message.Role != "assistant" || start.Message.Type != "message" {
		t.Fatalf("message_start message = %#v", start.Message)
	}
	if start.Message.Model != "claude-sonnet-4-5" {
		t.Fatalf("message_start model = %q, want the requested model", start.Message.Model)
	}
	if len(start.Message.Content) != 0 {
		t.Fatalf("message_start content = %#v, want empty", start.Message.Content)
	}

	blockStart := decodeEvent(t, events[1])
	if blockStart.ContentBlock == nil || blockStart.ContentBlock.Type != anthropic.BlockTypeText {
		t.Fatalf("content_block_start = %s, want a text block", events[1].data)
	}
	if blockStart.Index == nil || *blockStart.Index != 0 {
		t.Fatalf("content_block_start index = %v, want 0", blockStart.Index)
	}

	for i, wantText := range []string{"Hello", " world"} {
		delta := decodeEvent(t, events[2+i])
		if delta.Delta == nil || delta.Delta.Type != anthropic.DeltaTypeText || delta.Delta.Text != wantText {
			t.Fatalf("delta %d = %s, want text_delta %q", i, events[2+i].data, wantText)
		}
		if delta.Index == nil || *delta.Index != 0 {
			t.Fatalf("delta %d index = %v, want 0", i, delta.Index)
		}
	}

	stop := decodeEvent(t, events[4])
	if stop.Index == nil || *stop.Index != 0 {
		t.Fatalf("content_block_stop index = %v, want 0", stop.Index)
	}

	messageDelta := decodeEvent(t, events[5])
	if messageDelta.Delta == nil || messageDelta.Delta.StopReason == nil {
		t.Fatalf("message_delta = %s, want a stop_reason", events[5].data)
	}
	if *messageDelta.Delta.StopReason != anthropic.StopReasonEndTurn {
		t.Fatalf("stop_reason = %q, want end_turn", *messageDelta.Delta.StopReason)
	}
	if messageDelta.Usage == nil || *messageDelta.Usage != (anthropic.Usage{InputTokens: 7, OutputTokens: 2}) {
		t.Fatalf("message_delta usage = %#v, want input 7 / output 2", messageDelta.Usage)
	}
}

func TestWriteAnthropicStreamToolCall(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"toolu_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	events, _ := writeAnthropicStream(t, upstream, "")

	// A tool call is buffered until its arguments parse, then emitted as one
	// complete block, so upstream's two argument fragments become a single delta.
	want := []string{
		anthropic.EventMessageStart,
		anthropic.EventContentBlockStart,
		anthropic.EventContentBlockDelta,
		anthropic.EventContentBlockStop,
		anthropic.EventMessageDelta,
		anthropic.EventMessageStop,
	}
	if got := eventNames(events); !equalStrings(got, want) {
		t.Fatalf("event sequence = %v, want %v", got, want)
	}

	blockStart := decodeEvent(t, events[1])
	block := blockStart.ContentBlock
	if block == nil || block.Type != anthropic.BlockTypeToolUse {
		t.Fatalf("content_block_start = %s, want a tool_use block", events[1].data)
	}
	if block.ID != "toolu_1" || block.Name != "get_weather" {
		t.Fatalf("tool_use block = %#v", block)
	}
	if string(block.Input) != `{}` {
		t.Fatalf("tool_use input = %s, want an empty object at block start", block.Input)
	}

	delta := decodeEvent(t, events[2])
	if delta.Delta == nil || delta.Delta.Type != anthropic.DeltaTypeInputJSON {
		t.Fatalf("delta = %s, want input_json_delta", events[2].data)
	}
	if delta.Delta.PartialJSON != `{"city":"Paris"}` {
		t.Fatalf("partial_json = %q, want the reassembled arguments", delta.Delta.PartialJSON)
	}

	messageDelta := decodeEvent(t, events[4])
	if messageDelta.Delta == nil || messageDelta.Delta.StopReason == nil {
		t.Fatalf("message_delta = %s, want a stop_reason", events[4].data)
	}
	if *messageDelta.Delta.StopReason != anthropic.StopReasonToolUse {
		t.Fatalf("stop_reason = %q, want tool_use", *messageDelta.Delta.StopReason)
	}
}

func TestWriteAnthropicStreamTextThenToolCallUsesSequentialBlocks(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"content":"Let me check."}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"toolu_2","type":"function","function":{"name":"lookup","arguments":"{}"}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	events, _ := writeAnthropicStream(t, upstream, "")

	want := []string{
		anthropic.EventMessageStart,
		anthropic.EventContentBlockStart, // text, index 0
		anthropic.EventContentBlockDelta,
		anthropic.EventContentBlockStop,  // index 0 closes before the tool block opens
		anthropic.EventContentBlockStart, // tool_use, index 1
		anthropic.EventContentBlockDelta,
		anthropic.EventContentBlockStop,
		anthropic.EventMessageDelta,
		anthropic.EventMessageStop,
	}
	if got := eventNames(events); !equalStrings(got, want) {
		t.Fatalf("event sequence = %v, want %v", got, want)
	}

	textStop := decodeEvent(t, events[3])
	if textStop.Index == nil || *textStop.Index != 0 {
		t.Fatalf("first content_block_stop index = %v, want 0", textStop.Index)
	}

	toolStart := decodeEvent(t, events[4])
	if toolStart.Index == nil || *toolStart.Index != 1 {
		t.Fatalf("tool content_block_start index = %v, want 1", toolStart.Index)
	}
	if toolStart.ContentBlock == nil || toolStart.ContentBlock.Type != anthropic.BlockTypeToolUse {
		t.Fatalf("tool content_block_start = %s", events[4].data)
	}
	if string(toolStart.ContentBlock.Input) != `{}` {
		t.Fatalf("tool input = %s, want an empty object", toolStart.ContentBlock.Input)
	}

	// A tool_use always gets at least one input_json_delta, as the real API sends.
	toolDelta := decodeEvent(t, events[5])
	if toolDelta.Delta == nil || toolDelta.Delta.Type != anthropic.DeltaTypeInputJSON {
		t.Fatalf("tool delta = %s, want input_json_delta", events[5].data)
	}
	if toolDelta.Delta.PartialJSON != `{}` {
		t.Fatalf("tool partial_json = %q, want an empty object", toolDelta.Delta.PartialJSON)
	}

	toolStop := decodeEvent(t, events[6])
	if toolStop.Index == nil || *toolStop.Index != 1 {
		t.Fatalf("tool content_block_stop index = %v, want 1", toolStop.Index)
	}
}

// An upstream that streamed only the [DONE] sentinel produced a real, if empty,
// response. That is distinct from one that sent no frames at all, which
// TestWriteAnthropicStreamNoFramesIsAnError covers.
func TestWriteAnthropicStreamDoneOnlyIsWellFormed(t *testing.T) {
	events, _ := writeAnthropicStream(t, "data: [DONE]\n\n", "")

	want := []string{
		anthropic.EventMessageStart,
		anthropic.EventMessageDelta,
		anthropic.EventMessageStop,
	}
	if got := eventNames(events); !equalStrings(got, want) {
		t.Fatalf("event sequence = %v, want %v", got, want)
	}

	messageDelta := decodeEvent(t, events[1])
	if messageDelta.Delta == nil || messageDelta.Delta.StopReason == nil {
		t.Fatalf("message_delta = %s, want a stop_reason", events[1].data)
	}
	if *messageDelta.Delta.StopReason != anthropic.StopReasonEndTurn {
		t.Fatalf("stop_reason = %q, want end_turn", *messageDelta.Delta.StopReason)
	}
}

func TestWriteAnthropicStreamHandlesMissingTrailingDone(t *testing.T) {
	// llama.cpp normally terminates with [DONE]; a truncated stream must still
	// close out the Anthropic event sequence.
	upstream := `data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}` + "\n\n"

	events, _ := writeAnthropicStream(t, upstream, "")

	want := []string{
		anthropic.EventMessageStart,
		anthropic.EventContentBlockStart,
		anthropic.EventContentBlockDelta,
		anthropic.EventContentBlockStop,
		anthropic.EventMessageDelta,
		anthropic.EventMessageStop,
	}
	if got := eventNames(events); !equalStrings(got, want) {
		t.Fatalf("event sequence = %v, want %v", got, want)
	}
}

// Upstream may start a second tool call before the first one's arguments finish
// streaming, and may finish the second first. The emitted blocks must still
// follow upstream's index order, and neither may carry a partial fragment.
func TestWriteAnthropicStreamInterleavedToolCallsPreserveUpstreamOrder(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t0","type":"function","function":{"name":"a","arguments":"{\"x\":"}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"t1","type":"function","function":{"name":"b","arguments":"{\"y\":2}"}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	events, _ := writeAnthropicStream(t, upstream, "")

	// Tool call b completes first, but it is held back so that a, at the lower
	// upstream index, still becomes block 0.
	want := []string{
		anthropic.EventMessageStart,
		anthropic.EventContentBlockStart, // tool a, index 0
		anthropic.EventContentBlockDelta, // {"x":1}
		anthropic.EventContentBlockStop,
		anthropic.EventContentBlockStart, // tool b, index 1
		anthropic.EventContentBlockDelta, // {"y":2}
		anthropic.EventContentBlockStop,
		anthropic.EventMessageDelta,
		anthropic.EventMessageStop,
	}
	if got := eventNames(events); !equalStrings(got, want) {
		t.Fatalf("event sequence = %v, want %v", got, want)
	}

	for i, want := range []struct {
		index int
		name  string
		input string
	}{
		{0, "a", `{"x":1}`},
		{1, "b", `{"y":2}`},
	} {
		start := decodeEvent(t, events[1+i*3])
		if start.Index == nil || *start.Index != want.index {
			t.Fatalf("block %d start index = %v, want %d", i, start.Index, want.index)
		}
		if start.ContentBlock == nil || start.ContentBlock.Name != want.name {
			t.Fatalf("block %d = %#v, want name %q", i, start.ContentBlock, want.name)
		}

		delta := decodeEvent(t, events[2+i*3])
		if delta.Delta == nil || delta.Delta.PartialJSON != want.input {
			t.Fatalf("block %d input = %s, want %q", i, events[2+i*3].data, want.input)
		}
		if !json.Valid([]byte(delta.Delta.PartialJSON)) {
			t.Fatalf("block %d input %q is not valid JSON", i, delta.Delta.PartialJSON)
		}

		stop := decodeEvent(t, events[3+i*3])
		if stop.Index == nil || *stop.Index != want.index {
			t.Fatalf("block %d stop index = %v, want %d", i, stop.Index, want.index)
		}
	}
}

// OpenAI's tool_calls[].index is omitempty, so a runtime that never sends it is
// indistinguishable from one that always sends 0. Separate calls must still get
// separate blocks, or their JSON fragments concatenate into one invalid input.
func TestWriteAnthropicStreamSeparatesToolCallsWithoutIndex(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"delta":{"tool_calls":[{"id":"t0","type":"function","function":{"name":"first","arguments":"{\"x\":1}"}}]}}]}`,
		``,
		`data: {"choices":[{"delta":{"tool_calls":[{"id":"t1","type":"function","function":{"name":"second","arguments":"{\"y\":2}"}}]}}]}`,
		``,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	events, _ := writeAnthropicStream(t, upstream, "")

	var starts []anthropic.StreamEvent
	fragments := map[int]string{}
	for _, event := range events {
		decoded := decodeEvent(t, event)
		switch event.name {
		case anthropic.EventContentBlockStart:
			starts = append(starts, decoded)
		case anthropic.EventContentBlockDelta:
			if decoded.Delta != nil && decoded.Delta.Type == anthropic.DeltaTypeInputJSON {
				fragments[*decoded.Index] += decoded.Delta.PartialJSON
			}
		}
	}

	if len(starts) != 2 {
		t.Fatalf("content_block_start count = %d, want 2 separate tool blocks; sequence = %v", len(starts), eventNames(events))
	}
	for i, want := range []struct{ id, name string }{{"t0", "first"}, {"t1", "second"}} {
		block := starts[i].ContentBlock
		if block == nil || block.ID != want.id || block.Name != want.name {
			t.Fatalf("block %d = %#v, want id %q name %q", i, block, want.id, want.name)
		}
		if *starts[i].Index != i {
			t.Fatalf("block %d index = %d, want %d", i, *starts[i].Index, i)
		}
	}

	for index, want := range map[int]string{0: `{"x":1}`, 1: `{"y":2}`} {
		if fragments[index] != want {
			t.Fatalf("block %d reassembled to %q, want %q", index, fragments[index], want)
		}
	}
}

// Continuation fragments carry neither id nor name, so they must keep appending to
// the block already open for their index.
func TestWriteAnthropicStreamContinuesToolCallWithoutIDOrName(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t0","type":"function","function":{"name":"only","arguments":"{\"x\":"}}]}}]}`,
		``,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	events, _ := writeAnthropicStream(t, upstream, "")

	starts := 0
	reassembled := ""
	for _, event := range events {
		switch event.name {
		case anthropic.EventContentBlockStart:
			starts++
		case anthropic.EventContentBlockDelta:
			decoded := decodeEvent(t, event)
			if decoded.Delta != nil && decoded.Delta.Type == anthropic.DeltaTypeInputJSON {
				reassembled += decoded.Delta.PartialJSON
			}
		}
	}

	if starts != 1 {
		t.Fatalf("content_block_start count = %d, want 1; sequence = %v", starts, eventNames(events))
	}
	if reassembled != `{"x":1}` {
		t.Fatalf("input = %q, want the fragments joined into {\"x\":1}", reassembled)
	}
}

// toolBlock is one emitted tool_use block, reassembled from the event stream.
type toolBlock struct {
	index int
	id    string
	name  string
	input string
}

// collectToolBlocks reassembles the tool_use blocks from a stream and asserts the
// invariants that hold for every stream, whatever upstream sent:
//
//   - block indices start at 0 and rise by exactly one, which the SDK accumulators
//     enforce on content_block_start;
//   - a delta or stop only ever addresses a block that was started and not yet
//     stopped, since a delta after a stop silently corrupts the accumulated input;
//   - every block is stopped exactly once;
//   - every tool_use block carries a decodable JSON object as its input, and a
//     `name` field, without which a client cannot invoke anything;
//   - tool_use ids are unique, since a client's tool_result blocks key on them.
func collectToolBlocks(t *testing.T, events []sseEvent) []toolBlock {
	t.Helper()

	var blocks []toolBlock
	slots := map[int]int{}
	stops := map[int]int{}

	requireIndex := func(event sseEvent, decoded anthropic.StreamEvent) int {
		if decoded.Index == nil {
			t.Fatalf("%s without index: %s", event.name, event.data)
		}
		return *decoded.Index
	}

	for _, event := range events {
		decoded := decodeEvent(t, event)
		switch event.name {
		case anthropic.EventContentBlockStart:
			index := requireIndex(event, decoded)
			if index != len(slots) {
				t.Fatalf("content_block_start index = %d, want %d (indices must be gapless and ascending)", index, len(slots))
			}
			slots[index] = -1
			if decoded.ContentBlock != nil && decoded.ContentBlock.Type == anthropic.BlockTypeToolUse {
				// The decode-side type omits an absent name, so the raw payload is
				// what proves the required field was actually sent.
				if !strings.Contains(event.data, `"name"`) {
					t.Fatalf("tool_use content_block_start has no name field: %s", event.data)
				}
				slots[index] = len(blocks)
				blocks = append(blocks, toolBlock{
					index: index,
					id:    decoded.ContentBlock.ID,
					name:  decoded.ContentBlock.Name,
				})
			}
		case anthropic.EventContentBlockDelta:
			index := requireIndex(event, decoded)
			slot, started := slots[index]
			if !started {
				t.Fatalf("content_block_delta for index %d, which was never started", index)
			}
			if stops[index] > 0 {
				t.Fatalf("content_block_delta for index %d after it was stopped", index)
			}
			if slot >= 0 && decoded.Delta != nil && decoded.Delta.Type == anthropic.DeltaTypeInputJSON {
				blocks[slot].input += decoded.Delta.PartialJSON
			}
		case anthropic.EventContentBlockStop:
			index := requireIndex(event, decoded)
			if _, started := slots[index]; !started {
				t.Fatalf("content_block_stop for index %d, which was never started", index)
			}
			stops[index]++
		}
	}

	for index := range slots {
		if stops[index] != 1 {
			t.Fatalf("block %d stopped %d times, want exactly 1", index, stops[index])
		}
	}

	ids := map[string]bool{}
	for _, block := range blocks {
		input := block.input
		if input == "" {
			input = "{}"
		}
		var decoded map[string]any
		if err := json.Unmarshal([]byte(input), &decoded); err != nil {
			t.Fatalf("block %d input %q does not decode as an object: %v", block.index, input, err)
		}
		if ids[block.id] {
			t.Fatalf("block %d reuses tool_use id %q", block.index, block.id)
		}
		ids[block.id] = true
	}

	return blocks
}

// A tool call's id and name may arrive in any delta, and a name may itself be
// split. None of that may produce a second block, drop the name, or leave a block
// holding a fragment: identity is only committed once the arguments parse.
func TestWriteAnthropicStreamToolCallIdentityArrivesLate(t *testing.T) {
	cases := []struct {
		name     string
		upstream []string
		want     toolBlock
	}{
		{
			name: "id_arrives_after_name",
			upstream: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"type":"function","function":{"name":"f","arguments":"{\"a\":"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"arguments":"1}"}}]}}]}`,
			},
			want: toolBlock{id: "call_1", name: "f", input: `{"a":1}`},
		},
		{
			// The name shows up only in the second delta, after arguments began.
			name: "name_arrives_after_id",
			upstream: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"arguments":"{\"a\":"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"f","arguments":"1}"}}]}}]}`,
			},
			want: toolBlock{id: "call_1", name: "f", input: `{"a":1}`},
		},
		{
			name: "name_streamed_in_fragments_with_arguments",
			upstream: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"type":"function","function":{"name":"get_","arguments":"{\"a\":"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"weather","arguments":"1}"}}]}}]}`,
			},
			want: toolBlock{id: "toolu_get_weather", name: "get_weather", input: `{"a":1}`},
		},
		{
			// The same shape with no arguments at all, so nothing gates on the
			// arguments parsing. The name still has to arrive whole.
			name: "name_streamed_in_fragments_without_arguments",
			upstream: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"type":"function","function":{"name":"get_"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"weather"}}]}}]}`,
			},
			want: toolBlock{id: "toolu_get_weather", name: "get_weather", input: "{}"},
		},
		{
			// A repeated full name is a repeat, not a fragment.
			name: "name_repeated_in_full",
			upstream: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"only","arguments":"{\"a\":"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"only","arguments":"1}"}}]}}]}`,
			},
			want: toolBlock{id: "call_1", name: "only", input: `{"a":1}`},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			upstream := strings.Join(append(tc.upstream, `data: [DONE]`, ``), "\n\n")

			blocks := collectToolBlocks(t, mustWriteStream(t, upstream))

			if len(blocks) != 1 {
				t.Fatalf("emitted %d tool blocks, want 1: %#v", len(blocks), blocks)
			}
			got := blocks[0]
			if got.id != tc.want.id || got.name != tc.want.name || got.input != tc.want.input {
				t.Fatalf("block = %#v, want id %q name %q input %q", got, tc.want.id, tc.want.name, tc.want.input)
			}
		})
	}
}

// A tool call's arguments becoming parseable does not mean the call has ended:
// upstream may send further deltas for the same index. Treating parseability as
// the end fabricates a second, usually nameless, call out of those deltas.
func TestWriteAnthropicStreamTrailingToolCallDeltas(t *testing.T) {
	cases := []struct {
		name     string
		upstream []string
		want     toolBlock
	}{
		{
			// A padding delta repeating the identity with no arguments.
			name: "empty_arguments_with_identity",
			upstream: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","type":"function","function":{"name":"f","arguments":"{\"a\":"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","type":"function","function":{"name":"f","arguments":"1}"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","type":"function","function":{"name":"f","arguments":""}}]}}]}`,
			},
			want: toolBlock{id: "c0", name: "f", input: `{"a":1}`},
		},
		{
			// The same but anonymous, which previously became a nameless phantom call.
			name: "empty_arguments_anonymous",
			upstream: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":""}}]}}]}`,
			},
			want: toolBlock{id: "c0", name: "f", input: `{"a":1}`},
		},
		{
			name: "whitespace_arguments",
			upstream: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"  "}}]}}]}`,
			},
			want: toolBlock{id: "c0", name: "f", input: `{"a":1}`},
		},
		{
			// The name only shows up after the arguments are already whole.
			name: "name_after_complete_arguments",
			upstream: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","type":"function","function":{"arguments":"{\"a\":1}"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"f","arguments":""}}]}}]}`,
			},
			want: toolBlock{id: "c0", name: "f", input: `{"a":1}`},
		},
		{
			// A tool delta after the finish_reason chunk.
			name: "delta_after_finish_reason",
			upstream: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]}}]}`,
				`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":""}}]}}]}`,
			},
			want: toolBlock{id: "c0", name: "f", input: `{"a":1}`},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			upstream := strings.Join(append(tc.upstream, `data: [DONE]`, ``), "\n\n")

			blocks := collectToolBlocks(t, mustWriteStream(t, upstream))

			if len(blocks) != 1 {
				t.Fatalf("emitted %d tool blocks, want 1: %#v", len(blocks), blocks)
			}
			got := blocks[0]
			if got.id != tc.want.id || got.name != tc.want.name || got.input != tc.want.input {
				t.Fatalf("block = %#v, want id %q name %q input %q", got, tc.want.id, tc.want.name, tc.want.input)
			}
		})
	}
}

// Two parallel calls to the same tool with no upstream ids must not share an id:
// the client's tool_result blocks key on it.
func TestWriteAnthropicStreamDeduplicatesSynthesizedToolIDs(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"type":"function","function":{"name":"read_file","arguments":"{\"p\":\"a.txt\"}"}}]}}]}`,
		``,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":1,"type":"function","function":{"name":"read_file","arguments":"{\"p\":\"b.txt\"}"}}]}}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	// collectToolBlocks fails on duplicate ids, so reaching the assertions is
	// already most of the check.
	blocks := collectToolBlocks(t, mustWriteStream(t, upstream))

	if len(blocks) != 2 {
		t.Fatalf("emitted %d tool blocks, want 2: %#v", len(blocks), blocks)
	}
	if blocks[0].name != "read_file" || blocks[1].name != "read_file" {
		t.Fatalf("blocks = %#v, want both named read_file", blocks)
	}
	if blocks[0].id == blocks[1].id {
		t.Fatalf("both blocks use id %q", blocks[0].id)
	}
}

// A no-argument tool call followed by text must still be emitted before the text,
// so the blocks keep upstream's order.
func TestWriteAnthropicStreamNoArgumentToolCallPrecedesText(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","type":"function","function":{"name":"noargs","arguments":""}}]}}]}`,
		``,
		`data: {"choices":[{"delta":{"content":"trailing text"}}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	events, _ := writeAnthropicStream(t, upstream, "")

	var order []string
	for _, event := range events {
		if event.name != anthropic.EventContentBlockStart {
			continue
		}
		decoded := decodeEvent(t, event)
		order = append(order, decoded.ContentBlock.Type)
	}

	if !equalStrings(order, []string{anthropic.BlockTypeToolUse, anthropic.BlockTypeText}) {
		t.Fatalf("block order = %v, want the tool call before the text", order)
	}
}

// Distinct calls arriving at the same upstream index must become distinct blocks,
// including when upstream reuses an id or a name.
func TestWriteAnthropicStreamDistinctCallsAtSameIndex(t *testing.T) {
	cases := []struct {
		name     string
		upstream []string
		want     []toolBlock
	}{
		{
			name: "different_ids",
			upstream: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","type":"function","function":{"name":"a","arguments":"{\"x\":1}"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"b","arguments":"{\"y\":2}"}}]}}]}`,
			},
			want: []toolBlock{
				{id: "c0", name: "a", input: `{"x":1}`},
				{id: "c1", name: "b", input: `{"y":2}`},
			},
		},
		{
			// Upstream reuses one id across two genuinely different calls. The second
			// block's id is disambiguated, since a client replying with duplicate
			// tool_use_ids would be rejected on the next turn.
			name: "repeated_id_different_names",
			upstream: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","type":"function","function":{"name":"a","arguments":"{\"x\":1}"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","type":"function","function":{"name":"b","arguments":"{\"y\":2}"}}]}}]}`,
			},
			want: []toolBlock{
				{id: "c0", name: "a", input: `{"x":1}`},
				{id: "c0_2", name: "b", input: `{"y":2}`},
			},
		},
		{
			// Nothing to derive an id from, so the synthesized ids must still differ.
			name: "no_identity_at_all",
			upstream: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":1}"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"y\":2}"}}]}}]}`,
			},
			want: []toolBlock{
				{id: "toolu_localaik", name: "", input: `{"x":1}`},
				{id: "toolu_localaik_2", name: "", input: `{"y":2}`},
			},
		},
		{
			name: "three_calls_one_index",
			upstream: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","type":"function","function":{"name":"a","arguments":"{\"x\":1}"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"b","arguments":"{\"y\":2}"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c2","type":"function","function":{"name":"c","arguments":"{\"z\":3}"}}]}}]}`,
			},
			want: []toolBlock{
				{id: "c0", name: "a", input: `{"x":1}`},
				{id: "c1", name: "b", input: `{"y":2}`},
				{id: "c2", name: "c", input: `{"z":3}`},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			upstream := strings.Join(append(tc.upstream, `data: [DONE]`, ``), "\n\n")

			blocks := collectToolBlocks(t, mustWriteStream(t, upstream))

			if len(blocks) != len(tc.want) {
				t.Fatalf("emitted %d tool blocks, want %d: %#v", len(blocks), len(tc.want), blocks)
			}
			for i, want := range tc.want {
				if blocks[i].id != want.id || blocks[i].name != want.name || blocks[i].input != want.input {
					t.Fatalf("block %d = %#v, want id %q name %q input %q", i, blocks[i], want.id, want.name, want.input)
				}
			}
		})
	}
}

// Arguments that never parse must not reach the client as a broken input, and the
// call must not vanish either.
func TestWriteAnthropicStreamTruncatedToolArgumentsBecomeEmptyObject(t *testing.T) {
	cases := map[string]string{
		"truncated_object": `{"a":`,
		"trailing_comma":   `{"a":1,`,
		"only_brace":       `{`,
	}

	for name, arguments := range cases {
		t.Run(name, func(t *testing.T) {
			encoded, err := json.Marshal(arguments)
			if err != nil {
				t.Fatalf("encode arguments: %v", err)
			}
			upstream := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","type":"function","function":{"name":"f","arguments":` + string(encoded) + `}}]}}]}` +
				"\n\n" + "data: [DONE]\n\n"

			blocks := collectToolBlocks(t, mustWriteStream(t, upstream))

			if len(blocks) != 1 {
				t.Fatalf("emitted %d tool blocks, want 1: %#v", len(blocks), blocks)
			}
			if blocks[0].input != "{}" {
				t.Fatalf("input = %q, want the unparseable arguments replaced with an empty object", blocks[0].input)
			}
			if blocks[0].name != "f" {
				t.Fatalf("block = %#v, want the call preserved", blocks[0])
			}
		})
	}
}

// json.Valid accepts bare scalars, so a complete-but-not-object argument payload
// must not be forwarded as a tool input.
func TestWriteAnthropicStreamScalarToolArgumentsBecomeEmptyObject(t *testing.T) {
	for _, arguments := range []string{`null`, `1`, `"text"`, `[1,2]`, `true`} {
		t.Run(arguments, func(t *testing.T) {
			encoded, err := json.Marshal(arguments)
			if err != nil {
				t.Fatalf("encode arguments: %v", err)
			}
			upstream := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","type":"function","function":{"name":"f","arguments":` + string(encoded) + `}}]}}]}` +
				"\n\n" + "data: [DONE]\n\n"

			blocks := collectToolBlocks(t, mustWriteStream(t, upstream))

			if len(blocks) != 1 {
				t.Fatalf("emitted %d tool blocks, want 1: %#v", len(blocks), blocks)
			}
			input := blocks[0].input
			if input == "" {
				input = "{}"
			}
			var decoded map[string]any
			if err := json.Unmarshal([]byte(input), &decoded); err != nil {
				t.Fatalf("tool input %q does not decode as an object: %v", input, err)
			}
		})
	}
}

// A repeated name on every fragment must not be mistaken for a new call.
func TestWriteAnthropicStreamToolCallRepeatingNameStaysOneBlock(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t0","type":"function","function":{"name":"only","arguments":"{\"x\":"}}]}}]}`,
		``,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t0","type":"function","function":{"name":"only","arguments":"1}"}}]}}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	events, _ := writeAnthropicStream(t, upstream, "")

	starts := 0
	for _, event := range events {
		if event.name == anthropic.EventContentBlockStart {
			starts++
		}
	}
	if starts != 1 {
		t.Fatalf("content_block_start count = %d, want 1; sequence = %v", starts, eventNames(events))
	}
}

// A 200 with no SSE frames means upstream did not stream. Reporting a clean empty
// message there would present a broken upstream as a successful completion.
func TestWriteAnthropicStreamNoFramesIsAnError(t *testing.T) {
	cases := map[string]string{
		"empty_body":       "",
		"plain_json_reply": `{"choices":[{"message":{"content":"not a stream"}}]}`,
	}

	for name, upstream := range cases {
		t.Run(name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			err := WriteAnthropicStreamFromOpenAISSE(recorder, strings.NewReader(upstream), "")
			if err == nil {
				t.Fatal("expected an error when upstream sent no SSE frames")
			}

			events := parseAnthropicSSE(t, recorder.Body.String())
			if len(events) != 1 || events[0].name != anthropic.EventError {
				t.Fatalf("events = %v, want a single error event", eventNames(events))
			}
			if !strings.Contains(events[0].data, "no stream data") {
				t.Fatalf("error event = %s, want it to name the cause", events[0].data)
			}
		})
	}
}

func TestWriteAnthropicStreamTextAfterToolOpensNewBlock(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"content":"before"}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t0","type":"function","function":{"name":"a","arguments":"{}"}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"content":"after"}}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	events, _ := writeAnthropicStream(t, upstream, "")

	// Block indices must still be gapless and ascending: text 0, tool 1, text 2.
	var startIndices []int
	for _, event := range events {
		if event.name != anthropic.EventContentBlockStart {
			continue
		}
		decoded := decodeEvent(t, event)
		if decoded.Index == nil {
			t.Fatalf("content_block_start without index: %s", event.data)
		}
		startIndices = append(startIndices, *decoded.Index)
	}
	if len(startIndices) != 3 {
		t.Fatalf("content_block_start count = %d, want 3; sequence = %v", len(startIndices), eventNames(events))
	}
	for i, index := range startIndices {
		if index != i {
			t.Fatalf("start indices = %v, want 0,1,2", startIndices)
		}
	}

	// Every started block is stopped exactly once.
	stops := map[int]int{}
	for _, event := range events {
		if event.name != anthropic.EventContentBlockStop {
			continue
		}
		stops[*decodeEvent(t, event).Index]++
	}
	for index := range startIndices {
		if stops[index] != 1 {
			t.Fatalf("block %d stopped %d times, want 1", index, stops[index])
		}
	}
}

// The Messages API always sends "text": "" on a text content_block_start and
// always sends stop_reason / stop_sequence on message_delta, even when null.
func TestWriteAnthropicStreamEmitsPresentButNullFields(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	events, _ := writeAnthropicStream(t, upstream, "")

	var blockStart, messageDelta, blockDelta string
	for _, event := range events {
		switch event.name {
		case anthropic.EventContentBlockStart:
			blockStart = event.data
		case anthropic.EventContentBlockDelta:
			blockDelta = event.data
		case anthropic.EventMessageDelta:
			messageDelta = event.data
		}
	}

	if !strings.Contains(blockStart, `"text":""`) {
		t.Fatalf("content_block_start = %s, want an explicit empty text field", blockStart)
	}
	if !strings.Contains(messageDelta, `"stop_sequence":null`) {
		t.Fatalf("message_delta = %s, want stop_sequence present as null", messageDelta)
	}
	if !strings.Contains(messageDelta, `"stop_reason":"end_turn"`) {
		t.Fatalf("message_delta = %s, want stop_reason end_turn", messageDelta)
	}
	// A content_block_delta must not carry the terminal stop fields.
	if strings.Contains(blockDelta, "stop_reason") || strings.Contains(blockDelta, "stop_sequence") {
		t.Fatalf("content_block_delta = %s, want no stop fields", blockDelta)
	}
}

func TestWriteAnthropicStreamToolBlockStartOmitsText(t *testing.T) {
	upstream := `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t0","type":"function","function":{"name":"a","arguments":"{}"}}]}}]}` + "\n\n" +
		"data: [DONE]\n\n"

	events, _ := writeAnthropicStream(t, upstream, "")

	var blockStart string
	for _, event := range events {
		if event.name == anthropic.EventContentBlockStart {
			blockStart = event.data
		}
	}

	if strings.Contains(blockStart, `"text"`) {
		t.Fatalf("tool_use content_block_start = %s, want no text field", blockStart)
	}
	if !strings.Contains(blockStart, `"input":{}`) {
		t.Fatalf("tool_use content_block_start = %s, want input as an empty object", blockStart)
	}
}

func TestWriteAnthropicStreamDerivesMessageIDFromUpstream(t *testing.T) {
	upstream := `data: {"id":"chatcmpl-stream9","choices":[{"index":0,"delta":{"content":"hi"}}]}` + "\n\n" +
		"data: [DONE]\n\n"

	events, _ := writeAnthropicStream(t, upstream, "")

	start := decodeEvent(t, events[0])
	if start.Message == nil {
		t.Fatalf("message_start has no message: %s", events[0].data)
	}
	if start.Message.ID != "msg_stream9" {
		t.Fatalf("message_start id = %q, want msg_stream9", start.Message.ID)
	}
}

// Upstream can report choices out of order within a chunk; the message must
// follow one choice rather than whichever happens to be first.
func TestWriteAnthropicStreamFollowsLowestChoiceIndex(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"index":1,"delta":{"content":"WRONG"}},{"index":0,"delta":{"content":"right"}}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	events, _ := writeAnthropicStream(t, upstream, "")

	var texts []string
	for _, event := range events {
		if event.name != anthropic.EventContentBlockDelta {
			continue
		}
		decoded := decodeEvent(t, event)
		if decoded.Delta != nil {
			texts = append(texts, decoded.Delta.Text)
		}
	}

	if len(texts) != 1 || texts[0] != "right" {
		t.Fatalf("text deltas = %#v, want only the index-0 choice", texts)
	}
}

func TestWriteAnthropicStreamMalformedPayloadEmitsErrorEvent(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}`,
		``,
		`data: {not json`,
		``,
	}, "\n")

	recorder := httptest.NewRecorder()
	err := WriteAnthropicStreamFromOpenAISSE(recorder, strings.NewReader(upstream), "")
	if err == nil {
		t.Fatal("expected an error for a malformed upstream payload")
	}

	events := parseAnthropicSSE(t, recorder.Body.String())
	if len(events) == 0 {
		t.Fatal("no events written")
	}

	last := events[len(events)-1]
	if last.name != anthropic.EventError {
		t.Fatalf("last event = %q, want error; sequence = %v", last.name, eventNames(events))
	}
	errorEvent := decodeEvent(t, last)
	if errorEvent.Error == nil || errorEvent.Error.Message == "" {
		t.Fatalf("error event = %s, want a populated error", last.data)
	}
	if errorEvent.Error.Type != anthropic.ErrorTypeAPI {
		t.Fatalf("error type = %q, want %q", errorEvent.Error.Type, anthropic.ErrorTypeAPI)
	}
}

// failingWriter fails every write, standing in for a client that disconnected
// partway through a stream.
type failingWriter struct {
	header http.Header
	writes int
}

func (f *failingWriter) Header() http.Header {
	if f.header == nil {
		f.header = make(http.Header)
	}
	return f.header
}

func (f *failingWriter) Write([]byte) (int, error) {
	f.writes++
	return 0, errors.New("client gone")
}

func (f *failingWriter) WriteHeader(int) {}

// Once a write fails there is nowhere to report anything, so the translator must
// give up rather than keep formatting events for a dead connection.
func TestWriteAnthropicStreamStopsWritingAfterWriteFailure(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"content":"one"}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"content":"two"}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	writer := &failingWriter{}
	err := WriteAnthropicStreamFromOpenAISSE(writer, strings.NewReader(upstream), "")
	if err == nil {
		t.Fatal("expected the write failure to surface as an error")
	}
	if writer.writes != 1 {
		t.Fatalf("attempted %d writes, want 1: the first failure should stop the stream", writer.writes)
	}
}

func TestOpenAIToolCallsToAnthropicBlocksSkipsUnusableCalls(t *testing.T) {
	blocks := openAIToolCallsToAnthropicBlocks([]openaip.ToolCall{
		{ID: "keep", Type: "function", Function: &openaip.ToolCallFunction{Name: "ok", Arguments: `{}`}},
		{ID: "no_function", Type: "function"},
		{ID: "not_a_function", Type: "retrieval", Function: &openaip.ToolCallFunction{Name: "nope"}},
	})

	if len(blocks) != 1 {
		t.Fatalf("blocks = %#v, want only the usable function call", blocks)
	}
	if blocks[0].ID != "keep" || blocks[0].Name != "ok" {
		t.Fatalf("block = %#v", blocks[0])
	}
}

// TestWriteAnthropicStreamToolCallShapeMatrix is the regression net for the
// streaming state machine. It walks every combination of the ways upstreams
// legitimately differ — where a tool call's identity appears, whether an index is
// sent, how finely arguments are fragmented, whether a padding delta trails the
// call, whether text surrounds it — and asserts the same thing throughout: one
// block per intended call, each with the right name and input.
//
// Four review rounds each found a shape that broke a state machine whose own
// tests passed, because every fixture happened to share an assumption. Enumerating
// the dimensions is what stops that recurring.
func TestWriteAnthropicStreamToolCallShapeMatrix(t *testing.T) {
	type call struct {
		id   string
		name string
		args string
	}

	calls := []call{
		{id: "c0", name: "first", args: `{"a":1}`},
		{id: "c1", name: "second", args: `{"b":"two"}`},
	}

	// splitInto cuts s into n roughly equal pieces, so arguments arrive fragmented
	// the way a real token stream delivers them.
	splitInto := func(s string, n int) []string {
		if n <= 1 || len(s) < n {
			return []string{s}
		}
		pieces := make([]string, 0, n)
		size := len(s) / n
		for i := 0; i < n; i++ {
			start := i * size
			end := start + size
			if i == n-1 {
				end = len(s)
			}
			pieces = append(pieces, s[start:end])
		}
		return pieces
	}

	identityModes := []string{"with_first_fragment", "id_first_then_name", "name_first_then_id", "repeated_on_every_fragment"}
	indexModes := []string{"correct_index", "always_zero", "omitted"}

	for _, identity := range identityModes {
		for _, indexMode := range indexModes {
			for _, fragments := range []int{1, 2, 3} {
				for _, padding := range []bool{false, true} {
					for _, withText := range []bool{false, true} {
						name := fmt.Sprintf("%s/%s/frags=%d/pad=%v/text=%v", identity, indexMode, fragments, padding, withText)

						t.Run(name, func(t *testing.T) {
							// An upstream that omits the index or pins it to 0 gives no way to
							// separate two calls except a fresh argument object, so a call with
							// no arguments of its own cannot be told apart. Every call here has
							// arguments, so the shapes remain distinguishable.
							var chunks []string

							emit := func(callIndex int, delta string) {
								indexField := ""
								switch indexMode {
								case "correct_index":
									indexField = fmt.Sprintf(`"index":%d,`, callIndex)
								case "always_zero":
									indexField = `"index":0,`
								}
								chunks = append(chunks, `data: {"choices":[{"delta":{"tool_calls":[{`+indexField+delta+`}]}}]}`)
							}

							if withText {
								chunks = append(chunks, `data: {"choices":[{"delta":{"content":"before"}}]}`)
							}

							for i, c := range calls {
								pieces := splitInto(c.args, fragments)
								for p, piece := range pieces {
									encoded, err := json.Marshal(piece)
									if err != nil {
										t.Fatalf("encode fragment: %v", err)
									}

									id, toolName := "", ""
									switch identity {
									case "with_first_fragment":
										if p == 0 {
											id, toolName = c.id, c.name
										}
									case "id_first_then_name":
										if p == 0 {
											id = c.id
										}
										if p == min(1, len(pieces)-1) {
											toolName = c.name
										}
									case "name_first_then_id":
										if p == 0 {
											toolName = c.name
										}
										if p == min(1, len(pieces)-1) {
											id = c.id
										}
									case "repeated_on_every_fragment":
										id, toolName = c.id, c.name
									}

									parts := ""
									if id != "" {
										parts += fmt.Sprintf(`"id":%q,"type":"function",`, id)
									}
									if toolName != "" {
										parts += fmt.Sprintf(`"function":{"name":%q,"arguments":%s}`, toolName, encoded)
									} else {
										parts += fmt.Sprintf(`"function":{"arguments":%s}`, encoded)
									}
									emit(i, parts)
								}

								if padding {
									emit(i, `"function":{"arguments":""}`)
								}
							}

							if withText {
								chunks = append(chunks, `data: {"choices":[{"delta":{"content":"after"}}]}`)
							}
							chunks = append(chunks,
								`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
								`data: [DONE]`,
								``,
							)

							upstream := strings.Join(chunks, "\n\n")
							blocks := collectToolBlocks(t, mustWriteStream(t, upstream))

							if len(blocks) != len(calls) {
								t.Fatalf("emitted %d tool blocks, want %d\nblocks: %#v\nupstream:\n%s", len(blocks), len(calls), blocks, upstream)
							}
							for i, want := range calls {
								if blocks[i].name != want.name {
									t.Fatalf("block %d name = %q, want %q\nupstream:\n%s", i, blocks[i].name, want.name, upstream)
								}
								if blocks[i].input != want.args {
									t.Fatalf("block %d input = %q, want %q\nupstream:\n%s", i, blocks[i].input, want.args, upstream)
								}
							}
						})
					}
				}
			}
		}
	}
}

func equalStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
