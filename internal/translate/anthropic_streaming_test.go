package translate

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/harshaneel/localaik/internal/protocol/anthropic"
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

	// The first upstream delta carries empty arguments and must not produce a
	// partial_json event; only the two fragments below do.
	for i, wantFragment := range []string{`{"city":`, `"Paris"}`} {
		delta := decodeEvent(t, events[2+i])
		if delta.Delta == nil || delta.Delta.Type != anthropic.DeltaTypeInputJSON {
			t.Fatalf("delta %d = %s, want input_json_delta", i, events[2+i].data)
		}
		if delta.Delta.PartialJSON != wantFragment {
			t.Fatalf("delta %d partial_json = %q, want %q", i, delta.Delta.PartialJSON, wantFragment)
		}
	}

	messageDelta := decodeEvent(t, events[5])
	if messageDelta.Delta == nil || messageDelta.Delta.StopReason == nil {
		t.Fatalf("message_delta = %s, want a stop_reason", events[5].data)
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

	toolStop := decodeEvent(t, events[6])
	if toolStop.Index == nil || *toolStop.Index != 1 {
		t.Fatalf("tool content_block_stop index = %v, want 1", toolStop.Index)
	}
}

func TestWriteAnthropicStreamEmptyUpstreamStillWellFormed(t *testing.T) {
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
// streaming. Closing the first block at that point would stop it holding a
// half-written JSON fragment, so tool blocks stay open until the stream ends.
func TestWriteAnthropicStreamInterleavedToolCallsKeepBlocksOpen(t *testing.T) {
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

	want := []string{
		anthropic.EventMessageStart,
		anthropic.EventContentBlockStart, // tool a, index 0
		anthropic.EventContentBlockDelta, // {"x":
		anthropic.EventContentBlockStart, // tool b, index 1
		anthropic.EventContentBlockDelta, // {"y":2}
		anthropic.EventContentBlockDelta, // 1}  resumes index 0, still open
		anthropic.EventContentBlockStop,  // index 0
		anthropic.EventContentBlockStop,  // index 1
		anthropic.EventMessageDelta,
		anthropic.EventMessageStop,
	}
	if got := eventNames(events); !equalStrings(got, want) {
		t.Fatalf("event sequence = %v, want %v", got, want)
	}

	// No stop may precede the resumed delta for index 0.
	resumed := decodeEvent(t, events[5])
	if resumed.Index == nil || *resumed.Index != 0 {
		t.Fatalf("resumed delta index = %v, want 0", resumed.Index)
	}
	if resumed.Delta == nil || resumed.Delta.PartialJSON != "1}" {
		t.Fatalf("resumed delta = %s, want the trailing fragment", events[5].data)
	}

	// Stops close in ascending index order, once each.
	for i, wantIndex := range []int{0, 1} {
		stop := decodeEvent(t, events[6+i])
		if stop.Index == nil || *stop.Index != wantIndex {
			t.Fatalf("stop %d index = %v, want %d", i, stop.Index, wantIndex)
		}
	}

	// Reassembling the fragments per block must yield valid JSON objects.
	fragments := map[int]string{}
	for _, event := range events {
		if event.name != anthropic.EventContentBlockDelta {
			continue
		}
		decoded := decodeEvent(t, event)
		if decoded.Delta == nil || decoded.Delta.Type != anthropic.DeltaTypeInputJSON {
			continue
		}
		fragments[*decoded.Index] += decoded.Delta.PartialJSON
	}
	for index, want := range map[int]string{0: `{"x":1}`, 1: `{"y":2}`} {
		if fragments[index] != want {
			t.Fatalf("block %d reassembled to %q, want %q", index, fragments[index], want)
		}
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
