package translate

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/harshaneel/localaik/internal/protocol/anthropic"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

// WriteAnthropicStreamFromOpenAISSE translates an upstream OpenAI SSE stream
// into the Anthropic Messages event stream and writes it to w.
//
// The Anthropic stream is a strict event sequence rather than a flat sequence of
// chunks: message_start, then per content block a content_block_start / N
// content_block_delta / content_block_stop group, then message_delta and
// message_stop.
//
// Token usage is only as good as what upstream reports. llama.cpp may omit usage
// from streamed responses, in which case the counts are zero.
func WriteAnthropicStreamFromOpenAISSE(w http.ResponseWriter, body io.Reader, model string) error {
	writer := newAnthropicStreamWriter(w, model)
	writer.writeHeaders()

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var dataLines []string
	done := false

	for !done && scanner.Scan() {
		line := scanner.Text()
		if line != "" {
			if payload, ok := strings.CutPrefix(line, "data:"); ok {
				dataLines = append(dataLines, strings.TrimSpace(payload))
			}
			continue
		}

		err := writer.consume(dataLines)
		dataLines = nil
		if err == io.EOF {
			done = true
			break
		}
		if err != nil {
			writer.writeErrorEvent(err)
			return err
		}
	}

	if err := scanner.Err(); err != nil {
		writer.writeErrorEvent(err)
		return err
	}

	if !done {
		if err := writer.consume(dataLines); err != nil && err != io.EOF {
			writer.writeErrorEvent(err)
			return err
		}
	}

	// A 200 with no SSE frames at all means upstream did not stream: an empty
	// body, or a plain JSON reply to a streaming request. Closing out cleanly
	// there would report an empty completion as a success, so it is reported as
	// an error instead. A stream that sent frames but no finish_reason is a
	// different case and still closes cleanly.
	if !writer.sawFrame {
		err := errors.New("upstream returned no stream data")
		writer.writeErrorEvent(err)
		return err
	}

	return writer.finish()
}

// anthropicStreamWriter tracks the block bookkeeping the Anthropic event
// sequence requires.
//
// Block indices start at 0 and increase by exactly one per block, which the
// Anthropic SDK accumulators enforce on content_block_start. Delta and stop
// events address a block by index and may interleave across blocks that are
// still open, which is what makes the lifetime rules below safe:
//
//   - A text block is closed as soon as a tool block starts. Text arriving later
//     opens a fresh text block, exactly as the Messages API does.
//   - A tool block stays open until the stream ends. Upstream can resume an
//     earlier tool-call index after starting a later one, and closing a tool
//     block early would stop it holding a half-written JSON fragment.
type anthropicStreamWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	model   string

	started      bool
	finished     bool
	sawFrame     bool
	nextIndex    int
	openBlocks   []int
	textIndex    int
	toolBlocks   map[int]toolBlockRef
	messageID    string
	stopReason   string
	inputTokens  int
	outputTokens int
	writeErr     error
}

// toolBlockRef remembers which Anthropic block an upstream tool-call index maps
// to, along with enough about the call occupying it to tell a continuation from
// a different call. Identity matters because OpenAI's tool-call index is
// `omitempty`: an upstream that never sends it is indistinguishable from one
// that always sends 0, so index alone would merge separate calls into one block
// and concatenate their JSON.
//
// upstreamID is the id exactly as upstream sent it, kept apart from the id
// emitted downstream, which may have been synthesized from the name.
type toolBlockRef struct {
	index      int
	upstreamID string
	name       string
	arguments  string
}

// argumentsComplete reports whether the fragments seen so far form a whole JSON
// value. A call still mid-object cannot have ended, whatever its next delta says.
func (r toolBlockRef) argumentsComplete() bool {
	return r.arguments == "" || json.Valid([]byte(r.arguments))
}

// startsNewToolCall reports whether a tool-call delta belongs to a different call
// than the one already occupying its index.
//
// Arguments completeness gates the decision: a block holding a half-written JSON
// object is always treated as a continuation, because splitting there would leave
// two blocks each holding an unparseable fragment, and the SDK accumulators
// marshal a block when it stops. Only once the current call's arguments parse can
// a differing identity mean a genuinely new call.
//
// Ids win when either side has one. Names are only compared when neither does,
// and only when both are non-empty, so a name that arrives a delta later than the
// id does not look like a new call.
func startsNewToolCall(ref toolBlockRef, id, name string) bool {
	if !ref.argumentsComplete() {
		return false
	}
	if ref.upstreamID != "" || id != "" {
		return id != "" && ref.upstreamID != "" && id != ref.upstreamID
	}
	return name != "" && ref.name != "" && name != ref.name
}

func newAnthropicStreamWriter(w http.ResponseWriter, model string) *anthropicStreamWriter {
	flusher, _ := w.(http.Flusher)
	return &anthropicStreamWriter{
		w:          w,
		flusher:    flusher,
		model:      model,
		textIndex:  -1,
		toolBlocks: make(map[int]toolBlockRef),
	}
}

func (a *anthropicStreamWriter) writeHeaders() {
	a.w.Header().Set("Content-Type", "text/event-stream")
	a.w.Header().Set("Cache-Control", "no-cache")
	a.w.WriteHeader(http.StatusOK)
}

// consume handles one SSE frame's worth of data lines. It returns io.EOF when
// upstream signals [DONE].
func (a *anthropicStreamWriter) consume(dataLines []string) error {
	if len(dataLines) == 0 {
		return nil
	}
	a.sawFrame = true

	payload := strings.Join(dataLines, "\n")
	if payload == "[DONE]" {
		return io.EOF
	}

	var chunk openaip.StreamChunk
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		return fmt.Errorf("decode SSE payload: %w", err)
	}

	if a.messageID == "" && chunk.ID != "" {
		a.messageID = anthropicMessageID(chunk.ID)
	}
	a.ensureStarted()

	// The Messages API returns a single message, so the stream follows one
	// upstream choice. Selecting it by index rather than by position keeps the
	// text coherent if upstream ever reorders choices within a chunk.
	if choice, ok := primaryStreamChoice(chunk.Choices); ok {
		if text := extractTextFromOpenAIContent(choice.Delta.Content); text != "" {
			a.writeTextDelta(text)
		}
		for _, call := range choice.Delta.ToolCalls {
			a.writeToolCallDelta(call)
		}
		if reason := OpenAIFinishReasonToAnthropic(choice.FinishReason); reason != "" {
			a.stopReason = reason
		}
	}

	if chunk.Usage != nil {
		a.inputTokens = chunk.Usage.PromptTokens
		a.outputTokens = chunk.Usage.CompletionTokens
	}

	return a.writeErr
}

// primaryStreamChoice picks the choice the Anthropic message follows: index 0
// when present, otherwise the lowest index in the chunk.
func primaryStreamChoice(choices []openaip.StreamChoice) (openaip.StreamChoice, bool) {
	if len(choices) == 0 {
		return openaip.StreamChoice{}, false
	}

	best := choices[0]
	for _, choice := range choices[1:] {
		if choice.Index < best.Index {
			best = choice
		}
	}
	return best, true
}

func (a *anthropicStreamWriter) ensureStarted() {
	if a.started {
		return
	}
	a.started = true

	id := a.messageID
	if id == "" {
		id = anthropicMessageID("")
	}

	a.writeEvent(anthropic.EventMessageStart, anthropic.MessageStartEvent{
		Type: anthropic.EventMessageStart,
		Message: anthropic.MessagesResponse{
			ID:      id,
			Type:    "message",
			Role:    "assistant",
			Model:   anthropicResponseModel(a.model, ""),
			Content: []anthropic.ContentBlock{},
			Usage:   anthropic.Usage{InputTokens: a.inputTokens},
		},
	})
}

func (a *anthropicStreamWriter) writeTextDelta(text string) {
	if a.textIndex == -1 {
		a.textIndex = a.startBlock(anthropic.TextStreamBlock())
	}

	a.writeEvent(anthropic.EventContentBlockDelta, anthropic.ContentBlockDeltaEvent{
		Type:  anthropic.EventContentBlockDelta,
		Index: a.textIndex,
		Delta: anthropic.BlockDelta{
			Type: anthropic.DeltaTypeText,
			Text: text,
		},
	})
}

func (a *anthropicStreamWriter) writeToolCallDelta(call openaip.ToolCallDelta) {
	if call.Type != "" && call.Type != "function" {
		return
	}

	name := ""
	if call.Function != nil {
		name = call.Function.Name
	}

	ref, known := a.toolBlocks[call.Index]
	if known && startsNewToolCall(ref, call.ID, name) {
		known = false
	}

	if !known {
		// Any text so far is complete once a tool call begins.
		a.closeTextBlock()
		ref = toolBlockRef{
			index:      a.startBlock(anthropic.ToolUseStreamBlock(anthropicToolUseID(call.ID, name), name)),
			upstreamID: call.ID,
			name:       name,
		}
	} else {
		// Fill in identity that arrived a delta later than the block was opened.
		if ref.upstreamID == "" {
			ref.upstreamID = call.ID
		}
		if ref.name == "" {
			ref.name = name
		}
	}

	if call.Function != nil && call.Function.Arguments != "" {
		ref.arguments += call.Function.Arguments
	}
	a.toolBlocks[call.Index] = ref

	if call.Function == nil || call.Function.Arguments == "" {
		return
	}

	a.writeEvent(anthropic.EventContentBlockDelta, anthropic.ContentBlockDeltaEvent{
		Type:  anthropic.EventContentBlockDelta,
		Index: ref.index,
		Delta: anthropic.BlockDelta{
			Type:        anthropic.DeltaTypeInputJSON,
			PartialJSON: call.Function.Arguments,
		},
	})
}

// startBlock emits content_block_start for a new block and returns its index.
func (a *anthropicStreamWriter) startBlock(block anthropic.StreamContentBlock) int {
	index := a.nextIndex
	a.nextIndex++
	a.openBlocks = append(a.openBlocks, index)

	a.writeEvent(anthropic.EventContentBlockStart, anthropic.ContentBlockStartEvent{
		Type:         anthropic.EventContentBlockStart,
		Index:        index,
		ContentBlock: block,
	})

	return index
}

func (a *anthropicStreamWriter) closeTextBlock() {
	if a.textIndex == -1 {
		return
	}
	a.closeBlock(a.textIndex)
	a.textIndex = -1
}

func (a *anthropicStreamWriter) closeBlock(index int) {
	for i, open := range a.openBlocks {
		if open == index {
			a.openBlocks = append(a.openBlocks[:i], a.openBlocks[i+1:]...)
			break
		}
	}

	a.writeEvent(anthropic.EventContentBlockStop, anthropic.ContentBlockStopEvent{
		Type:  anthropic.EventContentBlockStop,
		Index: index,
	})
}

// finish closes the stream with the terminal message_delta and message_stop
// events. A stream whose only frame was [DONE] still yields a well-formed
// Anthropic stream describing an empty message; a stream with no frames at all
// never reaches here, since the caller reports that as an error.
func (a *anthropicStreamWriter) finish() error {
	if a.finished {
		return a.writeErr
	}
	a.finished = true

	a.ensureStarted()

	// openBlocks is append-only against a monotonic counter, so it is already in
	// ascending index order.
	for len(a.openBlocks) > 0 {
		a.closeBlock(a.openBlocks[0])
	}
	a.textIndex = -1

	stopReason := a.stopReason
	if stopReason == "" {
		stopReason = anthropic.StopReasonEndTurn
	}

	a.writeEvent(anthropic.EventMessageDelta, anthropic.MessageDeltaEvent{
		Type: anthropic.EventMessageDelta,
		Delta: anthropic.MessageDelta{
			StopReason: &stopReason,
		},
		Usage: anthropic.Usage{
			InputTokens:  a.inputTokens,
			OutputTokens: a.outputTokens,
		},
	})

	a.writeEvent(anthropic.EventMessageStop, anthropic.MessageStopEvent{
		Type: anthropic.EventMessageStop,
	})

	return a.writeErr
}

// writeErrorEvent surfaces a mid-stream failure to the client. The status line is
// already sent by this point, so an error event is the only way to report it.
func (a *anthropicStreamWriter) writeErrorEvent(cause error) {
	if a.finished {
		return
	}
	a.finished = true

	a.writeEvent(anthropic.EventError, anthropic.ErrorEvent{
		Type: anthropic.EventError,
		Error: anthropic.Error{
			Type:    anthropic.ErrorTypeAPI,
			Message: cause.Error(),
		},
	})
}

func (a *anthropicStreamWriter) writeEvent(name string, event any) {
	if a.writeErr != nil {
		return
	}

	encoded, err := json.Marshal(event)
	if err != nil {
		a.writeErr = fmt.Errorf("encode %s event: %w", name, err)
		return
	}

	if _, err := fmt.Fprintf(a.w, "event: %s\ndata: %s\n\n", name, encoded); err != nil {
		a.writeErr = err
		return
	}

	if a.flusher != nil {
		a.flusher.Flush()
	}
}
