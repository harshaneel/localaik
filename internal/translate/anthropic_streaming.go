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
// sequence requires. Block indices start at 0 and increase by exactly one per
// block, which the SDK accumulators enforce on content_block_start.
//
// Text streams through as it arrives. Tool calls do not: a content_block_start
// has to commit to the call's id and name, and upstream may still be revealing
// them. OpenAI spreads a tool call's identity and arguments across an
// unspecified number of deltas, and its `index` is `omitempty`, so an upstream
// that never sends one is indistinguishable from one that always sends 0.
// Committing early and then telling a continuation from a new call by comparing
// identity cannot be made correct: whichever way the comparison is tuned, some
// upstream either merges two calls into one block or splits one across two.
//
// So each tool call is buffered and emitted as a complete unit:
// content_block_start / one input_json_delta / content_block_stop.
//
// What ends a call is the subtle part. Its arguments becoming parseable is *not*
// the signal, however tempting: upstream may still send further deltas for the
// same index, whether a padding delta with empty arguments or identity that
// trails the arguments. Treating parseability as the end frees the slot too
// early and the next delta fabricates a second, usually nameless, call. The
// stream gives exactly three sound signals that a pending call is over:
//
//   - a tool delta for a higher index, since upstream does not go back;
//   - text, which never appears mid-way through a call's arguments;
//   - the end of the stream.
//
// Buffering also means a block is never left holding a half-written fragment,
// which is what the SDK accumulators fail to marshal when a block stops.
//
// The cost is that tool arguments reach the client in one input_json_delta
// rather than several. The SDK accumulators build the same result either way.
type anthropicStreamWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	model   string

	started      bool
	finished     bool
	sawFrame     bool
	nextIndex    int
	textIndex    int
	pendingTools map[int]*pendingToolCall
	toolUseIDs   *toolUseIDs
	messageID    string
	stopReason   string
	inputTokens  int
	outputTokens int
	writeErr     error
}

// pendingToolCall accumulates one upstream tool call until it can be emitted.
// upstreamID is the id as upstream sent it; the id emitted downstream may be
// synthesized from the name when upstream sends none.
type pendingToolCall struct {
	upstreamID string
	name       string
	arguments  string
}

// midObject reports whether the arguments seen so far are an unfinished JSON
// object. Such a call cannot have ended, so it is never flushed by a signal that
// merely suggests it might have.
//
// The object requirement is not incidental: json.Valid alone accepts bare
// scalars, so a malformed `arguments` of "1" would look finished and be
// forwarded as a tool input that is not an object.
func (p *pendingToolCall) midObject() bool {
	return p.arguments != "" && !isJSONObject(p.arguments)
}

// startsFreshArguments reports whether a delta begins a new JSON object while this
// call's arguments are already whole, which means upstream has moved on to another
// call at the same index.
//
// Upstreams that omit `index` put every call at 0, so this is the only thing
// separating two of them. It deliberately keys on the argument stream rather than
// on the id or name: identity can arrive at any point in a call, so comparing it
// mistakes late identity for a new call. And because a new object is required, a
// trailing delta that carries no arguments — a padding delta, or identity arriving
// after the arguments completed — reads as the continuation it is.
func (p *pendingToolCall) startsFreshArguments(arguments string) bool {
	if p.arguments == "" || p.midObject() {
		return false
	}
	return strings.HasPrefix(strings.TrimSpace(arguments), "{")
}

// observeName folds a name fragment into the call. Upstreams variously send the
// name once, repeat it in full on every delta, or (in principle) split it, so a
// repeat is ignored and anything else is appended.
func (p *pendingToolCall) observeName(name string) {
	switch {
	case name == "" || name == p.name:
		return
	case p.name == "":
		p.name = name
	default:
		p.name += name
	}
}

func newAnthropicStreamWriter(w http.ResponseWriter, model string) *anthropicStreamWriter {
	flusher, _ := w.(http.Flusher)
	return &anthropicStreamWriter{
		w:            w,
		flusher:      flusher,
		model:        model,
		textIndex:    -1,
		pendingTools: make(map[int]*pendingToolCall),
		toolUseIDs:   newToolUseIDs(),
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
	// Any tool call upstream had in flight is finished by the time it emits text,
	// so it is emitted first and the blocks keep upstream's order.
	a.flushSettledToolCalls()

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

	arguments := ""
	if call.Function != nil {
		arguments = call.Function.Arguments
	}

	// Upstream moving to a higher index ends the lower calls, unless one is still
	// mid-object: some upstreams start a later index and then come back to finish
	// an earlier one.
	a.flushSettledToolCallsBelow(call.Index)

	pending, ok := a.pendingTools[call.Index]
	if ok && pending.startsFreshArguments(arguments) {
		a.flushToolCall(call.Index)
		ok = false
	}
	if !ok {
		pending = &pendingToolCall{}
		a.pendingTools[call.Index] = pending
	}

	if call.ID != "" {
		pending.upstreamID = call.ID
	}
	if call.Function != nil {
		pending.observeName(call.Function.Name)
		pending.arguments += arguments
	}
}

// flushSettledToolCalls emits every pending call that is not mid-object, lowest
// index first. Called when text arrives: text never appears part-way through a
// call's arguments, so anything pending is over.
func (a *anthropicStreamWriter) flushSettledToolCalls() {
	a.flushSettledToolCallsBelow(-1)
}

// flushSettledToolCallsBelow emits pending calls that are not mid-object, lowest
// index first, and stops at the first index that is at or above limit or is still
// mid-object. Stopping rather than skipping keeps the emitted blocks in upstream's
// index order. A negative limit means no bound.
func (a *anthropicStreamWriter) flushSettledToolCallsBelow(limit int) {
	for {
		index, ok := a.lowestPendingToolIndex()
		if !ok {
			return
		}
		if limit >= 0 && index >= limit {
			return
		}
		if a.pendingTools[index].midObject() {
			return
		}
		a.flushToolCall(index)
	}
}

func (a *anthropicStreamWriter) lowestPendingToolIndex() (int, bool) {
	lowest := 0
	found := false
	for index := range a.pendingTools {
		if !found || index < lowest {
			lowest = index
			found = true
		}
	}
	return lowest, found
}

// flushToolCall emits one buffered call as a complete content block. Arguments
// that never became parseable are replaced with an empty object, so a block is
// never handed to the client holding JSON it cannot decode.
func (a *anthropicStreamWriter) flushToolCall(upstreamIndex int) {
	pending, ok := a.pendingTools[upstreamIndex]
	if !ok {
		return
	}
	delete(a.pendingTools, upstreamIndex)

	// Whatever text came before is complete once a tool call is emitted.
	a.closeTextBlock()

	// Arguments that never became a whole object are replaced, so a block is never
	// handed to the client holding input it cannot decode.
	input := string(anthropicToolInput(pending.arguments))

	index := a.startBlock(anthropic.ToolUseStreamBlock(
		a.toolUseIDs.assign(pending.upstreamID, pending.name),
		pending.name,
	))

	// The Messages API always sends at least one input_json_delta per tool_use, so
	// one is emitted even for an empty object rather than leaving the client to
	// infer the arguments from the absence of a delta.
	a.writeEvent(anthropic.EventContentBlockDelta, anthropic.ContentBlockDeltaEvent{
		Type:  anthropic.EventContentBlockDelta,
		Index: index,
		Delta: anthropic.BlockDelta{
			Type:        anthropic.DeltaTypeInputJSON,
			PartialJSON: input,
		},
	})

	a.closeBlock(index)
}

// startBlock emits content_block_start for a new block and returns its index.
func (a *anthropicStreamWriter) startBlock(block anthropic.StreamContentBlock) int {
	index := a.nextIndex
	a.nextIndex++

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

	// Any call whose arguments never parsed still becomes a block, so a tool call
	// upstream cut short is visible to the client rather than dropped.
	for {
		index, ok := a.lowestPendingToolIndex()
		if !ok {
			break
		}
		a.flushToolCall(index)
	}
	a.closeTextBlock()

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
