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

// WriteAnthropicStreamFromOpenAISSE translates an upstream OpenAI SSE stream into
// the Anthropic Messages event stream. See specs/design.md for the event sequence
// and the tool-call accumulation rules.
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

	// A 200 with no frames at all did not stream; reporting success would hide it.
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
// Text streams through as it arrives. Tool calls are accumulated and emitted at
// the end of the stream, each as a complete unit: content_block_start / one
// input_json_delta / content_block_stop.
//
// Accumulation follows OpenAI's streaming contract exactly, which is what every
// OpenAI client does: `tool_calls[].index` identifies the call, `arguments`
// fragments concatenate in arrival order, and `id`/`name` are sent once. An
// upstream that omits the index means index 0, so all its fragments belong to one
// call. "Two different calls at one index" is not something the contract permits,
// and trying to infer it from ids, names or argument shapes is what earlier
// versions of this file did — every such heuristic ended up splitting a call that
// was streaming normally, or merging two that were not.
//
// The one concession to non-conforming upstreams is dropping an exact restatement
// of a call already held, since gateways do resend the whole tool_calls array on a
// later chunk. Concatenating a call's arguments with themselves would leave the
// block unparseable, and the repeat carries no information either way.
//
// Deferring to end-of-stream also means a block is never stopped holding a
// half-written fragment, which is what the SDK accumulators fail to marshal.
//
// The cost is that a tool call's arguments reach the client in one
// input_json_delta rather than several. The SDK accumulators build the same result
// either way.
type anthropicStreamWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	model   string

	started          bool
	finished         bool
	sawFrame         bool
	nextIndex        int
	textIndex        int
	pendingTools     map[int]*pendingToolCall
	flushedTools     map[int]bool
	toolUseIDs       *toolUseIDs
	messageID        string
	finishReason     string
	emittedToolBlock bool
	inputTokens      int
	outputTokens     int
	writeErr         error
}

// pendingToolCall accumulates one upstream tool call until it can be emitted.
// upstreamID is the id as upstream sent it; the id emitted downstream may be
// synthesized from the name when upstream sends none.
type pendingToolCall struct {
	upstreamID string
	name       string
	arguments  string
}

// restates reports whether a delta repeats what this call already holds.
func (p *pendingToolCall) restates(id, name, arguments string) bool {
	if !identityMatches(id, p.upstreamID) || !identityMatches(name, p.name) {
		return false
	}
	trimmed := strings.TrimSpace(arguments)
	return trimmed != "" && trimmed == strings.TrimSpace(p.arguments)
}

// identityMatches treats an empty incoming value as consistent with any recorded one.
func identityMatches(incoming, recorded string) bool {
	return incoming == "" || incoming == recorded
}

// observe records the id and name, first non-empty value winning.
func (p *pendingToolCall) observe(id, name string) {
	if p.upstreamID == "" {
		p.upstreamID = id
	}
	if p.name == "" {
		p.name = name
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
		flushedTools: make(map[int]bool),
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
		if choice.FinishReason != "" {
			a.finishReason = choice.FinishReason
		}
	}

	if chunk.Usage != nil {
		a.inputTokens = chunk.Usage.PromptTokens
		a.outputTokens = chunk.Usage.CompletionTokens
	}

	return a.writeErr
}

// primaryStreamChoice picks the lowest-index choice, which the message follows.
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
	a.flushToolCalls()

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

	arguments, name := "", ""
	if call.Function != nil {
		arguments = call.Function.Arguments
		name = call.Function.Name
	}

	// A delta with no arguments for a call already emitted adds nothing.
	if a.flushedTools[call.Index] && strings.TrimSpace(arguments) == "" {
		return
	}

	pending, ok := a.pendingTools[call.Index]
	if !ok {
		pending = &pendingToolCall{}
		a.pendingTools[call.Index] = pending
	}

	if pending.restates(call.ID, name, arguments) {
		return
	}

	pending.observe(call.ID, name)
	pending.arguments += arguments
}

// flushToolCalls emits every accumulated call, lowest upstream index first.
func (a *anthropicStreamWriter) flushToolCalls() {
	for {
		index, ok := a.lowestPendingToolIndex()
		if !ok {
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

// flushToolCall emits one accumulated call as a complete content block.
func (a *anthropicStreamWriter) flushToolCall(upstreamIndex int) {
	pending, ok := a.pendingTools[upstreamIndex]
	if !ok {
		return
	}
	delete(a.pendingTools, upstreamIndex)
	a.flushedTools[upstreamIndex] = true

	// A nameless tool call cannot be invoked; see the README's known differences.
	if pending.name == "" {
		return
	}

	// Whatever text came before is complete once a tool call is emitted.
	a.closeTextBlock()

	// Arguments that never became a whole object are replaced, so a block is never
	// handed to the client holding input it cannot decode.
	input := string(anthropicToolInput(pending.arguments))

	a.emittedToolBlock = true
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

// finish writes the terminal message_delta and message_stop events.
func (a *anthropicStreamWriter) finish() error {
	if a.finished {
		return a.writeErr
	}
	a.finished = true

	a.ensureStarted()

	a.flushToolCalls()
	a.closeTextBlock()

	stopReason := AnthropicStopReason(a.finishReason, a.emittedToolBlock)
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
