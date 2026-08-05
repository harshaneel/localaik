package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/harshaneel/localaik/internal/protocol/anthropic"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
	"github.com/harshaneel/localaik/internal/translate"
)

func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	var req anthropic.MessagesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		anthropic.WriteError(w, http.StatusBadRequest, fmt.Sprintf("invalid Anthropic request: %v", err))
		return
	}

	openAIReq, err := translate.AnthropicRequestToOpenAI(r.Context(), req, s.pdfRenderer)
	if err != nil {
		anthropic.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}
	// Anthropic signals streaming with a body field, not a distinct route.
	openAIReq.Stream = req.Stream

	payload, err := json.Marshal(openAIReq)
	if err != nil {
		anthropic.WriteError(w, http.StatusInternalServerError, "failed to serialize upstream request")
		return
	}

	upstreamReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, s.upstreamChatURL, bytes.NewReader(payload))
	if err != nil {
		anthropic.WriteError(w, http.StatusInternalServerError, "failed to create upstream request")
		return
	}
	upstreamReq.Header.Set("Content-Type", "application/json")
	if req.Stream {
		upstreamReq.Header.Set("Accept", "text/event-stream")
	}

	resp, err := s.client.Do(upstreamReq)
	if err != nil {
		anthropic.WriteError(w, http.StatusBadGateway, fmt.Sprintf("failed to reach upstream %s", s.upstreamDisplay))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		body, _ := io.ReadAll(resp.Body)
		writeJSON(w, resp.StatusCode, translate.OpenAIErrorToAnthropic(resp.StatusCode, body))
		return
	}

	if req.Stream {
		_ = translate.WriteAnthropicStreamFromOpenAISSE(w, resp.Body, req.Model)
		return
	}

	var openAIResp openaip.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&openAIResp); err != nil {
		anthropic.WriteError(w, http.StatusBadGateway, fmt.Sprintf("failed to parse upstream response: %v", err))
		return
	}

	writeJSON(w, http.StatusOK, translate.OpenAIResponseToAnthropic(openAIResp, req.Model))
}

func (s *Server) handleAnthropicCountTokens(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	var req anthropic.MessagesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		anthropic.WriteError(w, http.StatusBadRequest, fmt.Sprintf("invalid count_tokens request: %v", err))
		return
	}
	if len(req.Messages) == 0 {
		anthropic.WriteError(w, http.StatusBadRequest, "messages: at least one message is required")
		return
	}

	count, upstreamStatus, upstreamBody, upstreamErr := s.countUpstreamTokens(r, translate.CountTokensTextFromAnthropic(req))
	if upstreamErr != nil {
		anthropic.WriteError(w, upstreamErr.status, upstreamErr.Error())
		return
	}
	if upstreamStatus >= http.StatusBadRequest {
		writeJSON(w, upstreamStatus, translate.OpenAIErrorToAnthropic(upstreamStatus, upstreamBody))
		return
	}

	writeJSON(w, http.StatusOK, anthropic.CountTokensResponse{InputTokens: count})
}
