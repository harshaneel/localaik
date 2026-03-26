package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/harshaneel/localaik/internal/protocol/gemini"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
	"github.com/harshaneel/localaik/internal/translate"
)

func (s *Server) handleGeminiGenerateContent(w http.ResponseWriter, r *http.Request, stream bool) {
	defer r.Body.Close()

	var geminiReq gemini.GenerateContentRequest
	if err := json.NewDecoder(r.Body).Decode(&geminiReq); err != nil {
		gemini.WriteError(w, http.StatusBadRequest, fmt.Sprintf("invalid Gemini request: %v", err))
		return
	}

	openAIReq, err := translate.GeminiRequestToOpenAI(r.Context(), geminiReq, s.pdfRenderer)
	if err != nil {
		gemini.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}
	openAIReq.Stream = stream

	payload, err := json.Marshal(openAIReq)
	if err != nil {
		gemini.WriteError(w, http.StatusInternalServerError, "failed to serialize upstream request")
		return
	}

	upstreamReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, s.upstreamChatURL, bytes.NewReader(payload))
	if err != nil {
		gemini.WriteError(w, http.StatusInternalServerError, "failed to create upstream request")
		return
	}
	upstreamReq.Header.Set("Content-Type", "application/json")
	if stream {
		upstreamReq.Header.Set("Accept", "text/event-stream")
	}

	resp, err := s.client.Do(upstreamReq)
	if err != nil {
		gemini.WriteError(w, http.StatusBadGateway, fmt.Sprintf("failed to reach upstream: %v", err))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		body, _ := io.ReadAll(resp.Body)
		writeJSON(w, resp.StatusCode, translate.OpenAIErrorToGemini(resp.StatusCode, body))
		return
	}

	if stream {
		_ = translate.WriteGeminiStreamFromOpenAISSE(w, resp.Body)
		return
	}

	var openAIResp openaip.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&openAIResp); err != nil {
		gemini.WriteError(w, http.StatusBadGateway, fmt.Sprintf("failed to parse upstream response: %v", err))
		return
	}

	writeJSON(w, http.StatusOK, translate.OpenAIResponseToGemini(openAIResp))
}
