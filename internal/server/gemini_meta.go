package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/harshaneel/localaik/internal/protocol/gemini"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
	"github.com/harshaneel/localaik/internal/translate"
)

func (s *Server) handleGeminiModelsList(w http.ResponseWriter, r *http.Request) {
	var upstream openaip.ModelList
	status, body, err := s.fetchUpstreamJSON(r, s.upstreamModelsURL, &upstream)
	if err != nil {
		gemini.WriteError(w, http.StatusBadGateway, fmt.Sprintf("failed to reach upstream: %v", err))
		return
	}
	if status >= http.StatusBadRequest {
		writeJSON(w, status, translate.OpenAIErrorToGemini(status, body))
		return
	}
	writeJSON(w, http.StatusOK, translate.OpenAIModelListToGemini(upstream))
}

func (s *Server) handleGeminiModelGet(w http.ResponseWriter, r *http.Request) {
	modelName := strings.TrimPrefix(r.URL.Path, "/v1beta/models/")
	if modelName == "" || strings.ContainsAny(modelName, "/:") {
		gemini.WriteError(w, http.StatusNotFound, "route not found")
		return
	}

	var upstream openaip.Model
	status, body, err := s.fetchUpstreamJSON(r, s.upstreamModelsURL+"/"+modelName, &upstream)
	if err != nil {
		gemini.WriteError(w, http.StatusBadGateway, fmt.Sprintf("failed to reach upstream: %v", err))
		return
	}
	if status >= http.StatusBadRequest {
		writeJSON(w, status, translate.OpenAIErrorToGemini(status, body))
		return
	}
	writeJSON(w, http.StatusOK, translate.OpenAIModelToGemini(upstream))
}

func (s *Server) handleGeminiCountTokens(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	var req gemini.CountTokensRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		gemini.WriteError(w, http.StatusBadRequest, fmt.Sprintf("invalid countTokens request: %v", err))
		return
	}

	upstreamPayload, err := json.Marshal(map[string]any{
		"content":     translate.CountTokensTextFromGemini(req.Contents),
		"add_special": false,
	})
	if err != nil {
		gemini.WriteError(w, http.StatusInternalServerError, "failed to serialize upstream request")
		return
	}

	upstreamReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, s.upstreamTokenizeURL, bytes.NewReader(upstreamPayload))
	if err != nil {
		gemini.WriteError(w, http.StatusInternalServerError, "failed to create upstream request")
		return
	}
	upstreamReq.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(upstreamReq)
	if err != nil {
		gemini.WriteError(w, http.StatusBadGateway, fmt.Sprintf("failed to reach upstream: %v", err))
		return
	}
	defer resp.Body.Close()

	body, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		gemini.WriteError(w, http.StatusBadGateway, fmt.Sprintf("failed to read upstream response: %v", readErr))
		return
	}
	if resp.StatusCode >= http.StatusBadRequest {
		writeJSON(w, resp.StatusCode, translate.OpenAIErrorToGemini(resp.StatusCode, body))
		return
	}

	var upstreamResp struct {
		Tokens []any `json:"tokens"`
	}
	if err := json.Unmarshal(body, &upstreamResp); err != nil {
		gemini.WriteError(w, http.StatusBadGateway, fmt.Sprintf("failed to parse upstream response: %v", err))
		return
	}

	writeJSON(w, http.StatusOK, gemini.CountTokensResponse{TotalTokens: len(upstreamResp.Tokens)})
}
