package server

import (
	"encoding/json"
	"fmt"
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
		gemini.WriteError(w, http.StatusBadGateway, fmt.Sprintf("failed to reach upstream %s", s.upstreamDisplay))
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
		gemini.WriteError(w, http.StatusBadGateway, fmt.Sprintf("failed to reach upstream %s", s.upstreamDisplay))
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

	count, upstreamStatus, upstreamBody, upstreamErr := s.countUpstreamTokens(r, translate.CountTokensTextFromGemini(req.Contents))
	if upstreamErr != nil {
		gemini.WriteError(w, upstreamErr.status, upstreamErr.Error())
		return
	}
	if upstreamStatus >= http.StatusBadRequest {
		writeJSON(w, upstreamStatus, translate.OpenAIErrorToGemini(upstreamStatus, upstreamBody))
		return
	}

	writeJSON(w, http.StatusOK, gemini.CountTokensResponse{TotalTokens: count})
}
