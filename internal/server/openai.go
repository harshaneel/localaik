package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

func (s *Server) handleOpenAIPassthrough(w http.ResponseWriter, r *http.Request, upstreamURL string) {
	req, err := http.NewRequestWithContext(r.Context(), r.Method, upstreamURL, r.Body)
	if err != nil {
		openaip.WriteError(w, http.StatusInternalServerError, "failed to create upstream request", "server_error")
		return
	}
	req.Header = cloneHeaders(r.Header)

	resp, err := s.client.Do(req)
	if err != nil {
		openaip.WriteError(w, http.StatusBadGateway, fmt.Sprintf("failed to reach upstream: %v", err), "server_error")
		return
	}
	defer resp.Body.Close()

	copyResponseHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)

	if flusher, ok := w.(http.Flusher); ok {
		_, _ = io.Copy(flushWriter{Writer: w, Flusher: flusher}, resp.Body)
		return
	}

	_, _ = io.Copy(w, resp.Body)
}

func (s *Server) handleOpenAIModelsList(w http.ResponseWriter, r *http.Request) {
	s.handleOpenAIPassthrough(w, r, s.upstreamModelsURL)
}

func (s *Server) handleOpenAIModelGet(w http.ResponseWriter, r *http.Request) {
	modelID := strings.TrimPrefix(r.URL.Path, "/v1/models/")
	if modelID == "" || strings.Contains(modelID, "/") {
		openaip.WriteError(w, http.StatusNotFound, "route not found", "invalid_request_error")
		return
	}
	s.handleOpenAIPassthrough(w, r, s.upstreamModelsURL+"/"+modelID)
}

type flushWriter struct {
	Writer  io.Writer
	Flusher http.Flusher
}

func (w flushWriter) Write(p []byte) (int, error) {
	n, err := w.Writer.Write(p)
	if err == nil {
		w.Flusher.Flush()
	}
	return n, err
}

// fetchUpstreamJSON issues a GET to upstreamURL and decodes the JSON response
// into dst. On non-2xx status it returns (status, body, nil) so the caller can
// translate the upstream error. On decode failure it returns (status, body, err)
// — body is the raw upstream payload but is malformed JSON, so callers should
// surface the error rather than the body. No request headers are forwarded;
// this is intentional so that client-side auth (Authorization, X-Goog-Api-Key)
// does not leak to upstream.
func (s *Server) fetchUpstreamJSON(r *http.Request, upstreamURL string, dst any) (int, []byte, error) {
	req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, upstreamURL, nil)
	if err != nil {
		return 0, nil, fmt.Errorf("create upstream request: %w", err)
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return 0, nil, fmt.Errorf("reach upstream: %w", err)
	}
	defer resp.Body.Close()

	body, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return resp.StatusCode, nil, fmt.Errorf("read upstream body: %w", readErr)
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return resp.StatusCode, body, nil
	}
	if err := json.Unmarshal(body, dst); err != nil {
		return resp.StatusCode, body, fmt.Errorf("decode upstream body: %w", err)
	}
	return resp.StatusCode, body, nil
}
