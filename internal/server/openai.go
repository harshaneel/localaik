package server

import (
	"fmt"
	"io"
	"net/http"

	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

func (s *Server) handleOpenAIChatCompletions(w http.ResponseWriter, r *http.Request) {
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, s.upstreamChatURL, r.Body)
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
