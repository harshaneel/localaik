package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// upstreamError pairs a failure talking to llama.cpp with the status to report.
type upstreamError struct {
	status  int
	message string
}

func (e *upstreamError) Error() string { return e.message }

// countUpstreamTokens tokenizes text via llama.cpp's /tokenize endpoint. A non-2xx
// upstream status is returned with the raw body and a nil error, for the caller to
// translate into its own shape.
func (s *Server) countUpstreamTokens(r *http.Request, text string) (count int, upstreamStatus int, upstreamBody []byte, err *upstreamError) {
	payload, marshalErr := json.Marshal(map[string]any{
		"content":     text,
		"add_special": false,
	})
	if marshalErr != nil {
		return 0, 0, nil, &upstreamError{http.StatusInternalServerError, "failed to serialize upstream request"}
	}

	req, reqErr := http.NewRequestWithContext(r.Context(), http.MethodPost, s.upstreamTokenizeURL, bytes.NewReader(payload))
	if reqErr != nil {
		return 0, 0, nil, &upstreamError{http.StatusInternalServerError, "failed to create upstream request"}
	}
	req.Header.Set("Content-Type", "application/json")

	resp, doErr := s.client.Do(req)
	if doErr != nil {
		return 0, 0, nil, &upstreamError{http.StatusBadGateway, fmt.Sprintf("failed to reach upstream %s", s.upstreamDisplay)}
	}
	defer resp.Body.Close()

	body, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return 0, resp.StatusCode, nil, &upstreamError{http.StatusBadGateway, fmt.Sprintf("failed to read upstream response: %v", readErr)}
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return 0, resp.StatusCode, body, nil
	}

	var upstream struct {
		Tokens []any `json:"tokens"`
	}
	if unmarshalErr := json.Unmarshal(body, &upstream); unmarshalErr != nil {
		return 0, resp.StatusCode, body, &upstreamError{http.StatusBadGateway, fmt.Sprintf("failed to parse upstream response: %v", unmarshalErr)}
	}

	return len(upstream.Tokens), resp.StatusCode, body, nil
}
