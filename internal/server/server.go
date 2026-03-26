package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"path"
	"strings"

	"github.com/harshaneel/localaik/internal/pdf"
	"github.com/harshaneel/localaik/internal/protocol/gemini"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

type Config struct {
	UpstreamBaseURL string
	HTTPClient      *http.Client
	PDFRenderer     pdf.Renderer
}

type Server struct {
	client            *http.Client
	pdfRenderer       pdf.Renderer
	upstreamChatURL   string
	upstreamHealthURL string
}

func New(cfg Config) (*Server, error) {
	if cfg.UpstreamBaseURL == "" {
		cfg.UpstreamBaseURL = "http://127.0.0.1:8080/v1"
	}

	parsed, err := url.Parse(cfg.UpstreamBaseURL)
	if err != nil {
		return nil, fmt.Errorf("parse upstream URL: %w", err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return nil, errors.New("upstream URL must include scheme and host")
	}

	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{}
	}

	renderer := cfg.PDFRenderer
	if renderer == nil {
		renderer = pdf.NewExecRenderer("pdftoppm")
	}

	return &Server{
		client:            client,
		pdfRenderer:       renderer,
		upstreamChatURL:   resolveURLPath(parsed, "chat/completions"),
		upstreamHealthURL: deriveHealthURL(parsed),
	}, nil
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch {
	case r.Method == http.MethodGet && r.URL.Path == "/health":
		s.handleHealth(w, r)
	case r.Method == http.MethodPost && r.URL.Path == "/v1/chat/completions":
		s.handleOpenAIChatCompletions(w, r)
	case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/v1beta/models/") && strings.HasSuffix(r.URL.Path, ":generateContent"):
		s.handleGeminiGenerateContent(w, r, false)
	case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/v1beta/models/") && strings.HasSuffix(r.URL.Path, ":streamGenerateContent"):
		s.handleGeminiGenerateContent(w, r, true)
	default:
		s.handleNotFound(w, r)
	}
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, s.upstreamHealthURL, nil)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"status": "error"})
		return
	}

	resp, err := s.client.Do(req)
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "unhealthy"})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "unhealthy"})
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleNotFound(w http.ResponseWriter, r *http.Request) {
	switch {
	case strings.HasPrefix(r.URL.Path, "/v1beta/"):
		gemini.WriteError(w, http.StatusNotFound, "route not found")
	case strings.HasPrefix(r.URL.Path, "/v1/"):
		openaip.WriteError(w, http.StatusNotFound, "route not found", "invalid_request_error")
	default:
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
	}
}

func resolveURLPath(base *url.URL, extra string) string {
	clone := *base
	basePath := strings.TrimSuffix(clone.Path, "/")
	clone.Path = joinURLPath(basePath, extra)
	return clone.String()
}

func deriveHealthURL(base *url.URL) string {
	clone := *base
	basePath := strings.TrimSuffix(clone.Path, "/")
	basePath = strings.TrimSuffix(basePath, "/v1")
	clone.Path = joinURLPath(basePath, "health")
	return clone.String()
}

func joinURLPath(parts ...string) string {
	joined := path.Join(parts...)
	if !strings.HasPrefix(joined, "/") {
		joined = "/" + joined
	}
	return joined
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func cloneHeaders(src http.Header) http.Header {
	dst := make(http.Header, len(src))
	for key, values := range src {
		if strings.EqualFold(key, "Authorization") ||
			strings.EqualFold(key, "Content-Length") ||
			strings.EqualFold(key, "Host") ||
			strings.EqualFold(key, "X-Goog-Api-Key") {
			continue
		}
		copied := make([]string, len(values))
		copy(copied, values)
		dst[key] = copied
	}
	return dst
}

func copyResponseHeaders(dst, src http.Header) {
	for key, values := range src {
		if strings.EqualFold(key, "Content-Length") {
			continue
		}
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}
