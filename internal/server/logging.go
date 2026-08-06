package server

import (
	"log"
	"net/http"
	"time"
)

// WithRequestLog wraps next so each request is logged as one access line:
// method, path, status and latency. Headers and bodies are never logged, so no
// credential or prompt content can leak through here. A nil logger disables
// logging and returns next unchanged.
func WithRequestLog(next http.Handler, logger *log.Logger) http.Handler {
	if logger == nil {
		return next
	}
	return &requestLogger{next: next, logger: logger}
}

type requestLogger struct {
	next   http.Handler
	logger *log.Logger
}

func (l *requestLogger) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
	l.next.ServeHTTP(rec, r)
	dur := time.Since(start).Round(time.Microsecond)

	// Health probes hit this every few seconds, so log them only when failing.
	if r.URL.Path == "/health" && rec.status < http.StatusBadRequest {
		return
	}

	if rec.status >= http.StatusBadRequest {
		l.logger.Printf("localaik  %s %s  %d %s  %s", r.Method, r.URL.Path, rec.status, http.StatusText(rec.status), dur)
		return
	}
	l.logger.Printf("localaik  %s %s  %d  %s", r.Method, r.URL.Path, rec.status, dur)
}

// statusRecorder captures the response status while preserving http.Flusher, so
// streaming responses still flush through the middleware.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (s *statusRecorder) WriteHeader(code int) {
	s.status = code
	s.ResponseWriter.WriteHeader(code)
}

func (s *statusRecorder) Flush() {
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
