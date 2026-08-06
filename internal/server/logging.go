package server

import (
	"log"
	"net/http"
	"strconv"
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

	// Deferred so a panicking handler still leaves an access line.
	defer func() {
		dur := time.Since(start).Round(time.Microsecond)

		// Health probes hit this every few seconds, so log them only when failing.
		if r.URL.Path == "/health" && rec.status < http.StatusBadRequest {
			return
		}

		method, path := sanitizeLogField(r.Method), sanitizeLogField(r.URL.Path)
		if rec.status >= http.StatusBadRequest {
			l.logger.Printf("localaik  %s %s  %d %s  %s", method, path, rec.status, http.StatusText(rec.status), dur)
			return
		}
		l.logger.Printf("localaik  %s %s  %d  %s", method, path, rec.status, dur)
	}()

	l.next.ServeHTTP(rec, r)
}

// sanitizeLogField quotes a value that carries control characters, so a decoded
// request path cannot forge extra log lines or inject terminal escapes.
func sanitizeLogField(s string) string {
	for _, c := range s {
		if c < 0x20 || c == 0x7f {
			return strconv.Quote(s)
		}
	}
	return s
}

// statusRecorder captures the response status while preserving http.Flusher, so
// streaming responses still flush through the middleware.
type statusRecorder struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

// WriteHeader records only the first status, matching net/http, so the logged
// status is the one the client actually received.
func (s *statusRecorder) WriteHeader(code int) {
	if s.wroteHeader {
		return
	}
	s.wroteHeader = true
	s.status = code
	s.ResponseWriter.WriteHeader(code)
}

// Write locks in the implicit 200, matching net/http, so a later WriteHeader
// cannot make the log disagree with the status the client received.
func (s *statusRecorder) Write(b []byte) (int, error) {
	s.wroteHeader = true
	return s.ResponseWriter.Write(b)
}

func (s *statusRecorder) Flush() {
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
