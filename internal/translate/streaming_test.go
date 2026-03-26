package translate

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestWriteGeminiStreamFromOpenAISSE(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"content":" world"}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	recorder := httptest.NewRecorder()
	if err := WriteGeminiStreamFromOpenAISSE(recorder, strings.NewReader(sse)); err != nil {
		t.Fatalf("WriteGeminiStreamFromOpenAISSE returned error: %v", err)
	}

	if contentType := recorder.Header().Get("Content-Type"); contentType != "text/event-stream" {
		t.Fatalf("content-type = %q, want text/event-stream", contentType)
	}

	got := recorder.Body.String()
	expectedParts := []string{
		`data:{"candidates":[{"content":{"role":"model","parts":[{"text":"hello"}]}}]}`,
		`data:{"candidates":[{"content":{"role":"model","parts":[{"text":" world"}]}}]}`,
		`data:{"candidates":[{"finishReason":"STOP"}]}`,
	}
	for _, expected := range expectedParts {
		if !strings.Contains(got, expected) {
			t.Fatalf("stream output missing %q\nbody=%s", expected, got)
		}
	}
}
