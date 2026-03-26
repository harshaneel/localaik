package translate

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

func WriteGeminiStreamFromOpenAISSE(w http.ResponseWriter, body io.Reader) error {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)

	flusher, _ := w.(http.Flusher)
	reader := bufio.NewScanner(body)
	reader.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var dataLines []string
	flushEvent := func() error {
		if len(dataLines) == 0 {
			return nil
		}
		payload := strings.Join(dataLines, "\n")
		dataLines = nil
		if payload == "[DONE]" {
			return io.EOF
		}

		var chunk openaip.StreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return fmt.Errorf("decode SSE payload: %w", err)
		}

		geminiChunk := OpenAIStreamChunkToGemini(chunk)
		if len(geminiChunk.Candidates) == 0 && geminiChunk.UsageMetadata == nil {
			return nil
		}

		encoded, err := json.Marshal(geminiChunk)
		if err != nil {
			return fmt.Errorf("encode Gemini chunk: %w", err)
		}

		if _, err := io.WriteString(w, "data:"); err != nil {
			return err
		}
		if _, err := w.Write(encoded); err != nil {
			return err
		}
		if _, err := io.WriteString(w, "\n\n"); err != nil {
			return err
		}
		if flusher != nil {
			flusher.Flush()
		}
		return nil
	}

	for reader.Scan() {
		line := reader.Text()
		if line == "" {
			err := flushEvent()
			if err == io.EOF {
				break
			}
			if err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}

	if err := reader.Err(); err != nil {
		return err
	}

	if err := flushEvent(); err != nil && err != io.EOF {
		return err
	}

	return nil
}
