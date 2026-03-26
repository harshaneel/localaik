//go:build docker_integration

package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	genaisdk "google.golang.org/genai"
)

var (
	imageServerReadyOnce sync.Once
	imageServerReadyErr  error
)

func TestImageHealth(t *testing.T) {
	waitForImageServer(t)

	baseURL := imageBaseURL()

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(baseURL + "/health")
	if err != nil {
		t.Fatalf("health request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("health status = %d, want 200; body=%s", resp.StatusCode, string(body))
	}
}

func TestImageOpenAISDKChat(t *testing.T) {
	waitForImageServer(t)

	baseURL := imageBaseURL()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client := openaisdk.NewClient(
		option.WithBaseURL(baseURL+"/v1/"),
		option.WithAPIKey("test"),
	)

	resp, err := client.Chat.Completions.New(ctx, openaisdk.ChatCompletionNewParams{
		Messages: []openaisdk.ChatCompletionMessageParamUnion{
			openaisdk.UserMessage("Reply with a short greeting."),
		},
		Model: openaisdk.ChatModelGPT4o,
	})
	if err != nil {
		t.Fatalf("Chat.Completions.New returned error: %v", err)
	}
	if len(resp.Choices) == 0 || strings.TrimSpace(resp.Choices[0].Message.Content) == "" {
		t.Fatalf("unexpected OpenAI response: %#v", resp)
	}
}

func TestImageGenAISDKGenerateContent(t *testing.T) {
	waitForImageServer(t)

	baseURL := imageBaseURL()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client, err := genaisdk.NewClient(ctx, &genaisdk.ClientConfig{
		APIKey:  "test",
		Backend: genaisdk.BackendGeminiAPI,
		HTTPOptions: genaisdk.HTTPOptions{
			BaseURL: baseURL,
		},
	})
	if err != nil {
		t.Fatalf("genai.NewClient returned error: %v", err)
	}

	resp, err := client.Models.GenerateContent(ctx, "gemini-test", genaisdk.Text("Reply with a short greeting."), nil)
	if err != nil {
		t.Fatalf("GenerateContent returned error: %v", err)
	}
	if strings.TrimSpace(resp.Text()) == "" {
		t.Fatalf("unexpected Gemini response: %#v", resp)
	}
	fmt.Println(resp.Text())
}

func TestImageGenAISDKStructuredOutputFromPDF(t *testing.T) {
	waitForImageServer(t)

	baseURL := imageBaseURL()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	client, err := genaisdk.NewClient(ctx, &genaisdk.ClientConfig{
		APIKey:  "test",
		Backend: genaisdk.BackendGeminiAPI,
		HTTPOptions: genaisdk.HTTPOptions{
			BaseURL: baseURL,
		},
	})
	if err != nil {
		t.Fatalf("genai.NewClient returned error: %v", err)
	}

	docBytes := mustReadStructuredOutputPDF(t)
	config := &genaisdk.GenerateContentConfig{
		Temperature:      genaisdk.Ptr[float32](0),
		MaxOutputTokens:  128,
		ResponseMIMEType: "application/json",
		ResponseSchema: &genaisdk.Schema{
			Type: genaisdk.TypeObject,
			Properties: map[string]*genaisdk.Schema{
				"name": {Type: genaisdk.TypeString},
				"city": {Type: genaisdk.TypeString},
				"year": {Type: genaisdk.TypeString},
			},
			Required: []string{"name", "city", "year"},
		},
	}

	resp, err := client.Models.GenerateContent(
		ctx,
		"gemini-test",
		[]*genaisdk.Content{{
			Parts: []*genaisdk.Part{
				{Text: "Extract the exact values after the labels NAME, CITY, and YEAR from this document. Return JSON with the required keys name, city, and year."},
				genaisdk.NewPartFromBytes(docBytes, "application/pdf"),
			},
		}},
		config,
	)
	if err != nil {
		t.Fatalf("GenerateContent returned error: %v", err)
	}

	t.Logf("structured output: %s", resp.Text())

	var got struct {
		Name string `json:"name"`
		City string `json:"city"`
		Year string `json:"year"`
	}
	if err := json.Unmarshal([]byte(resp.Text()), &got); err != nil {
		t.Fatalf("structured output was not valid JSON: %v; raw=%q", err, resp.Text())
	}

	if normalized := normalizeStructuredField(got.Name); normalized != "alice" {
		t.Fatalf("name = %q, want ALICE-like value", got.Name)
	}
	if normalized := normalizeStructuredField(got.City); normalized != "boston" {
		t.Fatalf("city = %q, want BOSTON-like value", got.City)
	}
	if normalized := normalizeStructuredField(got.Year); normalized != "2026" {
		t.Fatalf("year = %q, want 2026-like value", got.Year)
	}
}

func imageBaseURL() string {
	if value := os.Getenv("LOCALAIK_BASE_URL"); value != "" {
		return strings.TrimRight(value, "/")
	}
	return "http://127.0.0.1:18090"
}

func waitForImageServer(t *testing.T) {
	t.Helper()

	baseURL := imageBaseURL()
	imageServerReadyOnce.Do(func() {
		deadline := time.Now().Add(3 * time.Minute)
		client := &http.Client{Timeout: 5 * time.Second}

		var lastErr error
		lastStatus := 0

		for time.Now().Before(deadline) {
			resp, err := client.Get(baseURL + "/health")
			if err == nil {
				_, _ = io.Copy(io.Discard, resp.Body)
				resp.Body.Close()

				if resp.StatusCode == http.StatusOK {
					return
				}

				lastStatus = resp.StatusCode
				lastErr = nil
			} else {
				lastErr = err
			}

			time.Sleep(2 * time.Second)
		}

		if lastErr != nil {
			imageServerReadyErr = fmt.Errorf("image server at %s did not become healthy within 180s: %w", baseURL, lastErr)
			return
		}

		imageServerReadyErr = fmt.Errorf("image server at %s did not become healthy within 180s: last status %d", baseURL, lastStatus)
	})

	if imageServerReadyErr != nil {
		t.Fatal(imageServerReadyErr)
	}
}

func mustReadStructuredOutputPDF(t *testing.T) []byte {
	t.Helper()

	docPath := filepath.Join(t.TempDir(), "structured-output.pdf")
	if err := os.WriteFile(docPath, buildSimplePDF([]string{
		"NAME: ALICE",
		"CITY: BOSTON",
		"YEAR: 2026",
	}), 0o600); err != nil {
		t.Fatalf("write test pdf: %v", err)
	}

	docBytes, err := os.ReadFile(docPath)
	if err != nil {
		t.Fatalf("read test pdf: %v", err)
	}
	return docBytes
}

func buildSimplePDF(lines []string) []byte {
	var stream bytes.Buffer
	stream.WriteString("BT\n/F1 28 Tf\n72 720 Td\n")
	for i, line := range lines {
		if i > 0 {
			stream.WriteString("0 -36 Td\n")
		}
		fmt.Fprintf(&stream, "(%s) Tj\n", escapePDFText(line))
	}
	stream.WriteString("ET\n")

	objects := []string{
		"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
		"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
		"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
		fmt.Sprintf("4 0 obj\n<< /Length %d >>\nstream\n%sendstream\nendobj\n", stream.Len(), stream.String()),
		"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
	}

	var pdf bytes.Buffer
	pdf.WriteString("%PDF-1.4\n")

	offsets := []int{0}
	for _, obj := range objects {
		offsets = append(offsets, pdf.Len())
		pdf.WriteString(obj)
	}

	xrefOffset := pdf.Len()
	fmt.Fprintf(&pdf, "xref\n0 %d\n", len(offsets))
	pdf.WriteString("0000000000 65535 f \n")
	for _, offset := range offsets[1:] {
		fmt.Fprintf(&pdf, "%010d 00000 n \n", offset)
	}
	fmt.Fprintf(&pdf, "trailer\n<< /Root 1 0 R /Size %d >>\nstartxref\n%d\n%%%%EOF\n", len(offsets), xrefOffset)

	return pdf.Bytes()
}

func escapePDFText(value string) string {
	replacer := strings.NewReplacer(`\`, `\\`, "(", `\(`, ")", `\)`)
	return replacer.Replace(value)
}

func normalizeStructuredField(value string) string {
	var normalized strings.Builder
	for _, r := range strings.ToLower(value) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			normalized.WriteRune(r)
		}
	}
	return normalized.String()
}
