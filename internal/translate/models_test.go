package translate

import (
	"testing"

	"github.com/harshaneel/localaik/internal/protocol/gemini"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

func TestOpenAIModelListToGemini(t *testing.T) {
	in := openaip.ModelList{
		Object: "list",
		Data: []openaip.Model{
			{ID: "gemma-3-4b", Object: "model", OwnedBy: "llama.cpp"},
			{ID: "gemma-3-12b", Object: "model"},
		},
	}

	got := OpenAIModelListToGemini(in)

	if len(got.Models) != 2 {
		t.Fatalf("models = %#v, want 2 entries", got.Models)
	}
	if got.Models[0].Name != "models/gemma-3-4b" {
		t.Fatalf("model[0].Name = %q, want models/gemma-3-4b", got.Models[0].Name)
	}
	if got.Models[0].DisplayName != "gemma-3-4b" {
		t.Fatalf("model[0].DisplayName = %q, want gemma-3-4b", got.Models[0].DisplayName)
	}
	if len(got.Models[0].SupportedGenerationMethods) != 3 {
		t.Fatalf("model[0].SupportedGenerationMethods = %#v", got.Models[0].SupportedGenerationMethods)
	}
}

func TestOpenAIModelToGeminiSingle(t *testing.T) {
	got := OpenAIModelToGemini(openaip.Model{ID: "gemma-3-4b"})
	if got.Name != "models/gemma-3-4b" || got.DisplayName != "gemma-3-4b" {
		t.Fatalf("unexpected mapping: %#v", got)
	}
}

func TestCountTokensTextFromGemini(t *testing.T) {
	tests := []struct {
		name string
		in   []gemini.Content
		want string
	}{
		{
			name: "single text part",
			in: []gemini.Content{
				{Parts: []gemini.Part{{Text: "hello"}}},
			},
			want: "hello",
		},
		{
			name: "multiple parts joined with newline",
			in: []gemini.Content{
				{Parts: []gemini.Part{{Text: "hello"}}},
				{Parts: []gemini.Part{{Text: "world"}}},
			},
			want: "hello\nworld",
		},
		{
			name: "skips non-text parts",
			in: []gemini.Content{
				{Parts: []gemini.Part{
					{Text: "describe"},
					{InlineData: &gemini.Blob{MimeType: "image/png", Data: "AAA"}},
					{Text: "this image"},
				}},
			},
			want: "describe\nthis image",
		},
		{
			name: "empty input",
			in:   nil,
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CountTokensTextFromGemini(tt.in); got != tt.want {
				t.Fatalf("got %q, want %q", got, tt.want)
			}
		})
	}
}
