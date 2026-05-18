package translate

import (
	"strings"

	"github.com/harshaneel/localaik/internal/protocol/gemini"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

var defaultGenerationMethods = []string{
	"generateContent",
	"streamGenerateContent",
	"countTokens",
}

func OpenAIModelListToGemini(list openaip.ModelList) gemini.ListModelsResponse {
	out := gemini.ListModelsResponse{Models: make([]gemini.Model, 0, len(list.Data))}
	for _, m := range list.Data {
		out.Models = append(out.Models, OpenAIModelToGemini(m))
	}
	return out
}

func OpenAIModelToGemini(m openaip.Model) gemini.Model {
	return gemini.Model{
		Name:                       "models/" + m.ID,
		DisplayName:                m.ID,
		SupportedGenerationMethods: defaultGenerationMethods,
	}
}

// CountTokensTextFromGemini flattens a Gemini countTokens request body into a
// single text payload suitable for llama.cpp's /tokenize endpoint. Non-text
// parts (inline blobs, file refs, function calls/responses) are skipped — they
// are not tokenizable by the text tokenizer.
func CountTokensTextFromGemini(contents []gemini.Content) string {
	var b strings.Builder
	for _, content := range contents {
		for _, part := range content.Parts {
			if part.Text == "" {
				continue
			}
			if b.Len() > 0 {
				b.WriteByte('\n')
			}
			b.WriteString(part.Text)
		}
	}
	return b.String()
}
