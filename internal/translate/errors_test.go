package translate

import (
	"net/http"
	"testing"
)

func TestOpenAIErrorToGemini(t *testing.T) {
	body := []byte(`{"error":{"message":"bad request","type":"invalid_request_error"}}`)
	got := OpenAIErrorToGemini(http.StatusBadRequest, body)

	if got.Error.Code != http.StatusBadRequest {
		t.Fatalf("code = %d, want %d", got.Error.Code, http.StatusBadRequest)
	}
	if got.Error.Message != "bad request" {
		t.Fatalf("message = %q, want bad request", got.Error.Message)
	}
	if got.Error.Status != "INVALID_ARGUMENT" {
		t.Fatalf("status = %q, want INVALID_ARGUMENT", got.Error.Status)
	}
}
