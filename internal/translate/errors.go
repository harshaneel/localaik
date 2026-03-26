package translate

import (
	"encoding/json"
	"net/http"

	"github.com/harshaneel/localaik/internal/protocol/gemini"
	openaip "github.com/harshaneel/localaik/internal/protocol/openai"
)

func OpenAIErrorToGemini(statusCode int, body []byte) gemini.ErrorResponse {
	message := http.StatusText(statusCode)
	if message == "" {
		message = "upstream error"
	}

	var upstream openaip.ErrorResponse
	if err := json.Unmarshal(body, &upstream); err == nil && upstream.Error.Message != "" {
		message = upstream.Error.Message
	}

	var status string
	switch statusCode {
	case http.StatusBadRequest:
		status = "INVALID_ARGUMENT"
	case http.StatusUnauthorized:
		status = "UNAUTHENTICATED"
	case http.StatusForbidden:
		status = "PERMISSION_DENIED"
	case http.StatusNotFound:
		status = "NOT_FOUND"
	case http.StatusConflict:
		status = "ABORTED"
	case http.StatusTooManyRequests:
		status = "RESOURCE_EXHAUSTED"
	case http.StatusServiceUnavailable:
		status = "UNAVAILABLE"
	case http.StatusGatewayTimeout:
		status = "DEADLINE_EXCEEDED"
	case http.StatusInternalServerError, http.StatusBadGateway:
		status = "INTERNAL"
	default:
		if statusCode >= 500 {
			status = "INTERNAL"
		} else {
			status = "UNKNOWN"
		}
	}

	return gemini.ErrorResponse{
		Error: gemini.Error{
			Code:    statusCode,
			Message: message,
			Status:  status,
		},
	}
}
