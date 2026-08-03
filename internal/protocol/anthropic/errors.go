package anthropic

import (
	"encoding/json"
	"net/http"
)

// Error type values used by the Messages API.
const (
	ErrorTypeInvalidRequest  = "invalid_request_error"
	ErrorTypeAuthentication  = "authentication_error"
	ErrorTypePermission      = "permission_error"
	ErrorTypeNotFound        = "not_found_error"
	ErrorTypeRequestTooLarge = "request_too_large"
	ErrorTypeRateLimit       = "rate_limit_error"
	ErrorTypeAPI             = "api_error"
	ErrorTypeOverloaded      = "overloaded_error"
)

// WriteError emits an Anthropic-shaped error body with the HTTP status code and
// the error type Anthropic pairs with that status.
func WriteError(w http.ResponseWriter, statusCode int, message string) {
	writeJSON(w, statusCode, ErrorResponse{
		Type: EventError,
		Error: Error{
			Type:    ErrorTypeForHTTP(statusCode),
			Message: message,
		},
	})
}

// ErrorTypeForHTTP maps an HTTP status code to the Anthropic error type string.
func ErrorTypeForHTTP(statusCode int) string {
	switch statusCode {
	case http.StatusBadRequest:
		return ErrorTypeInvalidRequest
	case http.StatusUnauthorized:
		return ErrorTypeAuthentication
	case http.StatusForbidden:
		return ErrorTypePermission
	case http.StatusNotFound:
		return ErrorTypeNotFound
	case http.StatusRequestEntityTooLarge:
		return ErrorTypeRequestTooLarge
	case http.StatusTooManyRequests:
		return ErrorTypeRateLimit
	case 529:
		return ErrorTypeOverloaded
	default:
		if statusCode >= http.StatusInternalServerError {
			return ErrorTypeAPI
		}
		return ErrorTypeInvalidRequest
	}
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
