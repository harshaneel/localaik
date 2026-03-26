package gemini

import (
	"encoding/json"
	"net/http"
)

func WriteError(w http.ResponseWriter, statusCode int, message string) {
	writeJSON(w, statusCode, ErrorResponse{
		Error: Error{
			Code:    statusCode,
			Message: message,
			Status:  statusForHTTP(statusCode),
		},
	})
}

func statusForHTTP(statusCode int) string {
	switch statusCode {
	case http.StatusBadRequest:
		return "INVALID_ARGUMENT"
	case http.StatusUnauthorized:
		return "UNAUTHENTICATED"
	case http.StatusForbidden:
		return "PERMISSION_DENIED"
	case http.StatusNotFound:
		return "NOT_FOUND"
	case http.StatusConflict:
		return "ABORTED"
	case http.StatusTooManyRequests:
		return "RESOURCE_EXHAUSTED"
	case http.StatusServiceUnavailable:
		return "UNAVAILABLE"
	case http.StatusGatewayTimeout:
		return "DEADLINE_EXCEEDED"
	case http.StatusInternalServerError, http.StatusBadGateway:
		return "INTERNAL"
	default:
		if statusCode >= 500 {
			return "INTERNAL"
		}
		return "UNKNOWN"
	}
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
