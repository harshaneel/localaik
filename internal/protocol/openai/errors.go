package openai

import (
	"encoding/json"
	"net/http"
)

func WriteError(w http.ResponseWriter, statusCode int, message, errorType string) {
	writeJSON(w, statusCode, ErrorResponse{
		Error: Error{
			Message: message,
			Type:    errorType,
		},
	})
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
