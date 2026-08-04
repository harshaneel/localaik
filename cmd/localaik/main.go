package main

import (
	"flag"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/harshaneel/localaik/internal/pdf"
	"github.com/harshaneel/localaik/internal/server"
)

func resolveFlagDefault(envName, fallback string) string {
	if value := os.Getenv(envName); value != "" {
		return value
	}
	return fallback
}

func isValidAuthHeader(header string) bool {
	if header == "" {
		return false
	}
	idx := strings.Index(header, ":")
	if idx == -1 {
		return false
	}
	name := strings.TrimSpace(header[:idx])
	value := strings.TrimSpace(header[idx+1:])
	return name != "" && value != ""
}

func main() {
	port := flag.String("port", resolveFlagDefault("PORT", "8090"), "port to listen on")
	upstream := flag.String("upstream", resolveFlagDefault("LK_UPSTREAM", "http://127.0.0.1:8080/v1"), "upstream OpenAI-compatible base URL")
	flag.Parse()

	authHeader := os.Getenv("LK_UPSTREAM_AUTH_HEADER")
	if authHeader != "" && !isValidAuthHeader(authHeader) {
		log.Printf("localaik: LK_UPSTREAM_AUTH_HEADER is set but is not a \"Name: value\" header line; no credential will be sent upstream")
	}

	handler, err := server.New(server.Config{
		UpstreamBaseURL:    *upstream,
		UpstreamAuthHeader: authHeader,
		HTTPClient:         &http.Client{},
		PDFRenderer:        pdf.NewExecRenderer("pdftoppm"),
	})
	if err != nil {
		log.Fatalf("localaik: %v", err)
	}

	httpServer := &http.Server{
		Addr:              ":" + *port,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
	}

	log.Printf("localaik: listening on port %s", *port)
	log.Fatal(httpServer.ListenAndServe())
}
