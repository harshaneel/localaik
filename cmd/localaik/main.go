package main

import (
	"flag"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/harshaneel/localaik/internal/pdf"
	"github.com/harshaneel/localaik/internal/server"
)

func main() {
	defaultPort := os.Getenv("PORT")
	if defaultPort == "" {
		defaultPort = "8090"
	}

	port := flag.String("port", defaultPort, "port to listen on")
	upstream := flag.String("upstream", "http://127.0.0.1:8080/v1", "upstream OpenAI-compatible base URL")
	flag.Parse()

	handler, err := server.New(server.Config{
		UpstreamBaseURL: *upstream,
		HTTPClient:      &http.Client{},
		PDFRenderer:     pdf.NewExecRenderer("pdftoppm"),
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
