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

const defaultUpstream = "http://127.0.0.1:8080/v1"

func resolveFlagDefault(envName, fallback string) string {
	if value := os.Getenv(envName); value != "" {
		return value
	}
	return fallback
}

// resolveUpstream also reports where the value came from, so an image with no
// model server of its own can refuse to start on the default.
func resolveUpstream(flagValue string, flagSet bool, env string) (string, string) {
	switch {
	case flagSet:
		return flagValue, "flag"
	case env != "":
		return env, "LK_UPSTREAM"
	}
	return defaultUpstream, "default"
}

// requireEnv is an internal image marker set only in the proxy Dockerfile stage,
// so any non-empty value arms the check.
func upstreamRequiredButUnset(source, requireEnv string) bool {
	return source == "default" && requireEnv != ""
}

func flagWasSet(name string) bool {
	set := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == name {
			set = true
		}
	})
	return set
}

func main() {
	port := flag.String("port", resolveFlagDefault("PORT", "8090"), "port to listen on")
	upstreamFlag := flag.String("upstream", defaultUpstream, "upstream base URL speaking the OpenAI chat completions API")
	flag.Parse()

	upstream, source := resolveUpstream(*upstreamFlag, flagWasSet("upstream"), os.Getenv("LK_UPSTREAM"))
	if upstream == "" {
		log.Fatal("localaik: the upstream is set to an empty value; give a base URL that speaks the OpenAI chat completions API, for example http://llama.internal:8080/v1")
	}
	if upstreamRequiredButUnset(source, os.Getenv("LK_REQUIRE_UPSTREAM")) {
		log.Fatal("localaik: LK_UPSTREAM is not set. This image has no model server of its own, so it needs the base URL of one that speaks the OpenAI chat completions API, for example http://llama.internal:8080/v1")
	}
	log.Printf("localaik: upstream %s (%s)", server.RedactUpstream(upstream), source)

	authHeader := os.Getenv("LK_UPSTREAM_AUTH_HEADER")
	if authHeader != "" && !server.ValidUpstreamAuthHeader(authHeader) {
		log.Printf("localaik: LK_UPSTREAM_AUTH_HEADER is set but is not a valid \"Name: value\" header line; no credential will be sent upstream")
	}

	handler, err := server.New(server.Config{
		UpstreamBaseURL:    upstream,
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
