package main

import (
	"context"
	"fmt"
	"log"

	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  "test",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL: "http://localhost:8090",
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	resp, err := client.Models.GenerateContent(ctx,
		"localaik",
		genai.Text("What are three interesting facts about the Go programming language?"),
		nil,
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Text())
}
