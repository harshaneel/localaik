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
		genai.Text("List three popular programming languages with their year of creation and primary use case."),
		&genai.GenerateContentConfig{
			ResponseMIMEType: "application/json",
			ResponseSchema: &genai.Schema{
				Type: genai.TypeArray,
				Items: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"name": {Type: genai.TypeString},
						"year": {Type: genai.TypeInteger},
						"use_case": {Type: genai.TypeString},
					},
					Required: []string{"name", "year", "use_case"},
				},
			},
		},
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Text())
}
