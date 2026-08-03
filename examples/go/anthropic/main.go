package main

import (
	"context"
	"fmt"
	"log"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

func main() {
	ctx := context.Background()

	client := anthropic.NewClient(
		option.WithAPIKey("test"),
		option.WithBaseURL("http://localhost:8090/"),
	)

	message, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     "localaik",
		MaxTokens: 256,
		System: []anthropic.TextBlockParam{
			{Text: "You are a helpful assistant. Keep answers concise."},
		},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("What is the capital of France and what is it known for?")),
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	for _, block := range message.Content {
		if block.Type == "text" {
			fmt.Println(block.Text)
		}
	}
}
