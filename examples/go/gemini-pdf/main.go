package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"google.golang.org/genai"
)

// A small fake invoice so the example runs with no setup. Pass a path argument
// to use your own PDF instead.
//
//go:embed sample.pdf
var samplePDF []byte

type invoice struct {
	InvoiceNumber string `json:"invoice_number"`
	Vendor        string `json:"vendor"`
	AmountDue     string `json:"amount_due"`
	DueDate       string `json:"due_date"`
}

func main() {
	ctx := context.Background()

	pdfBytes := samplePDF
	if len(os.Args) > 1 {
		data, err := os.ReadFile(os.Args[1])
		if err != nil {
			log.Fatal(err)
		}
		pdfBytes = data
	}

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

	// A schema plus temperature 0 make the model read the fields off the page
	// rather than inventing plausible-looking ones.
	config := &genai.GenerateContentConfig{
		Temperature:      genai.Ptr[float32](0),
		ResponseMIMEType: "application/json",
		ResponseSchema: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"invoice_number": {Type: genai.TypeString},
				"vendor":         {Type: genai.TypeString},
				"amount_due":     {Type: genai.TypeString},
				"due_date":       {Type: genai.TypeString},
			},
			Required: []string{"invoice_number", "vendor", "amount_due", "due_date"},
		},
	}

	resp, err := client.Models.GenerateContent(ctx,
		"localaik",
		[]*genai.Content{{
			Parts: []*genai.Part{
				{Text: "Extract the invoice fields from this document."},
				genai.NewPartFromBytes(pdfBytes, "application/pdf"),
			},
		}},
		config,
	)
	if err != nil {
		log.Fatal(err)
	}

	var inv invoice
	if err := json.Unmarshal([]byte(resp.Text()), &inv); err != nil {
		log.Fatalf("response was not valid JSON: %v\nraw: %s", err, resp.Text())
	}

	fmt.Printf("invoice number: %s\n", inv.InvoiceNumber)
	fmt.Printf("vendor:         %s\n", inv.Vendor)
	fmt.Printf("amount due:     %s\n", inv.AmountDue)
	fmt.Printf("due date:       %s\n", inv.DueDate)
}
