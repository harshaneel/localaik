#!/bin/sh
curl -s http://localhost:8090/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: test" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "localaik",
    "max_tokens": 256,
    "system": "You are a helpful assistant. Keep answers concise.",
    "messages": [
      { "role": "user", "content": "What is the capital of France and what is it known for?" }
    ]
  }' | python3 -m json.tool
