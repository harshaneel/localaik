#!/bin/sh
curl -s http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "localaik",
    "messages": [
      { "role": "system", "content": "You are a helpful assistant. Keep answers concise." },
      { "role": "user", "content": "What is the capital of France and what is it known for?" }
    ]
  }' | python3 -m json.tool
