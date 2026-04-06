#!/bin/sh
curl -s http://localhost:8090/v1beta/models/localaik:generateContent \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [{ "text": "What are three interesting facts about the Go programming language?" }]
      }
    ]
  }' | python3 -m json.tool
