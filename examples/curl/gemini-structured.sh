#!/bin/sh
curl -s http://localhost:8090/v1beta/models/localaik:generateContent \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [{ "text": "List three popular programming languages with their year of creation and primary use case." }]
      }
    ],
    "generationConfig": {
      "responseMimeType": "application/json",
      "responseSchema": {
        "type": "ARRAY",
        "items": {
          "type": "OBJECT",
          "properties": {
            "name": { "type": "STRING" },
            "year": { "type": "INTEGER" },
            "use_case": { "type": "STRING" }
          },
          "required": ["name", "year", "use_case"]
        }
      }
    }
  }' | python3 -m json.tool
