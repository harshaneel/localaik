import { GoogleGenAI } from "@google/genai";

const client = new GoogleGenAI({
  apiKey: "test",
  httpOptions: { apiVersion: "v1beta", baseUrl: "http://localhost:8090" },
});

const response = await client.models.generateContent({
  model: "localaik",
  contents: "List three popular programming languages with their year of creation and primary use case.",
  config: {
    responseMimeType: "application/json",
    responseSchema: {
      type: "ARRAY",
      items: {
        type: "OBJECT",
        properties: {
          name: { type: "STRING" },
          year: { type: "INTEGER" },
          use_case: { type: "STRING" },
        },
        required: ["name", "year", "use_case"],
      },
    },
  },
});

console.log(response.text);
