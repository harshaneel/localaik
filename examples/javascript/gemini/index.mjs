import { GoogleGenAI } from "@google/genai";

const client = new GoogleGenAI({
  apiKey: "test",
  httpOptions: { apiVersion: "v1beta", baseUrl: "http://localhost:8090" },
});

const response = await client.models.generateContent({
  model: "localaik",
  contents: "What are three interesting facts about the Go programming language?",
});

console.log(response.text);
