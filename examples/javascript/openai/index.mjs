import OpenAI from "openai";

const client = new OpenAI({
  apiKey: "test",
  baseURL: "http://localhost:8090/v1",
});

const response = await client.chat.completions.create({
  model: "localaik",
  messages: [
    { role: "system", content: "You are a helpful assistant. Keep answers concise." },
    { role: "user", content: "What is the capital of France and what is it known for?" },
  ],
});

console.log(response.choices[0].message.content);
