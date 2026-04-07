from openai import OpenAI

client = OpenAI(api_key="test", base_url="http://localhost:8090/v1")

response = client.chat.completions.create(
    model="localaik",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Keep answers concise."},
        {"role": "user", "content": "What is the capital of France and what is it known for?"},
    ],
)

print(response.choices[0].message.content)
