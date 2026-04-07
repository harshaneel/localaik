from google import genai

client = genai.Client(
    api_key="test",
    http_options=genai.types.HttpOptions(
        api_version="v1beta",
        base_url="http://localhost:8090",
    ),
)

response = client.models.generate_content(
    model="localaik",
    contents="What are three interesting facts about the Go programming language?",
)

print(response.text)
