from google import genai
from google.genai import types

client = genai.Client(
    api_key="test",
    http_options=types.HttpOptions(
        api_version="v1beta",
        base_url="http://localhost:8090",
    ),
)

response = client.models.generate_content(
    model="localaik",
    contents="List three popular programming languages with their year of creation and primary use case.",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=types.Schema(
            type="ARRAY",
            items=types.Schema(
                type="OBJECT",
                properties={
                    "name": types.Schema(type="STRING"),
                    "year": types.Schema(type="INTEGER"),
                    "use_case": types.Schema(type="STRING"),
                },
                required=["name", "year", "use_case"],
            ),
        ),
    ),
)

print(response.text)
