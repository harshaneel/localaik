import com.google.genai.Client;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponse;
import com.google.genai.types.Schema;
import java.util.List;
import java.util.Map;

public class GeminiStructured {
    public static void main(String[] args) throws Exception {
        Client client = Client.builder()
                .apiKey("test")
                .baseUrl("http://localhost:8090")
                .build();

        Schema itemSchema = Schema.builder()
                .type("OBJECT")
                .properties(Map.of(
                        "name", Schema.builder().type("STRING").build(),
                        "year", Schema.builder().type("INTEGER").build(),
                        "use_case", Schema.builder().type("STRING").build()))
                .required(List.of("name", "year", "use_case"))
                .build();

        GenerateContentConfig config = GenerateContentConfig.builder()
                .responseMimeType("application/json")
                .responseSchema(Schema.builder()
                        .type("ARRAY")
                        .items(itemSchema)
                        .build())
                .build();

        GenerateContentResponse response = client.models.generateContent(
                "localaik",
                "List three popular programming languages with their year of creation and primary use case.",
                config);

        System.out.println(response.text());
    }
}
