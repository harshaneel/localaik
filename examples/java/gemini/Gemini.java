import com.google.genai.Client;
import com.google.genai.types.GenerateContentResponse;

public class Gemini {
    public static void main(String[] args) throws Exception {
        Client client = Client.builder()
                .apiKey("test")
                .baseUrl("http://localhost:8090")
                .build();

        GenerateContentResponse response = client.models.generateContent(
                "localaik",
                "What are three interesting facts about the Go programming language?",
                null);

        System.out.println(response.text());
    }
}
