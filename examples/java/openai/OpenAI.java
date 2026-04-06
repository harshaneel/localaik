import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionCreateParams;

public class OpenAI {
    public static void main(String[] args) {
        OpenAIClient client = OpenAIOkHttpClient.builder()
                .apiKey("test")
                .baseUrl("http://localhost:8090/v1/")
                .build();

        ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
                .model("localaik")
                .addSystemMessage("You are a helpful assistant. Keep answers concise.")
                .addUserMessage("What is the capital of France and what is it known for?")
                .build();

        ChatCompletion completion = client.chat().completions().create(params);

        System.out.println(completion.choices().get(0).message().content().orElse(""));
    }
}
