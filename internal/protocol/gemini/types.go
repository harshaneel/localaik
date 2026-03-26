package gemini

type GenerateContentRequest struct {
	Contents          []Content         `json:"contents,omitempty"`
	SystemInstruction *Content          `json:"systemInstruction,omitempty"`
	Tools             []Tool            `json:"tools,omitempty"`
	ToolConfig        *ToolConfig       `json:"toolConfig,omitempty"`
	GenerationConfig  *GenerationConfig `json:"generationConfig,omitempty"`
}

type GenerationConfig struct {
	TopP               *float64 `json:"topP,omitempty"`
	TopK               *float64 `json:"topK,omitempty"`
	CandidateCount     *int     `json:"candidateCount,omitempty"`
	Temperature        *float64 `json:"temperature,omitempty"`
	MaxOutputTokens    *int     `json:"maxOutputTokens,omitempty"`
	StopSequences      []string `json:"stopSequences,omitempty"`
	ResponseLogprobs   *bool    `json:"responseLogprobs,omitempty"`
	Logprobs           *int     `json:"logprobs,omitempty"`
	PresencePenalty    *float64 `json:"presencePenalty,omitempty"`
	FrequencyPenalty   *float64 `json:"frequencyPenalty,omitempty"`
	Seed               *int     `json:"seed,omitempty"`
	ResponseMimeType   string   `json:"responseMimeType,omitempty"`
	ResponseSchema     any      `json:"responseSchema,omitempty"`
	ResponseJSONSchema any      `json:"responseJsonSchema,omitempty"`
}

type Content struct {
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts,omitempty"`
}

type Part struct {
	Text                string               `json:"text,omitempty"`
	InlineData          *Blob                `json:"inlineData,omitempty"`
	FileData            *FileData            `json:"fileData,omitempty"`
	FunctionCall        *FunctionCall        `json:"functionCall,omitempty"`
	FunctionResponse    *FunctionResponse    `json:"functionResponse,omitempty"`
	ExecutableCode      *ExecutableCode      `json:"executableCode,omitempty"`
	CodeExecutionResult *CodeExecutionResult `json:"codeExecutionResult,omitempty"`
	ToolCall            *ToolCall            `json:"toolCall,omitempty"`
	ToolResponse        *ToolResponse        `json:"toolResponse,omitempty"`
}

type Blob struct {
	MimeType string `json:"mimeType,omitempty"`
	Data     string `json:"data,omitempty"`
}

type FileData struct {
	FileURI  string `json:"fileUri,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
}

type FunctionCall struct {
	ID   string         `json:"id,omitempty"`
	Args map[string]any `json:"args,omitempty"`
	Name string         `json:"name,omitempty"`
}

type FunctionResponse struct {
	ID       string                 `json:"id,omitempty"`
	Name     string                 `json:"name,omitempty"`
	Response map[string]any         `json:"response,omitempty"`
	Parts    []FunctionResponsePart `json:"parts,omitempty"`
}

type FunctionResponsePart struct {
	InlineData *Blob     `json:"inlineData,omitempty"`
	FileData   *FileData `json:"fileData,omitempty"`
}

type ExecutableCode struct {
	ID       string `json:"id,omitempty"`
	Code     string `json:"code,omitempty"`
	Language string `json:"language,omitempty"`
}

type CodeExecutionResult struct {
	ID      string `json:"id,omitempty"`
	Outcome string `json:"outcome,omitempty"`
	Output  string `json:"output,omitempty"`
}

type ToolCall struct {
	ID       string         `json:"id,omitempty"`
	ToolType string         `json:"toolType,omitempty"`
	Args     map[string]any `json:"args,omitempty"`
}

type ToolResponse struct {
	ID       string         `json:"id,omitempty"`
	ToolType string         `json:"toolType,omitempty"`
	Response map[string]any `json:"response,omitempty"`
}

type Tool struct {
	FunctionDeclarations []FunctionDeclaration `json:"functionDeclarations,omitempty"`
}

type FunctionDeclaration struct {
	Description          string `json:"description,omitempty"`
	Name                 string `json:"name,omitempty"`
	Parameters           any    `json:"parameters,omitempty"`
	ParametersJSONSchema any    `json:"parametersJsonSchema,omitempty"`
	Response             any    `json:"response,omitempty"`
	ResponseJSONSchema   any    `json:"responseJsonSchema,omitempty"`
}

type ToolConfig struct {
	FunctionCallingConfig *FunctionCallingConfig `json:"functionCallingConfig,omitempty"`
}

type FunctionCallingConfig struct {
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
	Mode                 string   `json:"mode,omitempty"`
}

type GenerateContentResponse struct {
	Candidates    []Candidate    `json:"candidates,omitempty"`
	UsageMetadata *UsageMetadata `json:"usageMetadata,omitempty"`
}

type Candidate struct {
	Content      *Content `json:"content,omitempty"`
	FinishReason string   `json:"finishReason,omitempty"`
	Index        int      `json:"index,omitempty"`
}

type UsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount,omitempty"`
	CandidatesTokenCount int `json:"candidatesTokenCount,omitempty"`
	TotalTokenCount      int `json:"totalTokenCount,omitempty"`
}

type ErrorResponse struct {
	Error Error `json:"error"`
}

type Error struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}
