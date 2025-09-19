# Google Genkit Go Example

This example demonstrates how to integrate the Foundry Local SDK with Google's [Genkit Go](https://firebase.google.com/docs/genkit/go/get-started) framework for AI-powered applications. It shows how to use Foundry Local as a backend for Genkit's OpenAI plugin, enabling you to run local models through Genkit's powerful AI generation framework.

## What This Example Shows

1. **Foundry Local Integration**
   - Starting a specific model using the convenience `StartModel()` function
   - Configuring the OpenAI endpoint for Genkit compatibility
   - Proper service cleanup and resource management

2. **Genkit Go Setup**
   - Initializing Genkit with the OpenAI compatibility plugin
   - Configuring the plugin to use Foundry Local as the backend
   - Defining model capabilities for optimal integration

3. **AI Text Generation**
   - Using Genkit's high-level `Generate()` function
   - Configuring prompts and model selection
   - Receiving and displaying generated content

4. **Model Definition Best Practices**
   - Explicitly defining model capabilities vs. dynamic registration
   - Understanding the difference between basic text and multi-modal models
   - Optimal configuration for single-modal text generation

## Key Features

- **Genkit Integration**: Seamless integration with Google's Genkit Go framework
- **Local Model Execution**: Run AI models locally while using Genkit's abstractions
- **OpenAI Compatibility**: Leverage existing OpenAI-compatible tools and patterns
- **Explicit Model Configuration**: Clear definition of model capabilities for better performance

## Prerequisites

- Foundry Local must be installed and available in your PATH
- The model `qwen2.5-1.5b` should be available in the catalog (or modify the `alias` variable to use a different model)
- Go 1.25.1 or later

## Dependencies

This example uses:
- `github.com/firebase/genkit/go` - Google Genkit Go framework
- `github.com/firebase/genkit/go/plugins/compat_oai` - OpenAI compatibility plugin for Genkit
- `github.com/joergjo/go-foundry-local/foundrylocal` - The Foundry Local SDK
- `github.com/openai/openai-go` - Official OpenAI Go client library

## Running the Example

```bash
go run main.go
```

Or build and run:

```bash
go build -o genkit-example
./genkit-example
```

## Expected Output

```
Using Foundry Local endpoint at http://localhost:5273/v1
> Write me a haiku

Morning dew glistens,
On petals soft and tender—
Nature's quiet song.
```

## Key Implementation Details

### Model Definition vs Dynamic Registration

The example demonstrates two approaches for model configuration:

**Recommended Approach (used in this example):**
```go
model := openAI.DefineModel(modelInfo.ID, ai.ModelOptions{
    Supports: &compat_oai.BasicText,
})
```

**Alternative Approach (commented out):**
```go
model := openAI.Model(g, modelInfo.ID)
```

The recommended approach explicitly defines the model's capabilities as `BasicText`, which is more accurate for text-only models and provides better performance and clarity compared to dynamic registration that assumes multi-modal capabilities.


## Comparison with Other Examples

This example differs from other examples in the repository:

- **vs. Chat Completion**: Uses Genkit's high-level abstractions instead of direct OpenAI API calls

Choose this example when you need:
- Integration with Google Genkit Go applications
- High-level AI generation abstractions
- Framework-based AI application development
- Structured AI workflows and pipelines