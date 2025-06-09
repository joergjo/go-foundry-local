# Chat Completion Example

This example demonstrates how to use the Foundry Local SDK with the official OpenAI Go client library for simple chat completions. It shows how to start a model and make a single chat completion request using the standard OpenAI API interface.

## What This Example Shows

1. **Model Setup**
   - Starting a specific model using the convenience `StartModel()` function
   - Retrieving model information for API configuration

2. **OpenAI Client Integration**
   - Using the official `openai-go` client library
   - Configuring the client to use Foundry Local as the backend
   - Making standard OpenAI-compatible API calls

3. **Simple Chat Completion**
   - Sending a single question to the AI model
   - Receiving a complete response (non-streaming)
   - Clean output formatting

## Key Features

- **Official OpenAI Client**: Uses the standard `openai-go` library for API compatibility
- **Simple Interface**: Minimal code for basic chat functionality
- **Seamless Integration**: Drop-in replacement for OpenAI API calls
- **Error Handling**: Basic error handling for common issues

## Prerequisites

- Foundry Local must be installed and available in your PATH
- The model `phi-3.5-mini` should be available in the catalog (or modify the `alias` variable to use a different model)

## Dependencies

This example uses:
- `github.com/joergjo/go-foundry-local/foundrylocal` - The Foundry Local SDK
- `github.com/openai/openai-go` - Official OpenAI Go client library

## Running the Example

```bash
go run main.go
```

Or build and run:

```bash
go build -o chat-example
./chat-example
```

## Expected Output

```
Using Foundry Local endpoint at http://localhost:5273/v1
> Write me a haiku

Silent morning dew
Glistens on the garden leaves—
Peace in simple things.
```

## Comparison with Streaming Example

This example differs from the `chat-completion-streaming` example in several ways:

- **Client Library**: Uses the official OpenAI Go client vs. direct HTTP calls
- **Response Type**: Single complete response vs. real-time streaming
- **Complexity**: Simpler code with higher-level abstractions
- **Use Case**: Better for simple Q&A vs. interactive conversations

Choose this example when you need:
- Simple, one-off completions
- Integration with existing OpenAI-based code
- Standard API compatibility
- Minimal implementation complexity
