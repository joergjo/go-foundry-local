# Chat Completion Streaming Example

This example demonstrates how to use the `go-foundry-local` SDK with streaming chat completions. It shows how to send a chat request to a locally running AI model and receive the response as a real-time stream using Server-Sent Events (SSE).

## What This Example Shows

1. **Model Setup**
   - Starting a specific model using the convenience `StartModel()` function
   - Retrieving model information for API calls

2. **Streaming Chat Completion**
   - Making HTTP requests directly to the Foundry Local REST API
   - Enabling streaming mode for real-time response generation
   - Processing Server-Sent Events (SSE) to receive streaming chunks
   - Handling completion termination signals

3. **Real-time Output**
   - Displaying AI responses as they are generated (character by character)
   - Proper handling of streaming data and error conditions

## Key Features

- **Real-time streaming**: See the AI response being generated live
- **Direct API usage**: Shows how to interact with Foundry Local's REST API
- **SSE handling**: Demonstrates proper Server-Sent Events processing
- **Error handling**: Robust error handling for network and parsing issues

## Prerequisites

- Foundry Local must be installed and available in your PATH
- The model `qwen2.5-1.5b` should be available in the catalog (or modify the `alias` variable to use a different model)

## Dependencies

This example uses:
- `github.com/joergjo/go-foundry-local/foundrylocal` - The Foundry Local SDK
- `github.com/tmaxmax/go-sse` - Server-Sent Events library for Go

## Running the Example

```bash
go run main.go
```

Or build and run:

```bash
go build -o streaming-example
./streaming-example
```

## Expected Output

The example will:
1. Start the specified model
2. Display the API endpoint being used
3. Send a "Write me a haiku" prompt to the model
4. Stream the response in real-time, showing each word as it's generated
5. Clean up by stopping the service

Example output:
```
Using Foundry Local endpoint at http://localhost:5273/v1/chat/completions
Cherry blossoms fall,
Gentle spring breeze carries dreams,
Peace in nature's song.
```

## Code Structure

- **`chunk` struct**: Represents the streaming response format from OpenAI-compatible APIs
- **`completionRequest` struct**: Defines the request payload for chat completions
- **Main function**: Orchestrates the entire flow from model startup to streaming response

## Alternative Approaches

While this example shows direct HTTP API usage, you could also use:
- The official OpenAI Go SDK (as shown in the `chat-completion` example)
- Other OpenAI-compatible client libraries that support streaming

> [This issue](https://github.com/microsoft/Foundry-Local/issues/144) currently prevents streaming to work with the official OpenAI SDK. 

This example is useful when you need fine-grained control over the HTTP requests or want to understand the underlying API interactions.
