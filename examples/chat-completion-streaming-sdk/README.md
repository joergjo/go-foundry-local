# Chat Completion Streaming (Official OpenAI SDK) Example

This example demonstrates how to use the `go-foundry-local` SDK together with the official OpenAI Go client library to stream chat completions from a locally running model. It shows how to start a model with Foundry Local, configure the OpenAI client to talk to the local endpoint, and then consume streaming response chunks.

## What This Example Shows

1. **Model Setup**
   - Starting a specific model using the convenience `StartModel(ctx, alias, device)` function (use `nil` for default device selection)
   - Retrieving model information for API calls with optional device filtering
   - Proper cleanup by stopping the Foundry Local service

2. **OpenAI Client Integration**
   - Using the official `openai-go/v3` client library
   - Configuring the client to use Foundry Local as the backend via `WithBaseURL()`
   - Using the Foundry Local API key with `WithAPIKey()`

3. **Streaming Chat Completion**
   - Requesting a streaming completion via `Chat.Completions.NewStreaming(...)`
   - Reading incremental chunks with `stream.Next()`
   - Printing streamed deltas as they arrive

## Key Features

- **Official OpenAI SDK**: Uses `openai-go/v3` for OpenAI-compatible API calls
- **Real-time streaming**: Prints the response while it’s generated
- **Minimal setup**: Only a few lines needed to redirect the client to Foundry Local
- **Clean shutdown**: Stops the Foundry Local service on exit

## Prerequisites

- Foundry Local must be installed and available in your PATH
- The model `qwen2.5-1.5b` should be available in the catalog (or modify the `alias` variable to use a different model)

## Dependencies

This example uses:
- `github.com/joergjo/go-foundry-local/foundrylocal` - The Foundry Local SDK
- `github.com/openai/openai-go/v3` - Official OpenAI Go client library (v3)

## Running the Example

```bash
go run main.go
```

Or build and run:

```bash
go build -o streaming-sdk-example
./streaming-sdk-example
```

## Expected Output

```
Using Foundry Local endpoint at http://localhost:5273/v1
> Write me a haiku

Silent morning dew
Glistens on the garden leaves—
Peace in simple things.
```

## Comparison with Other Examples

- **vs. `chat-completion`**: This example uses streaming (`NewStreaming`) instead of a single non-streaming response.
- **vs. `chat-completion-streaming`**: This example uses the official OpenAI Go SDK instead of direct HTTP + manual SSE handling.
