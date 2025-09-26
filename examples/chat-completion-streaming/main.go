package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/joergjo/go-foundry-local/foundrylocal"
	"github.com/tmaxmax/go-sse"
)

// chunk represents an OpenAI chat completion chunk, reduced to only those properties
// required for this demo. See https://platform.openai.com/docs/api-reference/chat-streaming.
type chunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason,omitzero"`
	} `json:"choices"`
}

// completionRequest represents a minimized JSON structure for an OpenAI chat completion request.
// See https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-local/reference/reference-rest#post-v1chatcompletions.
type completionRequest struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	Stream bool `json:"stream,omitzero"`
}

func main() {
	alias := "qwen2.5-1.5b"

	// Start the Foundry Local service with the specified model alias.
	m, err := foundrylocal.StartModel(context.Background(), alias, nil)
	if err != nil {
		panic(err)
	}
	defer func() {
		if err := m.StopService(context.Background()); err != nil {
			fmt.Printf("Error stopping Foundry Local service: %v\n", err)
		}
	}()

	// Get model information to retrieve the model ID.
	modelInfo, err := m.GetModelInfo(context.Background(), alias, nil)
	if err != nil {
		panic(fmt.Sprintf("Error getting model info: %v", err))
	}

	// Resolve the OpenAI endpoint.
	baseURL := m.Endpoint().JoinPath("chat", "completions").String()
	fmt.Printf("Using Foundry Local endpoint at %s\n", baseURL)

	// Prepare the chat completion request.
	completion := completionRequest{
		Model: modelInfo.ID,
		Messages: []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}{
			{
				Role:    "user",
				Content: "Write me a haiku",
			},
		},
		Stream: true, // Enable streaming
	}
	payload, err := json.Marshal(completion)
	if err != nil {
		panic(err)
	}

	// Create a context with a timeout for the request.
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute*2)
	defer cancel()

	// Make the HTTP request to the Foundry Local service.
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, baseURL, bytes.NewBuffer(payload))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.ApiKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	// Process the SSE stream response.
	var chunkData chunk
	for ev, err := range sse.Read(resp.Body, nil) {
		if err != nil {
			fmt.Printf("Error reading SSE stream: %v\n", err)
			break
		}
		if err := json.NewDecoder(strings.NewReader(ev.Data)).Decode(&chunkData); err != nil {
			fmt.Printf("Error decoding JSON: %v\n", err)
			continue // skip this chunk if there's an error
		}
		if chunkData.Choices[0].FinishReason == "stop" {
			break // stop processing if we hit the end of the response
		}
		fmt.Print(chunkData.Choices[0].Delta.Content)
	}
	fmt.Println()
}
