package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httputil"

	"github.com/joergjo/go-foundry-local/foundrylocal"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/shared"
)

type loggingTransport struct {
	InnerTransport http.RoundTripper
}

func (t *loggingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	bytes, _ := httputil.DumpRequestOut(r, true)
	resp, err := t.InnerTransport.RoundTrip(r)
	respBytes, _ := httputil.DumpResponse(resp, true)
	bytes = append(bytes, respBytes...)
	fmt.Printf("%s\n", bytes)
	return resp, err
}

func main() {
	lt := &loggingTransport{
		InnerTransport: http.DefaultTransport,
	}
	http.DefaultTransport = lt

	alias := "qwen2.5-1.5b"

	m, err := foundrylocal.StartModel(context.Background(), alias)
	if err != nil {
		panic(err)
	}
	defer func() {
		if err := m.StopService(context.Background()); err != nil {
			fmt.Printf("Error stopping Foundry Local service: %v\n", err)
		}
	}()

	modelInfo, err := m.GetModelInfo(context.Background(), alias)
	if err != nil {
		panic(fmt.Sprintf("Error getting model info: %v", err))
	}

	baseURL := m.Endpoint().String()
	fmt.Printf("Using Foundry Local endpoint at %s\n", baseURL)
	client := openai.NewClient(option.WithBaseURL(baseURL), option.WithAPIKey(m.ApiKey))

	question := "Write me a haiku"

	fmt.Print("> ")
	fmt.Println(question)
	fmt.Println()

	stream := client.Chat.Completions.NewStreaming(context.Background(), openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(question),
		},
		Model: shared.ChatModel(modelInfo.ID),
		Seed:  openai.Int(0),
	})

	for stream.Next() {
		completion := stream.Current()
		if len(completion.Choices) > 0 {
			fmt.Print(completion.Choices[0].Delta.Content)
		}
	}
	fmt.Println()

	if err := stream.Err(); err != nil {
		fmt.Println(err.Error())
	}
}
