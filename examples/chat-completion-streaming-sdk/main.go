package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httputil"
	"os"
	"time"

	"github.com/joergjo/go-foundry-local/foundrylocal"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

type loggingTransport struct {
	innerTransport http.RoundTripper
	file           *os.File
}

func (t *loggingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	bytes, _ := httputil.DumpRequestOut(r, true)
	resp, err := t.innerTransport.RoundTrip(r)
	respBytes, _ := httputil.DumpResponse(resp, true)
	bytes = append(bytes, respBytes...)
	t.file.Write(bytes)
	return resp, err
}

func main() {
	timestamp := time.Now().Format("2006-01-02T15-04-05.000")
	logFileName := fmt.Sprintf("httpdump-%s.log", timestamp)
	logFile, err := os.OpenFile(logFileName, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		panic(fmt.Sprintf("Failed to open log file: %v", err))
	}
	defer logFile.Close()

	lt := &loggingTransport{
		innerTransport: http.DefaultTransport,
		file:           logFile,
	}
	http.DefaultTransport = lt

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

	modelInfo, err := m.GetModelInfo(context.Background(), alias, nil)
	if err != nil {
		panic(fmt.Sprintf("Error getting model info: %v", err))
	}

	baseURL := m.Endpoint().String()
	fmt.Printf("Using Foundry Local endpoint at %s\n", baseURL)

	clients := make([]openai.Client, 2)
	clients[0] = openai.NewClient(option.WithBaseURL(baseURL), option.WithAPIKey(m.ApiKey))
	clients[1] = openai.NewClient()
	models := make([]string, 2)
	models[0] = modelInfo.ID
	models[1] = "gpt-5-mini"

	question := "Write me a haiku"

	fmt.Print("> ")
	fmt.Println(question)
	fmt.Println()

	for i, client := range clients {
		stream := client.Chat.Completions.NewStreaming(context.Background(), openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(question),
			},
			Model: models[i],
			Seed:  openai.Int(0),
		})

		for stream.Next() {
			completion := stream.Current()
			if len(completion.Choices) > 0 {
				switch completion.Choices[0].FinishReason {
				case "stop":
					break
				case "length":
					fmt.Print("\n[Truncated: max tokens reached]\n")
				case "content_filter":
					fmt.Print("\n[Truncated: content filtered]\n")
				default:
					fmt.Print(completion.Choices[0].Delta.Content)
				}
			}
		}
		fmt.Println()

		if err := stream.Err(); err != nil {
			fmt.Println(err.Error())
		}
	}
}
