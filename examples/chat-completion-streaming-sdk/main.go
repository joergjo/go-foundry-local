package main

import (
	"context"
	"fmt"

	"github.com/joergjo/go-foundry-local/foundrylocal"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

func main() {
	alias := "qwen2.5-1.5b"

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
	client := openai.NewClient(option.WithBaseURL(baseURL), option.WithAPIKey(m.ApiKey))

	question := "Write me a haiku"

	fmt.Print("> ")
	fmt.Println(question)
	fmt.Println()
	stream := client.Chat.Completions.NewStreaming(context.Background(), openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(question),
		},
		Model: modelInfo.ID,
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
