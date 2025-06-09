package main

import (
	"context"
	"fmt"

	"github.com/joergjo/go-foundry-local/foundrylocal"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

func main() {
	alias := "phi-3.5-mini"
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
	params := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(question),
		},
		Model: shared.ChatModel(modelInfo.ModelID),
		Seed:  openai.Int(0),
	}

	completion, err := client.Chat.Completions.New(context.Background(), params)

	if err != nil {
		panic(err)
	}

	fmt.Println(completion.Choices[0].Message.Content)
}
