package main

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	oai "github.com/firebase/genkit/go/plugins/compat_oai/openai"
	"github.com/joergjo/go-foundry-local/foundrylocal"
	"github.com/openai/openai-go/option"
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

	ctx := context.Background()

	// To use GenKit Go, we need to set its OpenAI plugin's base URL to Foundry Local's OpenAI endpoint.
	openAI := &oai.OpenAI{
		APIKey: m.ApiKey,
		Opts: []option.RequestOption{
			option.WithBaseURL(m.Endpoint().String()),
		},
	}
	g := genkit.Init(ctx, genkit.WithPlugins(openAI))
	// Specify the model and its capabilities
	model := openAI.DefineModel(modelInfo.ID, ai.ModelOptions{
		Supports: &compat_oai.BasicText,
	})
	// Note that the following code will also work
	// model := openAI.Model(g, modelInfo.ID)
	// because it registers our model dynamically in Genkit. But dynamically registered
	// OpenAI compatible models are assumed to be multi-modal, which is not true for our
	// model used in this sample. We don't use any multi-modal capabilities so the sample
	// will work fine, but it's calling DefineModel() and specifying the model's capabilitirs
	// is both more readable und easier to understand.

	question := "Write me a haiku"

	fmt.Print("> ")
	fmt.Println(question)
	fmt.Println()
	resp, err := genkit.Generate(ctx, g, ai.WithPrompt(question), ai.WithModel(model))
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Text())
}
