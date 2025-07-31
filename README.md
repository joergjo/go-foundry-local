# go-foundry-local

This is an _unofficial_ Go SDK for interacting with Microsoft's [Foundry Local](https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-local/what-is-foundry-local) runtime. It provides a simple, idiomatic Go interface for starting and stopping the Foundry Local runtime, managing AI models, and resolving Foundry Local's OpenAI compatible endpoints. 

[![Go Reference](https://pkg.go.dev/badge/github.com/joergjo/go-foundry-local/foundrylocal.svg)](https://pkg.go.dev/github.com/joergjo/go-foundry-local/foundrylocal)

This SDK has beeen ported to Go based on the already released [C#, Python, and JavaScript SDKs](https://github.com/microsoft/Foundry-Local).

>I hope that this module will be eventually superseded by an official SDK released by Microsoft, but there are currently [no plans for a Go SDK](https://github.com/microsoft/Foundry-Local/discussions/171).

## Features

- **Pure Go Implementation**: Uses only the Go standard library with no external dependencies
- **Model Management**: Download, start, stop, and query AI models
- **Runtime Control**: Start and stop the Foundry Local runtime
- **Progress Reporting**: Real-time progress updates for long-running operations
- **Well Documented**: Full GoDoc documentation for all public APIs

## Installation

```bash
go get github.com/joergjo/go-foundry-local/foundrylocal
```

## Compatibility Matrix

Foundry Local runtime version | Module version
------------------------------|-------------------
0.4.92                        | >= 0.2.0, < 0.3.0
0.5.117                       | >= 0.3.0

*v0.3.0 is coming soon!*

## Quick Start

```go
package main

import (
        "context"
        "fmt"
        "log"

        "github.com/joergjo/go-foundry-local/foundrylocal"
)

func main() {
	    // Create a new manager
	    manager := foundrylocal.NewManager()

	    // Start the Foundry Local runtime
	    ctx := context.Background()
	    if err := manager.StartService(ctx); err != nil {
		        log.Fatal("Failed to start Foundry Local:", err)
	    }
	    defer manager.StopService(ctx)

        // List available models
        models, err := manager.ListCatalogModels(ctx)
        if err != nil {
                log.Fatal("Failed to list models:", err)
        }

        fmt.Printf("Found %d models\n", len(models))
        for _, model := range models {
                fmt.Printf("- %s (%s)\n", model.Alias, model.ID)
        }
}
```

## Documentation

For complete API documentation, see the [GoDoc](https://pkg.go.dev/github.com/joergjo/go-foundry-local/foundrylocal).

## Examples

The `examples/` directory contains complete working examples:

### [Manager Example](examples/manager/)
Comprehensive demonstration of all Manager functionality including:
- Starting/stopping the runtime
- Downloading models with progress tracking
- Managing running models
- Error handling best practices

### [Chat Completion](examples/chat-completion/)
Shows how to use a model for chat completions with the official OpenAI Go client:
- Starting a model
- Configuring the OpenAI client for Foundry Local
- Making standard OpenAI-compatible API calls
- Simple, non-streaming responses

### [Streaming Chat Completion](examples/chat-completion-streaming/)
Demonstrates real-time streaming chat completions:
- Server-Sent Events (SSE) handling
- Real-time response display
- Stream termination handling

## Development

### Building

```bash
go build ./...
```

### Testing

```bash
go test -shuffle=on ./...
```

### Code Formatting

```bash
go fmt ./...
```

## Requirements

- Go 1.24.4 or later
- Foundry Local must be [installed and available in your PATH](https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-local/get-started)
- macOS or Windows (developed and tested on macOS Sequoia 15.5 and Windows 11 24H2)

## Contributing

1. Follow Go best practices and idiomatic patterns
2. Run `go fmt` before committing changes
3. Write unit tests for new functionality (use table-driven tests when possible)
4. Document public APIs using GoDoc comments
5. Ensure all tests pass: `go test -v -shuffle=on ./...`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
