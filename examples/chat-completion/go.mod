module github.com/joergjo/go-foundry-local/examples/chat-completion

go 1.24.4

require (
	github.com/joergjo/go-foundry-local/foundrylocal v0.0.0-20250926120018-4c65c0f7e212
	github.com/openai/openai-go/v2 v2.7.1
)

require (
	github.com/tidwall/gjson v1.18.0 // indirect
	github.com/tidwall/match v1.2.0 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
)

replace github.com/joergjo/go-foundry-local/foundrylocal => ../../foundrylocal
