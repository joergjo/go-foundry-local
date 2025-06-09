This is a Go based repository that provides an SDK for Foundry Local from Microsoft. 
It is primarily responsible for interacting with the Foundry Local runtime to start and stop the Foundry Local 
runtime and manage modelsl. This impplementation only requires to Go standard library and does not have 
any external dependencies.

## Code Standards

### Required Before Each Commit
- Run `go fmt` before committing any changes to ensure proper code formatting

### Development Flow
- Build: `go build`
- Test: `go test -shuffle=on ./...`

## Repository Structure
The repository provides a Go package `foundrylocal` that contains the SDK for Foundry Local
in the folder `./foundrylocal`. All test files are suffixed with `_test.go`. 
The folder `./examples` contains example code that demonstrates how to use the SDK.

## Key Guidelines
1. Follow Go best practices and idiomatic patterns
2. Maintain existing code structure and organization
4. Write unit tests for new functionality. Use table-driven unit tests when possible.
5. Document public APIs using GoDoc comments.