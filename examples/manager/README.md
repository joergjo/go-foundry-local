# Foundry Local Manager Example

This example demonstrates comprehensive usage of the `foundrylocal.Manager` type from the `go-foundry-local` SDK.

## What This Example Shows

1. **Service Management**
   - Creating a Manager instance
   - Starting and stopping the Foundry Local service
   - Checking service status

2. **Model Discovery**
   - Listing available models in the catalog
   - Getting detailed information about specific models
   - Checking model execution providers and device requirements

3. **Model Management**
   - Listing cached (downloaded) models
   - Downloading models with progress reporting
   - Loading models into memory for inference
   - Unloading models to free resources

4. **Service Information**
   - Getting the API endpoint URL
   - Finding the cache location

## Prerequisites

- Foundry Local must be installed and available in your PATH
- At least one model should be available in the catalog

## Running the Example

```bash
go run main.go
```

Or build and run:

```bash
go build -o manager-example
./manager-example
```

## Expected Output

The example will walk through each operation step by step, showing:
- Service startup confirmation
- API endpoint information
- List of available models
- Model download progress (if downloading)
- Loading and unloading operations
- Clean shutdown

## Key Methods Demonstrated

- `foundrylocal.NewManager()` - Create a new manager
- `manager.StartService()` - Start the Foundry Local service
- `manager.StopService()` - Stop the service
- `manager.ListCatalogModels()` - Get all available models
- `manager.GetModelInfo()` - Get specific model information
- `manager.DownloadModel()` - Download a model
- `manager.DownloadModelWithProgress()` - Download with progress reporting
- `manager.LoadModel()` - Load a model for inference
- `manager.UnloadModel()` - Unload a model from memory
- `manager.ListCachedModels()` - List downloaded models
- `manager.ListLoadedModels()` - List models in memory
