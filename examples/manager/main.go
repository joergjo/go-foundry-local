// Package main demonstrates the usage of the foundrylocal.Manager type.
//
// This example shows how to:
//   - Create and configure a Manager
//   - Start and stop the Foundry Local service
//   - List and browse available models
//   - Download models (with progress reporting)
//   - Load and unload models for inference
//   - Get service information and cache location
//
// Run this example with: go run main.go
//
// Prerequisites:
//   - Foundry Local must be installed and available in PATH
//   - At least one model should be available in the catalog
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/joergjo/go-foundry-local/foundrylocal"
)

// main demonstrates comprehensive usage of the foundrylocal.Manager.
// It walks through the complete lifecycle of working with Foundry Local:
// service management, model discovery, downloading, loading, and cleanup.
func main() {
	// Create a context with timeout for all operations
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	fmt.Println("=== Foundry Local Manager Example ===")
	fmt.Println()

	// Example 1: Basic manager creation and service lifecycle
	fmt.Println("1. Creating Manager and starting service...")
	manager := foundrylocal.NewManager()

	// Check if service is already running
	if manager.IsServiceRunning() {
		fmt.Println("   ✓ Service is already running")
	} else {
		fmt.Println("   • Starting Foundry Local service...")
		if err := manager.StartService(ctx); err != nil {
			log.Fatalf("Failed to start service: %v", err)
		}
		fmt.Println("   ✓ Service started successfully")
	}

	// Ensure we stop the service when done
	defer func() {
		fmt.Println("\n9. Cleaning up...")
		if err := manager.StopService(context.Background()); err != nil {
			fmt.Printf("   ⚠ Error stopping service: %v\n", err)
		} else {
			fmt.Println("   ✓ Service stopped successfully")
		}
	}()

	// Example 2: Get service information
	fmt.Println("\n2. Getting service information...")
	endpoint := manager.Endpoint()
	fmt.Printf("   • API endpoint: %s\n", endpoint.String())

	cacheLocation, err := manager.GetCacheLocation(ctx)
	if err != nil {
		log.Printf("   ⚠ Failed to get cache location: %v", err)
	} else {
		fmt.Printf("   • Cache location: %s\n", cacheLocation)
	}

	// Example 3: List available models in catalog
	fmt.Println("\n3. Listing available models...")
	catalogModels, err := manager.ListCatalogModels(ctx)
	if err != nil {
		log.Fatalf("Failed to list catalog models: %v", err)
	}

	fmt.Printf("   • Found %d models in catalog\n", len(catalogModels))
	if len(catalogModels) > 0 {
		fmt.Println("   • First few models:")
		index := min(3, len(catalogModels))
		for i, model := range catalogModels[:index] {
			fmt.Printf("     %d. %s (%s) - %s\n", i+1, model.DisplayName, model.ModelID, model.ProviderType)
		}
		if len(catalogModels) > 3 {
			fmt.Printf("     ... and %d more\n", len(catalogModels)-3)
		}
	}

	// Example 4: Get information about a specific model
	fmt.Println("\n4. Getting model information...")

	// Try to find a good model to use for the example
	var targetModel string
	possibleModels := []string{"phi-3.5-mini", "qwen2.5-0.5b", "Phi-3.5-mini-instruct-generic-gpu"}

	for _, modelName := range possibleModels {
		modelInfo, err := manager.GetModelInfo(ctx, modelName)
		if err != nil {
			fmt.Printf("   ⚠ Error checking model %s: %v\n", modelName, err)
			continue
		}
		if modelInfo.ModelID != "" {
			targetModel = modelName
			fmt.Printf("   ✓ Found model: %s\n", modelInfo.DisplayName)
			fmt.Printf("     • ID: %s\n", modelInfo.ModelID)
			fmt.Printf("     • Provider: %s\n", modelInfo.ProviderType)
			fmt.Printf("     • Size: %d MB\n", modelInfo.FileSizeMb)
			fmt.Printf("     • Execution Provider: %s\n", modelInfo.Runtime.ExecutionProvider)
			fmt.Printf("     • Device Type: %s\n", modelInfo.Runtime.DeviceType)
			break
		}
	}

	if targetModel == "" {
		// Fallback to first available model
		if len(catalogModels) > 0 {
			targetModel = catalogModels[0].ModelID
			fmt.Printf("   • Using first available model: %s\n", targetModel)
		} else {
			log.Fatal("No models available in catalog")
		}
	}

	// Example 5: List currently cached models
	fmt.Println("\n5. Checking cached models...")
	cachedModels, err := manager.ListCachedModels(ctx)
	if err != nil {
		log.Printf("   ⚠ Failed to list cached models: %v", err)
	} else {
		fmt.Printf("   • Found %d cached models\n", len(cachedModels))
		if len(cachedModels) > 0 {
			fmt.Println("   • Cached models:")
			for _, model := range cachedModels {
				fmt.Printf("     - %s (%s)\n", model.DisplayName, model.ModelID)
			}
		}
	}

	// Example 6: Download a model (with progress)
	fmt.Println("\n6. Downloading model (if not cached)...")

	// Check if model is already cached
	isCached := false
	for _, model := range cachedModels {
		if model.ModelID == targetModel {
			isCached = true
			break
		}
	}

	if isCached {
		fmt.Printf("   ✓ Model %s is already cached\n", targetModel)
	} else {
		fmt.Printf("   • Downloading model: %s\n", targetModel)

		// Download with progress reporting
		progressChan, err := manager.DownloadModelWithProgress(ctx, targetModel)
		if err != nil {
			log.Fatalf("Failed to start download: %v", err)
		}

		fmt.Print("   • Progress: ")
		lastPercentage := -1.0
		for progress := range progressChan {
			if progress.ErrorMessage != "" {
				fmt.Printf("\n   ✗ Download failed: %s\n", progress.ErrorMessage)
				log.Fatal("Download failed")
			}

			if progress.IsCompleted {
				fmt.Printf("\n   ✓ Download completed: %s\n", progress.ModelInfo.DisplayName)
				break
			}

			// Update progress display (avoid too frequent updates)
			if progress.Percentage-lastPercentage >= 5.0 {
				fmt.Printf("%.0f%% ", progress.Percentage)
				lastPercentage = progress.Percentage
			}
		}
	}

	// Example 7: List currently loaded models
	fmt.Println("\n7. Checking loaded models...")
	loadedModels, err := manager.ListLoadedModels(ctx)
	if err != nil {
		log.Printf("   ⚠ Failed to list loaded models: %v", err)
	} else {
		fmt.Printf("   • Found %d loaded models\n", len(loadedModels))
		if len(loadedModels) > 0 {
			fmt.Println("   • Loaded models:")
			for _, model := range loadedModels {
				fmt.Printf("     - %s (%s)\n", model.DisplayName, model.ModelID)
			}
		}
	}

	// Example 8: Load and unload a model
	fmt.Println("\n8. Loading and unloading model...")

	// Check if target model is already loaded
	isLoaded := false
	for _, model := range loadedModels {
		if model.ModelID == targetModel {
			isLoaded = true
			break
		}
	}

	if isLoaded {
		fmt.Printf("   • Model %s is already loaded\n", targetModel)
	} else {
		fmt.Printf("   • Loading model: %s\n", targetModel)
		loadedModel, err := manager.LoadModel(ctx, targetModel,
			foundrylocal.WithLoadTimeout(5*time.Minute))
		if err != nil {
			log.Printf("   ⚠ Failed to load model: %v", err)
		} else {
			fmt.Printf("   ✓ Model loaded successfully: %s\n", loadedModel.DisplayName)

			// Demonstrate unloading
			fmt.Printf("   • Unloading model: %s\n", targetModel)
			if err := manager.UnloadModel(ctx, targetModel); err != nil {
				log.Printf("   ⚠ Failed to unload model: %v", err)
			} else {
				fmt.Printf("   ✓ Model unloaded successfully\n")
			}
		}
	}

	// Example 9: Demonstrate convenience function
	fmt.Println("\n--- Bonus: Using StartModel convenience function ---")
	fmt.Println("• This function creates a manager, starts service, downloads and loads a model in one call")

	// Note: We'll skip actually running this since we already have a manager running
	fmt.Printf("• Example usage: foundrylocal.StartModel(ctx, \"%s\")\n", targetModel)

	fmt.Println("\n=== Example completed successfully! ===")
}
