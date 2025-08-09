// Package foundrylocal provides an SDK for interacting with the Microsoft Foundry Local runtime.
// For more information on Foundry Local see https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-local/what-is-foundry-local.
//
// This package enables developers to start and stop the Foundry Local runtime and perform operations such
// as downloading, loading, and unloading msall language models (SLM). After loading a model, an application
// can use the OpenAI compatible endpoints provided the Manager's Endpoint() method to interact with the model.
//
// This SDK supports both Windows and macOS, like the Foundry Local runtime itself.
//
// Basic usage:
//
//	// Create a new manager
//	manager := foundrylocal.NewManager()
//
//	// Start the Foundry Local service
//	ctx := context.Background()
//	if err := manager.StartService(ctx); err != nil {
//		log.Fatal(err)
//	}
//	defer manager.StopService(ctx)
//
//	// List available models
//	models, err := manager.ListCatalogModels(ctx)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Download and load a model
//	modelInfo, err := manager.DownloadModel(ctx, "qwen2.5-0.5b")
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	loadedModel, err := manager.LoadModel(ctx, "qwen2.5-0.5b")
//	if err != nil {
//		log.Fatal(err)
//	}
package foundrylocal

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/url"
	"os/exec"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"
)

// DeviceType represents the type of device used for model execution.
type DeviceType string

const (
	// DeviceTypeCPU indicates CPU-based execution.
	DeviceTypeCPU DeviceType = "CPU"
	// DeviceTypeGPU indicates GPU-based execution.
	DeviceTypeGPU = "GPU"
	// DeviceTypeNPU indicates Neural Processing Unit execution.
	DeviceTypeNPU = "NPU"
	// DeviceTypeInvalid represents an invalid or unknown device type.
	DeviceTypeInvalid = "Invalid"
)

// ExecutionProvider represents the execution provider for model inference.
type ExecutionProvider string

const (
	// ExecutionProviderInvalid represents an invalid or unknown execution provider.
	ExecutionProviderInvalid ExecutionProvider = "Invalid"
	// ExecutionProviderCPU indicates CPU-based execution.
	ExecutionProviderCPU = "CPU"
	// ExecutionProviderWebGPU indicates WebGPU-based execution.
	ExecutionProviderWebGPU = "WebGPU"
	// ExecutionProviderCUDA indicates CUDA-based execution for NVIDIA GPUs.
	ExecutionProviderCUDA = "CUDA"
	// ExecutionProviderQNN indicates Qualcomm Neural Network SDK execution.
	ExecutionProviderQNN = "QNN"
)

// testIsRunning is a regular expression used to extract service endpoint URLs from foundry status output.
var testIsRunning = regexp.MustCompile(`is running on (http://.*)\s+`)

// ErrModelNotInCatalog is returned when a requested model is not found in the Foundry Local catalog.
var ErrModelNotInCatalog = errors.New("model not found in catalog")

// ErrModelUpgradeFailed is returned when a model upgrade operation fails.
var ErrModelUpgradeFailed = errors.New("failed to upgrade model")

// Manager provides methods to interact with the Foundry Local runtime.
// It manages the lifecycle of the Foundry Local service and provides
// operations for model management including downloading, loading, and unloading models.
//
// The Manager maintains internal state including:
//   - HTTP client for API communication
//   - Service URL for the running Foundry Local instance
//   - Cached model catalog and mapping
//   - Execution provider priority configuration
//
// Example:
//
//	manager := foundrylocal.NewManager()
//	defer manager.StopService(context.Background())
//
//	if err := manager.StartService(context.Background()); err != nil {
//		log.Fatal(err)
//	}
type Manager struct {
	client        *http.Client
	serviceURL    *url.URL
	catalogModels []ModelInfo
	catalogMap    map[string]ModelInfo
	priorityMap   map[ExecutionProvider]int

	// ApiKey is the API key used for authentication with external services.
	// Default value is "OPENAI_API_KEY".
	ApiKey string

	// Logger is an optional logger for the Manager. If not set,
	// a logger using slog.DiscardHandler will be used.
	Logger *slog.Logger
}

// NewManager creates a new Manager instance with the specified options.
// If no options are provided, it uses OS-specific defaults for execution provider priorities.
//
// Example:
//
//	// Create with OS defaults (preferred method)
//	manager := foundrylocal.NewManager()
//
//	// Create with OS defaults explicitly set
//	manager := foundrylocal.NewManager(
//		foundrylocal.WithAutoConfigure(),
//	)
func NewManager(opts ...ManagerOption) *Manager {
	m := &Manager{
		ApiKey: "OPENAI_API_KEY",
	}

	for _, opt := range opts {
		opt(m)
	}

	if m.priorityMap == nil {
		WithAutoConfigure()(m)
	}
	if m.Logger == nil {
		m.Logger = slog.New(slog.DiscardHandler)
	}
	return m
}

// IsServiceRunning returns true if the Foundry Local service is currently running
// and accessible.
//
// Example:
//
//	if !manager.IsServiceRunning() {
//		if err := manager.StartService(ctx); err != nil {
//			log.Fatal(err)
//		}
//	}
func (m *Manager) IsServiceRunning() bool {
	return m.serviceURL != nil
}

// Endpoint returns the base URL for the Foundry Local OpenAI API endpoints.
// This method panics if the service is not running.
//
// Example:
//
//	if manager.IsServiceRunning() {
//		apiURL := manager.Endpoint()
//		fmt.Printf("API available at: %s\n", apiURL.String())
//	}
func (m *Manager) Endpoint() *url.URL {
	if m.serviceURL == nil {
		panic("serviceURL is not set")
	}
	return m.serviceURL.JoinPath("v1")
}

// StartModel is a convenience function that creates a new Manager, starts the service,
// and prepares the specified model for use by downloading and loading it.
// This is useful for quick setup scenarios where you want to get a model running
// with minimal setup code.
//
// The function performs the following steps:
//  1. Creates a new Manager with default settings
//  2. Starts the Foundry Local service
//  3. Looks up the model information
//  4. Downloads the model if not already cached
//  5. Loads the model for inference
//
// Example:
//
//	manager, err := foundrylocal.StartModel(ctx, "qwen2.5-0.5b")
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer manager.StopService(ctx)
func StartModel(ctx context.Context, aliasorModelID string) (*Manager, error) {
	m := NewManager()
	if err := m.StartService(ctx); err != nil {
		return nil, err
	}

	modelInfo, err := m.GetModelInfo(ctx, aliasorModelID)
	if err != nil {
		if errors.Is(err, ErrModelNotInCatalog) {
			m.Logger.ErrorContext(ctx, "model not found in catalog", "aliasOrModelID", aliasorModelID)
		}
		return nil, err
	}

	if _, err := m.DownloadModel(ctx, modelInfo.ID); err != nil {
		return nil, err
	}

	if _, err := m.LoadModel(ctx, aliasorModelID); err != nil {
		return nil, err
	}
	return m, nil
}

// StartService starts the Foundry Local service if it's not already running.
// This method is idempotent - calling it multiple times is safe.
//
// Example:
//
//	ctx := context.Background()
//	if err := manager.StartService(ctx); err != nil {
//		log.Fatalf("Failed to start service: %v", err)
//	}
func (m *Manager) StartService(ctx context.Context) error {
	if m.serviceURL != nil {
		m.Logger.InfoContext(ctx, "Foundry service is already running", "endpoint", m.serviceURL.String())
		return nil
	}

	endpoint, err := m.ensureServiceRunning(ctx)
	if err != nil {
		m.Logger.ErrorContext(ctx, "Foundry service did not start", "error", err)
		return err
	}

	m.serviceURL = endpoint
	m.client = &http.Client{
		Timeout: time.Duration(2) * time.Hour,
	}
	m.Logger.InfoContext(ctx, "Foundry service started successfully", "endpoint", m.serviceURL.String())
	return nil
}

// StopService stops the Foundry Local service if it's currently running.
// This method is idempotent - calling it multiple times is safe.
// After stopping, the Manager will need to be restarted before performing model operations.
//
// Example:
//
//	defer func() {
//		if err := manager.StopService(context.Background()); err != nil {
//			log.Printf("Error stopping service: %v", err)
//		}
//	}()
func (m *Manager) StopService(ctx context.Context) error {
	if m.serviceURL == nil {
		m.Logger.InfoContext(ctx, "Foundry service not running, nothing to stop")
		return nil
	}

	_, err := m.invokeFoundry(ctx, "service stop")
	m.serviceURL = nil
	m.client = nil
	m.Logger.InfoContext(ctx, "Foundry service stopped")
	return err
}

// ListCatalogModels returns all available models from the Foundry Local catalog.
// Results are cached after the first call until RefreshCatalog is called.
//
// Example:
//
//	models, err := manager.ListCatalogModels(ctx)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	for _, model := range models {
//		fmt.Printf("Model: %s (%s)\n", model.DisplayName, model.ID)
//	}
func (m *Manager) ListCatalogModels(ctx context.Context) ([]ModelInfo, error) {
	if m.catalogModels != nil {
		return m.catalogModels, nil
	}

	if err := m.StartService(ctx); err != nil {
		return nil, err
	}

	endpoint := m.serviceURL.JoinPath("foundry", "list")
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint.String(), nil)
	if err != nil {
		return nil, err
	}

	resp, err := m.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if !ensureSuccessStatusCode(resp) {
		return nil, fmt.Errorf("received non-success status code %d", resp.StatusCode)
	}

	var models []ModelInfo
	if err = json.NewDecoder(resp.Body).Decode(&models); err != nil || models == nil {
		return []ModelInfo{}, err
	}

	m.catalogModels = models
	return m.catalogModels, nil
}

// RefreshCatalog clears the cached model catalog and mapping,
// forcing the next call to ListCatalogModels or GetModelInfo to fetch fresh data.
//
// Example:
//
//	// Clear cache to get latest model information
//	manager.RefreshCatalog()
//	models, err := manager.ListCatalogModels(ctx)
func (m *Manager) RefreshCatalog() {
	m.catalogModels = nil
	m.catalogMap = nil
}

// GetModelInfo retrieves detailed information about a specific model by its ID or alias.
// Returns the model information, a boolean indicating if the model was found,
// and any error that occurred.
//
// The function uses a priority system when multiple models share the same alias,
// preferring models with higher-priority execution providers based on the Manager's
// configuration.
//
// Example:
//
//	modelInfo, err := manager.GetModelInfo(ctx, "qwen2.5-0.5b")
//	if err != nil {
//		if errors.Is(err, foundrylocal.ErrModelNotInCatalog) {
//			log.Fatal("Model not found in catalog")
//		}
//		log.Fatal(err)
//	}
//	fmt.Printf("Found model: %s\n", modelInfo.DisplayName)
func (m *Manager) GetModelInfo(ctx context.Context, aliasOrModelID string) (ModelInfo, error) {
	dict, err := m.getCatalogMap(ctx)
	if err != nil {
		return ModelInfo{}, err
	}
	model, ok := dict[aliasOrModelID]
	if !ok {
		if !strings.Contains(aliasOrModelID, ":") {
			prefix := strings.ToLower(aliasOrModelID) + ":"
			bestVersion := -1
			found := false

			for k, v := range dict {
				if strings.HasPrefix(strings.ToLower(k), prefix) {
					if version := GetVersion(k); version > bestVersion {
						bestVersion = version
						model = v
						found = true
					}
				}
			}

			if !found {
				return ModelInfo{}, ErrModelNotInCatalog
			}
		} else {
			// Input contains a version suffix (aliasOrModelID includes ':') but no exact match in catalog
			return ModelInfo{}, ErrModelNotInCatalog
		}
	}

	return model, nil
}

// GetCacheLocation returns the filesystem path where Foundry Local stores cached models.
//
// Example:
//
//	cachePath, err := manager.GetCacheLocation(ctx)
//	if err != nil {
//		log.Fatal(err)
//	}
//	fmt.Printf("Models cached at: %s\n", cachePath)
func (m *Manager) GetCacheLocation(ctx context.Context) (string, error) {
	if err := m.StartService(ctx); err != nil {
		return "", err
	}

	endpoint := m.serviceURL.JoinPath("openai", "status")
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint.String(), nil)
	if err != nil {
		return "", err
	}

	resp, err := m.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result struct {
		ModelDirPath string `json:"modelDirPath"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	if result.ModelDirPath == "" {
		return "", fmt.Errorf("model directory path not found in response")
	}
	return result.ModelDirPath, nil
}

// ListCachedModels returns information about all models currently cached locally.
// These are models that have been downloaded and are available for loading.
//
// Example:
//
//	cachedModels, err := manager.ListCachedModels(ctx)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	fmt.Printf("Found %d cached models:\n", len(cachedModels))
//	for _, model := range cachedModels {
//		fmt.Printf("  - %s\n", model.DisplayName)
//	}
func (m *Manager) ListCachedModels(ctx context.Context) ([]ModelInfo, error) {
	if err := m.StartService(ctx); err != nil {
		return nil, err
	}

	endpoint := m.serviceURL.JoinPath("openai", "models")
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint.String(), nil)
	if err != nil {
		return nil, err
	}

	resp, err := m.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("received non-success status code %d", resp.StatusCode)
	}

	var models []string
	if err = json.NewDecoder(resp.Body).Decode(&models); err != nil || models == nil {
		return []ModelInfo{}, err
	}
	return m.fetchModelInfo(ctx, models...)
}

// DownloadModel downloads a model to the local cache if it's not already present.
// By default, if the model is already cached, this operation is skipped.
// Use WithForceDownload() to re-download existing models.
//
// Supported options:
//   - WithToken(token): Provide authentication token for private models
//   - WithForceDownload(): Force re-download even if model exists locally
//
// Example:
//
//	// Basic download
//	modelInfo, err := manager.DownloadModel(ctx, "qwen2.5-0.5b")
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Download with authentication
//	modelInfo, err := manager.DownloadModel(ctx, "private-model",
//		foundrylocal.WithToken("your-token"))
//	if err != nil {
//		log.Fatal(err)
//	}
func (m *Manager) DownloadModel(ctx context.Context, aliasorModelID string, opts ...DownloadOption) (ModelInfo, error) {
	var config downloadConfig
	for _, opt := range opts {
		opt(&config)
	}

	modelInfo, err := m.GetModelInfo(ctx, aliasorModelID)
	if err != nil {
		return ModelInfo{}, err
	}

	localModels, err := m.ListCachedModels(ctx)
	if err != nil {
		return ModelInfo{}, err
	}
	if slices.ContainsFunc(localModels, matchAliasOrId(aliasorModelID)) && !config.force {
		m.Logger.InfoContext(ctx, "model already exists locally", "alias", modelInfo.Alias, "modelID", modelInfo.ID)
		return modelInfo, nil
	}

	request := DownloadRequest{
		Model: DownloadRequestModelInfo{
			Name:           modelInfo.ID,
			URI:            modelInfo.URI,
			ProviderType:   modelInfo.ProviderType + "Local",
			PromptTemplate: modelInfo.PromptTemplate,
		},
		Token:            config.token,
		IgnorePipeReport: true,
	}
	requestBody, err := json.Marshal(request)
	if err != nil {
		return ModelInfo{}, err
	}

	endpoint := m.serviceURL.JoinPath("openai", "download")
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), bytes.NewBuffer(requestBody))
	if err != nil {
		return ModelInfo{}, err
	}

	req.Header.Set("Content-Type", "application/json")
	m.Logger.InfoContext(ctx, "downloading model", "alias", modelInfo.Alias, "modelID", modelInfo.ID)
	resp, err := m.client.Do(req)
	if err != nil {
		return ModelInfo{}, err
	}
	defer resp.Body.Close()

	if !ensureSuccessStatusCode(resp) {
		return ModelInfo{}, fmt.Errorf("received non-success status code %d", resp.StatusCode)
	}

	responseBodyBytes, _ := io.ReadAll(resp.Body)
	responseBody := string(responseBodyBytes)
	// Find the last '{' to get the start of the JSON object
	jsonStart := strings.LastIndex(responseBody, "{")
	if jsonStart == -1 {
		return ModelInfo{}, fmt.Errorf("no JSON object found in response")
	}
	jsonPart := responseBody[jsonStart:]
	var jsonDoc struct {
		Success      bool   `json:"success"`
		ErrorMessage string `json:"errorMessage"`
	}

	if err := json.Unmarshal([]byte(jsonPart), &jsonDoc); err != nil {
		return ModelInfo{}, err
	}

	if !jsonDoc.Success {
		return ModelInfo{}, fmt.Errorf("failed to download model: %s", jsonDoc.ErrorMessage)
	}
	return modelInfo, nil
}

// LoadModel loads a previously downloaded model into memory for inference.
// The model must be downloaded first using DownloadModel.
// Loading may take significant time for large models.
//
// The function automatically selects the best execution provider based on:
//   - Available hardware (GPU, NPU, CPU)
//   - Model requirements
//   - Manager's execution provider priorities
//
// Supported options:
//   - WithLoadTimeout(duration): Set custom timeout (default: 10 minutes)
//
// Example:
//
//	// Load with default timeout
//	modelInfo, err := manager.LoadModel(ctx, "qwen2.5-0.5b")
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Load with custom timeout
//	modelInfo, err := manager.LoadModel(ctx, "large-model",
//		foundrylocal.WithLoadTimeout(30*time.Minute))
//	if err != nil {
//		log.Fatal(err)
//	}
func (m *Manager) LoadModel(ctx context.Context, aliasOrModelID string, opts ...LoadModelOption) (ModelInfo, error) {
	config := loadModelConfig{
		timeout: time.Minute * 10, // Default timeout
	}
	for _, opt := range opts {
		opt(&config)
	}

	modelInfo, err := m.GetModelInfo(ctx, aliasOrModelID)
	if err != nil {
		return ModelInfo{}, err
	}

	localModelInfo, err := m.ListCachedModels(ctx)
	if err != nil {
		return ModelInfo{}, err
	}

	if !slices.ContainsFunc(localModelInfo, matchAliasOrId(aliasOrModelID)) {
		return ModelInfo{}, fmt.Errorf("model %s not found in local models, download first", aliasOrModelID)
	}

	endpoint := *m.serviceURL.JoinPath("openai", "load", modelInfo.ID)

	params := url.Values{}
	params.Set("ttl", fmt.Sprintf("%d", int64(config.timeout.Seconds())))

	if modelInfo.Runtime.ExecutionProvider == ExecutionProviderCUDA || modelInfo.Runtime.ExecutionProvider == ExecutionProviderWebGPU {
		hasCUDASupport := slices.ContainsFunc(localModelInfo, func(mi ModelInfo) bool {
			return mi.Runtime.ExecutionProvider == ExecutionProviderCUDA
		})
		if hasCUDASupport {
			params.Set("ep", "cuda")
		} else {
			params.Set("ep", strings.ToLower(string(modelInfo.Runtime.ExecutionProvider)))
		}
	}

	endpoint.RawQuery = params.Encode()
	m.Logger.InfoContext(ctx, "loading model", "alias", modelInfo.Alias, "modelID", modelInfo.ID)
	resp, err := m.client.Get(endpoint.String())
	if err != nil {
		return ModelInfo{}, err
	}
	defer resp.Body.Close()

	if !ensureSuccessStatusCode(resp) {
		return ModelInfo{}, fmt.Errorf("received non-success status code %d", resp.StatusCode)
	}
	return modelInfo, nil
}

// DownloadModelWithProgress downloads a model and reports progress through a channel.
// This is useful for long-running downloads where you want to show progress to users.
// The returned channel will receive progress updates and will be closed when the
// operation completes (successfully or with an error).
//
// The progress channel receives ModelDownloadProgress structs containing:
//   - Percentage: Download progress (0-100)
//   - IsCompleted: Whether the operation is finished
//   - ModelInfo: Final model information (only when successfully completed)
//   - ErrorMessage: Error details (only when failed)
//
// Example:
//
//	progressChan, err := manager.DownloadModelWithProgress(ctx, "qwen2.5-0.5b")
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	for progress := range progressChan {
//		if progress.ErrorMessage != "" {
//			log.Fatalf("Download failed: %s", progress.ErrorMessage)
//		}
//		if progress.IsCompleted {
//			fmt.Println("Download completed!")
//			break
//		}
//		fmt.Printf("Progress: %.1f%%\n", progress.Percentage)
//	}
func (m *Manager) DownloadModelWithProgress(ctx context.Context, aliasOrModelID string, opts ...DownloadOption) (<-chan ModelDownloadProgress, error) {
	var config downloadConfig
	for _, opt := range opts {
		opt(&config)
	}
	progressChan := make(chan ModelDownloadProgress, 1)

	if m.client == nil {
		go func() {
			defer close(progressChan)
			progressChan <- NewDownloadError("service not started")
		}()
		return progressChan, nil
	}

	go func() {
		defer close(progressChan)

		modelInfo, err := m.GetModelInfo(ctx, aliasOrModelID)
		if err != nil {
			progressChan <- NewDownloadError(err.Error())
			return
		}

		localModels, err := m.ListCachedModels(ctx)
		if err != nil {
			progressChan <- NewDownloadError(err.Error())
			return
		}

		if slices.ContainsFunc(localModels, matchAliasOrId(aliasOrModelID)) && !config.force {
			progressChan <- NewDownloadCompleted(modelInfo)
			return
		}

		payload := DownloadRequest{
			Model: DownloadRequestModelInfo{
				Name:           modelInfo.ID,
				URI:            modelInfo.URI,
				ProviderType:   modelInfo.ProviderType + "Local",
				PromptTemplate: modelInfo.PromptTemplate,
			},
			Token:            config.token,
			IgnorePipeReport: true,
		}

		bodyBytes, err := json.Marshal(payload)
		if err != nil {
			progressChan <- NewDownloadError(err.Error())
			return
		}

		endpoint := m.serviceURL.JoinPath("openai", "download")
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), bytes.NewBuffer(bodyBytes))
		if err != nil {
			progressChan <- NewDownloadError(err.Error())
			return
		}

		req.Header.Set("Content-Type", "application/json")
		resp, err := m.client.Do(req)
		if err != nil {
			progressChan <- NewDownloadError(err.Error())
			return
		}
		defer resp.Body.Close()
		if !ensureSuccessStatusCode(resp) {
			progressChan <- NewDownloadError(fmt.Sprintf("received non-success status code %d", resp.StatusCode))
			return
		}

		scanner := bufio.NewScanner(resp.Body)
		var jsonBuilder strings.Builder
		var collectingJSON bool
		var completed bool

		for !completed && scanner.Scan() {
			select {
			case <-ctx.Done():
				progressChan <- NewDownloadError("context cancelled")
				return
			default:
			}
			line := scanner.Text()
			if strings.HasPrefix(strings.ToLower(line), "total") &&
				strings.Contains(line, "Downloading") &&
				strings.Contains(line, "%") {
				parts := strings.Fields(line)
				for _, part := range parts {
					if strings.HasSuffix(part, "%") {
						percentStr := strings.TrimSuffix(part, "%")
						var percent float64
						fmt.Sscanf(percentStr, "%f", &percent)
						progressChan <- NewDownloadProgress(percent)
						break
					}
				}
			} else if strings.Contains(line, "[DONE]") || strings.Contains(line, "All Completed") {
				collectingJSON = true
			} else if collectingJSON && strings.HasPrefix(strings.TrimSpace(line), "{") {
				// Start collecting JSON from the first '{' found
				jsonBuilder.WriteString(line + "\n")
			} else if collectingJSON && jsonBuilder.Len() > 0 {
				// Continue collecting JSON until we find the end
				jsonBuilder.WriteString(line + "\n")
				if strings.TrimSpace(line) == "}" {
					completed = true
					break
				}
			}
		}

		if err := scanner.Err(); err != nil && !errors.Is(err, io.EOF) {
			progressChan <- NewDownloadError(err.Error())
			return
		}

		if jsonBuilder.Len() == 0 {
			progressChan <- NewDownloadError("No completion response received")
			return
		}

		var result map[string]any
		if err := json.Unmarshal([]byte(jsonBuilder.String()), &result); err != nil {
			progressChan <- NewDownloadError(err.Error())
			return
		}
		if success, ok := result["success"].(bool); ok && success {
			progressChan <- NewDownloadCompleted(modelInfo)
		} else {
			msg := "unknown error"
			if m, ok := result["errorMessage"].(string); ok {
				msg = m
			}
			progressChan <- NewDownloadError(msg)
		}
	}()
	return progressChan, nil
}

// ListLoadedModels returns information about all models currently loaded in memory
// and available for inference.
//
// Example:
//
//	loadedModels, err := manager.ListLoadedModels(ctx)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	if len(loadedModels) == 0 {
//		fmt.Println("No models currently loaded")
//	} else {
//		fmt.Printf("Loaded models:\n")
//		for _, model := range loadedModels {
//			fmt.Printf("  - %s (%s)\n", model.DisplayName, model.ID)
//		}
//	}
func (m *Manager) ListLoadedModels(ctx context.Context) ([]ModelInfo, error) {
	endpoint := m.serviceURL.JoinPath("openai", "loadedmodels")
	resp, err := m.client.Get(endpoint.String())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if !ensureSuccessStatusCode(resp) {
		return nil, fmt.Errorf("received non-success status code %d", resp.StatusCode)
	}

	// Use a pointer to a slice to decode the JSON response
	// This allows us to umarshal a "null" response as nil.
	var names *[]string
	if err = json.NewDecoder(resp.Body).Decode(&names); err != nil || names == nil {
		return nil, err
	}
	return m.fetchModelInfo(ctx, *names...)
}

// UnloadModel removes a model from memory, freeing up resources.
// The model remains cached locally and can be loaded again later.
// This operation is forced, meaning it will unload the model even if it's currently in use.
//
// Example:
//
//	if err := manager.UnloadModel(ctx, "qwen2.5-0.5b"); err != nil {
//		log.Printf("Failed to unload model: %v", err)
//	} else {
//		fmt.Println("Model unloaded successfully")
//	}
func (m *Manager) UnloadModel(ctx context.Context, aliasOrModelID string) error {
	modelInfo, err := m.GetModelInfo(ctx, aliasOrModelID)
	if err != nil {
		return err
	}

	endpoint := m.serviceURL.JoinPath("openai", "unload", modelInfo.ID)
	params := url.Values{}
	params.Set("force", "true")
	endpoint.RawQuery = params.Encode()
	m.Logger.InfoContext(ctx, "unloading model", "alias", modelInfo.Alias, "modelID", modelInfo.ID)
	resp, err := m.client.Get(endpoint.String())
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if !ensureSuccessStatusCode(resp) {
		return fmt.Errorf("received non-success status code %d", resp.StatusCode)
	}
	return nil
}

// GetLatestModelInfo retrieves the latest version of a model by its alias or ID.
// If the input contains a version suffix (":"), it attempts to find an exact match first.
// Otherwise, it delegates to GetModelInfo to find the latest version.
//
// Example:
//
//	// Get latest version of a model by alias
//	modelInfo, err := manager.GetLatestModelInfo(ctx, "qwen2.5")
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Get specific version
//	modelInfo, err := manager.GetLatestModelInfo(ctx, "qwen2.5:1")
//	if err != nil {
//		log.Fatal(err)
//	}
func (m *Manager) GetLatestModelInfo(ctx context.Context, aliasOrModelID string) (ModelInfo, error) {
	if aliasOrModelID == "" {
		return ModelInfo{}, fmt.Errorf("aliasOrModelID cannot be empty")
	}

	catalog, err := m.getCatalogMap(ctx)
	if err != nil {
		return ModelInfo{}, err
	}

	// If alias or id without version
	if strings.Contains(aliasOrModelID, ":") {
		modelInfo, ok := catalog[aliasOrModelID]
		// If there is an exact match in catalog, return it directly
		if ok {
			return modelInfo, nil
		}

		// Otherwise, GetModelInfo will get the latest version
		return m.GetModelInfo(ctx, aliasOrModelID)
	}
	idWithoutVersion := strings.Split(aliasOrModelID, ":")[0]
	return m.GetModelInfo(ctx, idWithoutVersion)
}

func (m *Manager) IsModelUpgradable(ctx context.Context, aliasOrModelID string) (bool, error) {
	modelInfo, err := m.GetLatestModelInfo(ctx, aliasOrModelID)
	if err != nil {
		if errors.Is(err, ErrModelNotInCatalog) {
			// Model not found in catalog
			return false, nil
		}
		// Other error occurred while fetching model info
		return false, err
	}

	latestVersion := GetVersion(modelInfo.ID)
	if latestVersion == -1 {
		// Invalid version format in model ID
		return false, nil
	}

	cachedModels, err := m.ListCachedModels(ctx)
	if err != nil {
		// Error occurred while listing cached models
		return false, err
	}

	for _, cachedModel := range cachedModels {
		if strings.EqualFold(cachedModel.ID, modelInfo.ID) && GetVersion(cachedModel.ID) == latestVersion {
			// Cached model is already at latest version
			return false, nil
		}
	}

	// Latest version not in cache - upgrade available
	return true, nil
}

// UpgradeModel upgrades a model to its latest available version by downloading it.
// This is a convenience method that combines GetLatestModelInfo and DownloadModel.
// If a token is provided, it will be used for authentication when downloading private models.
//
// Example:
//
//	// Upgrade a public model
//	modelInfo, err := manager.UpgradeModel(ctx, "qwen2.5", "")
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Upgrade a private model with token
//	modelInfo, err := manager.UpgradeModel(ctx, "private-model", "your-token")
//	if err != nil {
//		log.Fatal(err)
//	}
func (m *Manager) UpgradeModel(ctx context.Context, aliasOrModelID, token string) (ModelInfo, error) {
	modelInfo, err := m.GetLatestModelInfo(ctx, aliasOrModelID)
	if err != nil {
		return ModelInfo{}, err
	}
	opts := []DownloadOption{}
	if token != "" {
		opts = append(opts, WithToken(token))
	}
	mi, err := m.DownloadModel(ctx, modelInfo.ID, opts...)
	if err != nil {
		return ModelInfo{}, ErrModelUpgradeFailed
	}
	return mi, nil

}

// fetchModelInfo retrieves ModelInfo structs for the given list of aliases or model IDs.
// Models not found in the catalog are skipped with a debug log entry rather than causing
// the entire operation to fail. This allows the method to work gracefully with cached
// or loaded models that may not be present in the current catalog.
func (m *Manager) fetchModelInfo(ctx context.Context, aliasesOrModelIDs ...string) ([]ModelInfo, error) {
	modelInfos := make([]ModelInfo, 0, len(aliasesOrModelIDs))
	for _, aliasOrID := range aliasesOrModelIDs {
		model, err := m.GetModelInfo(ctx, aliasOrID)
		if err != nil {
			if errors.Is(err, ErrModelNotInCatalog) {
				m.Logger.DebugContext(ctx, "model not found in catalog", "aliasOrID", aliasOrID)
				continue
			}
			return nil, err
		}
		modelInfos = append(modelInfos, model)
	}
	return modelInfos, nil
}

// getCatalogMap builds and caches a map from model IDs and aliases to ModelInfo structs.
// The map includes direct model ID lookups and alias resolution with priority-based selection.
// When multiple models share the same alias, the best candidate is chosen based on:
//  1. Execution provider priority (lower priority values are preferred)
//  2. Version number (higher versions are preferred as tie-breakers)
//
// The map is cached after the first call until RefreshCatalog is called.
func (m *Manager) getCatalogMap(ctx context.Context) (map[string]ModelInfo, error) {
	if m.catalogMap != nil {
		return m.catalogMap, nil
	}

	models, err := m.ListCatalogModels(ctx)
	if err != nil {
		return nil, err
	}

	dict := make(map[string]ModelInfo)
	aliasCandidates := make(map[string][]ModelInfo)

	for _, model := range models {
		dict[model.ID] = model
		if model.Alias != "" {
			// Use lower-cased key for case-insensitive grouping
			key := strings.ToLower(model.Alias)
			if _, ok := aliasCandidates[key]; !ok {
				aliasCandidates[key] = make([]ModelInfo, 0, 1)
			}
			aliasCandidates[key] = append(aliasCandidates[key], model)
		}
	}

	for k, v := range aliasCandidates {
		alias := k
		candidates := v
		bestCandidate := aggregate(candidates, func(best, current ModelInfo) ModelInfo {
			bestPriority, ok := m.priorityMap[best.Runtime.ExecutionProvider]
			if !ok {
				bestPriority = math.MaxInt
			}
			currentPriority, ok := m.priorityMap[current.Runtime.ExecutionProvider]
			if !ok {
				currentPriority = math.MaxInt
			}

			if currentPriority < bestPriority {
				return current
			}

			if currentPriority == bestPriority {
				bestVersion := GetVersion(best.ID)
				currentVersion := GetVersion(current.ID)
				if currentVersion > bestVersion {
					return current
				}
			}
			return best
		})
		dict[alias] = bestCandidate
	}

	m.catalogMap = dict
	return m.catalogMap, nil
}

// ensureServiceRunning starts the Foundry Local service if it's not already running
// and returns the service endpoint URL.
func (m *Manager) ensureServiceRunning(ctx context.Context) (*url.URL, error) {
	cmd := exec.CommandContext(ctx, "foundry", "service", "start")
	err := cmd.Run()
	if err != nil {
		return nil, err
	}
	return m.statusEndpoint(ctx)
}

// statusEndpoint retrieves the current service endpoint URL by calling 'foundry service status'
// and parsing the output for the running service URL.
func (m *Manager) statusEndpoint(ctx context.Context) (*url.URL, error) {
	statusResult, err := m.invokeFoundry(ctx, "service status")
	if err != nil {
		return nil, err
	}

	matches := testIsRunning.FindStringSubmatch(statusResult)
	if len(matches) < 2 {
		return nil, nil
	}

	endpoint, err := url.Parse(strings.TrimSpace(matches[1]))
	if err != nil {
		return nil, err
	}

	endpoint.Path = ""
	return endpoint, nil
}

// invokeFoundry executes the foundry command-line tool with the given arguments
// and returns the combined stdout and stderr output.
func (m *Manager) invokeFoundry(ctx context.Context, args string) (string, error) {
	cmdArgs := strings.Split(args, " ")
	cmd := exec.CommandContext(ctx, "foundry", cmdArgs...)
	bytes, err := cmd.CombinedOutput()
	return string(bytes), err
}

// GetVersion extracts the version number from a model ID that follows the format "name:version".
// Returns the version as an integer, or -1 if the model ID doesn't contain a valid version suffix.
//
// Example:
//
//	version := foundrylocal.GetVersion("qwen2.5-0.5b:2") // returns 2
//	version := foundrylocal.GetVersion("qwen2.5-0.5b")   // returns -1
func GetVersion(modelID string) int {
	if modelID == "" {
		return -1
	}

	parts := strings.Split(modelID, ":")
	if len(parts) < 2 {
		return -1
	}

	versionPart := parts[len(parts)-1]
	version, err := strconv.Atoi(versionPart)
	if err != nil {
		return -1 // Return -1 if version parsing fails
	}
	return version
}

// matchAliasOrId returns a predicate function that checks if a ModelInfo matches
// the given alias or model ID using case-insensitive comparison.
func matchAliasOrId(aliasOrModelID string) func(modelInfo ModelInfo) bool {
	return func(modelInfo ModelInfo) bool {
		return strings.EqualFold(modelInfo.ID, aliasOrModelID) ||
			strings.EqualFold(modelInfo.Alias, aliasOrModelID)
	}
}

// ensureSuccessStatusCode checks if an HTTP response has a success status code (2xx).
// Returns true for status codes in the range 200-299, false otherwise.
func ensureSuccessStatusCode(resp *http.Response) bool {
	return resp.StatusCode >= 200 && resp.StatusCode < 300
}

// aggregate applies a binary function to reduce a slice of items to a single value.
// The function starts with the first item and applies the function with each subsequent item.
// Panics if called with an empty slice.
func aggregate[T any](items []T, fn func(T, T) T) T {
	if len(items) == 0 {
		panic("aggregate called with empty items")
	}

	res := items[0]
	if len(items) > 1 {
		items = items[1:]
		for _, i := range items {
			res = fn(res, i)
		}
	}
	return res
}
