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
//	modelInfo, err := manager.DownloadModel(ctx, "qwen2.5-0.5b", nil)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	loadedModel, err := manager.LoadModel(ctx, "qwen2.5-0.5b", nil)
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

var (
	// version holds the current version of the SDK, set at build time.
	version string

	// testIsRunning is a regular expression used to extract service endpoint URLs from foundry status output.
	testIsRunning = regexp.MustCompile(`is running on (http://.*)\s+`)

	// ErrModelNotInCatalog is returned when a requested model is not found in the Foundry Local catalog.
	ErrModelNotInCatalog = errors.New("model not found in catalog")

	// ErrModelUpgradeFailed is returned when a model upgrade operation fails.
	ErrModelUpgradeFailed = errors.New("failed to upgrade model")

	// ErrReadLoadedModels is returned when the list of loaded models cannot be read.
	ErrReadLoadedModels = errors.New("failed to read loaded models")
)

type sdkRoundTripper struct {
	t http.RoundTripper
}

func (rt *sdkRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) {
	req := r.Clone(r.Context())
	req.Header.Set("User-Agent", "go-foundrylocal/"+version)
	return rt.t.RoundTrip(req)
}

// Manager provides methods to interact with the Foundry Local runtime.
// It manages the lifecycle of the Foundry Local service and provides
// operations for model management including downloading, loading, and unloading models.
//
// The Manager maintains internal state including:
//   - HTTP client for API communication
//   - Service URL for the running Foundry Local instance
//   - Cached model catalog and mapping
//   - OS specific configuration
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
	client             *http.Client
	serviceURL         *url.URL
	catalogModels      []ModelInfo
	useWindowsFallback bool

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
		ApiKey:             "OPENAI_API_KEY",
		useWindowsFallback: false,
	}

	// Make sure we always apply OS-specific defaults
	opts = append([]ManagerOption{WithAutoConfigure()}, opts...)
	for _, opt := range opts {
		opt(m)
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
// with minimal setup code. The device parameter allows callers to hint which
// device type (CPU, GPU, NPU) should be used; pass nil to let the SDK decide
// automatically.
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
//	manager, err := foundrylocal.StartModel(ctx, "qwen2.5-0.5b", nil)
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer manager.StopService(ctx)
func StartModel(ctx context.Context, aliasorModelID string, device *DeviceType) (*Manager, error) {
	m := NewManager()
	if err := m.StartService(ctx); err != nil {
		return nil, err
	}

	modelInfo, err := m.GetModelInfo(ctx, aliasorModelID, device)
	if err != nil {
		if errors.Is(err, ErrModelNotInCatalog) {
			m.Logger.ErrorContext(ctx, "model not found in catalog", "aliasOrModelID", aliasorModelID)
		}
		return nil, err
	}

	if _, err := m.DownloadModel(ctx, modelInfo.ID, device); err != nil {
		return nil, err
	}

	if _, err := m.LoadModel(ctx, aliasorModelID, device); err != nil {
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
		Timeout:   time.Duration(2) * time.Hour,
		Transport: &sdkRoundTripper{http.DefaultTransport},
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

	switch models {
	case nil:
		models = []ModelInfo{}
	default:
		m.catalogModels = models
	}

	// Override execution provider to CUDA for generic-gpu models if CUDA is available
	hasCUDA := slices.ContainsFunc(m.catalogModels, func(m ModelInfo) bool {
		return m.Runtime.ExecutionProvider == "CUDAExecutionProvider"
	})
	if hasCUDA {
		for i := range m.catalogModels {
			if strings.Contains(strings.ToLower(m.catalogModels[i].ID), "-generic-gpu") {
				m.catalogModels[i].EPOverride = "cuda"
			}
		}
	}

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
}

// GetModelInfo retrieves detailed information about a specific model by its ID or alias.
// The optional device parameter narrows alias matches to a preferred device type;
// pass nil to allow any device. The method returns the model metadata or
// ErrModelNotInCatalog if no match is found.
//
// The function uses a priority system when multiple models share the same alias,
// preferring models with higher-priority execution providers based on the Manager's
// configuration.
//
// Example:
//
//	modelInfo, err := manager.GetModelInfo(ctx, "qwen2.5-0.5b", nil)
//	if err != nil {
//		if errors.Is(err, foundrylocal.ErrModelNotInCatalog) {
//			log.Fatal("Model not found in catalog")
//		}
//		log.Fatal(err)
//	}
//	fmt.Printf("Found model: %s\n", modelInfo.DisplayName)
func (m *Manager) GetModelInfo(ctx context.Context, aliasOrModelID string, device *DeviceType) (ModelInfo, error) {
	catalog, err := m.ListCatalogModels(ctx)
	if err != nil {
		return ModelInfo{}, ErrModelNotInCatalog
	}

	// 1) Try to match by full ID exactly (with or without ':' for backwards compatibility)
	for _, model := range catalog {
		if strings.EqualFold(model.ID, aliasOrModelID) {
			return model, nil
		}
	}

	// 2) Try to match by ID prefix "<id>:" and pick the highest version
	prefix := strings.ToLower(aliasOrModelID) + ":"
	bestVersion := -1
	var best *ModelInfo

	for _, m := range catalog {
		if strings.HasPrefix(strings.ToLower(m.ID), prefix) {
			if version := GetVersion(m.ID); version > bestVersion {
				bestVersion = version
				best = &m
			}
		}
	}

	if best != nil {
		return *best, nil
	}

	// 3) Match by alias, optionally filtered by device
	var aliasMatches []ModelInfo
	for _, model := range catalog {
		if strings.EqualFold(model.Alias, aliasOrModelID) {
			aliasMatches = append(aliasMatches, model)
		}
	}

	if device != nil {
		aliasMatches = slices.DeleteFunc(aliasMatches, func(m ModelInfo) bool {
			return m.Runtime.DeviceType != *device
		})
	}

	if len(aliasMatches) == 0 {
		return ModelInfo{}, ErrModelNotInCatalog
	}

	// Catalog/list is assumed pre-sorted by service:
	// NPU → non-generic-GPU → generic-GPU → non-generic-CPU → CPU
	candidate := aliasMatches[0]
	if m.useWindowsFallback && strings.Contains(strings.ToLower(candidate.ID), "-generic-gpu") &&
		candidate.EPOverride == "" {
		for _, m := range catalog {
			if strings.EqualFold(m.Alias, aliasOrModelID) && m.Runtime.DeviceType == DeviceTypeCPU {
				candidate = m
				break
			}
		}
	}

	return candidate, nil
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
// The optional device parameter indicates the desired device type to match when
// resolving aliases; pass nil to accept any device. By default, if the model is
// already cached, this operation is skipped. Use WithForceDownload() to re-download
// existing models.
//
// Supported options:
//   - WithToken(token): Provide authentication token for private models
//   - WithForceDownload(): Force re-download even if model exists locally
//
// Example:
//
//	// Basic download
//	modelInfo, err := manager.DownloadModel(ctx, "qwen2.5-0.5b", nil)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Download with authentication
//	modelInfo, err := manager.DownloadModel(ctx, "private-model", nil,
//		foundrylocal.WithToken("your-token"))
//	if err != nil {
//		log.Fatal(err)
//	}
func (m *Manager) DownloadModel(ctx context.Context, aliasorModelID string, device *DeviceType, opts ...DownloadOption) (ModelInfo, error) {
	var config downloadConfig
	for _, opt := range opts {
		opt(&config)
	}

	modelInfo, err := m.GetModelInfo(ctx, aliasorModelID, device)
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
			Publisher:      modelInfo.Publisher,
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

	m.Logger.InfoContext(ctx, "downloading model", "alias", modelInfo.Alias, "modelID", modelInfo.ID)
	req.Header.Set("Content-Type", "application/json")
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
// The model must be downloaded first using DownloadModel. The optional device
// parameter allows targeting a specific device type; pass nil to use the
// Manager's default selection. Loading may take significant time for large models.
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
//	modelInfo, err := manager.LoadModel(ctx, "qwen2.5-0.5b", nil)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Load with custom timeout
//	modelInfo, err := manager.LoadModel(ctx, "large-model", nil,
//		foundrylocal.WithLoadTimeout(30*time.Minute))
//	if err != nil {
//		log.Fatal(err)
//	}
func (m *Manager) LoadModel(ctx context.Context, aliasOrModelID string, device *DeviceType, opts ...LoadModelOption) (ModelInfo, error) {
	config := loadModelConfig{
		timeout: time.Minute * 10, // Default timeout
	}
	for _, opt := range opts {
		opt(&config)
	}

	modelInfo, err := m.GetModelInfo(ctx, aliasOrModelID, device)
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
	// Note: The C# SDK still sets this value as "timeout", but the REST API specifies it as "ttl".
	// See https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-local/reference/reference-rest#get-openailoadname
	params.Set("ttl", fmt.Sprintf("%d", int64(config.timeout.Seconds())))

	if modelInfo.EPOverride != "" {
		params.Set("ep", modelInfo.EPOverride)
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
// The optional device parameter behaves like in DownloadModel. The returned channel
// will receive progress updates and will be closed when the operation completes
// (successfully or with an error).
//
// The progress channel receives ModelDownloadProgress structs containing:
//   - Percentage: Download progress (0-100)
//   - IsCompleted: Whether the operation is finished
//   - ModelInfo: Final model information (only when successfully completed)
//   - ErrorMessage: Error details (only when failed)
//
// Example:
//
//	progressChan, err := manager.DownloadModelWithProgress(ctx, "qwen2.5-0.5b", nil)
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
func (m *Manager) DownloadModelWithProgress(ctx context.Context, aliasOrModelID string, device *DeviceType, opts ...DownloadOption) (<-chan ModelDownloadProgress, error) {
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

		modelInfo, err := m.GetModelInfo(ctx, aliasOrModelID, device)
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
				Publisher:      modelInfo.Publisher,
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
	// This allows us to unmarshal a "null" response as nil.
	var names *[]string
	if err = json.NewDecoder(resp.Body).Decode(&names); err != nil || names == nil {
		return nil, ErrReadLoadedModels
	}
	return m.fetchModelInfo(ctx, *names...)
}

// UnloadModel removes a model from memory, freeing up resources.
// The model remains cached locally and can be loaded again later. The optional
// device parameter controls alias resolution; pass nil for the default behavior.
// Set force to true to unload the model even if it's currently in use.
//
// Example:
//
//	if err := manager.UnloadModel(ctx, "qwen2.5-0.5b", nil, true); err != nil {
//		log.Printf("Failed to unload model: %v", err)
//	} else {
//		fmt.Println("Model unloaded successfully")
//	}
func (m *Manager) UnloadModel(ctx context.Context, aliasOrModelID string, device *DeviceType, force bool) error {
	modelInfo, err := m.GetModelInfo(ctx, aliasOrModelID, device)
	if err != nil {
		return err
	}

	endpoint := m.serviceURL.JoinPath("openai", "unload", modelInfo.ID)
	params := url.Values{}
	params.Set("force", strconv.FormatBool(force))
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
// Otherwise, it delegates to GetModelInfo to find the latest version. The optional
// device parameter scopes alias lookups to a specific device type; pass nil to allow any.
//
// Example:
//
//	// Get latest version of a model by alias
//	modelInfo, err := manager.GetLatestModelInfo(ctx, "qwen2.5", nil)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Get specific version
//	modelInfo, err := manager.GetLatestModelInfo(ctx, "qwen2.5:1", nil)
//	if err != nil {
//		log.Fatal(err)
//	}
func (m *Manager) GetLatestModelInfo(ctx context.Context, aliasOrModelID string, device *DeviceType) (ModelInfo, error) {
	if aliasOrModelID == "" {
		return ModelInfo{}, fmt.Errorf("aliasOrModelID cannot be empty")
	}

	idWithoutVersion := strings.Split(aliasOrModelID, ":")[0]
	return m.GetModelInfo(ctx, idWithoutVersion, device)
}

// IsModelUpgradable checks if a model has a newer version available for upgrade.
// Returns true if an upgrade is available, false if the model is current or not found.
// The optional device parameter matches the semantics of GetLatestModelInfo; pass nil
// to consider any device. This method compares the locally cached version against
// the latest available version.
//
// Example:
//
//	upgradable, err := manager.IsModelUpgradable(ctx, "qwen2.5", nil)
//	if err != nil {
//		log.Printf("Error checking upgrade: %v", err)
//	} else if upgradable {
//		log.Println("Upgrade available")
//	}
func (m *Manager) IsModelUpgradable(ctx context.Context, aliasOrModelID string, device *DeviceType) (bool, error) {
	modelInfo, err := m.GetLatestModelInfo(ctx, aliasOrModelID, device)
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
// The optional device parameter filters alias lookups to the desired device type; pass nil
// to use the default behavior. If a token is provided, it will be used for authentication
// when downloading private models.
//
// Example:
//
//	// Upgrade a public model
//	modelInfo, err := manager.UpgradeModel(ctx, "qwen2.5", nil, "")
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Upgrade a private model with token
//	modelInfo, err := manager.UpgradeModel(ctx, "private-model", nil, "your-token")
//	if err != nil {
//		log.Fatal(err)
//	}
func (m *Manager) UpgradeModel(ctx context.Context, aliasOrModelID string, device *DeviceType, token string) (ModelInfo, error) {
	modelInfo, err := m.GetLatestModelInfo(ctx, aliasOrModelID, device)
	if err != nil {
		return ModelInfo{}, err
	}
	opts := []DownloadOption{}
	if token != "" {
		opts = append(opts, WithToken(token))
	}
	mi, err := m.DownloadModel(ctx, modelInfo.ID, device, opts...)
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
		model, err := m.GetModelInfo(ctx, aliasOrID, nil)
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
