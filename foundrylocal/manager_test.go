package foundrylocal

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

// newHandler creates a test HTTP handler that serves predefined routes.
// It matches incoming requests against the provided routes and serves
// the corresponding JSON response with the specified content type.
func newHandler(routes ...route) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for _, route := range routes {
			if r.URL.Path == route.path {
				if route.contentType != "" {
					w.Header().Set("Content-Type", route.contentType)
				}
				w.Write(route.json)
				return
			}
		}
		http.NotFound(w, r)
	})
}

// route represents a test HTTP route configuration.
// It defines the URL path, JSON response body, and content type
// for mock HTTP server responses in tests.
type route struct {
	path        string
	json        json.RawMessage
	contentType string
}

// TestListCatalogModel tests the ListCatalogModels method to ensure it correctly
// retrieves and parses the catalog of available models from the foundry service.
func TestListCatalogModel(t *testing.T) {
	model := json.RawMessage(`[{
			"name": "testModel",
			"alias": "alias",
			"uri": "http://model",
			"providerType": "huggingface"
		}]`)
	srv := httptest.NewServer(newHandler(route{"/foundry/list", model, "application/json"}))
	defer srv.Close()
	serviceURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	m := NewManager()
	m.serviceURL = serviceURL
	m.client = srv.Client()

	result, err := m.ListCatalogModels(t.Context())
	if err != nil {
		t.Fatalf("failed to list catalog models: %v", err)
	}
	if got, want := len(result), 1; got != want {
		t.Errorf("got %d models, want %d", got, want)
	}
	if got, want := result[0].ID, "testModel"; got != want {
		t.Errorf("got model ID %q, want %q", got, want)
	}
}

// TestRefreshCatalog tests the RefreshCatalog method to ensure it properly
// clears the cached catalog models and mapping, forcing fresh data retrieval
// on the next catalog operation.
func TestRefreshCatalog(t *testing.T) {
	m := NewManager()
	m.catalogModels = []ModelInfo{}
	m.catalogMap = make(map[string]ModelInfo)

	m.RefreshCatalog()
	if m.catalogMap != nil {
		t.Errorf("got catalogMap %v after refresh, want nil", m.catalogMap)
	}
	if m.catalogModels != nil {
		t.Errorf("got catalogModels %v after refresh, want nil", m.catalogModels)
	}
}

// TestGetModelInfo tests the GetModelInfo method to verify it correctly
// retrieves model information by both model ID and alias, and properly
// handles cases where models are not found.
func TestGetModelInfo(t *testing.T) {
	model := ModelInfo{
		ID:           "test-model-id",
		Alias:        "test-alias",
		URI:          "http://example.com",
		ProviderType: "huggingface",
		Runtime: Runtime{
			DeviceType:        DeviceTypeCPU,
			ExecutionProvider: ExecutionProviderCPU,
		},
	}

	m := NewManager()
	m.catalogModels = []ModelInfo{}
	m.catalogMap = map[string]ModelInfo{
		"test-model-id": model,
		"test-alias":    model,
	}

	tests := []struct {
		name           string
		aliasOrModelID string
		wantModelID    string
		err            error
	}{
		{
			name:           "find_model_by_id_ok",
			aliasOrModelID: "test-model-id",
			wantModelID:    "test-model-id",
			err:            nil,
		},
		{
			name:           "find_model_by_id_not_found",
			aliasOrModelID: "non-existent",
			wantModelID:    "",
			err:            ErrModelNotInCatalog,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, err := m.GetModelInfo(t.Context(), tc.aliasOrModelID)
			if got, want := err, tc.err; got != want {
				t.Fatalf("got error: %v, want: %v", err, tc.err)
			}
			if got, want := result.ID, tc.wantModelID; got != want {
				t.Errorf("got model ID %q, want %q", got, want)
			}
		})
	}
}

// TestGetModelInfoCUDAPriority tests the execution provider priority system
// when multiple models share the same alias. Verifies that CUDA models are
// preferred over CPU and WebGPU models when multiple variants exist.
func TestGetModelInfoCUDAPriority(t *testing.T) {
	phi4MiniModels := json.RawMessage(`[{
			"name": "Phi-4-mini-instruct-generic-cpu:1",
			"alias": "phi-4-mini",
			"uri": "http://example.com",
			"providerType": "huggingface",
			"runtime": {
				"deviceType": "cpu",
				"executionProvider": "CPU"
			}
		}, {
			"name": "Phi-4-mini-instruct-webgpu:1",
			"alias": "phi-4-mini",
			"uri": "http://example.com",
			"providerType": "huggingface",
			"runtime": {
				"deviceType": "webgpu",
				"executionProvider": "WEBGPU"
			}
		}, {
			"name": "Phi-4-mini-instruct-cuda-gpu:1",
			"alias": "phi-4-mini",
			"uri": "http://example.com",
			"providerType": "huggingface",
			"runtime": {
				"deviceType": "gpu",
				"executionProvider": "CUDA"
			}
		}]`)
	srv := httptest.NewServer(newHandler(route{"/foundry/list", phi4MiniModels, "application/json"}))
	defer srv.Close()
	serviceURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	m := NewManager()
	m.serviceURL = serviceURL
	m.client = srv.Client()

	tests := []struct {
		aliasOrModelID string
		wantModelID    string
	}{
		{
			aliasOrModelID: "Phi-4-mini-instruct-generic-cpu:1",
			wantModelID:    "Phi-4-mini-instruct-generic-cpu:1",
		},
		{
			aliasOrModelID: "Phi-4-mini-instruct-webgpu:1",
			wantModelID:    "Phi-4-mini-instruct-webgpu:1",
		},
		{
			aliasOrModelID: "Phi-4-mini-instruct-cuda-gpu:1",
			wantModelID:    "Phi-4-mini-instruct-cuda-gpu:1",
		},
		{
			aliasOrModelID: "phi-4-mini",
			wantModelID:    "Phi-4-mini-instruct-cuda-gpu:1",
		},
	}
	for _, tc := range tests {
		t.Run(tc.aliasOrModelID, func(t *testing.T) {
			result, err := m.GetModelInfo(t.Context(), tc.aliasOrModelID)
			if err != nil {
				t.Fatalf("failed to get model Info for %q: %v", tc.aliasOrModelID, err)
			}
			if got, want := result.ID, tc.wantModelID; got != want {
				t.Errorf("got model name %q, want %q", got, want)
			}
		})
	}
}

// TestGetModelInfoQNNPriority tests the execution provider priority system
// to ensure QNN (Qualcomm Neural Network) models have the highest priority
// and are selected over CUDA models when both variants exist with same alias.
func TestGetModelInfoQNNPriority(t *testing.T) {
	phi4MiniModels := json.RawMessage(`[{
			"name": "Phi-4-mini-instruct-qnn",
			"alias": "phi-4-mini",
			"uri": "http://example.com",
			"providerType": "huggingface",
			"runtime": {
				"deviceType": "npu",
				"executionProvider": "QNN"
			}
		}, {
			"name": "Phi-4-mini-instruct-cuda-gpu",
			"alias": "phi-4-mini",
			"uri": "http://example.com",
			"providerType": "huggingface",
			"runtime": {
				"deviceType": "gpu",
				"executionProvider": "CUDA"
			}
		}]`)
	srv := httptest.NewServer(newHandler(route{"/foundry/list", phi4MiniModels, "application/json"}))
	defer srv.Close()
	serviceURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	m := NewManager()
	m.serviceURL = serviceURL
	m.client = srv.Client()

	tests := []struct {
		aliasOrModelID string
		wantModelID    string
	}{
		{
			aliasOrModelID: "Phi-4-mini-instruct-qnn",
			wantModelID:    "Phi-4-mini-instruct-qnn",
		},
		{
			aliasOrModelID: "Phi-4-mini-instruct-cuda-gpu",
			wantModelID:    "Phi-4-mini-instruct-cuda-gpu",
		},
		{
			aliasOrModelID: "phi-4-mini",
			wantModelID:    "Phi-4-mini-instruct-qnn",
		},
	}
	for _, tc := range tests {
		t.Run(tc.aliasOrModelID, func(t *testing.T) {
			result, err := m.GetModelInfo(t.Context(), tc.aliasOrModelID)
			if err != nil {
				t.Fatalf("failed to get model Info for %q: %v", tc.aliasOrModelID, err)
			}
			if got, want := result.ID, tc.wantModelID; got != want {
				t.Errorf("got model name %q, want %q", got, want)
			}
		})
	}
}

// TestGetCacheLocation tests the GetCacheLocation method to verify it correctly
// retrieves the filesystem path where Foundry Local stores cached models.
func TestGetCacheLocation(t *testing.T) {
	json := json.RawMessage(`{"modelDirPath": "/models"}`)
	srv := httptest.NewServer(newHandler(route{"/openai/status", json, "application/json"}))
	defer srv.Close()
	serviceURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	m := NewManager()
	m.serviceURL = serviceURL
	m.client = srv.Client()

	path, err := m.GetCacheLocation(t.Context())
	if err != nil {
		t.Fatalf("failed to get cache location: %v", err)
	}
	if got, want := path, "/models"; got != want {
		t.Errorf("got cache location %q, want %q", got, want)
	}
}

// TestListCacheModels tests the ListCachedModels method to ensure it correctly
// retrieves information about all models currently cached locally and available
// for loading.
func TestListCacheModels(t *testing.T) {
	model := json.RawMessage(`["model1"]`)
	srv := httptest.NewServer(newHandler(route{"/openai/models", model, "application/json"}))
	defer srv.Close()
	serviceURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	m := NewManager()
	m.serviceURL = serviceURL
	m.client = srv.Client()
	m.catalogModels = []ModelInfo{
		{
			ID:           "model1",
			Alias:        "alias",
			URI:          "http://model",
			ProviderType: "huggingface",
		},
	}

	result, err := m.ListCachedModels(t.Context())
	if err != nil {
		t.Fatalf("failed to list cached models: %v", err)
	}
	if got, want := len(result), 1; got != want {
		t.Errorf("got %d models, want %d", got, want)
	}
	if got, want := result[0].ID, "model1"; got != want {
		t.Errorf("got model ID %q, want %q", got, want)
	}
}

// TestDownloadModel tests the DownloadModel method to verify it correctly
// downloads a model to the local cache and handles the download response
// properly including success status and error messages.
func TestDownloadModel(t *testing.T) {
	response := json.RawMessage(`some log text... {"success": true, "errorMessage": null}`)

	srv := httptest.NewServer(newHandler(
		route{"/openai/download", response, "application/json"},
		route{"/openai/models", json.RawMessage(`[]`), "application/json"}))
	defer srv.Close()
	serviceURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	modelID := "test-model"
	model := ModelInfo{
		ID:           modelID,
		Alias:        "alias1",
		URI:          "http://model.uri",
		ProviderType: "openai",
		Runtime: Runtime{
			DeviceType:        DeviceTypeCPU,
			ExecutionProvider: ExecutionProviderCPU,
		},
	}

	m := NewManager()
	m.serviceURL = serviceURL
	m.client = srv.Client()
	m.catalogMap = map[string]ModelInfo{
		modelID: model,
	}
	m.catalogModels = []ModelInfo{model}

	tests := []struct {
		name    string
		modelID string
		isError bool
	}{
		{
			name:    "download_model_success",
			modelID: modelID,
			isError: false,
		},
		{
			name:    "download_model_not_found",
			modelID: "non-existent",
			isError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, err := m.DownloadModel(t.Context(), tc.modelID)
			if tc.isError {
				if err == nil {
					t.Fatalf("got nil error for model ID %q, want error", tc.modelID)
				}
				return
			}
			if err != nil {
				t.Fatalf("failed to download model: %v", err)
			}
			if got, want := result.ID, tc.modelID; got != want {
				t.Errorf("got model ID %q, want %q", got, want)
			}
		})
	}
}

// TestLoadModel tests the LoadModel method to verify it correctly loads
// a previously downloaded model into memory for inference, including
// proper execution provider selection and timeout handling.
func TestLoadModel(t *testing.T) {
	modelID := "modelX"
	model := ModelInfo{
		ID:           modelID,
		Alias:        "aliasX",
		URI:          "http://model",
		ProviderType: "openai",
		Runtime: Runtime{
			DeviceType:        DeviceTypeCPU,
			ExecutionProvider: ExecutionProviderCPU,
		},
	}

	tests := []struct {
		name    string
		result  json.RawMessage
		isError bool
	}{
		{
			name:    "load_model_success",
			result:  json.RawMessage(fmt.Sprintf(`["%s"]`, modelID)),
			isError: false,
		},
		{
			name:    "load_model_not_found",
			result:  json.RawMessage(`[]`),
			isError: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(newHandler(
				route{"/openai/models", tc.result, "application/json"},
				route{"/openai/load/" + modelID, json.RawMessage(`{}`), "application/json"}))
			defer srv.Close()

			serviceURL, err := url.Parse(srv.URL)
			if err != nil {
				t.Fatalf("failed to parse service URL: %v", err)
			}

			m := NewManager()
			m.serviceURL = serviceURL
			m.client = srv.Client()
			m.catalogMap = map[string]ModelInfo{
				modelID:     model,
				model.Alias: model,
			}
			m.catalogModels = []ModelInfo{model}

			result, err := m.LoadModel(t.Context(), modelID)
			if tc.isError {
				if err == nil {
					t.Fatal("got nil error, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("failed to load model: %v", err)
			}
			if got, want := result.ID, modelID; got != want {
				t.Errorf("got model ID %q, want %q", got, want)
			}
		})
	}
}

// TestListLoadedModels tests the ListLoadedModels method to verify it correctly
// retrieves information about all models currently loaded in memory and
// available for inference.
func TestListLoadedModels(t *testing.T) {
	modelID := "modelX"
	model := ModelInfo{
		ID:           modelID,
		Alias:        "aliasX",
		URI:          "http://model",
		ProviderType: "openai",
		Runtime: Runtime{
			DeviceType:        DeviceTypeCPU,
			ExecutionProvider: ExecutionProviderCPU,
		},
	}

	tests := []struct {
		name    string
		result  json.RawMessage
		wantID  string
		wantLen int
	}{
		{
			name:    "load_model_success",
			result:  json.RawMessage(fmt.Sprintf(`["%s"]`, modelID)),
			wantID:  modelID,
			wantLen: 1,
		},
		{
			name:    "load_model_not_found",
			result:  json.RawMessage(`null`),
			wantID:  "",
			wantLen: 0,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(newHandler(
				route{"/openai/loadedmodels", tc.result, "application/json"}))
			defer srv.Close()

			serviceURL, err := url.Parse(srv.URL)
			if err != nil {
				t.Fatalf("failed to parse service URL: %v", err)
			}

			m := NewManager()
			m.serviceURL = serviceURL
			m.client = srv.Client()
			m.catalogMap = map[string]ModelInfo{
				modelID: model,
			}
			m.catalogModels = []ModelInfo{model}

			result, err := m.ListLoadedModels(t.Context())
			if err != nil {
				t.Fatalf("failed to list models: %v", err)
			}
			if got, want := len(result), tc.wantLen; got != want {
				t.Errorf("got %d loaded models, want %d", got, want)
			}
			if tc.wantLen == 0 {
				return
			}
			if got, want := result[0].ID, modelID; got != want {
				t.Errorf("got model ID %q, want %q", got, want)
			}
		})
	}
}

// TestUnloadModel tests the UnloadModel method to verify it correctly
// removes a model from memory, freeing up resources while keeping
// the model cached locally for future loading.
func TestUnloadModel(t *testing.T) {
	modelID := "modelY"
	model := ModelInfo{
		ID:           modelID,
		Alias:        "aliasY",
		URI:          "http://model",
		ProviderType: "huggingface",
		Runtime: Runtime{
			DeviceType:        DeviceTypeCPU,
			ExecutionProvider: ExecutionProviderCPU,
		},
	}

	urlPath := fmt.Sprintf("/openai/unload/%s", modelID)
	srv := httptest.NewServer(newHandler(
		route{urlPath, json.RawMessage{}, "application/json"}))
	defer srv.Close()

	serviceURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	m := NewManager()
	m.serviceURL = serviceURL
	m.client = srv.Client()
	m.catalogMap = map[string]ModelInfo{
		modelID: model,
	}
	m.catalogModels = []ModelInfo{model}

	if got, want := m.UnloadModel(t.Context(), modelID), error(nil); got != want {
		t.Errorf("got error %v, want %v", got, want)
	}
}

// TestDownloadModelWithProgressDownload tests the DownloadModelWithProgress method
// to verify it correctly reports download progress through a channel and handles
// the complete download process with progress updates.
func TestDownloadModelWithProgressDownload(t *testing.T) {
	modelID := "test-model"
	model := ModelInfo{
		ID:           modelID,
		Alias:        "alias1",
		URI:          "http://model.uri",
		ProviderType: "openai",
		Runtime: Runtime{
			DeviceType:        DeviceTypeCPU,
			ExecutionProvider: ExecutionProviderCPU,
		},
	}

	var buf bytes.Buffer
	buf.WriteString("Total 0.00% Downloading model.onnx.data\n")
	buf.WriteString("[DONE] All Completed!\n")
	buf.WriteString(`{"success": true, "errorMessage": null}`)

	srv := httptest.NewServer(newHandler(
		route{"/openai/download", buf.Bytes(), "application/json"},
		route{"/openai/models", json.RawMessage(`[]`), "application/json"}))
	defer srv.Close()

	serviceURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	m := NewManager()
	m.serviceURL = serviceURL
	m.client = srv.Client()
	m.catalogMap = map[string]ModelInfo{
		modelID: model,
	}
	m.catalogModels = []ModelInfo{model}

	progressChan, err := m.DownloadModelWithProgress(t.Context(), modelID)
	if err != nil {
		t.Fatalf("failed to download model with progress: %v", err)
	}

	progressList := []ModelDownloadProgress{}
	for progress := range progressChan {
		progressList = append(progressList, progress)
	}

	want := []struct {
		name         string
		percentage   float64
		isCompleted  bool
		modelInfo    ModelInfo
		errorMessage string
	}{
		{percentage: 0.0, isCompleted: false, modelInfo: ModelInfo{}, errorMessage: ""},
		{percentage: 100.0, isCompleted: true, modelInfo: model, errorMessage: ""},
	}

	for i, progress := range progressList {
		if got, want := progress.Percentage, want[i].percentage; got != want {
			t.Errorf("got percentage %.2f, want %.2f", got, want)
		}
		if got, want := progress.IsCompleted, want[i].isCompleted; got != want {
			t.Errorf("got isCompleted %v, want %v", got, want)
		}
		if got, want := progress.ModelInfo.ID, want[i].modelInfo.ID; got != want {
			t.Errorf("got model ID %q, want %q", got, want)
		}
		if got, want := progress.ErrorMessage, want[i].errorMessage; got != want {
			t.Errorf("got error message %q, want %q", got, want)
		}
	}
}

// TestDownloadModelWithProgressExistingModel tests the DownloadModelWithProgress method
// when attempting to download a model that already exists in the local cache,
// verifying it immediately returns completion without downloading.
func TestDownloadModelWithProgressExistingModel(t *testing.T) {
	modelID := "existing-model"
	model := ModelInfo{
		ID:           modelID,
		Alias:        "alias1",
		URI:          "http://model.uri",
		ProviderType: "openai",
		Runtime: Runtime{
			DeviceType:        DeviceTypeCPU,
			ExecutionProvider: ExecutionProviderCPU,
		},
	}

	srv := httptest.NewServer(newHandler(
		route{"/openai/models", json.RawMessage(fmt.Sprintf(`["%s"]`, modelID)), "application/json"}))
	defer srv.Close()

	serviceURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	m := NewManager()
	m.serviceURL = serviceURL
	m.client = srv.Client()
	m.catalogMap = map[string]ModelInfo{
		modelID: model,
	}
	m.catalogModels = []ModelInfo{model}

	progressChan, err := m.DownloadModelWithProgress(t.Context(), modelID)
	if err != nil {
		t.Fatalf("failed to download model with progress: %v", err)
	}

	progressList := []ModelDownloadProgress{}
	for progress := range progressChan {
		progressList = append(progressList, progress)
	}

	if got, want := len(progressList), 1; got != want {
		t.Fatalf("got %d progress updates, want %d", got, want)
	}
	if got, want := progressList[0].Percentage, 100.0; got != want {
		t.Errorf("got percentage %.2f, want %.2f", got, want)
	}
	if got, want := !progressList[0].IsCompleted, false; got != want {
		t.Errorf("got isCompleted %v, want %v", got, want)
	}
	if got, want := progressList[0].ModelInfo.ID, modelID; got != want {
		t.Errorf("got model ID %q, want %q", got, want)
	}
	if got, want := progressList[0].ErrorMessage, ""; got != want {
		t.Errorf("got error message %q, want %q", got, want)
	}
}

// TestDownloadModelWithProgressError tests the DownloadModelWithProgress method
// when a download fails, verifying it properly reports error status through
// the progress channel with appropriate error messaging.
func TestDownloadModelWithProgressError(t *testing.T) {
	modelID := "test-model"
	model := ModelInfo{
		ID:           modelID,
		Alias:        "alias1",
		URI:          "http://model.uri",
		ProviderType: "openai",
		Runtime: Runtime{
			DeviceType:        DeviceTypeCPU,
			ExecutionProvider: ExecutionProviderCPU,
		},
	}

	var buf bytes.Buffer
	buf.WriteString("[DONE] All Completed!\n")
	buf.WriteString(`{"success": false, "errorMessage": "Download error occurred."}`)

	srv := httptest.NewServer(newHandler(
		route{"/openai/download", buf.Bytes(), "application/json"},
		route{"/openai/models", json.RawMessage(`[]`), "application/json"}))
	defer srv.Close()

	serviceURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	m := NewManager()
	m.serviceURL = serviceURL
	m.client = srv.Client()
	m.catalogMap = map[string]ModelInfo{
		modelID: model,
	}
	m.catalogModels = []ModelInfo{model}

	progressChan, err := m.DownloadModelWithProgress(t.Context(), modelID)
	if err != nil {
		t.Fatalf("failed to download model with progress: %v", err)
	}

	progressList := []ModelDownloadProgress{}
	for progress := range progressChan {
		progressList = append(progressList, progress)
	}

	if got, want := len(progressList), 1; got != want {
		t.Fatalf("got %d progress updates, want %d", got, want)
	}
	if got, want := !progressList[0].IsCompleted, false; got != want {
		t.Errorf("got isCompleted %v, want %v", got, want)
	}
	if got, want := progressList[0].ErrorMessage, "Download error occurred."; got != want {
		t.Errorf("got error message %q, want %q", got, want)
	}
}

// TestGetVersion tests the GetVersion function to verify that version numbers
// are correctly extracted from model IDs, including cases with and without
// version suffixes, empty model IDs, and multiple colons in the ID.
func TestGetVersion(t *testing.T) {
	tests := []struct {
		name     string
		modelID  string
		expected int
	}{
		{
			name:     "model_id_with_version",
			modelID:  "test-model:1",
			expected: 1,
		},
		{
			name:     "model_id_without_version",
			modelID:  "test-model",
			expected: -1,
		},
		{
			name:     "empty_model_id",
			modelID:  "",
			expected: -1,
		},
		{
			name:     "model_id_with_empty_version",
			modelID:  "test-model:",
			expected: -1,
		},
		{
			name:     "model_id_with_multiple_colons",
			modelID:  "test:model:2",
			expected: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			version := GetVersion(tc.modelID)
			if version != tc.expected {
				t.Errorf("got version %d, want %d", version, tc.expected)
			}
		})
	}
}

func TestUpgradeModel(t *testing.T) {
	tests := []struct {
		name         string
		modelID      string
		alias        string
		token        string
		catalogJSON  json.RawMessage
		cacheJSON    json.RawMessage
		downloadJSON json.RawMessage
		err          error
	}{
		{
			name:    "upgrade_model_success",
			modelID: "model-1:2",
			alias:   "model-1",
			token:   "token",
			catalogJSON: json.RawMessage(`[{
				"name": "model-1:2",
				"alias": "model-1",
				"uri": "http://model.uri",
				"providerType": "openai",
				"runtime": {
					"deviceType": "cpu",
					"executionProvider": "CPU"
				}
			}]`),
			cacheJSON:    json.RawMessage(`[]`),
			downloadJSON: json.RawMessage(`{"success": true, "errorMessage": null}`),
			err:          nil,
		}, {
			name:         "upgrade_model_not_found",
			modelID:      "",
			alias:        "missing-model",
			token:        "",
			catalogJSON:  json.RawMessage(`[]`),
			cacheJSON:    json.RawMessage(`[]`),
			downloadJSON: json.RawMessage(`{"success": true, "errorMessage": null}`),
			err:          ErrModelNotInCatalog,
		}, {
			name:    "upgrade_model_download_error",
			modelID: "",
			alias:   "model-1",
			token:   "",
			catalogJSON: json.RawMessage(`[{
				"name": "model-1:2",
				"alias": "model-1",
				"uri": "http://model.uri",
				"providerType": "openai",
				"runtime": {
					"deviceType": "cpu",
					"executionProvider": "CPU"
				}
			}]`),
			cacheJSON:    json.RawMessage(`[]`),
			downloadJSON: json.RawMessage(`{"success": false, "errorMessage": "simulated download failure"}`),
			err:          ErrModelUpgradeFailed,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(newHandler(
				route{"/foundry/list", tc.catalogJSON, "application/json"},
				route{"/openai/models", tc.cacheJSON, "application/json"},
				route{"/openai/download", tc.downloadJSON, "application/json"}))
			defer srv.Close()

			serviceURL, err := url.Parse(srv.URL)
			if err != nil {
				t.Fatalf("failed to parse service URL: %v", err)
			}

			m := NewManager()
			m.serviceURL = serviceURL
			m.client = srv.Client()

			mi, err := m.UpgradeModel(t.Context(), tc.alias, tc.token)
			if got, want := err, tc.err; !errors.Is(got, want) {
				t.Fatalf("got error %v, want %v", got, want)
			}
			if got, want := mi.ID, tc.modelID; got != want {
				t.Errorf("got model ID %q, want %q", got, want)
			}
		})
	}
}

func TestIsModelUpgradeable(t *testing.T) {

	tests := []struct {
		name        string
		alias       string
		catalogJSON json.RawMessage
		cacheJSON   json.RawMessage
		want        bool
	}{
		{
			name:  "newer_version_available",
			alias: "model-1",
			catalogJSON: json.RawMessage(`[{
				"name": "model-1:2",
				"alias": "model-1",
				"uri": "http://model.uri",
				"providerType": "openai",
				"runtime": {
					"deviceType": "cpu",
					"executionProvider": "CPU"
				}
			}]`),
			cacheJSON: json.RawMessage(`["model-1:1"]`),
			want:      true,
		},
		{
			name:  "latest_version_cached",
			alias: "model-1",
			catalogJSON: json.RawMessage(`[{
				"name": "model-1:2",
				"alias": "model-1",
				"uri": "http://model.uri",
				"providerType": "openai",
				"runtime": {
					"deviceType": "cpu",
					"executionProvider": "CPU"
				}
			}]`),
			cacheJSON: json.RawMessage(`["model-1:2"]`),
			want:      false,
		}, {
			name:        "model_not_found",
			alias:       "missing-model",
			catalogJSON: json.RawMessage(`[]`),
			cacheJSON:   json.RawMessage(`[]`),
			want:        false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(newHandler(
				route{"/foundry/list", tc.catalogJSON, "application/json"},
				route{"/openai/models", tc.cacheJSON, "application/json"}))
			defer srv.Close()

			serviceURL, err := url.Parse(srv.URL)
			if err != nil {
				t.Fatalf("failed to parse service URL: %v", err)
			}

			m := NewManager()
			m.serviceURL = serviceURL
			m.client = srv.Client()

			got, err := m.IsModelUpgradable(t.Context(), tc.alias)
			if err != nil {
				t.Fatalf("failed to check if model is upgradeable: %v", err)
			}
			if got != tc.want {
				t.Errorf("got model to be upgradeable %t, want %t", got, tc.want)
			}
		})
	}
}
