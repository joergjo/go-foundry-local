package foundrylocal

import (
	"bytes"
	"encoding/json"
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
	if got, want := result[0].ModelID, "testModel"; got != want {
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
		ModelID:      "test-model-id",
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
			if got, want := result.ModelID, tc.wantModelID; got != want {
				t.Errorf("got model ID %q, want %q", result.ModelID, tc.wantModelID)
			}
		})
	}
}

// TestGetModelInfoCUDAPriority tests the execution provider priority system
// when multiple models share the same alias. Verifies that CUDA models are
// preferred over CPU and WebGPU models when multiple variants exist.
func TestGetModelInfoCUDAPriority(t *testing.T) {
	phi4MiniModels := json.RawMessage(`[{
			"name": "Phi-4-mini-instruct-generic-cpu",
			"alias": "phi-4-mini",
			"uri": "http://example.com",
			"providerType": "huggingface",
			"runtime": {
				"deviceType": "cpu",
				"executionProvider": "CPU"
			}
		}, {
			"name": "Phi-4-mini-instruct-webgpu",
			"alias": "phi-4-mini",
			"uri": "http://example.com",
			"providerType": "huggingface",
			"runtime": {
				"deviceType": "webgpu",
				"executionProvider": "WEBGPU"
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
			aliasOrModelID: "Phi-4-mini-instruct-generic-cpu",
			wantModelID:    "Phi-4-mini-instruct-generic-cpu",
		},
		{
			aliasOrModelID: "Phi-4-mini-instruct-webgpu",
			wantModelID:    "Phi-4-mini-instruct-webgpu",
		},
		{
			aliasOrModelID: "Phi-4-mini-instruct-cuda-gpu",
			wantModelID:    "Phi-4-mini-instruct-cuda-gpu",
		},
		{
			aliasOrModelID: "phi-4-mini",
			wantModelID:    "Phi-4-mini-instruct-cuda-gpu",
		},
	}
	for _, tc := range tests {
		t.Run(tc.aliasOrModelID, func(t *testing.T) {
			result, err := m.GetModelInfo(t.Context(), tc.aliasOrModelID)
			if err != nil {
				t.Fatalf("failed to get model Info for %q: %v", tc.aliasOrModelID, err)
			}
			if got := result.ModelID; got != tc.wantModelID {
				t.Errorf("got model name %q, want %q", got, tc.wantModelID)
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
			if got := result.ModelID; got != tc.wantModelID {
				t.Errorf("got model name %q, want %q", got, tc.wantModelID)
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
			ModelID:      "model1",
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
	if got, want := result[0].ModelID, "model1"; got != want {
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
		ModelID:      modelID,
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
			if got, want := result.ModelID, tc.modelID; got != want {
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
		ModelID:      modelID,
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
			if got, want := result.ModelID, modelID; got != want {
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
		ModelID:      modelID,
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
			if got, want := result[0].ModelID, modelID; got != want {
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
		ModelID:      modelID,
		Alias:        "aliasY",
		URI:          "http://model",
		ProviderType: "huggingface",
		Runtime: Runtime{
			DeviceType:        DeviceTypeCPU,
			ExecutionProvider: ExecutionProviderCPU,
		},
	}

	urlPath := fmt.Sprintf("/openai/unload/%s?force=true", modelID)
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
		ModelID:      modelID,
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
		if got, want := progress.ModelInfo.ModelID, want[i].modelInfo.ModelID; got != want {
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
		ModelID:      modelID,
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
	if got, want := progressList[0].ModelInfo.ModelID, modelID; got != want {
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
		ModelID:      modelID,
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
