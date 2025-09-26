package foundrylocal

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"testing"
)

// contentFn produces the raw bytes returned by a mocked HTTP route.
// It keeps tests concise by letting each route declare its own payload generator.
type contentFn func() []byte

// route represents a test HTTP route configuration.
// It defines the URL path, JSON response body, and content type
// for mock HTTP server responses in tests.
type route struct {
	contentFn
	path        string
	contentType string
}

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
				w.Write(route.contentFn())
				return
			}
		}
		http.NotFound(w, r)
	})
}

// mockCatalog returns a route that serves the catalog listing with optional CUDA
// entries so tests can exercise different device combinations.
func mockCatalog(includeCUDA bool) route {
	catalog := buildCatalog(includeCUDA)
	contentFn := func() []byte {
		json, _ := json.Marshal(catalog)
		return json
	}
	return route{
		contentFn: contentFn,
		path:      "/foundry/list",
	}
}

// mockLocalModels returns a route that serves the IDs of locally cached models
// to simulate Foundry Local's cache state.
func mockLocalModels(ids ...string) route {
	contentFn := func() []byte {
		json, _ := json.Marshal(ids)
		return json
	}
	return route{
		contentFn: contentFn,
		path:      "/openai/models",
	}
}

// mockLoadedModels returns a route that serves the IDs of models currently
// loaded in memory for inference.
func mockLoadedModels(ids ...string) route {
	contentFn := func() []byte {
		json, _ := json.Marshal(ids)
		return json
	}
	return route{
		contentFn: contentFn,
		path:      "/openai/loadedmodels",
	}
}

// mockJSON returns a route that serves the provided JSON payload at the given
// path without additional processing.
func mockJSON(path string, data json.RawMessage) route {
	contentFn := func() []byte {
		return data
	}
	return route{
		contentFn: contentFn,
		path:      path,
	}
}

// buildCatalog constructs synthetic catalog entries including CPU, GPU, and NPU
// variants so tests can validate filtering, overrides, and upgrades.
func buildCatalog(includeCUDA bool) []ModelInfo {
	common := ModelInfo{
		ProviderType:        "AzureFoundry",
		Version:             "1",
		ModelType:           "ONNX",
		PromptTemplate:      nil,
		Publisher:           "Microsoft",
		Task:                "chat-completion",
		FileSizeMB:          10403,
		ModelSettings:       ModelSettings{},
		SupportsToolCalling: false,
		License:             "MIT",
		LicenseDescription:  "License…",
		MaxOutputTokens:     1024,
		MinFLVersion:        "1.0.0",
	}

	list1 := []ModelInfo{
		{
			ID:             "model-1-generic-gpu:1",
			DisplayName:    "model-1-generic-gpu",
			URI:            "azureml://registries/azureml/models/model-1-generic-gpu/versions/1",
			Runtime:        Runtime{DeviceType: DeviceTypeGPU, ExecutionProvider: "WebGpuExecutionProvider"},
			Alias:          "model-1",
			ParentModelURI: "azureml://registries/azureml/models/model-1/versions/1",

			ProviderType:        common.ProviderType,
			Version:             common.Version,
			ModelType:           common.ModelType,
			PromptTemplate:      common.PromptTemplate,
			Publisher:           common.Publisher,
			Task:                common.Task,
			FileSizeMB:          common.FileSizeMB,
			ModelSettings:       common.ModelSettings,
			SupportsToolCalling: common.SupportsToolCalling,
			License:             common.License,
			LicenseDescription:  common.LicenseDescription,
			MaxOutputTokens:     common.MaxOutputTokens,
			MinFLVersion:        common.MinFLVersion,
		},
		{
			ID:             "model-1-generic-cpu:2",
			DisplayName:    "model-1-generic-cpu",
			URI:            "azureml://registries/azureml/models/model-1-generic-cpu/versions/2",
			Runtime:        Runtime{DeviceType: DeviceTypeCPU, ExecutionProvider: "CPUExecutionProvider"},
			Alias:          "model-1",
			ParentModelURI: "azureml://registries/azureml/models/model-1/versions/2",

			ProviderType:        common.ProviderType,
			Version:             common.Version,
			ModelType:           common.ModelType,
			PromptTemplate:      common.PromptTemplate,
			Publisher:           common.Publisher,
			Task:                common.Task,
			FileSizeMB:          common.FileSizeMB,
			ModelSettings:       common.ModelSettings,
			SupportsToolCalling: common.SupportsToolCalling,
			License:             common.License,
			LicenseDescription:  common.LicenseDescription,
			MaxOutputTokens:     common.MaxOutputTokens,
			MinFLVersion:        common.MinFLVersion,
		},
		{
			ID:             "model-1-generic-cpu:1",
			DisplayName:    "model-1-generic-cpu",
			URI:            "azureml://registries/azureml/models/model-1-generic-cpu/versions/1",
			Runtime:        Runtime{DeviceType: DeviceTypeCPU, ExecutionProvider: "CPUExecutionProvider"},
			Alias:          "model-1",
			ParentModelURI: "azureml://registries/azureml/models/model-1/versions/1",

			ProviderType:        common.ProviderType,
			Version:             common.Version,
			ModelType:           common.ModelType,
			PromptTemplate:      common.PromptTemplate,
			Publisher:           common.Publisher,
			Task:                common.Task,
			FileSizeMB:          common.FileSizeMB,
			ModelSettings:       common.ModelSettings,
			SupportsToolCalling: common.SupportsToolCalling,
			License:             common.License,
			LicenseDescription:  common.LicenseDescription,
			MaxOutputTokens:     common.MaxOutputTokens,
			MinFLVersion:        common.MinFLVersion,
		},
		{
			ID:             "model-2-npu:2",
			DisplayName:    "model-2-npu",
			URI:            "azureml://registries/azureml/models/model-2-npu/versions/2",
			Runtime:        Runtime{DeviceType: DeviceTypeNPU, ExecutionProvider: "QNNExecutionProvider"},
			Alias:          "model-2",
			ParentModelURI: "azureml://registries/azureml/models/model-2/versions/2",

			ProviderType:        common.ProviderType,
			Version:             common.Version,
			ModelType:           common.ModelType,
			PromptTemplate:      common.PromptTemplate,
			Publisher:           common.Publisher,
			Task:                common.Task,
			FileSizeMB:          common.FileSizeMB,
			ModelSettings:       common.ModelSettings,
			SupportsToolCalling: common.SupportsToolCalling,
			License:             common.License,
			LicenseDescription:  common.LicenseDescription,
			MaxOutputTokens:     common.MaxOutputTokens,
			MinFLVersion:        common.MinFLVersion,
		},
		{
			ID:             "model-2-npu:1",
			DisplayName:    "model-2-npu",
			URI:            "azureml://registries/azureml/models/model-2-npu/versions/1",
			Runtime:        Runtime{DeviceType: DeviceTypeNPU, ExecutionProvider: "QNNExecutionProvider"},
			Alias:          "model-2",
			ParentModelURI: "azureml://registries/azureml/models/model-2/versions/1",

			ProviderType:        common.ProviderType,
			Version:             common.Version,
			ModelType:           common.ModelType,
			PromptTemplate:      common.PromptTemplate,
			Publisher:           common.Publisher,
			Task:                common.Task,
			FileSizeMB:          common.FileSizeMB,
			ModelSettings:       common.ModelSettings,
			SupportsToolCalling: common.SupportsToolCalling,
			License:             common.License,
			LicenseDescription:  common.LicenseDescription,
			MaxOutputTokens:     common.MaxOutputTokens,
			MinFLVersion:        common.MinFLVersion,
		},
		{
			ID:             "model-2-generic-cpu:1",
			DisplayName:    "model-2-generic-cpu",
			URI:            "azureml://registries/azureml/models/model-2-generic-cpu/versions/1",
			Runtime:        Runtime{DeviceType: DeviceTypeCPU, ExecutionProvider: "CPUExecutionProvider"},
			Alias:          "model-2",
			ParentModelURI: "azureml://registries/azureml/models/model-2/versions/1",

			ProviderType:        common.ProviderType,
			Version:             common.Version,
			ModelType:           common.ModelType,
			PromptTemplate:      common.PromptTemplate,
			Publisher:           common.Publisher,
			Task:                common.Task,
			FileSizeMB:          common.FileSizeMB,
			ModelSettings:       common.ModelSettings,
			SupportsToolCalling: common.SupportsToolCalling,
			License:             common.License,
			LicenseDescription:  common.LicenseDescription,
			MaxOutputTokens:     common.MaxOutputTokens,
			MinFLVersion:        common.MinFLVersion,
		},
	}

	if includeCUDA {
		list1 = append(list1, ModelInfo{
			ID:             "model-3-cuda-gpu:1",
			DisplayName:    "model-3-cuda-gpu",
			URI:            "azureml://registries/azureml/models/model-3-cuda-gpu/versions/1",
			Runtime:        Runtime{DeviceType: DeviceTypeGPU, ExecutionProvider: "CUDAExecutionProvider"},
			Alias:          "model-3",
			ParentModelURI: "azureml://registries/azureml/models/model-3/versions/1",

			ProviderType:        common.ProviderType,
			Version:             common.Version,
			ModelType:           common.ModelType,
			PromptTemplate:      common.PromptTemplate,
			Publisher:           common.Publisher,
			Task:                common.Task,
			FileSizeMB:          common.FileSizeMB,
			ModelSettings:       common.ModelSettings,
			SupportsToolCalling: common.SupportsToolCalling,
			License:             common.License,
			LicenseDescription:  common.LicenseDescription,
			MaxOutputTokens:     common.MaxOutputTokens,
			MinFLVersion:        common.MinFLVersion,
		})
	}

	list2 := []ModelInfo{
		{
			ID:             "model-3-generic-gpu:1",
			DisplayName:    "model-3-generic-gpu",
			URI:            "azureml://registries/azureml/models/model-3-generic-gpu/versions/1",
			Runtime:        Runtime{DeviceType: DeviceTypeGPU, ExecutionProvider: "WebGpuExecutionProvider"},
			Alias:          "model-3",
			ParentModelURI: "azureml://registries/azureml/models/model-3/versions/1",

			ProviderType:        common.ProviderType,
			Version:             common.Version,
			ModelType:           common.ModelType,
			PromptTemplate:      common.PromptTemplate,
			Publisher:           common.Publisher,
			Task:                common.Task,
			FileSizeMB:          common.FileSizeMB,
			ModelSettings:       common.ModelSettings,
			SupportsToolCalling: common.SupportsToolCalling,
			License:             common.License,
			LicenseDescription:  common.LicenseDescription,
			MaxOutputTokens:     common.MaxOutputTokens,
			MinFLVersion:        common.MinFLVersion,
		},
		{
			ID:             "model-3-generic-cpu:1",
			DisplayName:    "model-3-generic-cpu",
			URI:            "azureml://registries/azureml/models/model-3-generic-cpu/versions/1",
			Runtime:        Runtime{DeviceType: DeviceTypeCPU, ExecutionProvider: "CPUExecutionProvider"},
			Alias:          "model-3",
			ParentModelURI: "azureml://registries/azureml/models/model-3/versions/1",

			ProviderType:        common.ProviderType,
			Version:             common.Version,
			ModelType:           common.ModelType,
			PromptTemplate:      common.PromptTemplate,
			Publisher:           common.Publisher,
			Task:                common.Task,
			FileSizeMB:          common.FileSizeMB,
			ModelSettings:       common.ModelSettings,
			SupportsToolCalling: common.SupportsToolCalling,
			License:             common.License,
			LicenseDescription:  common.LicenseDescription,
			MaxOutputTokens:     common.MaxOutputTokens,
			MinFLVersion:        common.MinFLVersion,
		},
		{
			ID:             "model-4-generic-gpu:1",
			DisplayName:    "model-4-generic-gpu",
			URI:            "azureml://registries/azureml/models/model-4-generic-gpu/versions/1",
			Runtime:        Runtime{DeviceType: DeviceTypeGPU, ExecutionProvider: "WebGpuExecutionProvider"},
			Alias:          "model-4",
			ParentModelURI: "azureml://registries/azureml/models/model-4/versions/1",

			ProviderType:        common.ProviderType,
			Version:             common.Version,
			ModelType:           common.ModelType,
			PromptTemplate:      common.PromptTemplate,
			Publisher:           common.Publisher,
			Task:                common.Task,
			FileSizeMB:          common.FileSizeMB,
			ModelSettings:       common.ModelSettings,
			SupportsToolCalling: common.SupportsToolCalling,
			License:             common.License,
			LicenseDescription:  common.LicenseDescription,
			MaxOutputTokens:     common.MaxOutputTokens,
			MinFLVersion:        common.MinFLVersion,
		},
	}
	return slices.Concat(list1, list2)
}

// TestListCatalogModel exercises ListCatalogModels to ensure it retrieves,
// parses, and caches catalog models while applying execution provider overrides.
func TestListCatalogModel(t *testing.T) {
	srv := httptest.NewServer(newHandler(mockCatalog(true)))
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
	if len(result) == 0 {
		t.Errorf("got no models, want non-empty list")
	}
	containsCUDA := slices.IndexFunc(result, func(m ModelInfo) bool {
		return m.Runtime.ExecutionProvider == "CUDAExecutionProvider"
	})
	if got, want := containsCUDA > 0, true; got != want {
		t.Fatalf("CUDA execution provider found %v, want %v", got, want)
	}

	containsGG := slices.IndexFunc(result, func(m ModelInfo) bool {
		return m.ID == "model-4-generic-gpu:1"
	})
	if got, want := containsGG > 0, true; got != want {
		t.Fatalf("model-4-generic-gpu:1 found %t, want %t", got, want)
	}

	if got, want := result[containsGG].EPOverride, "cuda"; got != want {
		t.Errorf("got EP override %q, want %q", got, want)
	}

	// Cache is used on second call
	again, err := m.ListCatalogModels(t.Context())
	if err != nil {
		t.Fatalf("failed to list catalog models: %v", err)
	}
	same := slices.EqualFunc(result, again, func(a, b ModelInfo) bool {
		// We only compare the fields that are different in the test data.
		return a.ID == b.ID && a.Alias == b.Alias && a.DisplayName == b.DisplayName &&
			a.URI == b.URI && a.Runtime.ExecutionProvider == b.Runtime.ExecutionProvider &&
			a.ParentModelURI == b.ParentModelURI
	})
	if got, want := same, true; got != want {
		t.Errorf("got same models on second call %t, want %t", got, want)
	}
}

// TestRefreshCatalog tests the RefreshCatalog method to ensure it properly
// clears the cached catalog models and mapping, forcing fresh data retrieval
// on the next catalog operation.
func TestRefreshCatalog(t *testing.T) {
	m := NewManager()
	m.catalogModels = []ModelInfo{}

	m.RefreshCatalog()
	if m.catalogModels != nil {
		t.Errorf("got non-nil catalogModels %v after refresh, want nil", m.catalogModels)
	}
}

// TestGetModelInfo verifies GetModelInfo resolves models by ID or alias,
// honors optional device filters, and surfaces not-found errors.
func TestGetModelInfo(t *testing.T) {
	gpuFilter := DeviceType(DeviceTypeGPU)
	cpuFilter := DeviceType(DeviceTypeCPU)
	npuFilter := DeviceType(DeviceTypeNPU)

	tests := []struct {
		name           string
		aliasOrModelID string
		wantModelID    string
		device         *DeviceType
		err            error
	}{
		{
			name:           "get_model_exact_id",
			aliasOrModelID: "model-1-generic-cpu:1",
			wantModelID:    "model-1-generic-cpu:1",
			device:         nil,
			err:            nil,
		},
		{
			name:           "get_model_latest_version",
			aliasOrModelID: "model-1-generic-cpu",
			wantModelID:    "model-1-generic-cpu:2",
			device:         nil,
			err:            nil,
		},
		{
			name:           "get_model_by_alias",
			aliasOrModelID: "model-2",
			wantModelID:    "model-2-npu:2",
			device:         nil,
			err:            nil,
		},
		{
			name:           "get_model_prefer_cuda",
			aliasOrModelID: "model-3",
			wantModelID:    "model-3-cuda-gpu:1",
			device:         nil,
			err:            nil,
		},
		{
			name:           "get_model_device_filter_gpu",
			aliasOrModelID: "model-1",
			wantModelID:    "model-1-generic-gpu:1",
			device:         &gpuFilter,
			err:            nil,
		},
		{
			name:           "get_model_device_filter_cpu",
			aliasOrModelID: "model-1",
			wantModelID:    "model-1-generic-cpu:2",
			device:         &cpuFilter,
			err:            nil,
		},
		{
			name:           "get_model_device_filter_npu_no_match",
			aliasOrModelID: "model-1",
			wantModelID:    "",
			device:         &npuFilter,
			err:            ErrModelNotInCatalog,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(newHandler(
				mockCatalog(true)))
			defer srv.Close()
			serviceURL, err := url.Parse(srv.URL)
			if err != nil {
				t.Fatalf("failed to parse service URL: %v", err)
			}

			m := NewManager()
			m.serviceURL = serviceURL
			m.client = srv.Client()

			result, err := m.GetModelInfo(t.Context(), tc.aliasOrModelID, tc.device)
			if got, want := err, tc.err; !errors.Is(got, want) {
				t.Fatalf("got error %v, want %v", got, want)
			}

			if got, want := result.ID, tc.wantModelID; got != want {
				t.Errorf("got model ID %q, want %q", got, want)
			}
		})
	}
}

// TestListCachedModels ensures ListCachedModels returns the expected set of
// locally cached models that are available for loading.
func TestListCachedModels(t *testing.T) {
	srv := httptest.NewServer(newHandler(
		mockCatalog(true),
		mockLocalModels("model-2-npu:1", "model-4-generic-gpu:1")))
	defer srv.Close()
	serviceURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	m := NewManager()
	m.serviceURL = serviceURL
	m.client = srv.Client()

	local, err := m.ListCachedModels(t.Context())
	if err != nil {
		t.Fatalf("failed to list cached models: %v", err)
	}
	if got, want := len(local), 2; got != want {
		t.Fatalf("got %d cached models, want %d", got, want)
	}
	if got, want := local[0].ID, "model-2-npu:1"; got != want {
		t.Errorf("got first cached model ID %q, want %q", got, want)
	}
	if got, want := local[1].ID, "model-4-generic-gpu:1"; got != want {
		t.Errorf("got second cached model ID %q, want %q", got, want)
	}
}

// TestListLoadedModels exercises ListLoadedModels to confirm it returns the
// models currently loaded in memory and maps malformed responses to errors.
func TestListLoadedModels(t *testing.T) {
	tests := []struct {
		name   string
		routes []route
		wantID string
		err    error
	}{
		{
			name:   "load_model_success",
			routes: []route{mockCatalog(true), mockLoadedModels("model-2-npu:1")},
			wantID: "model-2-npu:1",
			err:    nil,
		},
		{
			name:   "load_model_not_found",
			routes: []route{mockJSON("/openai/loadedmodels", json.RawMessage(`null`))},
			wantID: "",
			err:    ErrReadLoadedModels,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(newHandler(tc.routes...))
			defer srv.Close()
			serviceURL, err := url.Parse(srv.URL)
			if err != nil {
				t.Fatalf("failed to parse service URL: %v", err)
			}

			m := NewManager()
			m.serviceURL = serviceURL
			m.client = srv.Client()

			result, err := m.ListLoadedModels(t.Context())
			if err != nil {
				if got, want := err, tc.err; !errors.Is(got, want) {
					t.Fatalf("got error %v, want %v", got, want)
				}
				return
			}
			if got, want := result[0].ID, tc.wantID; got != want {
				t.Errorf("got model ID %q, want %q", got, want)
			}
		})
	}
}

// TestDownloadModel validates DownloadModel downloads the preferred model
// variant and propagates success or failure from the tail JSON payload.
func TestDownloadModel(t *testing.T) {
	tests := []struct {
		name        string
		routes      []route
		modelID     string
		wantModelID string
	}{
		{
			name: "download_model_success_parses_tail_json",
			routes: []route{
				mockCatalog(true),
				mockLocalModels(),
				mockJSON("/openai/download", json.RawMessage(`log... {"success": true, "errorMessage": null}`)),
			},
			modelID:     "model-3",
			wantModelID: "model-3-cuda-gpu:1",
		},
		{
			name: "download_model_returns_failure",
			routes: []route{
				mockCatalog(true),
				mockLocalModels(),
				mockJSON("/openai/download", json.RawMessage(`tail {"success": false, "errorMessage": "nope"`))},
			modelID:     "model-1",
			wantModelID: "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(newHandler(tc.routes...))
			defer srv.Close()
			serviceURL, err := url.Parse(srv.URL)
			if err != nil {
				t.Fatalf("failed to parse service URL: %v", err)
			}

			m := NewManager()
			m.serviceURL = serviceURL
			m.client = srv.Client()

			result, err := m.DownloadModel(t.Context(), tc.modelID, nil)
			if err != nil {
				if tc.wantModelID == "" {
					// We expected an error - TODO
					return
				}
				t.Fatalf("got error %v for model ID %q, want nil", err, tc.modelID)
			}

			if got, want := result.ID, tc.wantModelID; got != want {
				t.Errorf("got model ID %q, want %q", got, want)
			}
		})
	}
}

// TestDownloadModelCached ensures DownloadModel returns an already cached
// model when present and re-downloads it when forced.
func TestDownloadModelCached(t *testing.T) {
	srv := httptest.NewServer(newHandler(
		mockCatalog(true),
		mockLocalModels("model-2-npu:2")))
	defer srv.Close()
	serviceURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	m := NewManager()
	m.serviceURL = serviceURL
	m.client = srv.Client()

	cached, err := m.DownloadModel(t.Context(), "model-2", nil)
	if err != nil {
		t.Fatalf("got unexpected error downloading cached model: %v", err)
	}
	if got, want := cached.ID, "model-2-npu:2"; got != want {
		t.Fatalf("got cached model ID %q, want %q", got, want)
	}

	srv2 := httptest.NewServer(newHandler(
		mockCatalog(true),
		mockLocalModels("model-2-npu:2"),
		mockJSON("/openai/download", json.RawMessage(`{"success": true, "errorMessage": null}`))))
	defer srv2.Close()
	serviceURL, err = url.Parse(srv2.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	m.serviceURL = serviceURL
	m.client = srv2.Client()

	forced, err := m.DownloadModel(t.Context(), "model-2", nil, WithForceDownload())
	if err != nil {
		t.Fatalf("got unexpected error downloading cached model: %v", err)
	}
	if got, want := forced.ID, "model-2-npu:2"; got != want {
		t.Errorf("got cached model ID %q, want %q", got, want)
	}
}

// TestDownloadModelWithProgressDownload tests the DownloadModelWithProgress method
// to verify it correctly reports download progress through a channel and handles
// the complete download process with progress updates.
func TestDownloadModelWithProgressDownload(t *testing.T) {
	var buf bytes.Buffer
	buf.WriteString("Total 0.00% Downloading model.onnx.data\n")
	buf.WriteString("[DONE] All Completed!\n")
	buf.WriteString(`{"success": true, "errorMessage": null}`)

	srv := httptest.NewServer(newHandler(
		mockCatalog(true),
		mockLocalModels(),
		mockJSON("/openai/download", buf.Bytes())))
	defer srv.Close()
	serviceURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	m := NewManager()
	m.serviceURL = serviceURL
	m.client = srv.Client()

	progressChan, err := m.DownloadModelWithProgress(t.Context(), "model-3", nil)
	if err != nil {
		t.Fatalf("failed to download model with progress: %v", err)
	}

	progressList := []ModelDownloadProgress{}
	for progress := range progressChan {
		progressList = append(progressList, progress)
	}

	if got, want := len(progressList), 2; got != want {
		t.Fatalf("got %d progress updates, want %d", got, want)
	}
	if got, want := progressList[0].IsCompleted, false; got != want {
		t.Errorf("got isCompleted %t, want %t", got, want)
	}
	if got, want := progressList[0].Percentage, 0.0; got != want {
		t.Errorf("got percentage %.2f, want %.2f", got, want)
	}
	if got, want := progressList[1].IsCompleted, true; got != want {
		t.Errorf("got isCompleted %t, want %t", got, want)
	}
	if got, want := progressList[1].Percentage, 100.0; got != want {
		t.Errorf("got percentage %.2f, want %.2f", got, want)
	}
	if got, want := progressList[1].ModelInfo.ID, "model-3-cuda-gpu:1"; got != want {
		t.Errorf("got model ID %q, want %q", got, want)
	}
}

// TestDownloadModelWithProgressError tests the DownloadModelWithProgress method
// when a download fails, verifying it properly reports error status through
// the progress channel with appropriate error messaging.
func TestDownloadModelWithProgressError(t *testing.T) {
	var buf bytes.Buffer
	buf.WriteString("[DONE] All Completed!\n")
	buf.WriteString(`{"success": false, "errorMessage": "Download error occurred."}`)

	srv := httptest.NewServer(newHandler(
		mockCatalog(true),
		mockLocalModels(),
		mockJSON("/openai/download", buf.Bytes())))
	defer srv.Close()
	serviceURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	m := NewManager()
	m.serviceURL = serviceURL
	m.client = srv.Client()

	progressChan, err := m.DownloadModelWithProgress(t.Context(), "model-3", nil)
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
	if got, want := progressList[0].IsCompleted, true; got != want {
		t.Errorf("got isCompleted %t, want %t", got, want)
	}
	if got, want := progressList[0].ErrorMessage, "Download error occurred."; got != want {
		t.Errorf("got error message %q, want %q", got, want)
	}
}

// TestLoadModelEPOverride ensures LoadModel honors execution provider overrides
// derived from the catalog when selecting the model variant to load.
func TestLoadModelEPOverride(t *testing.T) {
	srv := httptest.NewServer(newHandler(
		mockCatalog(true),
		mockLocalModels("model-4-generic-gpu:1"),
		mockJSON("/openai/load/model-4-generic-gpu:1", json.RawMessage(`{}`))))
	defer srv.Close()
	serviceURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	m := NewManager()
	m.serviceURL = serviceURL
	m.client = srv.Client()

	// After ListCatalogModels runs, EPOverride for generic-gpu will be "cuda"
	// First call ensures the override is applied
	if _, err := m.ListCatalogModels(t.Context()); err != nil {
		t.Fatalf("failed to list catalog models: %v", err)
	}

	result, err := m.LoadModel(t.Context(), "model-4", nil)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	if got, want := result.ID, "model-4-generic-gpu:1"; got != want {
		t.Errorf("got model ID %q, want %q", got, want)
	}
}

// TestLoadModelError validates that LoadModel surfaces errors when the target
// model is absent from the local cache.
func TestLoadModelError(t *testing.T) {
	srv := httptest.NewServer(newHandler(
		mockCatalog(true),
		mockLocalModels()))
	defer srv.Close()
	serviceURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	m := NewManager()
	m.serviceURL = serviceURL
	m.client = srv.Client()

	_, err = m.LoadModel(t.Context(), "model-3", nil)
	if err == nil {
		t.Fatalf("got nil, want non-nil error")
	}

	errMsg := "not found in local models"
	if !strings.Contains(err.Error(), errMsg) {
		t.Errorf("got error %v, want %q", err, errMsg)
	}
}

// TestUnloadModel verifies UnloadModel issues the unload request successfully
// and reports any resulting error.
func TestUnloadModel(t *testing.T) {
	modelID := "model-2-npu:1"
	urlPath := fmt.Sprintf("/openai/unload/%s", modelID)
	srv := httptest.NewServer(newHandler(
		mockCatalog(true),
		mockJSON(urlPath, json.RawMessage(`{}`))))
	defer srv.Close()
	serviceURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("failed to parse service URL: %v", err)
	}

	m := NewManager()
	m.serviceURL = serviceURL
	m.client = srv.Client()

	if got, want := m.UnloadModel(t.Context(), modelID, nil, false), error(nil); got != want {
		t.Errorf("got error %v, want %v", got, want)
	}
}

// TestIsModelUpgradeable checks that IsModelUpgradable reports whether a local
// model is older than the catalog version and gracefully handles missing data.
func TestIsModelUpgradeable(t *testing.T) {
	tests := []struct {
		name    string
		modelID string
		routes  []route
		want    bool
	}{
		{
			name:    "cached_older_than_latest",
			modelID: "model-2",
			routes: []route{
				mockCatalog(true),
				mockLocalModels("model-2-npu:1"),
			},
			want: true,
		},
		{
			name:    "latest_version_cached",
			modelID: "model-2",
			routes: []route{
				mockCatalog(true),
				mockLocalModels("model-2-npu:2"),
			},
			want: false,
		},
		{
			name:    "model_missing_from_catalog",
			modelID: "model-2",
			routes: []route{
				mockJSON("/foundry/list", json.RawMessage(`[]`)),
				mockLocalModels("model-2-npu:1"),
			},
			want: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(newHandler(tc.routes...))
			defer srv.Close()
			serviceURL, err := url.Parse(srv.URL)
			if err != nil {
				t.Fatalf("failed to parse service URL: %v", err)
			}

			m := NewManager()
			m.serviceURL = serviceURL
			m.client = srv.Client()

			got, err := m.IsModelUpgradable(t.Context(), tc.modelID, nil)
			if err != nil {
				t.Fatalf("failed to check if model is upgradeable: %v", err)
			}
			if got != tc.want {
				t.Errorf("got model to be upgradeable %t, want %t", got, tc.want)
			}
		})
	}
}

// TestUpgradeModel validates UpgradeModel downloads the latest model variant
// when available and surfaces catalog or download failures.
func TestUpgradeModel(t *testing.T) {
	tests := []struct {
		name        string
		modelID     string
		wantModelID string
		routes      []route
		err         error
	}{
		{
			name:        "upgrade_model_success",
			modelID:     "model-3",
			wantModelID: "model-3-cuda-gpu:1",
			routes: []route{
				mockCatalog(true),
				mockLocalModels(),
				mockJSON("/openai/download", json.RawMessage(`{"success": true, "errorMessage": null}`)),
			},
			err: nil,
		},
		{
			name:        "upgrade_model_download_fails",
			modelID:     "model-3",
			wantModelID: "",
			routes: []route{
				mockCatalog(true),
				mockLocalModels(),
				mockJSON("/openai/download", json.RawMessage(`{"success": false, "errorMessage": "Simulated download failure."}`))},
			err: ErrModelUpgradeFailed,
		},
		{
			name:        "upgrade_model_not_found",
			modelID:     "missing-model",
			wantModelID: "",
			routes: []route{
				mockJSON("/foundry/list", json.RawMessage(`[]`)),
			},
			err: ErrModelNotInCatalog,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(newHandler(tc.routes...))
			defer srv.Close()
			serviceURL, err := url.Parse(srv.URL)
			if err != nil {
				t.Fatalf("failed to parse service URL: %v", err)
			}

			m := NewManager()
			m.serviceURL = serviceURL
			m.client = srv.Client()

			mi, err := m.UpgradeModel(t.Context(), tc.modelID, nil, "")
			if got, want := err, tc.err; !errors.Is(got, want) {
				t.Fatalf("got error %v, want %v", got, want)
			}
			if got, want := mi.ID, tc.wantModelID; got != want {
				t.Errorf("got model ID %q, want %q", got, want)
			}
		})
	}
}

// TestGetCacheLocation tests the GetCacheLocation method to verify it correctly
// retrieves the filesystem path where Foundry Local stores cached models.
func TestGetCacheLocation(t *testing.T) {
	srv := httptest.NewServer(newHandler(mockJSON("/openai/status", json.RawMessage(`{"modelDirPath": "/models"}`))))
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
