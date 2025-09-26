package foundrylocal

// PromptTemplate defines the format for prompts used with a model.
// Different models may require different prompt formatting to work optimally.
type PromptTemplate struct {
	// Assistant is the template for assistant responses.
	Assistant string `json:"assistant"`
	// Prompt is the template for user prompts.
	Prompt string `json:"prompt"`
}

// Runtime describes the execution environment and requirements for a model.
type Runtime struct {
	// DeviceType indicates the preferred device type for execution (CPU, GPU, NPU).
	DeviceType DeviceType `json:"deviceType"`
	// ExecutionProvider specifies the execution provider (CPU, CUDA, WebGPU, QNN).
	ExecutionProvider string `json:"executionProvider"`
}

// ModelSettings contains model-specific configuration parameters.
// The Parameters field is kept flexible to accommodate different model types.
type ModelSettings struct {
	// Parameters contains model-specific configuration as key-value pairs.
	Parameters []any `json:"parameters"`
}

// ModelInfo contains comprehensive information about a model available in Foundry Local.
// This includes metadata, runtime requirements, licensing information, and download details.
type ModelInfo struct {
	// ID is the unique identifier for the model.
	ID string `json:"name"`
	// DisplayName is the human-readable name of the model.
	DisplayName string `json:"displayName"`
	// ProviderType indicates the model provider (e.g., "Microsoft", "Meta").
	ProviderType string `json:"providerType"`
	// URI is the download location for the model.
	URI string `json:"uri"`
	// Version is the model version string.
	Version string `json:"version"`
	// ModelType describes the type of model (e.g., "Language Model").
	ModelType string `json:"modelType"`
	// PromptTemplate defines how to format prompts for this model.
	PromptTemplate *PromptTemplate `json:"promptTemplate,omitzero"`
	// Publisher is the organization that published the model.
	Publisher string `json:"publisher"`
	// Task describes the primary task the model is designed for.
	Task string `json:"task"`
	// Runtime specifies the execution requirements and preferences.
	Runtime Runtime `json:"runtime"`
	// FileSizeMB is the approximate download size in megabytes.
	FileSizeMB int64 `json:"fileSizeMb"`
	// ModelSettings contains model-specific configuration.
	ModelSettings ModelSettings `json:"modelSettings"`
	// Alias is an alternative name that can be used to reference this model.
	Alias string `json:"alias"`
	// SupportsToolCalling indicates if the model supports function/tool calling.
	SupportsToolCalling bool `json:"supportsToolCalling"`
	// License is the license identifier (e.g., "MIT", "Apache-2.0").
	License string `json:"license"`
	// LicenseDescription provides details about the license terms.
	LicenseDescription string `json:"licenseDescription"`
	// ParentModelURI references the base model if this is a fine-tuned variant.
	ParentModelURI string `json:"parentModelUri"`
	// MaxOutputTokens is the maximum number of tokens the model can generate in a single response.
	MaxOutputTokens int `json:"maxOutputTokens"`
	// MinFLVersion specifies the minimum Foundry Local version required to use this model.
	MinFLVersion string `json:"minFLVersion"`
	// EPOverride specifies an execution provider override to apply when loading the model.
	EPOverride string `json:"epOverride,omitzero"`
}

// DownloadRequestModelInfo contains the subset of model information needed for download requests.
// This is used internally when communicating with the Foundry Local service.
type DownloadRequestModelInfo struct {
	// Name is the model identifier.
	Name string `json:"Name"`
	// URI is the download location.
	URI string `json:"Uri"`
	// Publisher is the organization that published the model.
	Publisher string `json:"Publisher"`
	// ProviderType is the model provider with "Local" suffix.
	ProviderType string `json:"ProviderType"`
	// PromptTemplate defines the prompt formatting for the model.
	PromptTemplate *PromptTemplate `json:"PromptTemplate,omitzero"`
}

// DownloadRequest represents a request to download a model.
// This is used internally when communicating with the Foundry Local service.
type DownloadRequest struct {
	// Model contains the model information for download.
	Model DownloadRequestModelInfo `json:"Model"`
	// Token is the authentication token for private models.
	Token string `json:"token"`
	// IgnorePipeReport controls whether to ignore pipeline reporting.
	IgnorePipeReport bool `json:"IgnorePipeReport"`
}

// ModelDownloadProgress represents the progress of a model download operation.
// It's used by DownloadModelWithProgress to report real-time download status.
type ModelDownloadProgress struct {
	// Percentage is the download progress from 0.0 to 100.0.
	Percentage float64
	// IsCompleted indicates whether the download operation has finished.
	IsCompleted bool
	// ModelInfo contains the model information when download completes successfully.
	ModelInfo ModelInfo
	// ErrorMessage contains error details if the download failed.
	ErrorMessage string
}

// NewDownloadProgress creates a progress update with the specified percentage.
// Used internally to report download progress.
//
// Example:
//
//	progress := foundrylocal.NewDownloadProgress(45.5)
//	fmt.Printf("Download is %.1f%% complete\n", progress.Percentage)
func NewDownloadProgress(percentage float64) ModelDownloadProgress {
	return ModelDownloadProgress{
		Percentage:  percentage,
		IsCompleted: false,
	}
}

// NewDownloadCompleted creates a completion notification with the final model information.
// Used internally when a download completes successfully.
//
// Example:
//
//	completed := foundrylocal.NewDownloadCompleted(modelInfo)
//	fmt.Printf("Download completed: %s\n", completed.ModelInfo.DisplayName)
func NewDownloadCompleted(modelInfo ModelInfo) ModelDownloadProgress {
	return ModelDownloadProgress{
		Percentage:  100.0,
		IsCompleted: true,
		ModelInfo:   modelInfo,
	}
}

// NewDownloadError creates an error notification with the specified error message.
// Used internally when a download fails.
//
// Example:
//
//	error := foundrylocal.NewDownloadError("Network timeout")
//	fmt.Printf("Download failed: %s\n", error.ErrorMessage)
func NewDownloadError(errMessage string) ModelDownloadProgress {
	return ModelDownloadProgress{
		IsCompleted:  true,
		ErrorMessage: errMessage,
	}
}

// UpgradeBody contains the model information needed for upgrade requests.
// This is used internally when communicating with the Foundry Local service.
type UpgradeBody struct {
	// Name is the unique identifier for the model to be upgraded.
	Name string `json:"name"`
	// Version is the new version string for the model.
	URI string `json:"uri"`
	// Publisher is the organization that published the model.
	Publisher string `json:"publisher"`
	// ProviderType indicates the model provider (e.g., "Microsoft", "Meta").
	ProviderType string `json:"providerType"`
	// PromptTemplate defines how to format prompts for this model.
	PromptTemplate PromptTemplate `json:"promptTemplate"`
}

// UpgradeRequest represents a request to upgrade a model to a newer version.
// This is used internally when communicating with the Foundry Local service.
type UpgradeRequest struct {
	// Model contains the model information for upgrade.
	Model UpgradeBody `json:"model"`
	// Token is the authentication token for private models.
	Token string `json:"token"`
	// IgnorePipeReport controls whether to ignore pipeline reporting.
	IgnorePipeReport bool `json:"ignorePipeReport"`
}
