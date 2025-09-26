package foundrylocal

import (
	"log/slog"
	"runtime"
	"time"
)

// DownloadOption configures model download operations.
type DownloadOption func(*downloadConfig)

type downloadConfig struct {
	token string
	force bool
}

// WithToken configures the authentication token for model downloads.
// This is typically required when downloading models from private repositories
// like Hugging Face.
// Example:
//
//	modelInfo, err := manager.DownloadModel(ctx, "model-id", nil,
//		foundrylocal.WithToken("your-auth-token"))
func WithToken(token string) DownloadOption {
	return func(cfg *downloadConfig) {
		cfg.token = token
	}
}

// WithForceDownload forces re-download of a model even if it already exists locally.
// By default, if a model is already cached locally, the download operation will skip
// downloading and return the existing model information.
//
// Example:
//
//	modelInfo, err := manager.DownloadModel(ctx, "model-id", nil,
//		foundrylocal.WithForceDownload())
func WithForceDownload() DownloadOption {
	return func(cfg *downloadConfig) {
		cfg.force = true
	}
}

type loadModelConfig struct {
	timeout time.Duration
}

// WithLoadTimeout sets the timeout for loading a model.
// The default timeout is 10 minutes.
//
// Example:
//
//	modelInfo, err := manager.LoadModel(ctx, "model-id", nil,
//		foundrylocal.WithLoadTimeout(30*time.Minute))
func WithLoadTimeout(timeout time.Duration) LoadModelOption {
	return func(cfg *loadModelConfig) {
		cfg.timeout = timeout
	}
}

// LoadModelOption configures model loading operations.
type LoadModelOption func(*loadModelConfig)

// ManagerOption configures Manager instances during creation.
type ManagerOption func(*Manager)

// WithAutoConfigure configures the Manager with execution provider priorities
// appropriate for the current operating system. This automatically detects
// the OS and applies Windows or macOS defaults accordingly. Using this option
// is usually not required, as the Manager will use WithAutoConfigure() if no
// options are provided.
// Example:
//
//	manager := foundrylocal.NewManager(foundrylocal.WithAutoConfigure())
func WithAutoConfigure() ManagerOption {
	return func(m *Manager) {
		if runtime.GOOS == "windows" {
			WithWindowsFallback()(m)
			return
		}
	}
}

// WithWindowsFallback configures execution provider priorities optimized for Windows.
// For generic-GPU and has NO EpOverride, prefer CPU alias if available
// Example:
//
//	manager := foundrylocal.NewManager(foundrylocal.WithWindowsFallback())
func WithWindowsFallback() ManagerOption {
	return func(m *Manager) {
		m.useWindowsFallback = true
	}
}

// WithLogger sets a custom logger for the Manager.
//
// Example:
//
//	manager := foundrylocal.NewManager(
//		foundrylocal.WithLogger(slog.New(slog.NewTextHandler(os.Stderr, nil))))
func WithLogger(logger *slog.Logger) ManagerOption {
	return func(m *Manager) {
		m.Logger = logger
	}
}
