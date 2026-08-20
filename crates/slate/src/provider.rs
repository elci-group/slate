use crate::error::SlateError;
use crate::types::{Inference, Task};
use async_trait::async_trait;
use std::time::Duration;

/// Configuration for a generic OpenAI-compatible provider.
#[derive(Debug, Clone)]
pub struct OpenAiProviderConfig {
    pub name: String,
    pub base_url: String,
    pub api_key: String,
    pub default_model: String,
    pub timeout: Duration,
}

impl OpenAiProviderConfig {
    pub fn new(
        name: impl Into<String>,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        default_model: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            default_model: default_model.into(),
            timeout: Duration::from_secs(60),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Configuration for the Google Gemini provider.
#[derive(Debug, Clone)]
pub struct GeminiProviderConfig {
    pub name: String,
    pub api_key: String,
    pub default_model: String,
    pub timeout: Duration,
}

impl GeminiProviderConfig {
    pub fn new(
        api_key: impl Into<String>,
        default_model: impl Into<String>,
    ) -> Self {
        Self {
            name: "gemini".into(),
            api_key: api_key.into(),
            default_model: default_model.into(),
            timeout: Duration::from_secs(60),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Configuration for a local Ollama provider.
#[derive(Debug, Clone)]
pub struct OllamaProviderConfig {
    pub name: String,
    pub base_url: String,
    pub default_model: String,
    pub timeout: Duration,
}

impl OllamaProviderConfig {
    pub fn new(default_model: impl Into<String>) -> Self {
        Self {
            name: "ollama".into(),
            base_url: "http://localhost:11434".into(),
            default_model: default_model.into(),
            timeout: Duration::from_secs(120),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Provider-agnostic configuration holder.
#[derive(Debug, Clone)]
pub enum ProviderConfig {
    OpenAi(OpenAiProviderConfig),
    Gemini(GeminiProviderConfig),
    Ollama(OllamaProviderConfig),
}

/// Every upstream inference source implements this trait.
#[async_trait]
pub trait InferenceProvider: Send + Sync {
    /// Human-readable provider name (used for provenance and selection).
    fn name(&self) -> &str;

    /// Run a single inference against this provider.
    async fn infer(&self, task: &Task) -> Result<Inference, SlateError>;
}


