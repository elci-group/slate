use thiserror::Error;

#[derive(Debug, Error)]
pub enum SlateError {
    #[error("network request failed: {0}")]
    Network(#[from] reqwest::Error),

    #[error("serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("provider '{provider}' failed: {message}")]
    Provider { provider: String, message: String },

    #[error("provider '{0}' is not configured")]
    ProviderNotConfigured(String),

    #[error("provider '{0}' returned an empty response")]
    EmptyResponse(String),

    #[error("missing required environment variable: {0}")]
    MissingEnvVar(String),

    #[error("invalid configuration: {0}")]
    Config(String),
}

impl SlateError {
    pub fn provider(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Provider {
            provider: provider.into(),
            message: message.into(),
        }
    }

    pub fn config(message: impl Into<String>) -> Self {
        Self::Config(message.into())
    }
}
