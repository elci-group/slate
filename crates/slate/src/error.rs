use std::fmt;

#[derive(Debug)]
pub enum SlateError {
    Network(reqwest::Error),
    Serialization(serde_json::Error),
    Provider { provider: String, message: String },
    ProviderNotConfigured(String),
    EmptyResponse(String),
    MissingEnvVar(String),
    Config(String),
}

impl fmt::Display for SlateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlateError::Network(err) => write!(f, "network request failed: {err}"),
            SlateError::Serialization(err) => write!(f, "serialization failed: {err}"),
            SlateError::Provider { provider, message } => {
                write!(f, "provider '{provider}' failed: {message}")
            }
            SlateError::ProviderNotConfigured(provider) => {
                write!(f, "provider '{provider}' is not configured")
            }
            SlateError::EmptyResponse(provider) => {
                write!(f, "provider '{provider}' returned an empty response")
            }
            SlateError::MissingEnvVar(var) => {
                write!(f, "missing required environment variable: {var}")
            }
            SlateError::Config(message) => write!(f, "invalid configuration: {message}"),
        }
    }
}

impl std::error::Error for SlateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SlateError::Network(err) => Some(err),
            SlateError::Serialization(err) => Some(err),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for SlateError {
    fn from(err: reqwest::Error) -> Self {
        SlateError::Network(err)
    }
}

impl From<serde_json::Error> for SlateError {
    fn from(err: serde_json::Error) -> Self {
        SlateError::Serialization(err)
    }
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
