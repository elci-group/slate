use crate::provider::{GeminiProviderConfig, OllamaProviderConfig, OpenAiProviderConfig};
use crate::{GeminiProvider, OllamaProvider, OpenAiCompatibleProvider, ProviderConfig, SlateError};
use std::env;

/// Convenience builder for bootstrapping a Slate runtime from environment variables.
#[derive(Debug, Default, Clone)]
pub struct SlateConfig {
    pub providers: Vec<ProviderConfig>,
}

impl SlateConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_provider(mut self, config: ProviderConfig) -> Self {
        self.providers.push(config);
        self
    }

    /// Auto-detect configured providers from standard environment variables.
    pub fn from_env() -> Result<Self, SlateError> {
        let mut cfg = Self::new();

        // OpenAI
        if let Ok(key) = env::var("OPENAI_API_KEY") {
            cfg = cfg.with_provider(ProviderConfig::OpenAi(OpenAiProviderConfig::new(
                "openai",
                "https://api.openai.com/v1",
                key,
                env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into()),
            )));
        }

        // Groq (OpenAI-compatible)
        if let Ok(key) = env::var("GROQ_API_KEY") {
            cfg = cfg.with_provider(ProviderConfig::OpenAi(OpenAiProviderConfig::new(
                "groq",
                "https://api.groq.com/openai/v1",
                key,
                env::var("GROQ_MODEL").unwrap_or_else(|_| "llama3-70b-8192".into()),
            )));
        }

        // OpenRouter (OpenAI-compatible)
        if let Ok(key) = env::var("OPENROUTER_API_KEY") {
            cfg = cfg.with_provider(ProviderConfig::OpenAi(OpenAiProviderConfig::new(
                "openrouter",
                "https://openrouter.ai/api/v1",
                key,
                env::var("OPENROUTER_MODEL")
                    .unwrap_or_else(|_| "openai/gpt-4o-mini".into()),
            )));
        }

        // Kimi (Moonshot AI, OpenAI-compatible)
        if let Ok(key) = env::var("KIMI_API_KEY") {
            cfg = cfg.with_provider(ProviderConfig::OpenAi(OpenAiProviderConfig::new(
                "kimi",
                "https://api.moonshot.cn/v1",
                key,
                env::var("KIMI_MODEL").unwrap_or_else(|_| "moonshot-v1-8k".into()),
            )));
        }

        // Gemini
        if let Ok(key) = env::var("GEMINI_API_KEY") {
            cfg = cfg.with_provider(ProviderConfig::Gemini(GeminiProviderConfig::new(
                key,
                env::var("GEMINI_MODEL")
                    .unwrap_or_else(|_| "gemini-1.5-flash".into()),
            )));
        }

        // Ollama
        if env::var("OLLAMA_ENABLED").is_ok() || env::var("OLLAMA_HOST").is_ok() {
            let mut ollama = OllamaProviderConfig::new(
                env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3".into()),
            );
            if let Ok(host) = env::var("OLLAMA_HOST") {
                ollama = ollama.with_base_url(host);
            }
            cfg = cfg.with_provider(ProviderConfig::Ollama(ollama));
        }

        Ok(cfg)
    }

    pub fn build(self) -> Result<crate::engine::Slate, SlateError> {
        let mut slate = crate::engine::Slate::new();
        for provider_cfg in self.providers {
            match provider_cfg {
                ProviderConfig::OpenAi(c) => {
                    slate.register(OpenAiCompatibleProvider::new(c)?);
                }
                ProviderConfig::Gemini(c) => {
                    slate.register(GeminiProvider::new(c)?);
                }
                ProviderConfig::Ollama(c) => {
                    slate.register(OllamaProvider::new(c)?);
                }
            }
        }
        Ok(slate)
    }
}
