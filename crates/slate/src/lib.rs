pub mod config;
pub mod downstream;
pub mod engine;
pub mod error;
pub mod provider;
pub mod providers;
pub mod types;

pub use config::SlateConfig;
pub use downstream::{GrokReasoner, ReasoningOptions};
pub use engine::Slate;
pub use error::SlateError;
pub use provider::{GeminiProviderConfig, InferenceProvider, OllamaProviderConfig, OpenAiProviderConfig, ProviderConfig};
pub use providers::{GeminiProvider, OllamaProvider, OpenAiCompatibleProvider};
pub use types::{Inference, InferenceOutput, Provenance, ReasoningResult, SlateCorpus, Task};
