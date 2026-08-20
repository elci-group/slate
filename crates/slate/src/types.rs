use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use ulid::Ulid;

/// A task submitted to the inference network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Stable identifier for the task.
    pub id: String,
    /// The human-readable prompt or instruction.
    pub prompt: String,
    /// Optional system message / role directive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Additional parameters (temperature, max_tokens, etc.) in provider-agnostic form.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub params: HashMap<String, Value>,
}

impl Task {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            id: Ulid::new().to_string(),
            prompt: prompt.into(),
            system: None,
            params: HashMap::new(),
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn with_param(mut self, key: impl Into<String>, value: Value) -> Self {
        self.params.insert(key.into(), value);
        self
    }
}

/// Provenance metadata for a single inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// Provider adapter name, e.g. "groq" or "gemini".
    pub provider: String,
    /// Concrete model identifier returned by or sent to the provider.
    pub model: String,
    /// When the inference was received by Slate.
    pub timestamp: DateTime<Utc>,
    /// Base URL or endpoint used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// Provider-specific request parameters actually used.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub request_params: HashMap<String, Value>,
    /// Wall-clock latency in milliseconds.
    pub latency_ms: u64,
}

/// A normalised output from a single upstream inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InferenceOutput {
    /// Plain text completion.
    Text { content: String },
    /// Structured JSON output (already parsed).
    Json { content: Value },
    /// A provider-level failure captured as output so the corpus stays complete.
    Error { message: String },
}

/// A single inference collected by Slate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inference {
    /// Stable Slate-side identifier.
    pub id: String,
    /// Provenance of the inference.
    pub provenance: Provenance,
    /// Reference to the task that produced this inference.
    pub task_id: String,
    /// Normalised output.
    pub output: InferenceOutput,
    /// Provider-specific raw response snapshot (optional, for debugging).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<Value>,
    /// Free-form provider metadata.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, Value>,
}

impl Inference {
    pub fn text(&self) -> Option<&str> {
        match &self.output {
            InferenceOutput::Text { content } => Some(content),
            _ => None,
        }
    }

    pub fn is_error(&self) -> bool {
        matches!(self.output, InferenceOutput::Error { .. })
    }
}

/// A collection of inferences for downstream reasoning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SlateCorpus {
    /// The task all inferences relate to.
    pub task: Option<Task>,
    /// Inferences, in the order they were collected.
    pub inferences: Vec<Inference>,
}

impl SlateCorpus {
    pub fn new(task: Task) -> Self {
        Self {
            task: Some(task),
            inferences: Vec::new(),
        }
    }

    pub fn push(&mut self, inference: Inference) {
        self.inferences.push(inference);
    }

    pub fn extend(&mut self, other: SlateCorpus) {
        self.inferences.extend(other.inferences);
    }

    pub fn len(&self) -> usize {
        self.inferences.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inferences.is_empty()
    }

    pub fn successful(&self) -> Vec<&Inference> {
        self.inferences.iter().filter(|i| !i.is_error()).collect()
    }

    pub fn errors(&self) -> Vec<&Inference> {
        self.inferences.iter().filter(|i| i.is_error()).collect()
    }

    /// Group inferences by provider.
    pub fn by_provider(&self) -> HashMap<&str, Vec<&Inference>> {
        let mut map: HashMap<&str, Vec<&Inference>> = HashMap::new();
        for inf in &self.inferences {
            map.entry(&inf.provenance.provider).or_default().push(inf);
        }
        map
    }
}

/// Result returned by the downstream critical-reasoning layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningResult {
    /// The raw interpreted output from Grok.
    pub content: String,
    /// Provider/model that produced the reasoning.
    pub provenance: Provenance,
    /// Optional structured fields if the model emits JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured: Option<Value>,
}
