use crate::error::SlateError;
use crate::provider::OllamaProviderConfig;
use crate::types::{Inference, InferenceOutput, Provenance, Task};
use crate::InferenceProvider;
use async_trait::async_trait;
use chrono::Utc;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;

/// Adapter for a local Ollama instance.
pub struct OllamaProvider {
    config: OllamaProviderConfig,
    client: reqwest::Client,
}

impl OllamaProvider {
    pub fn new(config: OllamaProviderConfig) -> Result<Self, SlateError> {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()?;
        Ok(Self { config, client })
    }
}

#[async_trait]
impl InferenceProvider for OllamaProvider {
    fn name(&self) -> &str {
        &self.config.name
    }

    async fn infer(&self, task: &Task) -> Result<Inference, SlateError> {
        let started = Instant::now();
        let model = task
            .params
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(&self.config.default_model)
            .to_string();

        let mut body = json!({
            "model": model,
            "prompt": task.prompt,
            "stream": false,
        });

        if let Some(system) = &task.system {
            body["system"] = json!(system);
        }

        let mut request_params = HashMap::new();
        for (k, v) in &task.params {
            if k == "model" {
                continue;
            }
            body[k] = v.clone();
            request_params.insert(k.clone(), v.clone());
        }

        let url = format!("{}/api/generate", self.config.base_url.trim_end_matches('/'));
        let response = self.client.post(&url).json(&body).send().await?;
        let status = response.status();
        let raw: Value = response.json().await?;

        if !status.is_success() {
            return Ok(Inference {
                id: ulid::Ulid::new().to_string(),
                provenance: Provenance {
                    provider: self.name().to_string(),
                    model: model.clone(),
                    timestamp: Utc::now(),
                    endpoint: Some(url),
                    request_params,
                    latency_ms: started.elapsed().as_millis() as u64,
                },
                task_id: task.id.clone(),
                output: InferenceOutput::Error {
                    message: format!("HTTP {}: {}", status, raw),
                },
                raw_response: Some(raw),
                metadata: HashMap::new(),
            });
        }

        let content = raw
            .get("response")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        Ok(Inference {
            id: ulid::Ulid::new().to_string(),
            provenance: Provenance {
                provider: self.name().to_string(),
                model,
                timestamp: Utc::now(),
                endpoint: Some(url),
                request_params,
                latency_ms: started.elapsed().as_millis() as u64,
            },
            task_id: task.id.clone(),
            output: InferenceOutput::Text { content },
            raw_response: Some(raw),
            metadata: HashMap::new(),
        })
    }
}
