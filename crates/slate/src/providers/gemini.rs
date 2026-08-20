use crate::error::SlateError;
use crate::provider::GeminiProviderConfig;
use crate::types::{Inference, InferenceOutput, Provenance, Task};
use crate::InferenceProvider;
use async_trait::async_trait;
use chrono::Utc;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;

/// Adapter for Google Gemini's generateContent endpoint.
pub struct GeminiProvider {
    config: GeminiProviderConfig,
    client: reqwest::Client,
}

impl GeminiProvider {
    pub fn new(config: GeminiProviderConfig) -> Result<Self, SlateError> {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()?;
        Ok(Self { config, client })
    }
}

#[async_trait]
impl InferenceProvider for GeminiProvider {
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

        let mut parts = Vec::new();
        if let Some(system) = &task.system {
            // Gemini supports a `systemInstruction` field at the top level.
            parts.push(json!({"text": system}));
        }

        let contents = vec![json!({
            "role": "user",
            "parts": [{"text": &task.prompt}]
        })];

        let mut body = json!({
            "contents": contents,
        });

        // Add system instruction if present.
        if let Some(system) = &task.system {
            body["systemInstruction"] = json!({
                "parts": [{"text": system}]
            });
        }

        let mut request_params = HashMap::new();
        for (k, v) in &task.params {
            if k == "model" {
                continue;
            }
            body[k] = v.clone();
            request_params.insert(k.clone(), v.clone());
        }

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model, self.config.api_key
        );

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

        let content = extract_text(&raw).unwrap_or_default();

        Ok(Inference {
            id: ulid::Ulid::new().to_string(),
            provenance: Provenance {
                provider: self.name().to_string(),
                model,
                timestamp: Utc::now(),
                endpoint: Some(url.split('?').next().unwrap_or(&url).to_string()),
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

fn extract_text(raw: &Value) -> Option<String> {
    raw.get("candidates")?
        .as_array()?
        .first()?
        .get("content")?
        .get("parts")?
        .as_array()?
        .iter()
        .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
        .collect::<Vec<_>>()
        .join(" ")
        .into()
}
