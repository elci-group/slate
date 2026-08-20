use crate::error::SlateError;
use crate::provider::OpenAiProviderConfig;
use crate::types::{Inference, InferenceOutput, Provenance, Task};
use crate::InferenceProvider;
use async_trait::async_trait;
use chrono::Utc;
use reqwest::header::{self, HeaderMap};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;

/// Generic adapter for any OpenAI-compatible chat-completions endpoint.
/// Covers OpenAI, Groq, OpenRouter, and Kimi by swapping base URL + model.
pub struct OpenAiCompatibleProvider {
    config: OpenAiProviderConfig,
    client: reqwest::Client,
}

impl OpenAiCompatibleProvider {
    pub fn new(config: OpenAiProviderConfig) -> Result<Self, SlateError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            format!("Bearer {}", config.api_key)
                .parse()
                .map_err(|e| SlateError::config(format!("invalid api key header: {e}")))?,
        );
        headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(config.timeout)
            .build()?;

        Ok(Self { config, client })
    }
}

#[async_trait]
impl InferenceProvider for OpenAiCompatibleProvider {
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

        let mut messages = Vec::new();
        if let Some(system) = &task.system {
            messages.push(json!({"role": "system", "content": system}));
        }
        messages.push(json!({"role": "user", "content": &task.prompt}));

        let mut body = json!({
            "model": model,
            "messages": messages,
        });

        // Merge task params (excluding model) into the request body.
        let mut request_params = HashMap::new();
        for (k, v) in &task.params {
            if k == "model" {
                continue;
            }
            body[k] = v.clone();
            request_params.insert(k.clone(), v.clone());
        }

        let url = format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'));
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

        let content = extract_content(&raw).unwrap_or_default();

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

fn extract_content(raw: &Value) -> Option<String> {
    raw.get("choices")?
        .as_array()?
        .first()?
        .get("message")?
        .get("content")?
        .as_str()
        .map(|s| s.to_string())
}

