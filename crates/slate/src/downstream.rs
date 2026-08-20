use crate::error::SlateError;
use crate::types::{InferenceOutput, Provenance, ReasoningResult, SlateCorpus};
use chrono::Utc;
use reqwest::header::{self, HeaderMap};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Options controlling how Grok reasons over the inference corpus.
#[derive(Debug, Clone)]
pub struct ReasoningOptions {
    pub model: String,
    pub base_url: String,
    pub timeout: Duration,
    pub temperature: f32,
    pub max_tokens: u32,
}

impl Default for ReasoningOptions {
    fn default() -> Self {
        Self {
            model: "grok-2-latest".into(),
            base_url: "https://api.x.ai/v1".into(),
            timeout: Duration::from_secs(120),
            temperature: 0.0,
            max_tokens: 4096,
        }
    }
}

/// Downstream critical-reasoning layer backed by xAI Grok.
pub struct GrokReasoner {
    #[allow(dead_code)]
    api_key: String,
    options: ReasoningOptions,
    client: reqwest::Client,
}

impl GrokReasoner {
    pub fn new(api_key: impl Into<String>) -> Result<Self, SlateError> {
        Self::with_options(api_key, ReasoningOptions::default())
    }

    pub fn with_options(
        api_key: impl Into<String>,
        options: ReasoningOptions,
    ) -> Result<Self, SlateError> {
        let api_key = api_key.into();
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            format!("Bearer {}", api_key)
                .parse()
                .map_err(|e| SlateError::config(format!("invalid api key header: {e}")))?,
        );
        headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(options.timeout)
            .build()?;

        Ok(Self {
            api_key,
            options,
            client,
        })
    }

    /// Send the aggregated inference corpus to Grok and return its critical reasoning.
    pub async fn reason(&self, corpus: &SlateCorpus) -> Result<ReasoningResult, SlateError> {
        let started = Instant::now();
        let prompt = build_corpus_prompt(corpus);

        let body = json!({
            "model": self.options.model,
            "messages": [
                { "role": "system", "content": REASONING_SYSTEM_PROMPT },
                { "role": "user", "content": prompt }
            ],
            "temperature": self.options.temperature,
            "max_tokens": self.options.max_tokens,
        });

        let url = format!(
            "{}/chat/completions",
            self.options.base_url.trim_end_matches('/')
        );

        let response = self.client.post(&url).json(&body).send().await?;
        let status = response.status();
        let raw: Value = response.json().await?;

        if !status.is_success() {
            return Err(SlateError::provider(
                "grok",
                format!("HTTP {}: {}", status, raw),
            ));
        }

        let content = raw
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or_default()
            .to_string();

        let structured = raw
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("structured"))
            .cloned();

        Ok(ReasoningResult {
            content,
            provenance: Provenance {
                provider: "grok".into(),
                model: self.options.model.clone(),
                timestamp: Utc::now(),
                endpoint: Some(url),
                request_params: {
                    let mut p = HashMap::new();
                    p.insert("temperature".into(), json!(self.options.temperature));
                    p.insert("max_tokens".into(), json!(self.options.max_tokens));
                    p
                },
                latency_ms: started.elapsed().as_millis() as u64,
            },
            structured,
        })
    }
}

const REASONING_SYSTEM_PROMPT: &str = r#"You are a critical-reasoning layer.

You will be given an INFERENCE CORPUS produced by multiple independent upstream models and providers. Each inference entry includes provenance: provider, model, timestamp, latency, and the model's output.

Your job is not to generate a fresh answer from first principles, but to reason carefully across the accumulated inference. In your response, address the following:

1. Convergence — What do the inferences agree on?
2. Disagreement — Where do they diverge meaningfully?
3. Outliers — Are there any responses that look anomalous?
4. Confidence — Given the spread of answers, how confident should we be?
5. Contradictions — Are there direct logical contradictions?
6. Corroboration — Do independent providers/models support the same conclusion?
7. Provider/model-specific biases — Do any patterns map to specific sources?
8. Missing evidence — What would strengthen the conclusion?
9. Synthesis — What is the most reasonable overall conclusion?

Be concise but thorough. Cite specific inferences by their ID when relevant."#;

/// Build the provenance-aware corpus prompt shown to Grok.
pub fn build_corpus_prompt(corpus: &SlateCorpus) -> String {
    let mut lines = Vec::new();

    if let Some(task) = &corpus.task {
        lines.push(format!("TASK\nid: {}\nprompt: {}\n", task.id, task.prompt));
    }

    lines.push(format!("INFERENCE CORPUS ({} entries)\n", corpus.len()));

    for (idx, inference) in corpus.inferences.iter().enumerate() {
        let prov = &inference.provenance;
        lines.push(format!("INFERENCE #{:03}", idx + 1));
        lines.push(format!("id: {}", inference.id));
        lines.push(format!("provider: {}", prov.provider));
        lines.push(format!("model: {}", prov.model));
        lines.push(format!("timestamp: {}", prov.timestamp.to_rfc3339()));
        lines.push(format!("latency_ms: {}", prov.latency_ms));
        if let Some(endpoint) = &prov.endpoint {
            lines.push(format!("endpoint: {}", endpoint));
        }
        if !prov.request_params.is_empty() {
            if let Ok(json) = serde_json::to_string(&prov.request_params) {
                lines.push(format!("request_params: {}", json));
            }
        }
        lines.push(format!("task_id: {}", inference.task_id));

        match &inference.output {
            InferenceOutput::Text { content } => {
                lines.push("output:".into());
                for line in content.lines() {
                    lines.push(format!("> {}", line));
                }
            }
            InferenceOutput::Json { content } => {
                lines.push(format!("output (json): {}", content));
            }
            InferenceOutput::Error { message } => {
                lines.push(format!("output (error): {}", message));
            }
        }

        if !inference.metadata.is_empty() {
            if let Ok(json) = serde_json::to_string(&inference.metadata) {
                lines.push(format!("metadata: {}", json));
            }
        }

        lines.push(String::new());
    }

    lines.join("\n")
}

/// Build a minimal prompt that just lists inferences without the full reasoning directive.
/// Useful for debugging or custom downstream prompts.
pub fn build_compact_corpus_prompt(corpus: &SlateCorpus) -> String {
    build_corpus_prompt(corpus)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Inference, InferenceOutput, Provenance, SlateCorpus, Task};
    use chrono::Utc;

    fn sample_corpus() -> SlateCorpus {
        let mut corpus = SlateCorpus::new(Task::new("What is 2+2?"));
        corpus.push(Inference {
            id: "inf-1".into(),
            provenance: Provenance {
                provider: "groq".into(),
                model: "llama3-70b".into(),
                timestamp: Utc::now(),
                endpoint: Some("https://api.groq.com/openai/v1".into()),
                request_params: Default::default(),
                latency_ms: 120,
            },
            task_id: "task-1".into(),
            output: InferenceOutput::Text { content: "4".into() },
            raw_response: None,
            metadata: Default::default(),
        });
        corpus.push(Inference {
            id: "inf-2".into(),
            provenance: Provenance {
                provider: "gemini".into(),
                model: "gemini-1.5-flash".into(),
                timestamp: Utc::now(),
                endpoint: None,
                request_params: Default::default(),
                latency_ms: 200,
            },
            task_id: "task-1".into(),
            output: InferenceOutput::Text { content: "The answer is 4.".into() },
            raw_response: None,
            metadata: Default::default(),
        });
        corpus
    }

    #[test]
    fn corpus_prompt_includes_provenance() {
        let prompt = build_corpus_prompt(&sample_corpus());
        assert!(prompt.contains("INFERENCE #001"));
        assert!(prompt.contains("provider: groq"));
        assert!(prompt.contains("model: llama3-70b"));
        assert!(prompt.contains("INFERENCE #002"));
        assert!(prompt.contains("provider: gemini"));
    }

    #[test]
    fn corpus_prompt_includes_task() {
        let prompt = build_corpus_prompt(&sample_corpus());
        assert!(prompt.contains("What is 2+2?"));
    }

    #[test]
    fn corpus_prompt_quotes_outputs() {
        let prompt = build_corpus_prompt(&sample_corpus());
        assert!(prompt.contains("> 4"));
    }
}
