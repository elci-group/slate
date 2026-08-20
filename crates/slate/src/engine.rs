use crate::provider::InferenceProvider;
use crate::types::{Inference, SlateCorpus, Task};
use std::sync::Arc;

/// Slate inference clearinghouse.
///
/// Holds a registry of upstream `InferenceProvider`s and fans out tasks to them,
/// producing a provenance-preserving `SlateCorpus` for downstream reasoning.
pub struct Slate {
    providers: Vec<Arc<dyn InferenceProvider>>,
}

impl Default for Slate {
    fn default() -> Self {
        Self::new()
    }
}

impl Slate {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    /// Register an upstream provider.
    pub fn register<P>(&mut self, provider: P)
    where
        P: InferenceProvider + 'static,
    {
        self.providers.push(Arc::new(provider));
    }

    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }

    pub fn provider_names(&self) -> Vec<&str> {
        self.providers.iter().map(|p| p.name()).collect()
    }

    /// Run the task against every registered provider concurrently.
    pub async fn run(&self, task: &Task) -> SlateCorpus {
        self.run_selected(task, &[]).await
    }

    /// Run the task against a subset of providers.
    /// If `names` is empty, all providers are used.
    pub async fn run_selected(&self, task: &Task, names: &[&str]) -> SlateCorpus {
        let providers: Vec<Arc<dyn InferenceProvider>> = if names.is_empty() {
            self.providers.clone()
        } else {
            self.providers
                .iter()
                .filter(|p| names.contains(&p.name()))
                .cloned()
                .collect()
        };

        let mut corpus = SlateCorpus::new(task.clone());
        let mut handles = Vec::with_capacity(providers.len());

        for provider in providers {
            let task = task.clone();
            let handle = tokio::spawn(async move {
                let name = provider.name().to_string();
                match provider.infer(&task).await {
                    Ok(inference) => inference,
                    Err(err) => {
                        Inference {
                            id: ulid::Ulid::new().to_string(),
                            provenance: crate::types::Provenance {
                                provider: name.clone(),
                                model: "unknown".into(),
                                timestamp: chrono::Utc::now(),
                                endpoint: None,
                                request_params: Default::default(),
                                latency_ms: 0,
                            },
                            task_id: task.id,
                            output: crate::types::InferenceOutput::Error {
                                message: err.to_string(),
                            },
                            raw_response: None,
                            metadata: Default::default(),
                        }
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            if let Ok(inference) = handle.await {
                corpus.push(inference);
            }
            // Panicked tasks are dropped so one failing upstream does not abort the corpus.
        }

        corpus
    }

    /// Merge multiple corpora into one. The first non-None task is kept.
    pub fn batch(&self, corpora: Vec<SlateCorpus>) -> SlateCorpus {
        let mut merged = SlateCorpus::default();
        for mut corpus in corpora {
            if merged.task.is_none() {
                merged.task = corpus.task.take();
            }
            merged.inferences.append(&mut corpus.inferences);
        }
        merged
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::InferenceProvider;
    use crate::types::{InferenceOutput, Provenance, Task};
    use async_trait::async_trait;
    use chrono::Utc;

    struct MockProvider {
        name: String,
        output: String,
    }

    #[async_trait]
    impl InferenceProvider for MockProvider {
        fn name(&self) -> &str {
            &self.name
        }

        async fn infer(&self, task: &Task) -> Result<Inference, crate::error::SlateError> {
            Ok(Inference {
                id: ulid::Ulid::new().to_string(),
                provenance: Provenance {
                    provider: self.name.clone(),
                    model: "mock-model".into(),
                    timestamp: Utc::now(),
                    endpoint: None,
                    request_params: Default::default(),
                    latency_ms: 1,
                },
                task_id: task.id.clone(),
                output: InferenceOutput::Text {
                    content: self.output.clone(),
                },
                raw_response: None,
                metadata: Default::default(),
            })
        }
    }

    #[tokio::test]
    async fn slate_fans_out_to_all_providers() {
        let mut slate = Slate::new();
        slate.register(MockProvider {
            name: "alpha".into(),
            output: "answer a".into(),
        });
        slate.register(MockProvider {
            name: "beta".into(),
            output: "answer b".into(),
        });

        let corpus = slate.run(&Task::new("test prompt")).await;
        assert_eq!(corpus.len(), 2);
        assert!(corpus.successful().len() == 2);
    }

    #[tokio::test]
    async fn slate_selects_providers_by_name() {
        let mut slate = Slate::new();
        slate.register(MockProvider {
            name: "alpha".into(),
            output: "a".into(),
        });
        slate.register(MockProvider {
            name: "beta".into(),
            output: "b".into(),
        });

        let corpus = slate.run_selected(&Task::new("x"), &["beta"]).await;
        assert_eq!(corpus.len(), 1);
        assert_eq!(corpus.inferences[0].provenance.provider, "beta");
    }

    #[test]
    fn batching_merges_corpora() {
        let slate = Slate::new();
        let t1 = Task::new("task one");
        let t2 = Task::new("task two");

        let mut c1 = SlateCorpus::new(t1);
        c1.push(Inference {
            id: "i1".into(),
            provenance: Provenance {
                provider: "p1".into(),
                model: "m".into(),
                timestamp: Utc::now(),
                endpoint: None,
                request_params: Default::default(),
                latency_ms: 0,
            },
            task_id: "t1".into(),
            output: InferenceOutput::Text { content: "one".into() },
            raw_response: None,
            metadata: Default::default(),
        });

        let mut c2 = SlateCorpus::new(t2);
        c2.push(Inference {
            id: "i2".into(),
            provenance: Provenance {
                provider: "p2".into(),
                model: "m".into(),
                timestamp: Utc::now(),
                endpoint: None,
                request_params: Default::default(),
                latency_ms: 0,
            },
            task_id: "t2".into(),
            output: InferenceOutput::Text { content: "two".into() },
            raw_response: None,
            metadata: Default::default(),
        });

        let merged = slate.batch(vec![c1, c2]);
        assert_eq!(merged.len(), 2);
        assert!(merged.task.is_some());
    }
}
