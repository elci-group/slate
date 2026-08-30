# Slate

Slate is an **inference clearinghouse + reasoning terminal**.

Instead of treating models as a generation → critique → revision pipeline, Slate:

1. **Fans in** inference from multiple heterogeneous upstream providers.
2. **Preserves provenance** for every inference (provider, model, timestamp, latency, params).
3. **Normalises** outputs into a common corpus format.
4. Sends the corpus to **Grok** as a downstream **critical-reasoning layer**.

> **Inference → aggregation → critical reasoning**

## Architecture

```text
Groq ─────────┐
OpenAI ───────┤
Gemini ───────┤──→ Slate ─→ Grok ─→ interpreted conclusion
OpenRouter ───┤
Ollama ───────┤
Kimi ─────────┘
```

Slate does not ask Grok to rewrite a draft. It gives Grok an accumulated **inference corpus** and asks it to reason across it: convergence, disagreement, outliers, confidence, contradictions, corroboration, biases, missing evidence, and synthesis.

## Quick Start

Copy `.env.example` to `.env`, fill in your API keys, then:

```bash
cargo run --bin slate -- "What is the capital of France?"
```

With a specific provider subset:

```bash
cargo run --bin slate -- -p groq,gemini --no-reason "Explain Rust lifetimes"
```

Show the exact corpus prompt that would be sent to Grok:

```bash
cargo run --bin slate -- --show-prompt "Is machine learning overhyped?"
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key |
| `GROQ_API_KEY` | Groq API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `KIMI_API_KEY` | Moonshot Kimi API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `XAI_API_KEY` | xAI Grok API key (downstream reasoning) |
| `OLLAMA_ENABLED` | Set to any value to enable local Ollama |
| `OLLAMA_HOST` | Ollama base URL (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | Ollama model name (default: `llama3`) |
| `*_MODEL` | Override default model per provider |

## Project Layout

```
crates/slate      # Core library: types, providers, engine, reasoning
crates/slate-cli  # CLI demo binary
```

## Key Abstractions

- `InferenceProvider` trait — every upstream source implements this.
- `Slate` engine — fans out tasks concurrently and builds a `SlateCorpus`.
- `SlateCorpus` — a provenance-preserving collection of inferences.
- `GrokReasoner` — downstream critical-reasoning layer over the corpus.

## Status

This is an MVP. It covers fan-in, provenance, normalisation, batching, and downstream reasoning. Out of scope: persistence, streaming, rate-limit management, complex retry policies, and a REST API.
