use anyhow::{Context, Result};
use clap::Parser;
use slate::{GrokReasoner, SlateConfig, Task};
use std::env;

#[derive(Parser, Debug)]
#[command(name = "slate")]
#[command(about = "Inference clearinghouse + reasoning terminal")]
struct Args {
    /// The task prompt to run through the inference network.
    #[arg(required = true)]
    prompt: Vec<String>,

    /// Optional system message.
    #[arg(short, long)]
    system: Option<String>,

    /// Only use these providers (comma-separated).
    #[arg(short, long, value_delimiter = ',')]
    providers: Vec<String>,

    /// Skip the downstream Grok reasoning step.
    #[arg(long)]
    no_reason: bool,

    /// Print the raw corpus prompt that would be sent to Grok.
    #[arg(long)]
    show_prompt: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let prompt = args.prompt.join(" ");

    let config = SlateConfig::from_env()
        .context("failed to load Slate configuration from environment")?;

    if config.providers.is_empty() {
        eprintln!("No providers configured. Set at least one of:");
        eprintln!("  OPENAI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, KIMI_API_KEY,");
        eprintln!("  GEMINI_API_KEY, or OLLAMA_ENABLED / OLLAMA_HOST");
        std::process::exit(1);
    }

    let slate = config
        .build()
        .context("failed to build Slate engine")?;

    println!("Configured providers: {:?}", slate.provider_names());

    let mut task = Task::new(prompt);
    if let Some(system) = args.system {
        task = task.with_system(system);
    }

    let provider_refs: Vec<&str> = args.providers.iter().map(|s| s.as_str()).collect();
    let corpus = if provider_refs.is_empty() {
        slate.run(&task).await
    } else {
        slate.run_selected(&task, &provider_refs).await
    };

    println!("\n=== Inference Corpus ===");
    println!("{}", serde_json::to_string_pretty(&corpus)?);

    if args.show_prompt {
        println!("\n=== Grok Corpus Prompt ===");
        println!("{}", slate::downstream::build_corpus_prompt(&corpus));
    }

    if !args.no_reason {
        if let Ok(key) = env::var("XAI_API_KEY") {
            println!("\n=== Grok Critical Reasoning ===");
            let reasoner = GrokReasoner::new(key)?;
            match reasoner.reason(&corpus).await {
                Ok(result) => {
                    println!("{}", result.content);
                }
                Err(e) => {
                    eprintln!("Grok reasoning failed: {}", e);
                    std::process::exit(1);
                }
            }
        } else {
            println!("\nSkipping Grok reasoning (set XAI_API_KEY to enable).");
        }
    }

    Ok(())
}
