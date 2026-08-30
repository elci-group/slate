mod curly_expand;
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

fn __curly_original_main() {
    if let Err(err) = tokio::runtime::Runtime::new().unwrap().block_on(run()) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    let prompt = args.prompt.join(" ");

    let config = SlateConfig::from_env()
        .map_err(|e| format!("failed to load Slate configuration from environment: {e}"))?;

    if config.providers.is_empty() {
        eprintln!("No providers configured. Set at least one of:");
        eprintln!("  OPENAI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, KIMI_API_KEY,");
        eprintln!("  GEMINI_API_KEY, or OLLAMA_ENABLED / OLLAMA_HOST");
        std::process::exit(1);
    }

    let slate = config
        .build()
        .map_err(|e| format!("failed to build Slate engine: {e}"))?;

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

fn main() {
    let raw_args: Vec<String> = std::env::args().collect();
    let mut positions: Vec<usize> = Vec::new();
    let mut fields: Vec<Vec<String>> = Vec::new();
    for (__i, __a) in raw_args.iter().enumerate() {
        if __a == "--system" {
            if let Some(__v) = raw_args.get(__i + 1) {
                positions.push(__i + 1);
                fields.push(curly_expand::expand_or_literal(__v));
            }
            break;
        } else if let Some(__v) = __a.strip_prefix("--system=") {
            positions.push(__i);
            fields.push(
                curly_expand::expand_or_literal(__v)
                    .into_iter()
                    .map(|v| format!("--system={}", v))
                    .collect(),
            );
            break;
        }
    }
    for (__i, __a) in raw_args.iter().enumerate() {
        if __a == "--providers" {
            if let Some(__v) = raw_args.get(__i + 1) {
                positions.push(__i + 1);
                fields.push(curly_expand::expand_or_literal(__v));
            }
            break;
        } else if let Some(__v) = __a.strip_prefix("--providers=") {
            positions.push(__i);
            fields.push(
                curly_expand::expand_or_literal(__v)
                    .into_iter()
                    .map(|v| format!("--providers={}", v))
                    .collect(),
            );
            break;
        }
    }

    if fields.is_empty() || fields.iter().all(|f| f.len() <= 1) {
        __curly_original_main();
        return;
    }

    let combos = curly_expand::cartesian(&fields);
    let exe = std::env::current_exe().expect("resolve current exe");
    let mut had_failure = false;
    for combo in &combos {
        let mut new_args = raw_args.clone();
        for (slot, value) in positions.iter().zip(combo.iter()) {
            new_args[*slot] = value.clone();
        }
        let status = std::process::Command::new(&exe)
            .args(&new_args[1..])
            .status()
            .expect("failed to re-exec self");
        if !status.success() {
            had_failure = true;
        }
    }
    if had_failure {
        std::process::exit(1);
    }
}
