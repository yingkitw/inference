use crate::error::{InfluenceError, Result};
use crate::local::{LocalModel, LocalModelConfig};
use crate::local::DevicePreference;
use std::io::{self, Write};
use std::path::Path;
use tracing::info;

mod service;
mod server;

pub use service::LlmService;

/// Serve local LLM over a web API (REST + SSE)
pub async fn serve(
    model_path: Option<&Path>,
    port: u16,
    device: &str,
    device_index: usize,
) -> Result<()> {
    server::serve(model_path, port, device, device_index).await
}

/// Generate text using a local LLM model
pub async fn generate(
    prompt: &str,
    system: Option<&str>,
    model_path: Option<&Path>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    repeat_penalty: f32,
    device: &str,
    device_index: usize,
) -> Result<()> {
    info!("Generating response for prompt: {}", prompt);

    // Require local model path
    let path = model_path.ok_or_else(|| InfluenceError::InvalidConfig(
        "Model path is required for generation. Use --model-path <path> to specify a local model directory.".to_string()
    ))?;

    info!("Using local model from: {}", path.display());

    let device_preference: DevicePreference = device.parse()?;

    let config = LocalModelConfig {
        model_path: path.to_path_buf(),
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        max_seq_len: max_tokens * 2, // Give some room for the prompt
        device_preference,
        device_index,
        ..Default::default()
    };

    let mut local_model = LocalModel::load(config).await?;
    println!("\n--- Local Generation ---");

    let effective_prompt = match system {
        Some(system_prompt) if !system_prompt.trim().is_empty() => {
            format!(
                "System: {}\n\nUser: {}\n\nAssistant:",
                system_prompt.trim(),
                prompt
            )
        }
        _ => prompt.to_string(),
    };

    local_model.generate_stream(&effective_prompt, max_tokens, temperature).await?;
    println!("\n--- End ---\n");

    Ok(())
}

pub async fn embed(text: &str, model_path: &Path, device: &str, device_index: usize) -> Result<()> {
    info!("Generating embedding");

    let device_preference: DevicePreference = device.parse()?;

    let config = LocalModelConfig {
        model_path: model_path.to_path_buf(),
        device_preference,
        device_index,
        ..Default::default()
    };

    let mut local_model = LocalModel::load(config).await?;
    let embedding = local_model.embed_text(text).await?;

    println!("{}", serde_json::to_string(&embedding).map_err(|e| InfluenceError::JsonError(e))?);
    
    Ok(())
}

/// Interactive chat mode with conversation history
pub async fn chat(
    model_path: &Path,
    system: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    repeat_penalty: f32,
    device: &str,
    device_index: usize,
) -> Result<()> {
    info!("Starting interactive chat mode");

    let device_preference: DevicePreference = device.parse()?;

    let config = LocalModelConfig {
        model_path: model_path.to_path_buf(),
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        max_seq_len: 4096, // Fixed max context length for chat
        device_preference,
        device_index,
        ..Default::default()
    };

    let mut local_model = LocalModel::load(config).await?;

    println!("╭─────────────────────────────────────────────────────────────╮");
    println!("│  Interactive Chat Mode                                        │");
    println!("│  Type your messages and press Enter to send                  │");
    println!("│  Type 'quit', 'exit', or Ctrl+C to exit                       │");
    println!("╰─────────────────────────────────────────────────────────────╯");
    println!();

    // Initialize conversation history
    let mut conversation_history: Vec<String> = Vec::new();

    // Add system prompt if provided
    if let Some(system_prompt) = system {
        if !system_prompt.trim().is_empty() {
            conversation_history.push(format!("System: {}", system_prompt.trim()));
        }
    }

    loop {
        // Print user prompt
        print!("You: ");
        io::stdout().flush()?;

        // Read user input
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)
            .map_err(|e| InfluenceError::LocalModelError(format!("Failed to read input: {}", e)))?;

        let user_input = user_input.trim();

        // Check for exit commands
        if user_input.eq_ignore_ascii_case("quit") || user_input.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }

        // Skip empty input
        if user_input.is_empty() {
            continue;
        }

        // Add user message to history
        conversation_history.push(format!("User: {}", user_input));

        // Build prompt from conversation history
        let conversation_prompt = conversation_history.join("\n");
        let full_prompt = format!("{}\nAssistant:", conversation_prompt);

        // Generate response
        print!("Assistant: ");
        io::stdout().flush()?;

        let response = local_model.generate_text(&full_prompt, max_tokens, temperature).await?;

        println!("{}", response);

        // Add assistant response to history
        conversation_history.push(format!("Assistant: {}", response));
        println!();

        // Prevent conversation from growing too large
        // Keep last 10 turns (20 messages: user + assistant pairs)
        if conversation_history.len() > 20 {
            // Keep system prompt if exists and last 18 messages
            if system.is_some() && !system.unwrap().is_empty() {
                conversation_history = vec![conversation_history[0].clone()]
                    .into_iter()
                    .chain(conversation_history.into_iter().skip(2).take(18))
                    .collect();
            } else {
                conversation_history = conversation_history.into_iter().skip(2).take(20).collect();
            }
        }
    }

    Ok(())
}
