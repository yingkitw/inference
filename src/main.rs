mod cli;
mod config;
mod download;
mod error;
mod format;
mod influencer;
mod local;
mod models;
mod output;
mod search;

use clap::Parser;
use cli::{Cli, Commands};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use dotenvy::dotenv;
use error::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file if present (silently ignore if not found)
    let _ = dotenv();

    let cli = Cli::parse();

    // Set log level based on command - suppress logs for generate/chat for cleaner output
    let log_level = match &cli.command {
        Commands::Generate { .. } | Commands::Chat { .. } => "influence=warn",
        _ => "influence=info",
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| log_level.into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    match cli.command {
        Commands::Download { model, mirror, output } => {
            let mirror_url = mirror.or_else(|| Some(config::get_mirror()));
            let output_dir = output.or_else(config::get_output_dir);
            download::download_model(&model, mirror_url.as_deref(), output_dir.as_deref()).await?;
        }
        Commands::Search { query, limit, author } => {
            search::search_models(&query, limit, author.as_deref(), None).await?;
        }
        Commands::Serve { model_path, port, device, device_index } => {
            let model = model_path.or_else(config::get_model_path);
            influencer::serve(model.as_deref(), port, &device, device_index).await?;
        }
        Commands::Generate {
            prompt,
            system,
            model_path,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            device,
            device_index,
        } => {
            let model = model_path.or_else(config::get_model_path);
            influencer::generate(
                &prompt,
                system.as_deref(),
                model.as_deref(),
                max_tokens,
                temperature,
                top_p,
                top_k,
                repeat_penalty,
                &device,
                device_index,
            ).await?;
        }
        Commands::Chat {
            model_path,
            system,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            device,
            device_index,
            session,
            save_on_exit,
        } => {
            influencer::chat(
                &model_path,
                system.as_deref(),
                max_tokens,
                temperature,
                top_p,
                top_k,
                repeat_penalty,
                &device,
                device_index,
                session.as_deref(),
                save_on_exit.as_deref(),
            ).await?;
        }
        Commands::Embed { text, model_path, device, device_index } => {
            influencer::embed(&text, &model_path, &device, device_index).await?;
        }
        Commands::List { models_dir } => {
            let models_dir_path = models_dir.as_deref();
            let models = models::list_models(models_dir_path)?;
            let formatter = output::OutputFormatter::new();

            if models.is_empty() {
                formatter.print_warning("No models found.");
                formatter.print_markdown("\n**To download a model:**\n\n```bash\ninfluence download -m <model-name>\n```\n\n**Example:**\n\n```bash\ninfluence download -m TinyLlama/TinyLlama-1.1B-Chat-v1.0\n```\n");
            } else {
                models::display_models(&models, &formatter);
            }
        }
        Commands::Deploy {
            model_path,
            port,
            device,
            device_index,
            detached,
        } => {
            let model = model_path.or_else(config::get_model_path);
            let formatter = output::OutputFormatter::new();

            if detached {
                formatter.print_header("Deploying Model (Background Mode)");
                formatter.print_info(&format!("Server will be accessible at: http://localhost:{}", port));
                formatter.print_markdown("\n**To stop the server later:**\n\n```bash\nps aux | grep influence\nkill <pid>\n```\n");
            }

            influencer::serve(model.as_deref(), port, &device, device_index).await?;

            if detached {
                formatter.print_success("Model deployed successfully!");
                formatter.print_markdown(&format!("\n**Test the deployment:**\n\n```bash\ncurl http://localhost:{}/health\n```\n", port));
            }
        }
        Commands::Config => {
            let formatter = output::OutputFormatter::new();
            formatter.print_header("Configuration Settings");
            
            let model_path = config::get_model_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "Not set".to_string());
            let output_dir = config::get_output_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "./models".to_string());
            let top_k = config::get_top_k()
                .map(|k| k.to_string())
                .unwrap_or_else(|| "Not set".to_string());
            
            let config_table = format!(
r#"
### Model Settings
- **Model Path:** `{}`
- **Output Directory:** `{}`
- **Mirror URL:** `{}`

### Generation Parameters
- **Temperature:** `{}`
- **Top-P:** `{}`
- **Top-K:** `{}`
- **Repeat Penalty:** `{}`
- **Max Tokens:** `{}`

### Device Settings
- **Device:** `{}`
- **Device Index:** `{}`

### Server Settings
- **Port:** `{}`

### Environment Variables
Set these in your `.env` file or environment:
- `INFLUENCE_MODEL_PATH` - Default model path
- `INFLUENCE_OUTPUT_DIR` - Download output directory
- `INFLUENCE_MIRROR` - HuggingFace mirror URL
- `INFLUENCE_TEMPERATURE` - Generation temperature
- `INFLUENCE_TOP_P` - Top-p sampling threshold
- `INFLUENCE_TOP_K` - Top-k sampling limit
- `INFLUENCE_REPEAT_PENALTY` - Repetition penalty
- `INFLUENCE_MAX_TOKENS` - Maximum tokens to generate
- `INFLUENCE_DEVICE` - Compute device (auto/cpu/metal/cuda)
- `INFLUENCE_DEVICE_INDEX` - GPU device index
- `INFLUENCE_PORT` - Server port
"#,
                model_path,
                output_dir,
                config::get_mirror(),
                config::get_temperature(),
                config::get_top_p(),
                top_k,
                config::get_repeat_penalty(),
                config::get_max_tokens(),
                config::get_device(),
                config::get_device_index(),
                config::get_port(),
            );
            
            formatter.print_markdown(&config_table);
        }
    }

    Ok(())
}
