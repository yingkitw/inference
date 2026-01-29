mod cli;
mod config;
mod download;
mod error;
mod format;
mod influencer;
mod local;
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

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "influence=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

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
            ).await?;
        }
        Commands::Embed { text, model_path, device, device_index } => {
            influencer::embed(&text, &model_path, &device, device_index).await?;
        }
    }

    Ok(())
}
