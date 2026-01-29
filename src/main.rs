mod cli;
mod download;
mod search;
mod influencer;
mod local;
mod error;

use clap::Parser;
use cli::{Cli, Commands};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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
            download::download_model(&model, mirror.as_deref(), output.as_deref()).await?;
        }
        Commands::Search { query, limit, author } => {
            search::search_models(&query, limit, author.as_deref(), None).await?;
        }
        Commands::Serve { model_path, port } => {
            influencer::serve(model_path.as_deref(), port).await?;
        }
        Commands::Generate { prompt, system, model_path, max_tokens, temperature, device, device_index } => {
            influencer::generate(
                &prompt,
                system.as_deref(),
                model_path.as_deref(),
                max_tokens,
                temperature,
                &device,
                device_index,
            ).await?;
        }
    }

    Ok(())
}
