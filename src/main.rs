mod cli;
mod download;
mod search;
mod influencer;
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
        Commands::Generate { prompt, model_path, max_tokens, temperature } => {
            influencer::generate(&prompt, model_path.as_deref(), max_tokens, temperature).await?;
        }
    }

    Ok(())
}
