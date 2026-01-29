mod cli;
mod config;
mod download;
mod error;
mod format;
mod influencer;
mod local;
mod models;
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

            if models.is_empty() {
                println!("No models found.");
                println!("\nTo download a model, use:");
                println!("  influence download -m <model-name>");
                println!("\nExample:");
                println!("  influence download -m TinyLlama/TinyLlama-1.1B-Chat-v1.0");
            } else {
                models::display_models(&models);
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

            if detached {
                println!("🚀 Deploying model in background mode...");
                println!("   Server will be accessible at: http://localhost:{}", port);
                println!("\nTo stop the server later, find the process ID:");
                println!("  ps aux | grep influence");
                println!("\nThen kill it:");
                println!("  kill <pid>");
                println!("\nStarting background server...\n");
            }

            influencer::serve(model.as_deref(), port, &device, device_index).await?;

            if detached {
                println!("\n✅ Model deployed successfully!");
                println!("\nTest the deployment:");
                println!("  curl http://localhost:{}/health", port);
            }
        }
    }

    Ok(())
}
