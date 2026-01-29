use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "influence")]
#[command(about = "Download and serve LLM models as influencer", long_about = None)]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    #[command(about = "Download model from HuggingFace mirror")]
    Download {
        #[arg(short, long, help = "Model name (e.g., 'ibm/granite-4-h-small')")]
        model: String,

        #[arg(short = 'r', long, help = "Mirror URL (default: hf-mirror.com)")]
        mirror: Option<String>,

        #[arg(short, long, help = "Output directory")]
        output: Option<PathBuf>,
    },

    #[command(about = "Search for models on HuggingFace")]
    Search {
        #[arg(help = "Search query")]
        query: String,

        #[arg(short, long, default_value = "20", help = "Maximum number of results")]
        limit: usize,

        #[arg(short, long, help = "Filter by author/organization")]
        author: Option<String>,
    },

    #[command(about = "Serve LLM as influencer")]
    Serve {
        #[arg(short, long, help = "Path to model directory")]
        model_path: Option<PathBuf>,

        #[arg(short, long, default_value = "8080", help = "Port to serve on")]
        port: u16,
    },

    #[command(about = "Generate text using the LLM")]
    Generate {
        #[arg(help = "Prompt text")]
        prompt: String,

        #[arg(short, long, help = "Path to model directory")]
        model_path: Option<PathBuf>,

        #[arg(long, default_value = "512", help = "Maximum tokens to generate")]
        max_tokens: usize,

        #[arg(long, default_value = "0.7", help = "Temperature for generation")]
        temperature: f32,
    },
}
