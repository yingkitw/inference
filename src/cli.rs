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

        #[arg(long, help = "Optional system prompt")]
        system: Option<String>,

        #[arg(short, long, help = "Path to model directory")]
        model_path: Option<PathBuf>,

        #[arg(long, default_value = "512", help = "Maximum tokens to generate")]
        max_tokens: usize,

        #[arg(long, default_value = "0.7", help = "Temperature for generation")]
        temperature: f32,

        #[arg(long, default_value = "auto", help = "Compute device: auto|cpu|metal|cuda")]
        device: String,

        #[arg(long, default_value = "0", help = "Device index (GPU ordinal) when using metal/cuda")]
        device_index: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_command_parsing() {
        let args = vec!["influence", "download", "-m", "test/model", "-r", "https://example.com", "-o", "/tmp/models"];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Download { model, mirror, output } => {
                assert_eq!(model, "test/model");
                assert_eq!(mirror, Some("https://example.com".to_string()));
                assert_eq!(output, Some(PathBuf::from("/tmp/models")));
            }
            _ => panic!("Expected Download command"),
        }
    }

    #[test]
    fn test_search_command_parsing() {
        let args = vec!["influence", "search", "llama", "--limit", "10", "--author", "meta"];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Search { query, limit, author } => {
                assert_eq!(query, "llama");
                assert_eq!(limit, 10);
                assert_eq!(author, Some("meta".to_string()));
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_search_default_limit() {
        let args = vec!["influence", "search", "query"];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Search { limit, .. } => {
                assert_eq!(limit, 20);
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_generate_command_parsing() {
        let args = vec![
            "influence",
            "generate",
            "Hello world",
            "--system",
            "You are a helpful assistant.",
            "--model-path",
            "/models",
            "--max-tokens",
            "100",
            "--temperature",
            "0.5",
            "--device",
            "cpu",
            "--device-index",
            "1",
        ];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Generate { prompt, system, model_path, max_tokens, temperature, device, device_index } => {
                assert_eq!(prompt, "Hello world");
                assert_eq!(system, Some("You are a helpful assistant.".to_string()));
                assert_eq!(model_path, Some(PathBuf::from("/models")));
                assert_eq!(max_tokens, 100);
                assert_eq!(temperature, 0.5);
                assert_eq!(device, "cpu");
                assert_eq!(device_index, 1);
            }
            _ => panic!("Expected Generate command"),
        }
    }

    #[test]
    fn test_generate_default_values() {
        let args = vec!["influence", "generate", "test"];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Generate { max_tokens, temperature, device, device_index, .. } => {
                assert_eq!(max_tokens, 512);
                assert_eq!(temperature, 0.7);
                assert_eq!(device, "auto");
                assert_eq!(device_index, 0);
            }
            _ => panic!("Expected Generate command"),
        }
    }

    #[test]
    fn test_serve_command_parsing() {
        let args = vec!["influence", "serve", "--model-path", "/models", "--port", "9000"];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Serve { model_path, port } => {
                assert_eq!(model_path, Some(PathBuf::from("/models")));
                assert_eq!(port, 9000);
            }
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_serve_default_port() {
        let args = vec!["influence", "serve"];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_ok());
        let cli = cli.unwrap();
        match cli.command {
            Commands::Serve { port, .. } => {
                assert_eq!(port, 8080);
            }
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_invalid_command() {
        let args = vec!["influence", "invalid-command"];
        let cli = Cli::try_parse_from(args);

        assert!(cli.is_err());
    }
}
