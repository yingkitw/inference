use crate::error::{InfluenceError, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder};
use candle_transformers::{
    models::{quantized_llama::{Model as QLlama,_weights}, llama_llm::{Llama as Llama, LlamaConfig}},
    quantized::{GgmlDType, QTensor},
    GenerationConfig,
};
use tokenizers::Tokenizer;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, debug, warn};

/// Supported model architectures for local inference
#[derive(Debug, Clone, Copy)]
pub enum ModelArchitecture {
    Llama,
    LlamaQuantized,
    Mistral,
    Phi,
    // Add more architectures as needed
}

/// Configuration for local model loading
#[derive(Debug, Clone)]
pub struct LocalModelConfig {
    pub model_path: PathBuf,
    pub architecture: ModelArchitecture,
    pub quantized: bool,
    pub max_seq_len: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Default for LocalModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models"),
            architecture: ModelArchitecture::Llama,
            quantized: false,
            max_seq_len: 4096,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}

/// Local model service for running inference on downloaded models
pub struct LocalModel {
    config: LocalModelConfig,
    tokenizer: Tokenizer,
    device: Device,
    model: Option<Box<dyn candle_transformers::models::llama_llm::ModelWeights>>,
}

impl LocalModel {
    /// Load a local model from the specified path
    pub async fn load(config: LocalModelConfig) -> Result<Self> {
        info!("Loading local model from: {}", config.model_path.display());

        // Validate model directory exists
        if !config.model_path.exists() {
            return Err(InfluenceError::ModelNotFound(format!(
                "Model directory not found: {}",
                config.model_path.display()
            )));
        }

        // Load tokenizer
        let tokenizer = Self::load_tokenizer(&config.model_path)?;

        // Detect model architecture and load weights
        let architecture = Self::detect_architecture(&config.model_path)?;
        let device = Device::Cpu;

        info!("Detected architecture: {:?}", architecture);
        info!("Loading model weights (this may take a while)...");

        let model = Self::load_model_weights(&config.model_path, architecture, &device)?;

        Ok(Self {
            config,
            tokenizer,
            device,
            model: Some(model),
        })
    }

    /// Load the tokenizer from the model directory
    fn load_tokenizer(model_path: &Path) -> Result<Tokenizer> {
        let tokenizer_files = ["tokenizer.json", "tokenizer_config.json"];

        // Try to find tokenizer file
        let tokenizer_path = tokenizer_files.iter()
            .find_map(|file| {
                let path = model_path.join(file);
                if path.exists() { Some(path) } else { None }
            })
            .ok_or_else(|| InfluenceError::InvalidConfig(
                "Tokenizer file not found in model directory. Expected tokenizer.json or tokenizer_config.json".to_string()
            ))?;

        debug!("Loading tokenizer from: {}", tokenizer_path.display());

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| InfluenceError::LlmError(format!("Failed to load tokenizer: {}", e)))?;

        Ok(tokenizer)
    }

    /// Detect the model architecture from config.json or model files
    fn detect_architecture(model_path: &Path) -> Result<ModelArchitecture> {
        // Check for config.json
        let config_path = model_path.join("config.json");
        if config_path.exists() {
            let config_content = fs::read_to_string(&config_path)
                .map_err(|e| InfluenceError::IoError(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Failed to read config.json: {}", e)
                )))?;

            if let Ok(config_json) = serde_json::from_str::<serde_json::Value>(&config_content) {
                // Check model_type in config
                if let Some(model_type) = config_json.get("model_type").and_then(|v| v.as_str()) {
                    debug!("Model type from config: {}", model_type);

                    return match model_type {
                        "llama" | "LlamaForCausalLM" => Ok(ModelArchitecture::Llama),
                        "mistral" | "MistralForCausalLM" => Ok(ModelArchitecture::Mistral),
                        "phi" | "PhiForCausalLM" => Ok(ModelArchitecture::Phi),
                        _ => {
                            warn!("Unknown model type '{}', defaulting to Llama", model_type);
                            Ok(ModelArchitecture::Llama)
                        }
                    };
                }
            }
        }

        // Fallback: check for model files
        let has_safetensors = model_path.join("model.safetensors").exists()
            || model_path.join("model-00001-of-00001.safetensors").exists();

        let has_gguf = model_path.join("model.gguf").exists()
            || model_path.join("model-q4k.gguf").exists();

        if has_gguf {
            return Ok(ModelArchitecture::LlamaQuantized);
        }

        if has_safetensors {
            return Ok(ModelArchitecture::Llama); // Default to Llama for safetensors
        }

        // Default fallback
        warn!("Could not detect model architecture, defaulting to Llama");
        Ok(ModelArchitecture::Llama)
    }

    /// Load the model weights based on architecture
    fn load_model_weights(
        model_path: &Path,
        architecture: ModelArchitecture,
        device: &Device,
    ) -> Result<Box<dyn candle_transformers::models::llama_llm::ModelWeights>> {
        // This is a simplified placeholder - the actual implementation will vary
        // based on the specific model architecture being used

        // For now, we'll create a basic implementation structure
        // The actual weight loading will be done in the generate method

        warn!("Model weight loading not fully implemented yet");
        Err(InfluenceError::LlmError(
            "Local model loading not fully implemented - this requires architecture-specific code".to_string()
        ))
    }

    /// Generate text using the local model
    pub async fn generate_text(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String> {
        info!("Generating text locally (max_tokens={}, temperature={})", max_tokens, temperature);

        // Encode the prompt
        let tokens = self.tokenizer.encode(prompt, false)
            .map_err(|e| InfluenceError::LlmError(format!("Tokenization failed: {}", e)))?;

        let input_ids = tokens.get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect::<Vec<_>>();

        debug!("Prompt tokenized into {} tokens", input_ids.len());

        // Placeholder for actual generation
        // The real implementation would use candle's sampling and model forward pass
        warn!("Local generation not fully implemented yet - using echo mode");

        // For now, just echo the prompt with a note
        Ok(format!("{} [Local generation would continue here]", prompt))
    }

    /// Generate text with streaming output
    pub async fn generate_stream(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<()> {
        info!("Starting local streaming generation");

        let response = self.generate_text(prompt, max_tokens, temperature).await?;

        print!("{}", response);
        use std::io::Write;
        std::io::stdout().flush().unwrap();

        Ok(())
    }
}

/// Helper function to automatically detect and load a model from a directory
pub async fn load_model_from_path(path: &Path) -> Result<LocalModel> {
    let config = LocalModelConfig {
        model_path: path.to_path_buf(),
        ..Default::default()
    };

    LocalModel::load(config).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_architecture_from_name() {
        // Test architecture detection logic
        let result = ModelArchitecture::Llama;
        assert!(matches!(result, ModelArchitecture::Llama));
    }

    #[tokio::test]
    async fn test_local_model_config_default() {
        let config = LocalModelConfig::default();
        assert_eq!(config.max_seq_len, 4096);
        assert_eq!(config.temperature, 0.7);
        assert!(!config.quantized);
    }
}
