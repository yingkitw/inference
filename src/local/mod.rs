use crate::error::{InfluenceError, Result};
use tokenizers::Tokenizer;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use tracing::{info, warn};
use candle_core::{Device, Tensor, DType};
use std::io::Write;

mod backends;
mod device;

pub use backends::LocalBackend;
pub use device::get_device;

#[derive(Debug, Clone, Copy)]
pub enum ModelArchitecture {
    Llama,
    LlamaQuantized,
    Mistral,
    Phi,
    Granite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

impl FromStr for DevicePreference {
    type Err = InfluenceError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            other => Err(InfluenceError::InvalidConfig(format!(
                "Invalid device '{}'. Use one of: auto, cpu, metal, cuda",
                other
            ))),
        }
    }
}

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
    pub device_preference: DevicePreference,
    pub device_index: usize,
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
            device_preference: DevicePreference::Auto,
            device_index: 0,
        }
    }
}

pub struct LocalModel {
    config: LocalModelConfig,
    tokenizer: Tokenizer,
    backend: Option<LocalBackend>,
    device: Device,
}

impl LocalModel {
    pub async fn load(mut config: LocalModelConfig) -> Result<Self> {
        info!("Loading local model from: {}", config.model_path.display());

        if !config.model_path.exists() {
            return Err(InfluenceError::ModelNotFound(format!(
                "Model directory not found: {}",
                config.model_path.display()
            )));
        }

        let tokenizer = Self::load_tokenizer(&config.model_path)?;
        let architecture = Self::detect_architecture(&config.model_path)?;
        config.architecture = architecture;

        info!("Detected architecture: {:?}", architecture);

        let device = get_device(config.device_preference, config.device_index)?;
        info!("Using device: {:?}", device);

        let backend = match architecture {
            ModelArchitecture::Llama | ModelArchitecture::LlamaQuantized => {
                LocalBackend::load_llama(&config, &device)?
            }
            ModelArchitecture::Mistral => LocalBackend::load_mistral(&config, &device)?,
            _ => {
                warn!("Architecture {:?} not yet fully implemented", architecture);
                None
            }
        };

        if backend.is_some() {
            info!("Model loaded successfully with full inference capability!");
        } else {
            info!("Model structure loaded (placeholder mode - no .safetensors files found)");
        }

        Ok(Self {
            config,
            tokenizer,
            backend,
            device,
        })
    }

    fn load_tokenizer(model_path: &Path) -> Result<Tokenizer> {
        let tokenizer_files = ["tokenizer.json", "tokenizer_config.json"];
        let tokenizer_path = tokenizer_files.iter()
            .find_map(|file| {
                let path = model_path.join(file);
                if path.exists() { Some(path) } else { None }
            })
            .ok_or_else(|| InfluenceError::InvalidConfig(
                "Tokenizer file not found".to_string()
            ))?;

        Tokenizer::from_file(tokenizer_path)
            .map_err(|e| InfluenceError::LocalModelError(format!("Failed to load tokenizer: {}", e)))
    }

    fn detect_architecture(model_path: &Path) -> Result<ModelArchitecture> {
        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            return Ok(ModelArchitecture::Llama);
        }

        let config_content = fs::read_to_string(&config_path)
            .map_err(|e| InfluenceError::LocalModelError(format!("Failed to read config: {}", e)))?;

        let config: serde_json::Value = serde_json::from_str(&config_content)
            .map_err(|e| InfluenceError::LocalModelError(format!("Failed to parse config: {}", e)))?;

        let model_type = config.get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("llama");

        // Check for encoder-only models (not suitable for text generation)
        if model_type == "bert" || model_type == "roberta" || model_type == "albert" {
            return Err(InfluenceError::LocalModelError(
                format!(
                    "Unsupported model type '{}': This is an encoder-only model.\n\
                    \n\
                    Encoder-only models (BERT, RoBERTa, ALBERT) are designed for:\n\
                    - Text classification\n\
                    - Named entity recognition\n\
                    - Question answering\n\
                    - Embeddings\n\
                    \n\
                    They CANNOT generate text. For text generation, use:\n\
                    - Llama models (meta-llama/Llama-2-7b-hf)\n\
                    - Mistral models (mistralai/Mistral-7B-v0.1)\n\
                    - GPT models (decoder-only architectures)\n\
                    \n\
                    Try: cargo run -- download -m TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    model_type
                )
            ));
        }

        match model_type {
            "llama" => Ok(ModelArchitecture::Llama),
            "mistral" => Ok(ModelArchitecture::Mistral),
            "phi" => Ok(ModelArchitecture::Phi),
            "granite" => Ok(ModelArchitecture::Granite),
            _ => {
                warn!("Unknown model type '{}', defaulting to Llama", model_type);
                Ok(ModelArchitecture::Llama)
            }
        }
    }

    pub async fn generate_text(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String> {
        info!("Generating (max_tokens={}, temp={})", max_tokens, temperature);

        let tokens = self.tokenizer.encode(prompt, false)
            .map_err(|e| InfluenceError::LocalModelError(format!("Tokenization failed: {}", e)))?;
        let input_ids: Vec<u32> = tokens.get_ids().to_vec();

        let eos_token = self.get_eos_token();

        let backend = self.backend.as_mut().ok_or_else(|| InfluenceError::LocalModelError(
            format!("Model not loaded. Ensure .safetensors files are in: {}", self.config.model_path.display())
        ))?;

        let generated = match backend {
            LocalBackend::Llama { model, cache } => {
                // Process the prompt to fill the cache
                let prompt_tensor = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;
                let logits = model.forward(&prompt_tensor, 0, cache)?;

                // logits shape: [batch=1, vocab_size] (model returns logits for last token only)
                let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                let last_logits = &logits_vec[0];

                // Apply temperature and find argmax
                let mut next = 0u32;
                let mut max_val = f32::NEG_INFINITY;
                for (i, &logit) in last_logits.iter().enumerate() {
                    let scaled = if temperature > 0.0 { logit / temperature } else { logit };
                    if scaled > max_val {
                        max_val = scaled;
                        next = i as u32;
                    }
                }

                let mut generated = vec![next];

                // Generate remaining tokens one at a time
                for idx in 1..max_tokens {
                    if let Some(eos) = eos_token {
                        if next == eos { break; }
                    }

                    let tensor = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
                    let logits = model.forward(&tensor, input_ids.len() + idx - 1, cache)?;

                    // Single token: [batch=1, vocab]
                    let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                    let token_logits = &logits_vec[0];

                    let mut max_val = f32::NEG_INFINITY;
                    for (i, &logit) in token_logits.iter().enumerate() {
                        let scaled = if temperature > 0.0 { logit / temperature } else { logit };
                        if scaled > max_val {
                            max_val = scaled;
                            next = i as u32;
                        }
                    }

                    generated.push(next);
                }

                generated
            }
            LocalBackend::Mistral { .. } => {
                return Err(InfluenceError::LocalModelError(
                    "Mistral backend generation not yet implemented".to_string()
                ));
            }
        };

        // Decode tokens with proper spacing
        let mut result = String::new();
        let mut skipped_special = 0;

        for (i, &token_id) in generated.iter().enumerate() {
            let raw_token = self.tokenizer.id_to_token(token_id);

            if let Some(ref token) = raw_token {
                if token == "</s>" || token == "<s>" || token == "<unk>" {
                    skipped_special += 1;
                    continue;
                }
            }

            let token_str = self.tokenizer.decode(&[token_id], false)
                .map_err(|e| InfluenceError::LocalModelError(format!("Token decode failed: {}", e)))?;

            if let Some(raw) = raw_token {
                if raw.starts_with('▁') {
                    let actual_index = i - skipped_special;
                    if actual_index > 0 && !result.is_empty() {
                        result.push(' ');
                    }
                }
            }

            result.push_str(&token_str);
        }

        Ok(result.trim().to_string())
    }

    fn get_eos_token(&self) -> Option<u32> {
        self.tokenizer.token_to_id("</s>")
            .or_else(|| self.tokenizer.token_to_id("<EOS>"))
    }

    fn print_stream_token(tokenizer: &Tokenizer, token_id: u32, started: &mut bool) -> Result<()> {
        if let Some(raw) = tokenizer.id_to_token(token_id) {
            if raw == "</s>" || raw == "<s>" || raw == "<unk>" {
                return Ok(());
            }

            if raw.starts_with('▁') && *started {
                print!(" ");
            }
        }

        let text = tokenizer.decode(&[token_id], false)?;
        if text.is_empty() {
            return Ok(());
        }

        print!("{}", text);
        std::io::stdout().flush().unwrap();
        *started = true;
        Ok(())
    }

    pub async fn generate_stream(&mut self, prompt: &str, max_tokens: usize, temp: f32) -> Result<()> {
        let tokens = self.tokenizer.encode(prompt, false)?;
        let input_ids: Vec<u32> = tokens.get_ids().to_vec();
        let eos_token = self.get_eos_token();

        let backend = self.backend.as_mut().ok_or_else(|| InfluenceError::LocalModelError(
            format!("Model not loaded. Ensure .safetensors files are in: {}", self.config.model_path.display())
        ))?;

        match backend {
            LocalBackend::Llama { model, cache } => {
                // Process prompt
                let prompt_tensor = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;
                let logits = model.forward(&prompt_tensor, 0, cache)?;

                let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                let last_logits = &logits_vec[0];

                let mut next = 0u32;
                let mut max_val = f32::NEG_INFINITY;
                for (i, &logit) in last_logits.iter().enumerate() {
                    let scaled = if temp > 0.0 { logit / temp } else { logit };
                    if scaled > max_val {
                        max_val = scaled;
                        next = i as u32;
                    }
                }

                let mut started = false;
                Self::print_stream_token(&self.tokenizer, next, &mut started)?;

                // Generate remaining tokens
                for idx in 1..max_tokens {
                    if let Some(eos) = eos_token {
                        if next == eos { break; }
                    }

                    let tensor = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
                    let logits = model.forward(&tensor, input_ids.len() + idx - 1, cache)?;

                    let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                    let token_logits = &logits_vec[0];

                    let mut max_val = f32::NEG_INFINITY;
                    for (i, &logit) in token_logits.iter().enumerate() {
                        let scaled = if temp > 0.0 { logit / temp } else { logit };
                        if scaled > max_val {
                            max_val = scaled;
                            next = i as u32;
                        }
                    }

                    Self::print_stream_token(&self.tokenizer, next, &mut started)?;
                }
            }
            LocalBackend::Mistral { .. } => {
                return Err(InfluenceError::LocalModelError(
                    "Mistral backend generation not yet implemented".to_string()
                ));
            }
        }

        println!();
        Ok(())
    }
}

impl crate::influencer::LlmService for LocalModel {
    async fn generate_text(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String> {
        self.generate_text(prompt, max_tokens, temperature).await
    }

    async fn generate_stream(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<()> {
        self.generate_stream(prompt, max_tokens, temperature).await
    }
}

pub async fn load_model_from_path(path: &Path) -> Result<LocalModel> {
    LocalModel::load(LocalModelConfig {
        model_path: path.to_path_buf(),
        ..Default::default()
    }).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture() {
        assert!(matches!(ModelArchitecture::Llama, ModelArchitecture::Llama));
    }

    #[tokio::test]
    async fn test_config_default() {
        let cfg = LocalModelConfig::default();
        assert_eq!(cfg.max_seq_len, 4096);
        assert_eq!(cfg.temperature, 0.7);
    }
}
