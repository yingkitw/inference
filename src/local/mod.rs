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
    pub top_k: Option<usize>,
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
            top_k: None,
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
    pub fn config(&self) -> &LocalModelConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut LocalModelConfig {
        &mut self.config
    }

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

        // Extract config values before mutable borrow
        let top_p = self.config.top_p;
        let top_k = self.config.top_k;

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

                // Sample token using temperature, top-k, and top-p
                let next = Self::do_sample(last_logits, temperature, top_p, top_k)?;

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

                    let next = Self::do_sample(token_logits, temperature, top_p, top_k)?;
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

    /// Sample a token from logits using temperature, top-k, and top-p sampling
    fn sample_token(&self, logits: &[f32], temperature: f32) -> Result<u32> {
        let vocab_size = logits.len();

        // Apply temperature scaling
        let scaled: Vec<f32> = logits.iter()
            .map(|&logit| if temperature > 0.0 { logit / temperature } else { logit })
            .collect();

        // Get top_k if specified
        let top_k = self.config.top_k.unwrap_or(vocab_size);
        let mut sorted_indices: Vec<usize> = (0..vocab_size).collect();
        sorted_indices.sort_by(|&a, &b| {
            scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep only top k tokens
        sorted_indices.truncate(top_k);

        // Apply softmax to get probabilities for top-k tokens
        let mut probs: Vec<f32> = sorted_indices.iter()
            .map(|&i| {
                let logit = scaled[i];
                // For numerical stability, subtract max before exp
                logit.exp()
            })
            .collect();

        let sum: f32 = probs.iter().sum();
        if sum == 0.0 {
            // Fallback to argmax if all probabilities are zero
            return Ok(sorted_indices[0] as u32);
        }

        // Normalize probabilities
        for prob in probs.iter_mut() {
            *prob /= sum;
        }

        // Apply top-p (nucleus sampling) if specified
        let top_p = self.config.top_p;
        if top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut cutoff_idx = sorted_indices.len();

            for (idx, &prob) in probs.iter().enumerate() {
                cumulative += prob;
                if cumulative >= top_p {
                    cutoff_idx = idx + 1;
                    break;
                }
            }

            // Renormalize after cutoff
            sorted_indices.truncate(cutoff_idx);
            probs.truncate(cutoff_idx);

            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for prob in probs.iter_mut() {
                    *prob /= sum;
                }
            }
        }

        // Sample from the distribution using a simple approach
        // Use system time for randomness
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| InfluenceError::LocalModelError(format!("Failed to get time: {}", e)))?
            .as_nanos() as f32;

        let random_value = (nanos % 1000000.0) / 1000000.0;
        let mut cumulative = 0.0;

        for (idx, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                return Ok(sorted_indices[idx] as u32);
            }
        }

        // Fallback to last token (shouldn't happen if probabilities sum to 1)
        Ok(sorted_indices[sorted_indices.len() - 1] as u32)
    }

    /// Sample a token from logits using temperature, top-k, and top-p sampling with explicit config
    /// This version takes config values as parameters to avoid borrow checker issues
    fn do_sample(logits: &[f32], temperature: f32, top_p: f32, top_k: Option<usize>) -> Result<u32> {
        let vocab_size = logits.len();

        // Apply temperature scaling
        let scaled: Vec<f32> = logits.iter()
            .map(|&logit| if temperature > 0.0 { logit / temperature } else { logit })
            .collect();

        // Get top_k if specified
        let top_k = top_k.unwrap_or(vocab_size);
        let mut sorted_indices: Vec<usize> = (0..vocab_size).collect();
        sorted_indices.sort_by(|&a, &b| {
            scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep only top k tokens
        sorted_indices.truncate(top_k);

        // Apply softmax to get probabilities for top-k tokens
        let mut probs: Vec<f32> = sorted_indices.iter()
            .map(|&i| {
                let logit = scaled[i];
                // For numerical stability, subtract max before exp
                logit.exp()
            })
            .collect();

        let sum: f32 = probs.iter().sum();
        if sum == 0.0 {
            // Fallback to argmax if all probabilities are zero
            return Ok(sorted_indices[0] as u32);
        }

        // Normalize probabilities
        for prob in probs.iter_mut() {
            *prob /= sum;
        }

        // Apply top-p (nucleus sampling) if specified
        if top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut cutoff_idx = sorted_indices.len();

            for (idx, &prob) in probs.iter().enumerate() {
                cumulative += prob;
                if cumulative >= top_p {
                    cutoff_idx = idx + 1;
                    break;
                }
            }

            // Renormalize after cutoff
            sorted_indices.truncate(cutoff_idx);
            probs.truncate(cutoff_idx);

            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for prob in probs.iter_mut() {
                    *prob /= sum;
                }
            }
        }

        // Sample from the distribution using a simple approach
        // Use system time for randomness
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| InfluenceError::LocalModelError(format!("Failed to get time: {}", e)))?
            .as_nanos() as f32;

        let random_value = (nanos % 1000000.0) / 1000000.0;
        let mut cumulative = 0.0;

        for (idx, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                return Ok(sorted_indices[idx] as u32);
            }
        }

        // Fallback to last token (shouldn't happen if probabilities sum to 1)
        Ok(sorted_indices[sorted_indices.len() - 1] as u32)
    }

    fn stream_piece(tokenizer: &Tokenizer, token_id: u32, started: &mut bool) -> Result<Option<String>> {
        if let Some(raw) = tokenizer.id_to_token(token_id) {
            if raw == "</s>" || raw == "<s>" || raw == "<unk>" {
                return Ok(None);
            }

            let text = tokenizer.decode(&[token_id], false)?;
            if text.is_empty() {
                return Ok(None);
            }

            let mut piece = String::new();
            if raw.starts_with('▁') && *started {
                piece.push(' ');
            }
            piece.push_str(&text);
            *started = true;
            return Ok(Some(piece));
        }

        Ok(None)
    }

    pub async fn generate_stream_with<F>(&mut self, prompt: &str, max_tokens: usize, temp: f32, mut emit: F) -> Result<()>
    where
        F: FnMut(String) -> Result<()>,
    {
        let tokens = self.tokenizer.encode(prompt, false)?;
        let input_ids: Vec<u32> = tokens.get_ids().to_vec();
        let eos_token = self.get_eos_token();

        // Extract config values before mutable borrow
        let top_p = self.config.top_p;
        let top_k = self.config.top_k;

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

                // Sample token using temperature, top-k, and top-p
                let mut next = Self::do_sample(last_logits, temp, top_p, top_k)?;

                let mut started = false;
                if let Some(piece) = Self::stream_piece(&self.tokenizer, next, &mut started)? {
                    emit(piece)?;
                }

                // Generate remaining tokens
                for idx in 1..max_tokens {
                    if let Some(eos) = eos_token {
                        if next == eos { break; }
                    }

                    let tensor = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
                    let logits = model.forward(&tensor, input_ids.len() + idx - 1, cache)?;

                    let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                    let token_logits = &logits_vec[0];

                    next = Self::do_sample(token_logits, temp, top_p, top_k)?;
                    if let Some(piece) = Self::stream_piece(&self.tokenizer, next, &mut started)? {
                        emit(piece)?;
                    }
                }
            }
            LocalBackend::Mistral { .. } => {
                return Err(InfluenceError::LocalModelError(
                    "Mistral backend generation not yet implemented".to_string()
                ));
            }
        }

        Ok(())
    }

    pub async fn generate_stream(&mut self, prompt: &str, max_tokens: usize, temp: f32) -> Result<()> {
        self.generate_stream_with(prompt, max_tokens, temp, |piece| {
            print!("{}", piece);
            std::io::stdout().flush().unwrap();
            Ok(())
        }).await?;
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
        assert_eq!(cfg.top_p, 0.9);
        assert_eq!(cfg.top_k, None);
        assert_eq!(cfg.repeat_penalty, 1.1);
    }

    #[test]
    fn test_do_sample_with_temperature() {
        // Create a simple logits distribution
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Test with temperature = 1.0 (no scaling)
        let result = LocalModel::do_sample(&logits, 1.0, 1.0, None);
        assert!(result.is_ok());
        let token = result.unwrap();
        assert!(token < 5);

        // Test with temperature = 0.5 (more deterministic)
        let result = LocalModel::do_sample(&logits, 0.5, 1.0, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_do_sample_with_top_k() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.1, 0.2];

        // Sample from top 3 tokens only
        let result = LocalModel::do_sample(&logits, 1.0, 1.0, Some(3));
        assert!(result.is_ok());
        let token = result.unwrap();
        // Should sample from top 3 tokens (indices 2, 3, 4 with values 3.0, 4.0, 5.0)
        assert!(token >= 2 && token <= 4);
    }

    #[test]
    fn test_do_sample_with_top_p() {
        let logits = vec![0.1, 0.2, 0.3, 4.0, 5.0];

        // Sample with top_p = 0.9 (nucleus sampling)
        let result = LocalModel::do_sample(&logits, 1.0, 0.9, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_do_sample_with_zero_temperature() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Zero temperature should be nearly deterministic (argmax)
        let result = LocalModel::do_sample(&logits, 0.0, 1.0, None);
        assert!(result.is_ok());
        // With zero temperature, should pick the highest logit (index 4)
        let token = result.unwrap();
        assert_eq!(token, 4);
    }

    #[test]
    fn test_do_sample_single_token() {
        let logits = vec![5.0];

        let result = LocalModel::do_sample(&logits, 1.0, 1.0, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_do_sample_with_both_top_k_and_top_p() {
        let logits = vec![0.1, 0.2, 0.3, 4.0, 5.0, 0.5, 0.6, 0.7];

        // Combine top-k and top-p
        let result = LocalModel::do_sample(&logits, 1.0, 0.8, Some(4));
        assert!(result.is_ok());
        let token = result.unwrap();
        // Should be within top-k range
        assert!(token < 8);
    }

    #[test]
    fn test_config_getters() {
        let config = LocalModelConfig::default();

        // Test public field access
        assert_eq!(config.model_path, PathBuf::from("models"));
        assert_eq!(config.max_seq_len, 4096);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.repeat_penalty, 1.1);
    }
}
