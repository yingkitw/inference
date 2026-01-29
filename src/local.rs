use crate::error::{InfluenceError, Result};
use tokenizers::Tokenizer;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Llama, Config as LlamaConfig, Cache};
use std::io::Write;

#[derive(Debug, Clone, Copy)]
pub enum ModelArchitecture {
    Llama,
    LlamaQuantized,
    Mistral,
    Phi,
    Granite,
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

pub struct LocalModel {
    config: LocalModelConfig,
    tokenizer: Tokenizer,
    model: Option<Llama>,
    cache: Option<Cache>,
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

        // Try to use GPU if available (Metal on macOS, CUDA on Linux/Windows)
        let device = Self::get_device()?;
        info!("Using device: {:?}", device);

        let (model, cache) = match architecture {
            ModelArchitecture::Llama | ModelArchitecture::LlamaQuantized => {
                Self::load_llama_model(&config, &device)?
            }
            _ => {
                warn!("Architecture {:?} not yet fully implemented", architecture);
                (None, None)
            }
        };

        if model.is_some() {
            info!("Model loaded successfully with full inference capability!");
        } else {
            info!("Model structure loaded (placeholder mode - no .safetensors files found)");
        }

        Ok(Self {
            config,
            tokenizer,
            model,
            cache,
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

    fn load_llama_model(config: &LocalModelConfig, device: &Device) -> Result<(Option<Llama>, Option<Cache>)> {
        info!("Loading Llama model weights...");

        let config_path = config.model_path.join("config.json");
        if !config_path.exists() {
            return Err(InfluenceError::LocalModelError("config.json not found".to_string()));
        }

        let config_content = fs::read_to_string(&config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&config_content)
            .map_err(|e| InfluenceError::LocalModelError(format!("Failed to parse config: {}", e)))?;

        // Check for unsupported architectures
        if let Some(layer_types) = config_json.get("layer_types") {
            if layer_types.is_array() {
                return Err(InfluenceError::LocalModelError(
                    format!(
                        "Unsupported model architecture: This model uses mixed Mamba+Attention layers.\n\
                        \n\
                        The current implementation only supports standard Llama/Mistral transformer models.\n\
                        \n\
                        This model (GraniteMoeHybrid) requires a specialized implementation.\n\
                        \n\
                        Supported models:\n\
                        - Standard Llama models (meta-llama/Llama-2-7b-hf, etc.)\n\
                        - Standard Mistral models (mistralai/Mistral-7B-v0.1, etc.)\n\
                        - Pure transformer-based Granite models\n\
                        \n\
                        For this model, consider using:\n\
                        - transformers library with Python\n\
                        - vLLM for high-performance serving\n\
                        - llama.cpp (if GGUF version available)"
                    )
                ));
            }
        }

        // Extract parameters from config.json
        let vocab_size = config_json.get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(32000) as usize;
        
        let hidden_size = config_json.get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize;
        
        let intermediate_size = config_json.get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(11008) as usize;
        
        let num_hidden_layers = config_json.get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        
        let num_attention_heads = config_json.get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        
        let num_key_value_heads = config_json.get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .or_else(|| Some(num_attention_heads as u64))
            .unwrap() as usize;
        
        let rms_norm_eps = config_json.get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);
        
        let rope_theta = config_json.get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0) as f32;

        // Create LlamaConfig with actual model parameters
        let llama_config = LlamaConfig {
            hidden_size,
            intermediate_size,
            vocab_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            rms_norm_eps,
            rope_theta,
            use_flash_attn: false,
            ..LlamaConfig::config_7b_v2(false)
        };
        
        info!("Config: vocab={}, hidden={}, layers={}, heads={}", 
            vocab_size, hidden_size, num_hidden_layers, num_attention_heads);

        let weight_files = Self::find_weight_files(&config.model_path)?;
        if weight_files.is_empty() {
            warn!("No .safetensors files found");
            return Ok((None, None));
        }

        info!("Loading {} weight file(s)...", weight_files.len());

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, device)
                .map_err(|e| InfluenceError::LocalModelError(format!("Failed to load weights: {}", e)))?
        };

        let model = Llama::load(vb, &llama_config)
            .map_err(|e| InfluenceError::LocalModelError(format!("Failed to create model: {}", e)))?;

        let cache = Cache::new(true, DType::F32, &llama_config, device)
            .map_err(|e| InfluenceError::LocalModelError(format!("Failed to create cache: {}", e)))?;

        info!("Model and cache initialized");
        Ok((Some(model), Some(cache)))
    }

    fn get_device() -> Result<Device> {
        // Try Metal (macOS GPU) first
        #[cfg(feature = "metal")]
        {
            match Device::new_metal(0) {
                Ok(device) => {
                    info!("✓ Metal GPU available");
                    return Ok(device);
                }
                Err(e) => {
                    warn!("Metal GPU not available: {}, falling back to CPU", e);
                }
            }
        }
        
        // Try CUDA (NVIDIA GPU) on other platforms
        #[cfg(feature = "cuda")]
        {
            match Device::new_cuda(0) {
                Ok(device) => {
                    info!("✓ CUDA GPU available");
                    return Ok(device);
                }
                Err(e) => {
                    warn!("CUDA GPU not available: {}, falling back to CPU", e);
                }
            }
        }
        
        // Fall back to CPU
        info!("Using CPU");
        Ok(Device::Cpu)
    }

    fn find_weight_files(model_path: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        if let Ok(entries) = fs::read_dir(model_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "safetensors" {
                        files.push(path);
                    }
                }
            }
        }
        files.sort();
        Ok(files)
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

        if self.model.is_none() {
            return Ok(format!(
                "OK Tokenizer ready (vocab: {})\nX Model weights not loaded\n\nTo enable inference, ensure .safetensors files are in: {}",
                self.tokenizer.get_vocab_size(true),
                self.config.model_path.display()
            ));
        }

        let mut generated = Vec::new();
        let model = self.model.as_ref().unwrap();
        let eos_token = self.get_eos_token();
        let cache = self.cache.as_mut().unwrap();

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
        
        generated.push(next);
        
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

        // Decode generated tokens without skipping special tokens to preserve spacing
        let decoded = self.tokenizer.decode(&generated, false)
            .map_err(|e| InfluenceError::LocalModelError(format!("Decode failed: {}", e)))?;
        
        // Replace SentencePiece underscore (▁) with regular spaces
        Ok(decoded.replace('▁', " "))
    }

    fn get_eos_token(&self) -> Option<u32> {
        self.tokenizer.token_to_id("</s>")
            .or_else(|| self.tokenizer.token_to_id("<EOS>"))
    }

    pub async fn generate_stream(&mut self, prompt: &str, max_tokens: usize, temp: f32) -> Result<()> {
        if self.model.is_none() {
            let resp = self.generate_text(prompt, max_tokens, temp).await?;
            print!("{}", resp);
            std::io::stdout().flush().unwrap();
            return Ok(());
        }

        let tokens = self.tokenizer.encode(prompt, false)?;
        let input_ids: Vec<u32> = tokens.get_ids().to_vec();
        let model = self.model.as_ref().unwrap();
        let eos_token = self.get_eos_token();
        let cache = self.cache.as_mut().unwrap();

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
        
        let text = self.tokenizer.decode(&[next], false)?;
        print!("{}", text);
        std::io::stdout().flush().unwrap();
        
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

            let text = self.tokenizer.decode(&[next], false)?;
            print!("{}", text);
            std::io::stdout().flush().unwrap();
        }
        println!();
        Ok(())
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
