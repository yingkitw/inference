use crate::error::{InfluenceError, Result};
use crate::local::LocalModelConfig;
use std::fs;
use std::path::Path;
use tracing::{info, warn};
use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Llama, Config as LlamaConfig, Cache};
use candle_transformers::models::mistral::{Config as MistralConfig, Model as MistralModel};

/// Local model backend enum supporting multiple architectures
pub enum LocalBackend {
    Llama { model: Llama, cache: Cache },
    Mistral { model: MistralModel },
}

impl LocalBackend {
    /// Load a Llama backend from model weights
    pub fn load_llama(config: &LocalModelConfig, device: &Device) -> Result<Option<Self>> {
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

        let weight_files = find_weight_files(&config.model_path)?;
        if weight_files.is_empty() {
            warn!("No .safetensors files found");
            return Ok(None);
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
        Ok(Some(LocalBackend::Llama { model, cache }))
    }

    /// Load a Mistral backend from model weights
    pub fn load_mistral(config: &LocalModelConfig, device: &Device) -> Result<Option<Self>> {
        info!("Loading Mistral model weights...");

        let config_path = config.model_path.join("config.json");
        if !config_path.exists() {
            return Err(InfluenceError::LocalModelError("config.json not found".to_string()));
        }

        let config_content = fs::read_to_string(&config_path)?;
        let mistral_cfg: MistralConfig = serde_json::from_str(&config_content)
            .map_err(|e| InfluenceError::LocalModelError(format!("Failed to parse config: {}", e)))?;

        let weight_files = find_weight_files(&config.model_path)?;
        if weight_files.is_empty() {
            warn!("No .safetensors files found");
            return Ok(None);
        }

        info!("Loading {} weight file(s)...", weight_files.len());
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, device)
                .map_err(|e| InfluenceError::LocalModelError(format!("Failed to load weights: {}", e)))?
        };

        let model = MistralModel::new(&mistral_cfg, vb)
            .map_err(|e| InfluenceError::LocalModelError(format!("Failed to create model: {}", e)))?;

        info!("Model initialized");
        Ok(Some(LocalBackend::Mistral { model }))
    }
}

/// Find all .safetensors files in the model directory
fn find_weight_files(model_path: &Path) -> Result<Vec<std::path::PathBuf>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_weight_files_empty() {
        let result = find_weight_files(Path::new("/nonexistent/path"));
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
