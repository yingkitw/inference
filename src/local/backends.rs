use crate::error::{InfluenceError, Result};
use crate::local::LocalModelConfig;
use std::fs;
use std::path::Path;
use std::time::Instant;
use tracing::{info, warn};
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Llama, Config as LlamaConfig, Cache};
use candle_transformers::models::mistral::{Config as MistralConfig, Model as MistralModel};
use candle_transformers::models::mamba::{Config as MambaConfig, Model as MambaModel};
use candle_transformers::models::granitemoehybrid::{
    GraniteMoeHybrid, GraniteMoeHybridConfig, GraniteMoeHybridInternalConfig,
};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};

/// Local model backend enum supporting multiple architectures
pub enum LocalBackend {
    Llama { model: Llama, config: LlamaConfig },
    Mistral { model: MistralModel },
    Mamba { model: MambaModel, config: MambaConfig },
    GraniteMoeHybrid { model: GraniteMoeHybrid, config: GraniteMoeHybridInternalConfig },
    Bert { model: BertModel },
    #[cfg(feature = "gguf")]
    Gguf { backend: super::gguf_backend::GgufBackend },
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

        // On Metal, the first few decode steps can be much slower due to kernel compilation.
        // Warm up a few single-token forward passes with an increasing position to reduce
        // visible latency for the first generated words.
        let warmup_tokens: usize = std::env::var("INFLUENCE_WARMUP_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(6);

        if warmup_tokens > 0 {
            if matches!(device, Device::Metal(_)) {
                info!("Metal warmup: running {} decode step(s)...", warmup_tokens);
                let t_warm = Instant::now();
                let mut warm_cache = Cache::new(true, DType::F32, &llama_config, device)
                    .map_err(|e| InfluenceError::LocalModelError(format!("Failed to create warmup cache: {}", e)))?;
                let warm_token: u32 = 0;
                for pos in 0..warmup_tokens {
                    let tensor = Tensor::new(&[warm_token], device)?
                        .unsqueeze(0)?;
                    let _ = model.forward(&tensor, pos, &mut warm_cache)?;
                }
                info!("Metal warmup: done in {} ms", t_warm.elapsed().as_millis());
            }
        }

        info!("Model initialized");
        Ok(Some(LocalBackend::Llama { model, config: llama_config }))
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

    pub fn load_mamba(config: &LocalModelConfig, device: &Device) -> Result<Option<Self>> {
        info!("Loading Mamba model weights...");

        let config_path = config.model_path.join("config.json");
        if !config_path.exists() {
            return Err(InfluenceError::LocalModelError("config.json not found".to_string()));
        }

        let config_content = fs::read_to_string(&config_path)?;
        let mamba_cfg: MambaConfig = serde_json::from_str(&config_content)
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

        let model = MambaModel::new(&mamba_cfg, vb)
            .map_err(|e| InfluenceError::LocalModelError(format!("Failed to create model: {}", e)))?;

        info!("Model initialized");
        Ok(Some(LocalBackend::Mamba { model, config: mamba_cfg }))
    }

    pub fn load_granite_moe_hybrid(config: &LocalModelConfig, device: &Device) -> Result<Option<Self>> {
        info!("Loading GraniteMoeHybrid model weights...");

        let config_path = config.model_path.join("config.json");
        if !config_path.exists() {
            return Err(InfluenceError::LocalModelError("config.json not found".to_string()));
        }

        let config_content = fs::read_to_string(&config_path)?;
        let cfg: GraniteMoeHybridConfig = serde_json::from_str(&config_content)
            .map_err(|e| InfluenceError::LocalModelError(format!("Failed to parse config: {}", e)))?;
        let internal_cfg = cfg.into_config(false);

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

        let model = GraniteMoeHybrid::load(vb, &internal_cfg)
            .map_err(|e| InfluenceError::LocalModelError(format!("Failed to create model: {}", e)))?;

        info!("Model initialized");
        Ok(Some(LocalBackend::GraniteMoeHybrid { model, config: internal_cfg }))
    }

    pub fn load_bert(config: &LocalModelConfig, device: &Device) -> Result<Option<Self>> {
        info!("Loading BERT-family model weights...");

        let config_path = config.model_path.join("config.json");
        if !config_path.exists() {
            return Err(InfluenceError::LocalModelError("config.json not found".to_string()));
        }

        let config_content = fs::read_to_string(&config_path)?;
        let bert_cfg: BertConfig = serde_json::from_str(&config_content)
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

        let model = BertModel::load(vb, &bert_cfg)
            .map_err(|e| InfluenceError::LocalModelError(format!("Failed to create model: {}", e)))?;

        info!("Model initialized");
        Ok(Some(LocalBackend::Bert { model }))
    }

    /// Load a GGUF backend from a GGUF file
    #[cfg(feature = "gguf")]
    pub fn load_gguf(config: &LocalModelConfig, device: &Device) -> Result<Option<Self>> {
        use std::fs;
        info!("Loading GGUF model...");

        // Find GGUF files in the model directory
        let gguf_files: Vec<_> = fs::read_dir(&config.model_path)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.path()
                    .extension()
                    .map_or(false, |ext| ext == "gguf")
            })
            .collect();

        if gguf_files.is_empty() {
            return Ok(None);
        }

        if gguf_files.len() > 1 {
            warn!("Multiple GGUF files found, using first: {}",
                  gguf_files[0].path().display());
        }

        let gguf_path = &gguf_files[0].path();
        info!("Found GGUF file: {}", gguf_path.display());

        let backend = super::gguf_backend::GgufBackend::load(config, gguf_path)?;

        info!("GGUF model loaded successfully (quantization: {})",
              backend.quantization());
        Ok(Some(LocalBackend::Gguf { backend }))
    }

    /// Load a GGUF backend (stub when feature is not enabled)
    #[cfg(not(feature = "gguf"))]
    pub fn load_gguf(_config: &LocalModelConfig, _device: &Device) -> Result<Option<Self>> {
        Ok(None)
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
