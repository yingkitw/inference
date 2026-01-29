//! GGUF (quantized model) backend
//!
//! This module provides support for loading GGUF format models,
//! which offer significant memory savings through quantization.
//!
//! Note: Full inference support is under development.
//! Currently this module handles GGUF file detection and metadata parsing.

use crate::error::{InfluenceError, Result};
use crate::local::LocalModelConfig;
use std::path::Path;
use tracing::info;

/// GGUF backend for quantized model inference
#[cfg(feature = "gguf")]
pub struct GgufBackend {
    context_size: usize,
    quantization: String,
    gguf_path: std::path::PathBuf,
}

/// GGUF backend stub (when GGUF feature is not enabled)
#[cfg(not(feature = "gguf"))]
pub struct GgufBackend {
    _private: (),
}

#[cfg(feature = "gguf")]
impl GgufBackend {
    /// Load a GGUF model from the given path
    pub fn load(config: &LocalModelConfig, gguf_path: &Path) -> Result<Self> {
        info!("Loading GGUF model from: {}", gguf_path.display());

        // Detect quantization format from filename
        let quantization = Self::detect_quantization(gguf_path)?;
        info!("Detected quantization: {}", quantization);

        // For now, just store metadata. Full GGUF inference support
        // using llama.cpp will be added in a future update.
        info!("GGUF metadata loaded. Full inference support coming soon.");

        Ok(Self {
            gguf_path: gguf_path.to_path_buf(),
            context_size: config.max_seq_len,
            quantization,
        })
    }

    /// Detect quantization format from GGUF filename
    fn detect_quantization(path: &Path) -> Result<String> {
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| InfluenceError::GgufParsingError("Invalid filename".to_string()))?;

        let filename_lower = filename.to_lowercase();

        let quant = if filename_lower.contains("q2_k") {
            "Q2_K"
        } else if filename_lower.contains("q4_k_m") {
            "Q4_K_M"
        } else if filename_lower.contains("q4_k") {
            "Q4_K"
        } else if filename_lower.contains("q5_k_m") {
            "Q5_K_M"
        } else if filename_lower.contains("q5_k") {
            "Q5_K"
        } else if filename_lower.contains("q6_k") {
            "Q6_K"
        } else if filename_lower.contains("q8_0") {
            "Q8_0"
        } else if filename_lower.contains("f16") {
            "F16"
        } else {
            "Unknown"
        };

        Ok(quant.to_string())
    }

    /// Get the quantization format
    pub fn quantization(&self) -> &str {
        &self.quantization
    }

    /// Get the context size
    pub fn context_size(&self) -> usize {
        self.context_size
    }

    /// Get the GGUF file path
    pub fn path(&self) -> &Path {
        &self.gguf_path
    }
}

#[cfg(not(feature = "gguf"))]
impl GgufBackend {
    /// Load a GGUF model (stub when feature is not enabled)
    pub fn load(_config: &LocalModelConfig, _gguf_path: &Path) -> Result<Self> {
        Err(InfluenceError::InvalidConfig(
            "GGUF support not enabled. Build with --features gguf".to_string()
        ))
    }

    /// Get the quantization format (stub)
    pub fn quantization(&self) -> &str {
        "N/A"
    }

    /// Get the context size (stub)
    pub fn context_size(&self) -> usize {
        0
    }

    /// Get the GGUF file path (stub)
    pub fn path(&self) -> &Path {
        Path::new("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "gguf")]
    fn test_detect_quantization() {
        // Test all supported quantization formats
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model-q2_k.gguf")).unwrap(),
            "Q2_K"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model-q4_k.gguf")).unwrap(),
            "Q4_K"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model-q4_k_m.gguf")).unwrap(),
            "Q4_K_M"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model-q5_k.gguf")).unwrap(),
            "Q5_K"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model-q5_k_m.gguf")).unwrap(),
            "Q5_K_M"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model-q6_k.gguf")).unwrap(),
            "Q6_K"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model-q8_0.gguf")).unwrap(),
            "Q8_0"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model-f16.gguf")).unwrap(),
            "F16"
        );
        // Test case insensitivity
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("MODEL-Q2_K.GGUF")).unwrap(),
            "Q2_K"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("Model-Q4_K_M.GgUF")).unwrap(),
            "Q4_K_M"
        );
    }

    #[test]
    #[cfg(feature = "gguf")]
    fn test_detect_quantization_unknown() {
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model.gguf")).unwrap(),
            "Unknown"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model.bin")).unwrap(),
            "Unknown"
        );
    }

    #[test]
    #[cfg(feature = "gguf")]
    fn test_detect_quantization_invalid_path() {
        assert!(GgufBackend::detect_quantization(Path::new("")).is_err());
        assert!(GgufBackend::detect_quantization(Path::new("/")).is_err());
    }

    #[test]
    #[cfg(feature = "gguf")]
    fn test_gguf_backend_methods() {
        let config = LocalModelConfig {
            max_seq_len: 2048,
            ..Default::default()
        };

        // Note: This will fail to actually load the model since it's not a real GGUF file,
        // but we can test the metadata detection
        let result = GgufBackend::load(&config, Path::new("/fake/path/model-q4_k_m.gguf"));
        assert!(result.is_ok());

        let backend = result.unwrap();
        assert_eq!(backend.quantization(), "Q4_K_M");
        assert_eq!(backend.context_size(), 2048);
        assert_eq!(backend.path(), Path::new("/fake/path/model-q4_k_m.gguf"));
    }

    #[test]
    #[cfg(not(feature = "gguf"))]
    fn test_gguf_disabled() {
        let config = LocalModelConfig::default();
        let result = GgufBackend::load(&config, Path::new("test.gguf"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("GGUF support not enabled"));
    }
}
