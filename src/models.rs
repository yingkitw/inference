//! Model management - listing, deploying, and managing local models

use crate::error::{InfluenceError, Result};
use std::fs;
use std::path::{Path, PathBuf};
use serde_json::Value;
use tracing::info;

/// Information about a local model
#[derive(Debug, Clone)]
pub struct LocalModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub architecture: String,
    pub format: ModelFormat,
    pub size_bytes: u64,
    pub file_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelFormat {
    SafeTensors,
    GGUF { quantization: String },
    Unknown,
}

/// List all models in the models directory
pub fn list_models(models_dir: Option<&Path>) -> Result<Vec<LocalModelInfo>> {
    let search_path = models_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./models"));

    info!("Listing models in: {}", search_path.display());

    if !search_path.exists() {
        return Ok(vec![]);
    }

    let mut models = Vec::new();

    let entries = fs::read_dir(&search_path)
        .map_err(|e| InfluenceError::ModelNotFound(format!("Failed to read models directory: {}", e)))?;

    for entry in entries.flatten() {
        let path = entry.path();

        // Skip if not a directory
        if !path.is_dir() {
            continue;
        }

        // Check for model files
        let model_info = analyze_model_directory(&path)?;
        if let Some(info) = model_info {
            models.push(info);
        }
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(models)
}

/// Analyze a model directory to determine its format and metadata
fn analyze_model_directory(path: &Path) -> Result<Option<LocalModelInfo>> {
    let name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("Unknown")
        .to_string();

    let mut size_bytes = 0u64;
    let mut file_count = 0usize;
    let mut has_safetensors = false;
    let mut has_gguf = false;
    let mut gguf_quantization = None;
    let mut architecture = String::from("Unknown");

    // Read directory contents
    let entries = fs::read_dir(path)
        .map_err(|e| InfluenceError::LocalModelError(format!("Failed to read directory: {}", e)))?;

    for entry in entries.flatten() {
        let file_path = entry.path();

        // Skip directories
        if file_path.is_dir() {
            continue;
        }

        // Get metadata
        if let Ok(metadata) = file_path.metadata() {
            size_bytes += metadata.len();
            file_count += 1;
        }

        // Check for GGUF files
        if let Some(ext) = file_path.extension() {
            if ext == "gguf" {
                has_gguf = true;
                // Try to detect quantization from filename
                if let Some(filename) = file_path.file_name().and_then(|n| n.to_str()) {
                    let filename_lower = filename.to_lowercase();
                    if filename_lower.contains("q2_k") {
                        gguf_quantization = Some("Q2_K");
                    } else if filename_lower.contains("q4_k_m") {
                        gguf_quantization = Some("Q4_K_M");
                    } else if filename_lower.contains("q4_k") {
                        gguf_quantization = Some("Q4_K");
                    } else if filename_lower.contains("q5_k_m") {
                        gguf_quantization = Some("Q5_K_M");
                    } else if filename_lower.contains("q5_k") {
                        gguf_quantization = Some("Q5_K");
                    } else if filename_lower.contains("q6_k") {
                        gguf_quantization = Some("Q6_K");
                    } else if filename_lower.contains("q8_0") {
                        gguf_quantization = Some("Q8_0");
                    } else if filename_lower.contains("f16") {
                        gguf_quantization = Some("F16");
                    }
                }
            } else if ext == "safetensors" {
                has_safetensors = true;
            }
        }

        // Try to read architecture from config.json
        if file_path.file_name() == Some(std::ffi::OsStr::new("config.json")) {
            if let Ok(content) = fs::read_to_string(&file_path) {
                if let Ok(config) = serde_json::from_str::<Value>(&content) {
                    if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
                        architecture = model_type.to_string();
                    }
                }
            }
        }
    }

    // Determine model format
    let format = if has_gguf {
        ModelFormat::GGUF {
            quantization: gguf_quantization.unwrap_or("Unknown").to_string()
        }
    } else if has_safetensors {
        ModelFormat::SafeTensors
    } else {
        ModelFormat::Unknown
    };

    // Only include if it has model files
    if matches!(format, ModelFormat::Unknown) {
        return Ok(None);
    }

    Ok(Some(LocalModelInfo {
        name,
        path: path.to_path_buf(),
        architecture,
        format,
        size_bytes,
        file_count,
    }))
}

/// Display model information in a formatted table
pub fn display_models(models: &[LocalModelInfo], formatter: &crate::output::OutputFormatter) {
    if models.is_empty() {
        formatter.print_warning("No models found in the models directory.");
        formatter.print_markdown("\n**To download a model:**\n\n```bash\ninfluence download -m <model-name>\n```\n\n**Example:**\n\n```bash\ninfluence download -m TinyLlama/TinyLlama-1.1B-Chat-v1.0\n```\n");
        return;
    }

    formatter.print_header("Local Models");

    for model in models.iter() {
        let format_str = match &model.format {
            ModelFormat::SafeTensors => "SafeTensors".to_string(),
            ModelFormat::GGUF { quantization } => format!("GGUF ({})", quantization),
            ModelFormat::Unknown => "Unknown".to_string(),
        };

        let size_mb = model.size_bytes / (1024 * 1024);
        let size_gb = size_mb / 1024;

        let size_str = if size_gb > 0 {
            format!("{} GB", size_gb)
        } else {
            format!("{} MB", size_mb)
        };

        formatter.print_model_info(
            &model.name,
            &model.path.display().to_string(),
            &format_str,
            &model.architecture,
            &size_str,
            model.file_count,
        );
    }

    formatter.print_info(&format!("Total: {} model(s)", models.len()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_list_models_empty_directory() {
        let tmp = TempDir::new().unwrap();
        let models = list_models(Some(tmp.path())).unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn test_list_models_nonexistent_directory() {
        let models = list_models(Some(Path::new("/nonexistent/path"))).unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn test_analyze_model_directory_with_safetensors() {
        let tmp = TempDir::new().unwrap();
        let model_dir = tmp.path().join("test-model");
        fs::create_dir(&model_dir).unwrap();

        // Create fake safetensors file
        fs::write(model_dir.join("model.safetensors"), b"fake content").unwrap();

        // Create config.json
        fs::write(
            model_dir.join("config.json"),
            r#"{"model_type":"llama"}"#,
        ).unwrap();

        let info = analyze_model_directory(&model_dir).unwrap();
        assert!(info.is_some());

        let model_info = info.unwrap();
        assert_eq!(model_info.name, "test-model");
        assert!(matches!(model_info.format, ModelFormat::SafeTensors));
        assert_eq!(model_info.architecture, "llama");
    }

    #[test]
    fn test_analyze_model_directory_with_gguf() {
        let tmp = TempDir::new().unwrap();
        let model_dir = tmp.path().join("test-model");
        fs::create_dir(&model_dir).unwrap();

        // Create fake GGUF file
        fs::write(model_dir.join("model-q4_k_m.gguf"), b"fake content").unwrap();

        let info = analyze_model_directory(&model_dir).unwrap();
        assert!(info.is_some());

        let model_info = info.unwrap();
        assert_eq!(model_info.name, "test-model");
        assert!(matches!(model_info.format, ModelFormat::GGUF { .. }));

        if let ModelFormat::GGUF { quantization } = model_info.format {
            assert_eq!(quantization, "Q4_K_M");
        }
    }
}
