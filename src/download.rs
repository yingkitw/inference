use crate::error::{InfluenceError, Result};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt;
use tracing::info;

const DEFAULT_MIRROR: &str = "https://hf-mirror.com";

#[derive(Debug, Deserialize)]
struct RepoFile {
    path: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(rename = "type")]
    file_type: String,
}

#[derive(Debug, Deserialize)]
struct RepoTreeResponse {
    #[serde(default)]
    content: Option<Vec<RepoFile>>,
}

pub async fn download_model(
    model: &str,
    mirror: Option<&str>,
    output: Option<&Path>,
) -> Result<()> {
    let mirror_url = mirror.unwrap_or(DEFAULT_MIRROR);
    let output_dir = get_output_dir(model, output)?;

    info!("Downloading model '{}' from {}", model, mirror_url);
    info!("Output directory: {}", output_dir.display());

    fs::create_dir_all(&output_dir).await?;

    let client = Client::builder()
        .user_agent("influence/0.1.0")
        .build()?;

    // Check if model exists before downloading
    check_model_exists(&client, mirror_url, model).await?;

    // Fetch the list of files to download (with fallback)
    let files_to_download = get_model_files(&client, mirror_url, model).await?;

    for file in files_to_download {
        let url = format!("{}/{}/resolve/main/{}", mirror_url, model, file);
        let file_path = output_dir.join(&file);

        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        info!("Downloading: {}", file);
        download_file(&client, &url, &file_path).await
            .map_err(|e| InfluenceError::DownloadError(format!("Failed to download {}: {}", file, e)))?;
    }

    info!("Model downloaded successfully to: {}", output_dir.display());
    Ok(())
}

fn get_output_dir(model: &str, output: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = output {
        return Ok(path.to_path_buf());
    }

    let dirs = directories::ProjectDirs::from("com", "influence", "influence")
        .ok_or_else(|| InfluenceError::InvalidConfig("Cannot determine home directory".into()))?;

    let model_name = model.replace('/', "_");
    Ok(dirs.data_dir().join("models").join(model_name))
}

async fn check_model_exists(client: &Client, mirror_url: &str, model: &str) -> Result<()> {
    // Try to fetch the model info to check if it exists
    let info_url = format!("{}/api/models/{}", mirror_url, model);

    let response = client
        .head(&info_url)
        .send()
        .await
        .map_err(|e| InfluenceError::DownloadError(format!("Failed to check model availability: {}", e)))?;

    if response.status().is_success() {
        return Ok(());
    }

    // Also try checking if config.json exists (some mirrors don't support the API endpoint)
    let config_url = format!("{}/{}/resolve/main/config.json", mirror_url, model);
    let response = client
        .head(&config_url)
        .send()
        .await
        .map_err(|e| InfluenceError::DownloadError(format!("Failed to check model availability: {}", e)))?;

    if !response.status().is_success() {
        return Err(InfluenceError::DownloadError(format!(
            "Model '{}' not found. Please verify the model name.\n\nHint: Model names should be in the format 'org/model-name' (e.g., 'bert-base-uncased', 'google/flan-t5-small').\nYou can search for models at https://hf-mirror.com or https://huggingface.co/models",
            model
        )));
    }

    Ok(())
}

/// Fetches the list of files for a model from the HuggingFace API
async fn fetch_model_files(client: &Client, mirror_url: &str, model: &str) -> Result<Vec<String>> {
    let tree_url = format!("{}/api/models/{}/tree/main", mirror_url, model);

    info!("Fetching file list from HuggingFace API...");

    let response = client
        .get(&tree_url)
        .send()
        .await
        .map_err(|e| InfluenceError::DownloadError(format!("Failed to fetch model files: {}", e)))?;

    if !response.status().is_success() {
        return Err(InfluenceError::DownloadError(format!(
            "HTTP {} when fetching file list from {}",
            response.status(),
            tree_url
        )));
    }

    let files: Vec<RepoFile> = response
        .json()
        .await
        .map_err(|e| InfluenceError::DownloadError(format!("Failed to parse file list: {}", e)))?;

    // Filter to only files (not directories), excluding hidden files
    let model_files: Vec<String> = files
        .into_iter()
        .filter(|f| f.file_type == "file" && !f.path.starts_with('.'))
        .map(|f| f.path)
        .collect();

    if model_files.is_empty() {
        return Err(InfluenceError::DownloadError(
            "No files found for this model".to_string(),
        ));
    }

    info!("Found {} files to download", model_files.len());
    Ok(model_files)
}

/// Gets the list of files to download, using dynamic discovery with fallback
async fn get_model_files(client: &Client, mirror_url: &str, model: &str) -> Result<Vec<String>> {
    // Try dynamic discovery first
    match fetch_model_files(client, mirror_url, model).await {
        Ok(files) => Ok(files),
        Err(e) => {
            info!("Dynamic file discovery failed: {}", e);
            info!("Falling back to hardcoded file list for this model");

            // Fallback to hardcoded lists
            if model.contains("granite") {
                Ok(vec![
                    "config.json".to_string(),
                    "tokenizer.json".to_string(),
                    "tokenizer_config.json".to_string(),
                    "special_tokens_map.json".to_string(),
                    "model.safetensors".to_string(),
                ])
            } else {
                Ok(vec![
                    "config.json".to_string(),
                    "tokenizer.json".to_string(),
                    "tokenizer_config.json".to_string(),
                    "pytorch_model.bin".to_string(),
                ])
            }
        }
    }
}

async fn download_file(client: &Client, url: &str, path: &Path) -> Result<()> {
    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        return Err(InfluenceError::DownloadError(format!(
            "HTTP {}: {}",
            response.status(),
            url
        )));
    }

    let total_size = response.content_length().unwrap_or(0);

    let pb = if total_size > 0 {
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        Some(pb)
    } else {
        None
    };

    let mut file = File::create(path).await?;
    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        if let Some(ref pb) = pb {
            pb.set_position(downloaded);
        }
    }

    if let Some(pb) = pb {
        pb.finish_with_message("Downloaded");
    }

    file.sync_all().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_get_output_dir() {
        let temp_dir = TempDir::new().unwrap();
        let output = get_output_dir("ibm/granite", Some(temp_dir.path())).unwrap();
        assert_eq!(output, temp_dir.path());
    }
}
