use crate::error::{InfluenceError, Result};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt;
use tracing::{info, warn};

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

    let total_files = files_to_download.len();
    info!("Starting download of {} files...", total_files);

    let overall_progress = ProgressBar::new(total_files as u64);
    overall_progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files ({percent}%)")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut skipped_files: Vec<String> = Vec::new();
    let mut successful_downloads: usize = 0;

    for file in &files_to_download {
        let url = format!("{}/{}/resolve/main/{}", mirror_url, model, file);
        let file_path = output_dir.join(file);

        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        info!("Downloading: {}", file);

        match download_file(&client, &url, &file_path).await {
            Ok(()) => {
                successful_downloads += 1;
            }
            Err(e) => {
                // Check if this is a 403 Forbidden error (gated model)
                if e.to_string().contains("403") {
                    warn!("Skipping {} (access forbidden - this may be a gated model file)", file);
                    skipped_files.push(file.clone());
                } else {
                    // For other errors, still report but continue
                    warn!("Failed to download {}: {}", file, e);
                    skipped_files.push(file.clone());
                }
            }
        }

        overall_progress.inc(1);
    }

    overall_progress.finish_with_message(format!("Download complete: {}/{} files", successful_downloads, total_files));

    if !skipped_files.is_empty() {
        warn!("Skipped files: {}", skipped_files.join(", "));
        warn!("Some files could not be downloaded. This model may be gated or require authentication.");
        warn!("Visit https://huggingface.co/{}/request-access to request access if needed.", model);
    }

    if successful_downloads == 0 {
        return Err(InfluenceError::DownloadError(
            format!("No files could be downloaded. The model '{}' may be gated or require authentication. Visit https://huggingface.co/{}/request-access to request access.", model, model)
        ));
    }

    info!("Model downloaded successfully to: {}", output_dir.display());
    Ok(())
}

fn get_output_dir(model: &str, output: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = output {
        return Ok(path.to_path_buf());
    }

    // Use a local working folder "models" instead of system directory
    let model_name = model.replace('/', "_");
    Ok(PathBuf::from("models").join(model_name))
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
    let file_name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file");

    let pb = if total_size > 0 {
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})\n   {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message(format!("Downloading: {}", file_name));
        Some(pb)
    } else {
        // For files without known size, show a spinner
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap()
        );
        pb.set_message(format!("Downloading: {} (size unknown)", file_name));
        Some(pb)
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

            // Update message periodically with progress percentage
            if total_size > 0 {
                let percent = (downloaded as f64 / total_size as f64 * 100.0) as u64;
                pb.set_message(format!("{}: {}%", file_name, percent));
            }
        }
    }

    if let Some(pb) = pb {
        pb.finish_with_message(format!("✓ {}", file_name));
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
