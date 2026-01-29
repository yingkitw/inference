use crate::error::{InfluenceError, Result};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use std::path::{Path, PathBuf};
use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt;
use tracing::info;

const DEFAULT_MIRROR: &str = "https://hf-mirror.com";

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

    let files_to_download = get_model_files(model);

    for file in files_to_download {
        let url = format!("{}/{}/resolve/main/{}", mirror_url, model, file);
        let file_path = output_dir.join(file);

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

fn get_model_files(model: &str) -> Vec<&'static str> {
    if model.contains("granite") {
        vec![
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors",
        ]
    } else {
        vec![
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "pytorch_model.bin",
        ]
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

    #[test]
    fn test_get_model_files() {
        let granite_files = get_model_files("ibm/granite-4-h-small");
        assert!(granite_files.contains(&"config.json"));
        assert!(granite_files.contains(&"model.safetensors"));

        let other_files = get_model_files("other/model");
        assert!(other_files.contains(&"config.json"));
        assert!(other_files.contains(&"pytorch_model.bin"));
    }

    #[tokio::test]
    async fn test_get_output_dir() {
        let temp_dir = TempDir::new().unwrap();
        let output = get_output_dir("ibm/granite", Some(temp_dir.path())).unwrap();
        assert_eq!(output, temp_dir.path());
    }
}
