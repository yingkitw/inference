use crate::error::{InfluenceError, Result};
use reqwest::Client;
use serde::Deserialize;
use tracing::info;

const DEFAULT_MIRROR: &str = "https://hf-mirror.com";

#[derive(Debug, Deserialize)]
struct ModelInfo {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    modelId: Option<String>,
    #[serde(alias = "author")]  // Handle both "author" and "authorId"
    #[serde(default)]
    author_name: Option<String>,
    #[serde(default)]
    downloads: Option<u64>,
    #[serde(default)]
    likes: Option<u64>,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    library_name: Option<String>,
}

pub async fn search_models(
    query: &str,
    limit: usize,
    author: Option<&str>,
    mirror: Option<&str>,
) -> Result<()> {
    let mirror_url = mirror.unwrap_or(DEFAULT_MIRROR);
    let client = Client::builder()
        .user_agent("influence/0.1.0")
        .build()?;

    info!("Searching for models with query: '{}'", query);

    // Build search URL with query parameters
    let mut search_url = format!(
        "{}/api/models?search={}&limit={}",
        mirror_url,
        urlencoding::encode(query),
        limit
    );

    if let Some(author_filter) = author {
        search_url.push_str(&format!("&author={}", urlencoding::encode(author_filter)));
    }

    let response = client
        .get(&search_url)
        .send()
        .await
        .map_err(|e| InfluenceError::DownloadError(format!("Failed to search models: {}", e)))?;

    if !response.status().is_success() {
        return Err(InfluenceError::DownloadError(format!(
            "HTTP {} when searching models",
            response.status()
        )));
    }

    // HuggingFace API returns an array directly, not wrapped in an object
    let search_result: Vec<ModelInfo> = response
        .json()
        .await
        .map_err(|e| InfluenceError::DownloadError(format!("Failed to parse search results: {}", e)))?;

    if search_result.is_empty() {
        println!("No models found matching '{}'", query);
        return Ok(());
    }

    println!("\nFound {} models:\n", search_result.len());

    for (index, model) in search_result.iter().enumerate() {
        let model_id = model.id.as_ref()
            .or(model.modelId.as_ref())
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        println!("{}. {}", index + 1, model_id);

        if let Some(author) = &model.author_name {
            println!("   Author: {}", author);
        }

        if let Some(pipeline) = &model.pipeline_tag {
            println!("   Task: {}", pipeline);
        }

        if let Some(downloads) = model.downloads {
            println!("   Downloads: {}", downloads);
        }

        if let Some(likes) = model.likes {
            println!("   Likes: {}", likes);
        }

        if let Some(library) = &model.library_name {
            println!("   Library: {}", library);
        }

        // Show download command
        println!("   Download: cargo run -- download --model {}", model_id);
        println!();
    }

    Ok(())
}
