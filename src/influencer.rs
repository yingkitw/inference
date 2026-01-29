use crate::error::{InfluenceError, Result};
use crate::local::{LocalModel, LocalModelConfig};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{info, error, debug};
use reqwest::Client;
use futures::StreamExt;

const WATSONX_URL: &str = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation_stream";

#[derive(Debug, Serialize, Deserialize)]
pub struct InfluencerConfig {
    pub api_key: String,
    pub project_id: String,
    pub model_id: String,
    pub url: Option<String>,
}

#[derive(Debug, Serialize)]
struct GenerateRequest {
    model_id: String,
    input: String,
    parameters: Parameters,
    project_id: String,
}

#[derive(Debug, Serialize)]
struct Parameters {
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: i32,
}

#[derive(Debug, Deserialize)]
struct GenerateResponse {
    results: Vec<GenerateResult>,
}

#[derive(Debug, Deserialize)]
struct GenerateResult {
    generated_text: String,
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    results: Option<Vec<StreamResult>>,
}

#[derive(Debug, Deserialize)]
struct StreamResult {
    generated_text: Option<String>,
}

pub trait LlmService {
    async fn generate_text(&self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String>;
    async fn generate_stream(&self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<()>;
}

pub struct WatsonXService {
    client: Client,
    config: InfluencerConfig,
}

impl WatsonXService {
    pub fn new(config: InfluencerConfig) -> Result<Self> {
        let client = Client::builder()
            .user_agent("influence/0.1.0")
            .build()
            .map_err(|e| InfluenceError::LlmError(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            config,
        })
    }

    fn format_granite_prompt(&self, prompt: &str) -> String {
        format!(
            "<|start_of_role|>user<|end_of_role|>{}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
            prompt
        )
    }

    async fn get_access_token(&self) -> Result<String> {
        Ok(self.config.api_key.clone())
    }
}

impl LlmService for WatsonXService {
    async fn generate_text(&self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String> {
        let formatted_prompt = if self.config.model_id.contains("granite") {
            self.format_granite_prompt(prompt)
        } else {
            prompt.to_string()
        };

        let request = GenerateRequest {
            model_id: self.config.model_id.clone(),
            input: formatted_prompt,
            parameters: Parameters {
                max_new_tokens: max_tokens,
                temperature,
                top_p: 0.9,
                top_k: 50,
            },
            project_id: self.config.project_id.clone(),
        };

        let token = self.get_access_token().await?;
        let url = self.config.url.as_deref().unwrap_or(WATSONX_URL).replace("_stream", "");

        debug!("Sending request to: {}", url);

        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| InfluenceError::LlmError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(InfluenceError::LlmError(format!(
                "API error {}: {}",
                status, text
            )));
        }

        let result: GenerateResponse = response
            .json()
            .await
            .map_err(|e| InfluenceError::LlmError(format!("Failed to parse response: {}", e)))?;

        result.results
            .first()
            .map(|r| r.generated_text.clone())
            .ok_or_else(|| InfluenceError::LlmError("No text generated".into()))
    }

    async fn generate_stream(&self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<()> {
        let formatted_prompt = if self.config.model_id.contains("granite") {
            self.format_granite_prompt(prompt)
        } else {
            prompt.to_string()
        };

        let request = GenerateRequest {
            model_id: self.config.model_id.clone(),
            input: formatted_prompt,
            parameters: Parameters {
                max_new_tokens: max_tokens,
                temperature,
                top_p: 0.9,
                top_k: 50,
            },
            project_id: self.config.project_id.clone(),
        };

        let token = self.get_access_token().await?;
        let url = self.config.url.as_deref().unwrap_or(WATSONX_URL);

        debug!("Sending streaming request to: {}", url);

        let response = self.client
            .post(url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| InfluenceError::LlmError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(InfluenceError::LlmError(format!(
                "API error {}: {}",
                status, text
            )));
        }

        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    for line in text.lines() {
                        if line.starts_with("data: ") {
                            let json_str = &line[6..];
                            if json_str.trim() == "[DONE]" {
                                break;
                            }
                            if let Ok(chunk) = serde_json::from_str::<StreamChunk>(json_str) {
                                if let Some(results) = chunk.results {
                                    for result in results {
                                        if let Some(text) = result.generated_text {
                                            print!("{}", text);
                                            use std::io::Write;
                                            std::io::stdout().flush().unwrap();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Stream error: {}", e);
                    return Err(InfluenceError::LlmError(format!("Stream error: {}", e)));
                }
            }
        }
        println!();

        Ok(())
    }
}

fn load_config(_model_path: Option<&Path>) -> Result<InfluencerConfig> {
    let api_key = std::env::var("WATSONX_API_KEY")
        .map_err(|_| InfluenceError::InvalidConfig("WATSONX_API_KEY not set".into()))?;
    
    let project_id = std::env::var("WATSONX_PROJECT_ID")
        .map_err(|_| InfluenceError::InvalidConfig("WATSONX_PROJECT_ID not set".into()))?;

    let model_id = std::env::var("WATSONX_MODEL_ID")
        .unwrap_or_else(|_| "ibm/granite-4-h-small".to_string());

    Ok(InfluencerConfig {
        api_key,
        project_id,
        model_id,
        url: None,
    })
}

pub async fn serve(model_path: Option<&Path>, port: u16) -> Result<()> {
    info!("Starting influencer service on port {}", port);
    
    let config = load_config(model_path)?;
    let service = WatsonXService::new(config)?;

    info!("Service initialized with model: {}", service.config.model_id);
    info!("Service is ready. Use the 'generate' command to test it.");

    Ok(())
}

pub async fn generate(
    prompt: &str,
    model_path: Option<&Path>,
    max_tokens: usize,
    temperature: f32,
) -> Result<()> {
    info!("Generating response for prompt: {}", prompt);

    // Check if local model path is provided
    if let Some(path) = model_path {
        info!("Using local model from: {}", path.display());

        let config = LocalModelConfig {
            model_path: path.to_path_buf(),
            temperature,
            max_seq_len: max_tokens * 2, // Give some room for the prompt
            ..Default::default()
        };

        let mut local_model = LocalModel::load(config).await?;
        println!("\n--- Local Generation ---");
        local_model.generate_stream(prompt, max_tokens, temperature).await?;
        println!("\n--- End ---\n");
    } else {
        // Use WatsonX cloud service
        let config = load_config(None)?;
        let service = WatsonXService::new(config)?;

        info!("Using WatsonX cloud model: {}", service.config.model_id);
        println!("\n--- Response ---");

        service.generate_stream(prompt, max_tokens, temperature).await?;

        println!("\n--- End ---\n");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_granite_prompt() {
        let config = InfluencerConfig {
            api_key: "test".to_string(),
            project_id: "test".to_string(),
            model_id: "ibm/granite-4-h-small".to_string(),
            url: None,
        };
        
        let service = WatsonXService::new(config).unwrap();
        let formatted = service.format_granite_prompt("Hello");
        
        assert!(formatted.contains("<|start_of_role|>user<|end_of_role|>"));
        assert!(formatted.contains("Hello"));
        assert!(formatted.contains("<|start_of_role|>assistant<|end_of_role|>"));
    }

    #[test]
    fn test_load_config_missing_env() {
        unsafe {
            std::env::remove_var("WATSONX_API_KEY");
            std::env::remove_var("WATSONX_PROJECT_ID");
        }
        
        let result = load_config(None);
        assert!(result.is_err());
    }
}
