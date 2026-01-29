use thiserror::Error;

#[derive(Error, Debug)]
pub enum InfluenceError {
    #[error("Download failed: {0}")]
    DownloadError(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("LLM error: {0}")]
    LlmError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Local model error: {0}")]
    LocalModelError(String),

    #[error("Candle error: {0}")]
    CandleError(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
}

impl From<candle_core::Error> for InfluenceError {
    fn from(err: candle_core::Error) -> Self {
        InfluenceError::CandleError(err.to_string())
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for InfluenceError {
    fn from(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        InfluenceError::TokenizerError(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, InfluenceError>;
