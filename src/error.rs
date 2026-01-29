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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_error_display() {
        let err = InfluenceError::DownloadError("Failed to download".to_string());
        assert_eq!(err.to_string(), "Download failed: Failed to download");
    }

    #[test]
    fn test_model_not_found_display() {
        let err = InfluenceError::ModelNotFound("model not found".to_string());
        assert_eq!(err.to_string(), "Model not found: model not found");
    }

    #[test]
    fn test_invalid_config_display() {
        let err = InfluenceError::InvalidConfig("invalid config".to_string());
        assert_eq!(err.to_string(), "Invalid configuration: invalid config");
    }

    #[test]
    fn test_llm_error_display() {
        let err = InfluenceError::LlmError("LLM failed".to_string());
        assert_eq!(err.to_string(), "LLM error: LLM failed");
    }

    #[test]
    fn test_local_model_error_display() {
        let err = InfluenceError::LocalModelError("Model load failed".to_string());
        assert_eq!(err.to_string(), "Local model error: Model load failed");
    }

    #[test]
    fn test_candle_error_display() {
        let err = InfluenceError::CandleError("Candle error".to_string());
        assert_eq!(err.to_string(), "Candle error: Candle error");
    }

    #[test]
    fn test_tokenizer_error_display() {
        let err = InfluenceError::TokenizerError("Tokenizer failed".to_string());
        assert_eq!(err.to_string(), "Tokenizer error: Tokenizer failed");
    }

    #[test]
    fn test_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let inf_err: InfluenceError = io_err.into();
        assert!(matches!(inf_err, InfluenceError::IoError(_)));
        assert!(inf_err.to_string().contains("file not found"));
    }

    #[test]
    fn test_error_from_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json")
            .unwrap_err();
        let inf_err: InfluenceError = json_err.into();
        assert!(matches!(inf_err, InfluenceError::JsonError(_)));
    }

    #[test]
    fn test_result_type_alias() {
        fn returns_ok() -> Result<String> {
            Ok("success".to_string())
        }
        fn returns_err() -> Result<String> {
            Err(InfluenceError::DownloadError("test".to_string()))
        }

        assert!(returns_ok().is_ok());
        assert!(returns_err().is_err());
    }
}
