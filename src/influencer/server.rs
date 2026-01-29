use crate::error::{InfluenceError, Result};
use crate::local::{DevicePreference, LocalModel, LocalModelConfig};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    response::sse::{Event, Sse},
    routing::post,
    Json, Router,
};
use axum::body::Body;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::{
    convert::Infallible,
    net::SocketAddr,
    path::Path,
    sync::Arc,
    time::Duration,
};
use tokio::sync::{mpsc, Mutex};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info};

#[derive(Clone)]
pub struct AppState {
    model: Arc<Mutex<LocalModel>>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct OllamaOptions {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub num_predict: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaGenerateRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaEmbeddingsRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,
}

#[derive(Debug, Serialize)]
pub struct OllamaGenerateResponse {
    pub model: String,
    pub response: String,
    pub done: bool,
    pub created_at: String,
}

#[derive(Debug, Serialize)]
pub struct OllamaEmbeddingsResponse {
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct OllamaTagsResponse {
    pub models: Vec<OllamaTagModel>,
}

#[derive(Debug, Serialize)]
pub struct OllamaTagModel {
    pub name: String,
    pub model: String,
    pub modified_at: String,
}

fn build_effective_prompt(prompt: &str, system: Option<&str>) -> String {
    match system {
        Some(system_prompt) if !system_prompt.trim().is_empty() => format!(
            "System: {}\n\nUser: {}\n\nAssistant:",
            system_prompt.trim(),
            prompt
        ),
        _ => prompt.to_string(),
    }
}

fn apply_config_overrides(mut base: LocalModelConfig, req: &GenerateRequest) -> LocalModelConfig {
    if let Some(t) = req.temperature {
        base.temperature = t;
    }
    if let Some(p) = req.top_p {
        base.top_p = p;
    }
    if let Some(k) = req.top_k {
        base.top_k = Some(k);
    }
    if let Some(rp) = req.repeat_penalty {
        base.repeat_penalty = rp;
    }
    if let Some(mt) = req.max_tokens {
        base.max_seq_len = mt.saturating_mul(2);
    }
    base
}

fn now_millis_string() -> String {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => d.as_millis().to_string(),
        Err(_) => "0".to_string(),
    }
}

fn apply_ollama_options(mut base: LocalModelConfig, opts: Option<&OllamaOptions>) -> (LocalModelConfig, usize, f32) {
    let mut max_tokens = 512usize;
    let mut temperature = base.temperature;

    if let Some(opts) = opts {
        if let Some(t) = opts.temperature {
            base.temperature = t;
            temperature = t;
        }
        if let Some(p) = opts.top_p {
            base.top_p = p;
        }
        if let Some(k) = opts.top_k {
            base.top_k = Some(k);
        }
        if let Some(rp) = opts.repeat_penalty {
            base.repeat_penalty = rp;
        }
        if let Some(np) = opts.num_predict {
            max_tokens = np;
            base.max_seq_len = np.saturating_mul(2);
        }
    }

    (base, max_tokens, temperature)
}

async fn generate_handler(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> std::result::Result<Json<GenerateResponse>, (StatusCode, String)> {
    let effective_prompt = build_effective_prompt(&req.prompt, req.system.as_deref());

    let mut model = state.model.lock().await;

    let cfg = apply_config_overrides(model.config().clone(), &req);
    *model.config_mut() = cfg;

    let max_tokens = req.max_tokens.unwrap_or(512);
    let temperature = req.temperature.unwrap_or(model.config().temperature);

    model
        .generate_text(&effective_prompt, max_tokens, temperature)
        .await
        .map(|text| Json(GenerateResponse { text }))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

async fn generate_stream_handler(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> std::result::Result<Sse<impl tokio_stream::Stream<Item = std::result::Result<Event, Infallible>>>, (StatusCode, String)> {
    let effective_prompt = build_effective_prompt(&req.prompt, req.system.as_deref());

    let (tx, rx) = mpsc::channel::<String>(64);

    let max_tokens = req.max_tokens.unwrap_or(512);

    let state_clone = state.clone();
    tokio::spawn(async move {
        let mut model = state_clone.model.lock().await;

        let cfg = apply_config_overrides(model.config().clone(), &req);
        *model.config_mut() = cfg;

        let temperature = req.temperature.unwrap_or(model.config().temperature);

        let res = model
            .generate_stream_with(&effective_prompt, max_tokens, temperature, |piece| {
                let _ = tx.try_send(piece);
                Ok(())
            })
            .await;

        if let Err(e) = res {
            error!("stream generation failed: {}", e);
        }
    });

    let stream = ReceiverStream::new(rx).map(|chunk| Ok(Event::default().event("token").data(chunk)));

    Ok(Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new().interval(Duration::from_secs(10)).text("keep-alive"),
    ))
}

async fn ollama_generate_handler(
    State(state): State<AppState>,
    Json(req): Json<OllamaGenerateRequest>,
) -> std::result::Result<Response, (StatusCode, String)> {
    let model_name = req.model.clone().unwrap_or_else(|| "influence".to_string());
    let stream = req.stream.unwrap_or(false);
    let effective_prompt = build_effective_prompt(&req.prompt, req.system.as_deref());

    if !stream {
        let mut model = state.model.lock().await;
        let (cfg, max_tokens, temperature) = apply_ollama_options(model.config().clone(), req.options.as_ref());
        *model.config_mut() = cfg;

        let text = model
            .generate_text(&effective_prompt, max_tokens, temperature)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        let resp = OllamaGenerateResponse {
            model: model_name,
            response: text,
            done: true,
            created_at: now_millis_string(),
        };
        return Ok(Json(resp).into_response());
    }

    let (tx, rx) = mpsc::channel::<Bytes>(64);
    let state_clone = state.clone();
    tokio::spawn(async move {
        let mut model = state_clone.model.lock().await;
        let (cfg, max_tokens, temperature) = apply_ollama_options(model.config().clone(), req.options.as_ref());
        *model.config_mut() = cfg;

        let send_line = |obj: &OllamaGenerateResponse| {
            if let Ok(line) = serde_json::to_string(obj) {
                let _ = tx.try_send(Bytes::from(format!("{}\n", line)));
            }
        };

        let res = model
            .generate_stream_with(&effective_prompt, max_tokens, temperature, |piece| {
                let obj = OllamaGenerateResponse {
                    model: model_name.clone(),
                    response: piece,
                    done: false,
                    created_at: now_millis_string(),
                };
                send_line(&obj);
                Ok(())
            })
            .await;

        if let Err(e) = res {
            error!("ollama stream generation failed: {}", e);
        }

        let done_obj = OllamaGenerateResponse {
            model: model_name,
            response: String::new(),
            done: true,
            created_at: now_millis_string(),
        };
        send_line(&done_obj);
    });

    let stream = ReceiverStream::new(rx).map(Ok::<Bytes, Infallible>);
    let mut resp = Response::new(Body::from_stream(stream));
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/x-ndjson"),
    );
    Ok(resp)
}

async fn ollama_embeddings_handler(
    State(state): State<AppState>,
    Json(req): Json<OllamaEmbeddingsRequest>,
) -> std::result::Result<Json<OllamaEmbeddingsResponse>, (StatusCode, String)> {
    let mut model = state.model.lock().await;
    let emb = model
        .embed_text(&req.prompt)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(OllamaEmbeddingsResponse { embedding: emb }))
}

async fn ollama_tags_handler(
    State(state): State<AppState>,
) -> std::result::Result<Json<OllamaTagsResponse>, (StatusCode, String)> {
    let model = state.model.lock().await;
    let name = model
        .config()
        .model_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "influence".to_string());
    let resp = OllamaTagsResponse {
        models: vec![OllamaTagModel {
            name: name.clone(),
            model: name,
            modified_at: now_millis_string(),
        }],
    };
    Ok(Json(resp))
}

pub fn build_app(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_headers(Any)
        .allow_methods(Any);

    Router::new()
        .route("/v1/generate", post(generate_handler))
        .route("/v1/generate_stream", post(generate_stream_handler))
        .route("/api/generate", post(ollama_generate_handler))
        .route("/api/embeddings", post(ollama_embeddings_handler))
        .route("/api/tags", post(ollama_tags_handler))
        .layer(cors)
        .with_state(state)
}

pub async fn serve(model_path: Option<&Path>, port: u16, device: &str, device_index: usize) -> Result<()> {
    let path = model_path.ok_or_else(|| {
        InfluenceError::InvalidConfig(
            "Model path is required for serving. Use --model-path <path> to specify a local model directory.".to_string(),
        )
    })?;

    let device_preference: DevicePreference = device.parse()?;

    let config = LocalModelConfig {
        model_path: path.to_path_buf(),
        device_preference,
        device_index,
        ..Default::default()
    };

    let model = LocalModel::load(config).await?;

    let state = AppState {
        model: Arc::new(Mutex::new(model)),
    };

    let app = build_app(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("Serving web API on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| InfluenceError::IoError(e))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| InfluenceError::LocalModelError(e.to_string()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{Request, header};
    use serde_json::json;
    use tempfile::tempdir;
    use tower::ServiceExt;

    fn write_minimal_llama_config(dir: &std::path::Path) {
        // Minimal config for architecture detection + llama loader parsing.
        // No weights are provided in tests (placeholder mode).
        let cfg = json!({
            "model_type": "llama",
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0
        });
        std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&cfg).unwrap()).unwrap();
    }

    fn write_minimal_tokenizer(dir: &std::path::Path) {
        // Write a minimal tokenizer.json directly. We only need LocalModel::load
        // to successfully parse it during tests; generation is expected to fail
        // due to missing weights.
        let tokenizer = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": { "<unk>": 0, "hello": 1, "world": 2 },
    "unk_token": "<unk>"
  }
}"#;
        std::fs::write(dir.join("tokenizer.json"), tokenizer.as_bytes()).unwrap();
    }

    async fn build_test_app() -> Router {
        let dir = tempdir().unwrap();
        write_minimal_llama_config(dir.path());
        write_minimal_tokenizer(dir.path());

        let config = LocalModelConfig {
            model_path: dir.path().to_path_buf(),
            device_preference: DevicePreference::Cpu,
            device_index: 0,
            ..Default::default()
        };

        let model = LocalModel::load(config).await.unwrap();
        let state = AppState {
            model: Arc::new(Mutex::new(model)),
        };

        build_app(state)
    }

    #[tokio::test]
    async fn test_ollama_tags_ok() {
        let app = build_test_app().await;

        let req = Request::builder()
            .method("POST")
            .uri("/api/tags")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_ollama_generate_non_stream_returns_error_without_weights() {
        let app = build_test_app().await;

        let body = serde_json::to_vec(&json!({
            "prompt": "hello",
            "stream": false
        }))
        .unwrap();

        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_ollama_generate_stream_returns_ndjson_and_done() {
        let app = build_test_app().await;

        let body = serde_json::to_vec(&json!({
            "prompt": "hello",
            "stream": true
        }))
        .unwrap();

        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let content_type = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert_eq!(content_type, "application/x-ndjson");

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.contains("\"done\":true"));
    }
}
