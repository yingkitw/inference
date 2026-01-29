use crate::error::{InfluenceError, Result};
use crate::local::{DevicePreference, LocalModel, LocalModelConfig};
use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
    routing::post,
    Json, Router,
};
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
                let _ = tx.blocking_send(piece);
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

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_headers(Any)
        .allow_methods(Any);

    let app = Router::new()
        .route("/v1/generate", post(generate_handler))
        .route("/v1/generate_stream", post(generate_stream_handler))
        .layer(cors)
        .with_state(state);

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
