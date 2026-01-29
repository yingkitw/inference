# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Influence** is a privacy-first, local LLM inference CLI tool written in Rust. It downloads models from HuggingFace and runs them entirely on the local machine with optional GPU acceleration (Metal on macOS, CUDA on Linux/Windows).

## Common Commands

### Building

```bash
# Default build (Metal GPU on macOS)
cargo build --release

# Build with GGUF quantized model support
cargo build --release --features gguf,metal

# Build with CUDA (Linux/Windows)
cargo build --release --features cuda

# Build without GPU (CPU-only)
cargo build --release --no-default-features

# Build with accelerate (macOS CPU acceleration)
cargo build --release --features accelerate
```

### Testing

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name

# Run GGUF-specific tests (requires gguf feature)
cargo test --features gguf gguf
```

### Running

```bash
# With debug logging
RUST_LOG=influence=debug cargo run -- generate "Hello" --model-path ./models/ModelName

# Using the built binary
./target/release/influence generate "Hello" --model-path ./models/ModelName
```

### Environment Configuration

Copy `.env.example` to `.env` and set defaults. Environment variables are loaded via `dotenvy` at startup.

## Architecture

### Modular Command Structure

The codebase follows a **one-command-per-module** pattern:

- **[main.rs](src/main.rs)** - Entry point, Tokio runtime, command routing
- **[cli.rs](src/cli.rs)** - Clap command definitions with extensive tests
- **[download.rs](src/download.rs)** - HuggingFace model download
- **[search.rs](src/search.rs)** - HuggingFace API model search
- **[influencer/](src/influencer/)** - High-level inference orchestration
  - **[mod.rs](src/influencer/mod.rs)** - Interactive chat mode with slash commands
  - **[server.rs](src/influencer/server.rs)** - Axum web API server
  - **[service.rs](src/influencer/service.rs)** - LlmService trait
- **[local/](src/local/)** - Core local inference implementation
  - **[mod.rs](src/local/mod.rs)** - LocalModel, architecture detection, generation
  - **[backends.rs](src/local/backends.rs)** - Model backend variants (Llama, Mistral, Mamba, BERT, GGUF)
  - **[device.rs](src/local/device.rs)** - Device management (CPU/Metal/CUDA)
  - **[gguf_backend.rs](src/local/gguf_backend.rs)** - GGUF model wrapper (feature-gated)
- **[models.rs](src/models.rs)** - Model listing and analysis
- **[config.rs](src/config.rs)** - Environment variable configuration
- **[error.rs](src/error.rs)** - Centralized error types with `thiserror`
- **[format.rs](src/format.rs)** - Output formatting utilities

### Backend Pattern

Different model architectures use a backend enum in **[local/backends.rs](src/local/backends.rs)**:

```rust
pub enum LocalBackend {
    Llama { model: Llama, config: LlamaConfig },
    Mistral { model: Mistral, config: MistralConfig },
    Mamba { model: Mamba, config: MambaConfig },
    GraniteMoeHybrid { model: GraniteMoeHybrid, config: GraniteMoeHybridConfig },
    Bert { model: Bert, config: BertConfig },
    #[cfg(feature = "gguf")]
    Gguf { backend: GGUFBackend },
}
```

When adding support for a new architecture:
1. Add variant to `LocalBackend` enum
2. Add loader in `LocalBackend::load_*`
3. Add match arm in `LocalModel::generate_text()` and `generate_stream_with()`
4. Update `ModelArchitecture` enum and detection in **[local/mod.rs](src/local/mod.rs)**

### Architecture Detection Flow

In **[local/mod.rs](src/local/mod.rs)**, `detect_architecture()`:

1. **GGUF files** (`.gguf`) are checked first if `gguf` feature is enabled
2. Falls back to `config.json` parsing
3. Detects unsupported architectures (MoE, Mamba in GraniteMoeHybrid)
4. Returns appropriate `ModelArchitecture` variant

### Device Management

**[local/device.rs](src/local/device.rs)** handles device selection:

- `auto` - Auto-detects GPU (Metal → CUDA → CPU)
- `cpu` - CPU only
- `metal` - macOS Metal GPU
- `cuda` - NVIDIA CUDA GPU

Use `get_device(preference, index)` to get the appropriate Candle Device.

### Chat Mode Architecture

Interactive chat (**[influencer/mod.rs](src/influencer/mod.rs)**) uses a state machine:

- **Slash commands**: `/help`, `/clear`, `/save`, `/load`, `/history`, `/set`, `/quit`
- **Session persistence**: JSON format with messages, timestamps, system prompt
- **Conversation history**: Keeps last 10 turns (20 messages) to manage memory
- **Runtime parameter adjustment**: `/set temperature 0.9`

## Key Patterns

### Error Handling

All modules use the centralized `InfluenceError` enum in **[error.rs](src/error.rs)**. Use `thiserror` for type-safe errors:

```rust
use crate::error::{InfluenceError, Result};

pub fn some_function() -> Result<()> {
    Err(InfluenceError::ModelNotFound("path".to_string()))
}
```

### Configuration Priority

1. CLI arguments (highest)
2. Environment variables (`.env` file)
3. Built-in defaults

### Sampling Implementation

Token sampling in **[local/mod.rs](src/local/mod.rs)** uses `do_sample()`:

- Temperature scaling
- Top-k filtering
- Top-p (nucleus) sampling
- Zero temperature → greedy argmax

### KV Cache Behavior

- **Fresh cache per request** - Each `generate` or API call starts with a clean cache
- No cross-request cache reuse (stateless generation)
- Cache is freed after generation completes

## GGUF Support (Experimental)

GGUF quantized model support is feature-gated:

- **Current status**: Detection and metadata parsing working
- **Inference**: Not yet implemented (returns error if attempted)
- **Detection priority**: GGUF files checked before `config.json`
- **Quantization format**: Detected from filename (Q4_K_M, Q8_0, etc.)

To enable: `cargo build --release --features gguf,metal`

## Testing Strategy

- **Unit tests**: Inline in each module (79 total tests)
- **Integration tests**: Manual testing via `test.sh`
- **Feature flag tests**: GGUF tests only run with `gguf` feature
- **CLI parsing tests**: Comprehensive in **[cli.rs](src/cli.rs)**

## Model Requirements

Each model directory must contain:

- `config.json` - Model architecture and parameters
- `tokenizer.json` or `tokenizer_config.json` - Tokenizer
- `*.safetensors` - Model weights (memory-mapped) OR
- `*.gguf` - Quantized model weights

## Supported Architectures

- ✅ **Llama** (Llama 2/3, TinyLlama)
- ✅ **Mistral** (detection only, inference not implemented)
- ✅ **Mamba**
- ✅ **GraniteMoeHybrid** (attention-only configs)
- ✅ **BERT** (encoder-only, embeddings only)
- ✅ **Phi**
- ✅ **Granite**
- ❌ **Mixture of Experts (MoE)** - Not supported
- ❌ **GraniteMoeHybrid with Mamba layers** - Not supported

## Adding New Commands

1. Add variant to `Commands` enum in **[cli.rs](src/cli.rs)**
2. Create new module in `src/`
3. Add handler in **[main.rs](src/main.rs)** match statement
4. Add tests in **[cli.rs](src/cli.rs)**

## Important Files

- **[Cargo.toml](Cargo.toml)** - Dependencies and feature flags
- **[.env.example](.env.example)** - Environment variable template
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed technical architecture
- **[README.md](README.md)** - User-facing documentation
- **[SPEC.md](SPEC.md)** - Implementation specifications
- **[TODO.md](TODO.md)** - Feature roadmap
