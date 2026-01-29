# Influence

A Rust CLI application for downloading models from HuggingFace mirrors and running local LLM inference with Candle.

## Features

- **Search models** on HuggingFace with filters
- **Download models** from HuggingFace mirrors with progress tracking
- **Full local inference** with candle-transformers (Llama architecture)
- **Streaming text generation** with real-time token output
- **Model architecture detection** (Llama, Mistral, Phi, Granite)
- **CPU inference** with support for .safetensors model weights
- **CLI interface** with simple commands
- **No external API dependencies** - 100% local inference

## Quick Start

```bash
# 1. Search for a model
cargo run -- search "llama" --limit 5

# 2. Download a model
cargo run -- download -m TinyLlama/TinyLlama-1.1B-Chat-v1.0

# 3. Generate text with the local model
cargo run -- generate "What is Rust?" --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0
```

## Installation

```bash
# Build from source
cargo build --release

# The binary will be at target/release/influence
```

## Usage

### Search for models

Search for models on HuggingFace:

```bash
cargo run -- search "llama"
```

With filters:

```bash
cargo run -- search "text-generation" --limit 10 --author meta-llama
```

### Download a model

Download a model from HuggingFace mirror:

```bash
cargo run -- download -m TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

With custom mirror and output directory:

```bash
cargo run -- download -m ibm/granite-4-h-small -r https://hf-mirror.com -o ./models
```

### Generate text with local model

Generate text using a downloaded local model:

```bash
cargo run -- generate "What is Rust programming language?" --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0
```

With custom parameters:

```bash
cargo run -- generate "Explain quantum computing" --model-path ./models/meta-llama_Llama-2-7b --max-tokens 1024 --temperature 0.8
```

**Note:** The `--model-path` parameter is required and should point to a directory containing:
- `tokenizer.json` or `tokenizer_config.json`
- `config.json`
- Model weights (`.safetensors` files)

## Local Inference Details

The CLI includes complete local inference implementation using candle-transformers:
- Loads Llama-architecture models from .safetensors weights
- Performs forward pass with KV caching for efficiency
- Samples tokens with temperature control
- Streams output token-by-token for real-time generation
- Runs on CPU (Mac acceleration available with `--features accelerate`)

### Supported Model Architectures

- OK Standard Llama models (meta-llama/Llama-2-7b-hf, TinyLlama, etc.)
- OK Standard Mistral models (mistralai/Mistral-7B-v0.1, etc.)
- OK Pure transformer-based models with Llama architecture
- X Mamba/Hybrid models (GraniteMoeHybrid, etc.) - requires specialized implementation
- X MoE (Mixture of Experts) models - not yet supported

### Recommended Models

For testing and development:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` - Small (~1GB), fast for testing
- `microsoft/phi-2` - Compact model with good performance
- `mistralai/Mistral-7B-v0.1` - Larger but capable model

## Commands

### `search`

Search for models on HuggingFace.

**Arguments:**
- `<QUERY>` - Search query

**Options:**
- `-l, --limit <LIMIT>` - Maximum number of results (default: 20)
- `-a, --author <AUTHOR>` - Filter by author/organization

### `download`

Download a model from HuggingFace mirror.

**Options:**
- `-m, --model <MODEL>` - Model name (e.g., 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
- `-r, --mirror <MIRROR>` - Mirror URL (default: hf-mirror.com)
- `-o, --output <OUTPUT>` - Output directory (default: ./models/)

### `generate`

Generate text using a local LLM.

**Arguments:**
- `<PROMPT>` - Prompt text

**Options:**
- `-m, --model-path <MODEL_PATH>` - Path to local model directory (required)
- `--max-tokens <MAX_TOKENS>` - Maximum tokens to generate (default: 512)
- `--temperature <TEMPERATURE>` - Temperature for generation (default: 0.7)

**Note:** The `--model-path` parameter is required. Local inference only - no cloud API dependency.

## Architecture

The application follows a modular architecture:

- **`cli`** - Command-line interface using Clap
- **`search`** - Model search on HuggingFace
- **`download`** - Model downloading from HuggingFace mirrors with dynamic file discovery
- **`local`** - Local model loading and inference with candle-transformers
- **`error`** - Error handling with thiserror

## Testing

Run tests:

```bash
cargo test
```

Run with logging:

```bash
RUST_LOG=influence=debug cargo test
```

## Dependencies

- **clap** - CLI parsing
- **tokio** - Async runtime
- **reqwest** - HTTP client
- **candle-core/candle-nn/candle-transformers** - ML inference framework
- **tokenizers** - Tokenization from HuggingFace
- **tracing** - Logging
- **indicatif** - Progress bars

## License

MIT
