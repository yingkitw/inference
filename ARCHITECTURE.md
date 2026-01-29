# Architecture

## Overview

Influence is a modular Rust CLI application for downloading HuggingFace models and running local LLM inference. The architecture prioritizes simplicity, testability, and local-only operation (no cloud API dependencies).

## Module Structure

```
influence/
├── src/
│   ├── main.rs           # Entry point, CLI routing
│   ├── cli.rs            # Command definitions using Clap
│   ├── download.rs       # Model download from HuggingFace
│   ├── search.rs         # Model search via HuggingFace API
│   ├── local.rs          # Local model inference with Candle
│   ├── influencer.rs     # Command generation logic
│   └── error.rs          # Centralized error types
├── examples/
│   └── basic_usage.rs    # Usage examples
└── tests/                # Integration tests
```

## Component Diagram

```
┌─────────────────────────────────────────┐
│              CLI (main.rs)              │
│         Command Routing Layer           │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
   ┌────────┐ ┌──────┐ ┌──────────┐
   │Download│ │Search│ │Generate  │
   │Module  │ │Module│ │Module    │
   └────┬───┘ └───┬──┘ └────┬─────┘
        │         │         │
        │         │         └───────┐
        │         │                 │
   ┌────▼────────▼─────────────────▼────┐
   │         Local Inference Layer       │
   │              (local.rs)             │
   │  ┌──────────────────────────────┐  │
   │  │  LocalModel                  │  │
   │  │  - Tokenizer loading         │  │
   │  │  - Architecture detection    │  │
   │  │  - Weight loading (.safetensors)│ │
   │  │  - Forward pass (Candle)     │  │
   │  │  - Token sampling            │  │
   │  │  - Streaming output          │  │
   │  └──────────────────────────────┘  │
   └────────────────────────────────────┘
        │
        ▼
   ┌────────────────────────────────────┐
   │         Dependencies               │
   │  - candle-transformers (inference) │
   │  - tokenizers (tokenization)       │
   │  - reqwest (HTTP client)           │
   └────────────────────────────────────┘
```

## Key Design Patterns

### 1. Modular Commands

Each command is a separate module with a single responsibility:
- `download.rs` - Model downloading
- `search.rs` - Model search
- `local.rs` - Local inference
- `influencer.rs` - Generation orchestration

### 2. Centralized Error Handling

Using `thiserror` for type-safe error handling:

```rust
pub enum InfluenceError {
    DownloadError(String),
    ModelNotFound(String),
    InvalidConfig(String),
    LocalModelError(String),
    IoError(std::io::Error),
    HttpError(reqwest::Error),
    JsonError(serde_json::Error),
    CandleError(String),
    TokenizerError(String),
}
```

### 3. Async/Await

All I/O operations are async for better concurrency:
- HTTP downloads
- File operations
- Model loading

### 4. Architecture Detection

Automatic model architecture detection from `config.json`:
- Parses `model_type` field
- Detects unsupported architectures (Mamba, MoE)
- Provides helpful error messages

### 5. Streaming Generation

Real-time token-by-token output:
- Tokenizes prompt
- Runs forward pass with KV cache
- Samples tokens with temperature
- Streams output as tokens are generated

## Data Flow

### Download Command

```
CLI Input
  → Parse Args
  → Determine Output Dir
  → Create HTTP Client
  → Check Model Exists
  → Fetch File List (from HuggingFace API)
  → For each file:
      → Download with progress bar
      → Save to disk
  → Complete
```

### Generate Command (Local Inference)

```
CLI Input
  → Parse Args (require --model-path)
  → Load LocalModel
      → Load tokenizer
      → Detect architecture from config.json
      → Load .safetensors weights
      → Initialize KV cache
  → Tokenize prompt
  → Run inference loop:
      → Forward pass through model
      → Apply temperature scaling
      → Sample next token (argmax)
      → Check EOS token
      → Stream token to stdout
  → Complete
```

## Model Loading Details

### Architecture Detection Flow

```
Read config.json
  → Parse JSON
  → Check for layer_types (detect Mamba/MoE)
  → Check model_type field
      ├── "llama" → Llama architecture
      ├── "mistral" → Mistral architecture
      ├── "phi" → Phi architecture
      ├── "granite" → Granite architecture
      └── Unknown → Default to Llama with warning
```

### Supported Model Files

Each model directory must contain:
- `tokenizer.json` or `tokenizer_config.json` - Tokenizer configuration
- `config.json` - Model architecture and parameters
- `*.safetensors` - Model weights (one or more files)

### Inference Pipeline

```
1. Load Model
   - Parse config.json for model parameters
   - Create LlamaConfig with actual dimensions
   - Memory-map .safetensors files
   - Build model graph

2. Process Prompt
   - Tokenize input text
   - Create input tensor
   - Run forward pass to fill cache

3. Generate Tokens
   - Get logits from last token
   - Apply temperature scaling
   - Sample token (argmax)
   - Append to output
   - Check EOS condition
   - Repeat until max_tokens or EOS
```

## Configuration

### No Configuration Files

The CLI follows the KISS principle:
- All configuration via command-line arguments
- No config files to manage
- Predictable behavior

### Environment Variables

No environment variables are required for basic local inference.

Optional environment variables:
- `INFLUENCE_WARMUP_TOKENS` (macOS + Metal + Llama): controls a short warmup during model load to reduce first-token latency caused by Metal kernel compilation. Set to `0` to disable.

## Logging

Uses `tracing` for structured logging:
- `info` - User-facing progress messages
- `debug` - Internal operations details
- `warn` - Non-fatal issues (unsupported architectures)
- `error` - Fatal errors

Enable debug logging:
```bash
RUST_LOG=influence=debug cargo run -- generate "Hello" --model-path ./model
```

## Dependencies

### Core
- `tokio` - Async runtime
- `clap` - CLI parsing
- `reqwest` - HTTP client for downloads/search

### ML Inference
- `candle-core` - Core ML operations
- `candle-nn` - Neural network components
- `candle-transformers` - Transformer models
- `tokenizers` - HuggingFace tokenizers

### Utilities
- `tracing` - Logging
- `indicatif` - Progress bars
- `serde` - Serialization

### Error Handling
- `anyhow` - Error context
- `thiserror` - Error types

## Performance Considerations

### Current Optimizations
1. **KV Caching**: Cache key/value tensors for generated tokens
2. **Memory Mapping**: Use mmap for .safetensors files
3. **Streaming Output**: Display tokens as they're generated
4. **Async I/O**: Non-blocking operations

### Performance Characteristics
- **Memory**: Model size + cache + tokenizer
- **CPU-bound**: Inference runs on CPU (GPU support planned)
- **Single-threaded**: No parallel inference yet

## Security Considerations

1. **No Remote Execution**: Pure local inference
2. **HTTPS Only**: All downloads use HTTPS
3. **File Permissions**: Respects system permissions
4. **Input Validation**: CLI arguments validated
5. **No API Keys**: No credentials to leak

## Extensibility

### Adding New Model Architectures

1. Add variant to `ModelArchitecture` enum
2. Implement architecture-specific loader
3. Add detection logic in `detect_architecture()`
4. Update config parsing if needed

### Adding New Commands

1. Add variant to `Commands` enum in `cli.rs`
2. Create new module in `src/`
3. Add handler in `main.rs`
4. Update documentation

### Adding New Sampling Methods

1. Extend `generate_text()` method
2. Add CLI option for sampling method
3. Implement sampling algorithm (top-k, nucleus, etc.)
4. Update tests

## Maintenance

- **Single Responsibility**: Each module has one clear purpose
- **DRY**: No code duplication
- **KISS**: Simple, straightforward implementations
- **Documentation**: Inline docs for public APIs
- **Tests**: Unit tests for core functionality

## Known Limitations

1. **No batch processing**: Single prompt only
2. **Some architectures not supported**: MoE models are rejected; GraniteMoeHybrid with Mamba layers is rejected
