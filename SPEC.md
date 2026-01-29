# Influence - Specification

## Overview

Influence is a Rust CLI application for downloading models from HuggingFace mirrors and running local LLM inference with Candle. The tool prioritizes privacy and local-only operation with no cloud API dependencies.

## Version

**Current Version:** 0.1.0

## Architecture

### Components

1. **CLI Module** (`cli.rs`)
   - Command-line interface using Clap derive API
   - Four main commands: search, download, serve, generate
   - Type-safe argument parsing

2. **Download Module** (`download.rs`)
   - Downloads models from HuggingFace mirrors
   - Dynamic file discovery via HuggingFace API
   - Fallback to hardcoded file lists
   - Progress tracking with indicatif
   - Handles gated models gracefully
   - Automatic directory management

3. **Search Module** (`search.rs`)
   - Queries HuggingFace model API
   - Filters by author/organization
   - Displays model metadata
   - Shows download commands

4. **Local Module** (`local.rs`)
   - Local model loading and inference
   - Tokenizer loading from HuggingFace format
   - Model architecture detection
   - Llama-style model loading from .safetensors
   - Forward pass with KV caching
   - Temperature-based token sampling
   - Streaming token generation

5. **Influencer Module** (`influencer.rs`)
   - Command generation orchestration
   - Requires local model path
   - No cloud API dependency

6. **Error Module** (`error.rs`)
   - Centralized error handling with thiserror
   - Custom error types for each failure mode
   - Helpful error messages

### Design Principles

- **DRY**: No code duplication
- **KISS**: Simple, straightforward implementation
- **SoC**: Clear separation of concerns
- **Local-First**: No cloud API dependencies for generation
- **Maintainability**: Modular structure with clear responsibilities

## API Design

### CLI Commands

#### `search`
```bash
influence search <query> [options]
```

Search for models on HuggingFace.

**Arguments:**
- `query` - Search query string

**Options:**
- `-l, --limit <n>` - Maximum results (default: 20)
- `-a, --author <org>` - Filter by author/organization

**Output:**
- Model ID
- Author
- Downloads/Likes count
- Pipeline tag
- Download command

#### `download`
```bash
influence download -m <model> [options]
```

Download a model from HuggingFace mirror.

**Required:**
- `-m, --model <model>` - Model name (e.g., 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')

**Options:**
- `-r, --mirror <url>` - Mirror URL (default: hf-mirror.com)
- `-o, --output <path>` - Output directory (default: ./models/)

**Process:**
1. Validate model exists on mirror
2. Fetch file list from HuggingFace API
3. Download files with progress bars
4. Handle 403 errors for gated models
5. Save to local directory

**Downloaded Files:**
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer configuration
- `tokenizer_config.json` - Additional tokenizer settings
- `special_tokens_map.json` - Special token mappings
- `*.safetensors` - Model weights

#### `generate`
```bash
influence generate <prompt> [options]
```

Generate text using local LLM inference.

**Required:**
- `prompt` - Text prompt
- `-m, --model-path <path>` - Path to local model directory

**Options:**
- `--max-tokens <n>` - Maximum tokens to generate (default: 512)
- `--temperature <t>` - Sampling temperature (default: 0.7)
- `--system <text>` - Optional system prompt prepended before user prompt
- `--device <auto|cpu|metal|cuda>` - Compute backend selection (default: auto)
- `--device-index <n>` - GPU ordinal for metal/cuda backends (default: 0)

**Process:**
1. Load tokenizer from model directory
2. Detect model architecture from config.json
3. Load .safetensors weights
4. Initialize KV cache
5. Tokenize prompt
6. Generate tokens with temperature sampling
7. Stream output to stdout

### `serve`

Serve the local model over HTTP.

**Endpoints:**
- `POST /v1/generate` - returns JSON `{ "text": "..." }`
- `POST /v1/generate_stream` - returns SSE events `event: token` with token chunks in `data:`

**Error Conditions:**
- Model path not provided
- Model directory not found
- Missing tokenizer files
- Missing config.json
- Missing .safetensors files
- Unsupported model architecture (Mamba, MoE, encoder-only)

#### `serve`
```bash
influence serve [options]
```

Placeholder for future HTTP server functionality.

**Options:**
- `-m, --model-path <path>` - Path to model directory
- `-p, --port <n>` - Port to serve on (default: 8080)

**Status:** Not yet implemented

## Configuration

### No Configuration Files

The CLI follows the KISS principle:
- All configuration via command-line arguments
- No config files to manage
- No environment variables required
- Predictable behavior

### Model Directory Structure

Models are stored in a flat structure:
```
models/
└── <org>_<model-name>/
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── model.safetensors (or sharded model-*.safetensors)
    └── README.md (optional)
```

## Data Flow

### Download Flow

```
Parse CLI Arguments
  → Validate model name
  → Determine output directory
  → Create HTTP client
  → Check model exists (HEAD request)
  → Fetch file list (HuggingFace API or fallback)
  → For each file:
      → Check if already exists
      → Download with progress bar
      → Save to disk
      → Verify file integrity
  → Complete
```

### Generate Flow

```
Parse CLI Arguments
  → Validate --model-path provided
  → Read model directory
  → Load tokenizer (tokenizer.json)
  → Read config.json
  → Detect architecture (model_type field)
  → Check for unsupported architectures
  → Load .safetensors weights (mmap)
  → Build model graph
  → Initialize KV cache
  → Tokenize prompt
  → Generate tokens:
      → Forward pass
      → Get logits for last token
      → Apply temperature
      → Sample token (argmax)
      → Check EOS
      → Stream token to stdout
  → Complete
```

## Model Support

### Supported Architectures

**Currently Supported:**
- Llama (meta-llama/Llama-2-7b-hf, TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- Mistral (mistralai/Mistral-7B-v0.1)
- Phi (microsoft/phi-2)
- Granite (pure transformer variants)

**Not Supported:**
- Mamba/Hybrid models (e.g., GraniteMoeHybrid)
- Mixture of Experts (MoE) models
- Encoder-only models (BERT, RoBERTa, ALBERT)

### Model File Requirements

Each model directory must contain:
- `config.json` - Model architecture and hyperparameters
- `tokenizer.json` or `tokenizer_config.json` - Tokenizer
- `*.safetensors` - Model weights (memory-mapped)

### Architecture Detection

Detection logic in `local.rs`:
1. Read `config.json`
2. Check for `layer_types` field (indicates Mamba/MoE)
3. Parse `model_type` field
4. Map to architecture enum
5. Return helpful error for unsupported types

## Error Handling

### Error Types

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

### Error Messages

- User-friendly descriptions
- Actionable suggestions
- Model-specific guidance
- Architecture incompatibility explanations

## Testing Strategy

### Unit Tests
- Individual function testing
- Error case handling
- Configuration parsing

### Integration Tests
- End-to-end command testing
- Mock HTTP servers for downloads
- Model loading with test fixtures

### Test Coverage
- Core functionality
- Error paths
- Architecture detection

## Performance Considerations

### Current Optimizations
1. **KV Caching**: Reuse computed key/value tensors
2. **Memory Mapping**: mmap .safetensors files
3. **Async I/O**: Non-blocking downloads
4. **Streaming Output**: Display tokens as generated

### Performance Characteristics
- **Memory**: Model size + cache + tokenizer
- **Compute**: CPU-bound (GPU support planned)
- **Latency**: First token slower, subsequent tokens faster (due to cache)

## Security Considerations

1. **No Remote Code Execution**: Pure local inference
2. **HTTPS Only**: All downloads use HTTPS
3. **File Permissions**: Respects system permissions
4. **No API Keys**: No credentials to leak
5. **Input Validation**: All arguments validated

## Future Enhancements

### Priority 1 (Usability)
- Better sampling methods (top-k, nucleus)
- Chat mode with conversation history
- System prompt support
- Repetition penalty control

### Priority 2 (Performance)
- GPU support (CUDA/Metal)
- Batch generation
- Model caching
- Quantized model support

### Priority 3 (Features)
- HTTP API server
- Configuration file support
- Download resume capability
- Model validation after download
- Integration test suite

## Dependencies

### Core
- `tokio` 1.42 - Async runtime
- `clap` 4.5 - CLI parsing
- `reqwest` 0.12 - HTTP client

### ML Inference
- `candle-core` 0.9 - Core ML operations
- `candle-nn` 0.9 - Neural network components
- `candle-transformers` 0.9 - Transformer models
- `tokenizers` 0.21 - HuggingFace tokenizers

### Utilities
- `tracing` 0.1 - Logging
- `indicatif` 0.17 - Progress bars
- `serde` 1.0 - Serialization
- `anyhow` 1.0 - Error context
- `thiserror` 2.0 - Error types
