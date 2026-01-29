# Influence - Feature Overview

## Core Features

### 1. Model Search 🔍

Search for models on HuggingFace with powerful filtering:

```bash
# Basic search
cargo run -- search "llama"

# Filter by author
cargo run -- search "text-generation" --author meta-llama

# Limit results
cargo run -- search "granite" --limit 5
```

**Features:**
- Real-time search via HuggingFace API
- Filter by author/organization
- Display model metadata (downloads, likes, task type)
- Show direct download commands

### 2. Model Download 📥

Download models from HuggingFace mirrors with advanced features:

```bash
# Basic download
cargo run -- download -m ibm/granite-4-h-small

# Custom output directory
cargo run -- download -m meta-llama/Llama-2-7b -o ./my-models

# Custom mirror
cargo run -- download -m mistralai/Mistral-7B-v0.1 -r https://hf-mirror.com
```

**Features:**
- Dynamic file discovery from HuggingFace API
- Progress bars for each file
- Overall progress tracking
- Graceful handling of gated models
- Automatic model existence verification
- Resume capability (skip already downloaded files)
- Smart error messages with helpful hints

**Downloaded Files:**
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer vocabulary
- `tokenizer_config.json` - Tokenizer settings
- `special_tokens_map.json` - Special tokens
- Model weights (`.safetensors`, `.gguf`, or `.bin`)

### 3. Local Model Generation 🤖

Generate text using downloaded local models:

```bash
# Basic generation
cargo run -- generate "What is Rust?" --model-path ./models/ibm_granite-4-h-small

# With parameters
cargo run -- generate "Explain AI" \
  --model-path ./models/ibm_granite-4-h-small \
  --max-tokens 1024 \
  --temperature 0.8
```

**Features:**
- Automatic model architecture detection (Llama, Mistral, Phi, Granite)
- Tokenizer loading and validation
- Model file verification
- Informative output about model status
- Ready for full inference implementation

**Supported Architectures:**
- Llama / Llama 2
- Mistral
- Phi
- IBM Granite
- Generic models (fallback to Llama)

**Supported Model Formats:**
- SafeTensors (`.safetensors`) - Recommended
- GGUF (`.gguf`) - Quantized models
- PyTorch (`.bin`) - Standard format

### 4. Model Architecture Detection 🔬

Automatically detects model architecture from:
1. `config.json` model_type field
2. Model file extensions
3. Fallback to sensible defaults

Detected architectures:
- `LlamaForCausalLM` → Llama
- `MistralForCausalLM` → Mistral
- `PhiForCausalLM` → Phi
- GGUF files → LlamaQuantized
- SafeTensors → Llama (default)

## Technical Features

### Error Handling
- Custom error types for different failure modes
- Helpful error messages with actionable suggestions
- Graceful degradation for partial failures
- Gated model detection and guidance

### Progress Tracking
- Per-file download progress bars
- Overall download progress
- File size and transfer rate display
- Completion percentage

### Model Validation
- Pre-download existence checks
- Tokenizer file validation
- Config file parsing
- Architecture compatibility checks

### Logging
- Structured logging with `tracing`
- Multiple log levels (info, debug, warn, error)
- Configurable via `RUST_LOG` environment variable

## Workflow Examples

### Complete Workflow
```bash
# 1. Search for models
cargo run -- search "granite" --author ibm --limit 5

# 2. Download selected model
cargo run -- download -m ibm/granite-4-h-small

# 3. Generate with local model
cargo run -- generate "Write hello world in Rust" \
  --model-path ./models/ibm_granite-4-h-small
```

### Batch Download
```bash
# Download multiple models
cargo run -- download -m ibm/granite-4-h-small
cargo run -- download -m meta-llama/Llama-2-7b-hf
cargo run -- download -m mistralai/Mistral-7B-v0.1
```

### Custom Configuration
```bash
# Use custom directories and parameters
cargo run -- download -m ibm/granite-4-h-small -o /data/models
cargo run -- generate "Explain Rust" \
  --model-path /data/models/ibm_granite-4-h-small \
  --max-tokens 2048 \
  --temperature 0.5
```

## Future Enhancements

### Planned Features
- Full local inference with candle-transformers
- GPU acceleration support
- Quantization support (4-bit, 8-bit)
- Batch processing
- HTTP API server
- Model caching and versioning
- Resume interrupted downloads
- Parallel file downloads

### Potential Integrations
- llama.cpp for quantized inference
- ONNX Runtime for cross-platform inference
- vLLM for high-performance serving
- OpenAI-compatible API endpoint

## Performance

### Download Performance
- Streaming downloads (no memory overhead)
- Concurrent file downloads (future)
- Resume capability for large files
- Mirror selection for optimal speed

### Model Loading
- Fast tokenizer loading with `tokenizers` crate
- Lazy model weight loading
- Memory-efficient architecture detection
- Minimal startup overhead

## Security

### Best Practices
- No hardcoded credentials
- Environment variable configuration
- HTTPS-only downloads
- File permission respect
- Input validation

### Gated Models
- Automatic detection of gated models
- Clear guidance for access requests
- Graceful failure with helpful messages
- Link to HuggingFace access request page

## Compatibility

### Rust Edition
- Edition 2024
- Latest stable Rust

### Dependencies
- `clap` - CLI parsing
- `tokio` - Async runtime
- `reqwest` - HTTP client
- `tokenizers` - Tokenizer support
- `candle-*` - ML framework (ready for full inference)
- `serde` - Serialization
- `tracing` - Logging

### Platform Support
- macOS ✓
- Linux ✓
- Windows ✓ (untested)

## Documentation

- `README.md` - Overview and quick start
- `USAGE.md` - Detailed usage guide
- `SPEC.md` - Technical specification
- `ARCHITECTURE.md` - Architecture details
- `TODO.md` - Task tracking
- `FEATURES.md` - This file

## Testing

- Unit tests for core functionality
- Integration tests (planned)
- Example programs
- Demo script (`demo.sh`)

Run tests:
```bash
cargo test
```

Run demo:
```bash
./demo.sh
```
