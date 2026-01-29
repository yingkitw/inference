# Influence

> Privacy-first local LLM inference - Download models from HuggingFace and run them entirely on your machine.

## Why Influence?

**The Problem:** Most LLM tools require cloud APIs, expensive subscriptions, or complex Python setups. Your data leaves your machine, you pay per token, and you're locked into someone else's infrastructure.

**The Solution:** Influence gives you:
- **Complete privacy** - All inference happens locally on your machine
- **No API costs** - Pay once (in compute) and use forever
- **No vendor lock-in** - Models are downloaded to your disk
- **Simplicity** - Single binary, no Python, no virtual environments
- **GPU acceleration** - Metal support for macOS (CUDA coming soon)

## What Makes It Different?

| Feature | Influence | Cloud APIs (OpenAI, etc.) | Python Tools |
|---------|-----------|---------------------------|--------------|
| Privacy | 100% local | Data sent to servers | Local but complex |
| Cost | Free (after download) | Pay per token | Free but complex setup |
| Setup | Single binary | API key required | Python, pip, venv |
| GPU Support | Metal (macOS) | Server-side | Hard to configure |
| Offline Use | Yes | No | Yes |

## Quick Start

```bash
# Build from source
git clone https://github.com/yingkitw/influence.git
cd influence
cargo build --release

# Search for a model
./target/release/influence search "tinyllama" --limit 5

# Download a model (~1GB for TinyLlama)
./target/release/influence download -m TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Generate text locally (with Metal GPU on macOS)
./target/release/influence generate "Explain quantum computing in simple terms" \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
  --device metal
```

## Usage Examples

### Example 1: Quick Question Answering

```bash
# Ask a factual question
influence generate "What are the main differences between Rust and C++?" \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
  --max-tokens 256
```

**Benefit:** Get instant answers without:
- Opening a browser
- Waiting for cloud API responses
- Paying per token
- Sending your queries to third parties

### Example 2: Code Generation

```bash
# Generate code with higher temperature for creativity
influence generate "Write a Rust function to merge two sorted vectors" \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
  --temperature 0.8 \
  --max-tokens 512
```

**Benefit:** Generate code locally with:
- No rate limits
- No API keys to manage
- Full context control
- Works offline

### Example 3: Content Creation

```bash
# Generate blog post or documentation
influence generate "Write a technical introduction to vector databases" \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
  --max-tokens 1024
```

**Benefit:** Create content without:
- Using cloud services
- Exposing your ideas to third parties
- Worrying about content policies

## Current Status

**Working:**
- Model search and download from Hugging Face
- Local inference with Llama-architecture models (including TinyLlama, Llama 2/3)
- GPU acceleration (Metal on macOS, CUDA on Linux/Windows)
- Streaming generation with fresh KV cache per request
- Interactive chat mode with conversation history
- Web API server (REST + SSE streaming)
- Top-k, top-p sampling and temperature control
- Repetition penalty
- System prompt support
- Metal GPU warmup for reduced first-token latency
- Ollama-compatible API endpoints (`/api/generate`, `/api/embeddings`, `/api/tags`)

**Tested Models:**
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` - Working perfectly
- Other Llama-architecture models - Supported

## Installation

### Build from Source

```bash
# Clone the repository
git clone https://github.com/yingkitw/influence.git
cd influence

# Build release binary with Metal support (macOS)
cargo build --release

# The binary will be at target/release/influence
./target/release/influence --help
```

**Features:**
- `metal` (default) - Metal GPU acceleration for macOS
- `accelerate` - CPU acceleration for macOS
- `cuda` - CUDA support for NVIDIA GPUs (placeholder)

**Build without GPU:**
```bash
cargo build --release --no-default-features
```

## Configuration

Influence supports configuration via environment variables for convenience. Create a `.env` file in the project root:

```bash
cp .env.example .env
# Edit .env with your preferred defaults
```

**Available environment variables:**

```bash
# Model Configuration
INFLUENCE_MODEL_PATH=./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0

# Generation Parameters
INFLUENCE_TEMPERATURE=0.7
INFLUENCE_TOP_P=0.9
INFLUENCE_TOP_K=
INFLUENCE_REPEAT_PENALTY=1.1
INFLUENCE_MAX_TOKENS=512

# Device Configuration
INFLUENCE_DEVICE=auto
INFLUENCE_DEVICE_INDEX=0

# Server Configuration
INFLUENCE_PORT=8080

# Performance Tuning
INFLUENCE_WARMUP_TOKENS=6

# Download Configuration
INFLUENCE_MIRROR=https://hf-mirror.com
INFLUENCE_OUTPUT_DIR=./models
```

**Priority:** CLI arguments > Environment variables > Built-in defaults

## Command Reference

### `search` - Find Models on HuggingFace

```bash
influence search <query> [options]
```

**Examples:**

```bash
# Search for llama models
influence search "llama"

# Search with filters
influence search "text-generation" --limit 10 --author meta-llama

# Search for small models
influence search "1b" --limit 5
```

**Options:**
- `-l, --limit <N>` - Max results (default: 20)
- `-a, --author <ORG>` - Filter by author

### `download` - Download Model from HuggingFace

```bash
influence download -m <model> [options]
```

**Examples:**

```bash
# Download TinyLlama (recommended for testing)
influence download -m TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Download to custom location
influence download -m microsoft/phi-2 -o ~/models

# Use custom mirror
influence download -m ibm/granite-4-h-small -r https://hf-mirror.com
```

**Options:**
- `-m, --model <MODEL>` - Model name (required)
- `-r, --mirror <URL>` - Mirror URL (default: hf-mirror.com)
- `-o, --output <PATH>` - Output directory (default: ./models/)

### `generate` - Generate Text Locally

```bash
influence generate <prompt> [options]
```

**Examples:**

```bash
# Basic generation (with explicit model path)
influence generate "What is machine learning?" \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0

# Or use .env configuration (set INFLUENCE_MODEL_PATH)
influence generate "What is machine learning?"

# With custom parameters
influence generate "Explain async/await" \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
  --max-tokens 512 \
  --temperature 0.7

# Lower temperature for more focused output
influence generate "Summarize: Rust is a systems programming language" \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
  --temperature 0.3 \
  --max-tokens 100
```

**Options:**
- `-m, --model-path <PATH>` - Path to model directory (required)
- `--max-tokens <N>` - Max tokens to generate (default: 512)
- `--temperature <0.0-2.0>` - Sampling temperature (default: 0.7)
  - Lower (0.1-0.3): More focused, deterministic
  - Higher (0.7-1.0): More creative, diverse

## Recommended Models

### For Testing & Development

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~1GB | Fast | Testing, quick experiments |
| `microsoft/phi-2` | ~2GB | Medium | Quality vs speed balance |
| `mistralai/Mistral-7B-v0.1` | ~14GB | Slower | Production-quality output |

### Why TinyLlama?

```bash
# Download and try TinyLlama first
influence download -m TinyLlama/TinyLlama-1.1B-Chat-v1.0
influence generate "Hello, world!" \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0
```

**Benefits:**
- Fast downloads (~1GB)
- Quick inference (even on CPU)
- Good quality for many tasks
- Great for learning and experimentation

## Benefits Over Alternatives

### vs Cloud APIs (OpenAI, Anthropic, etc.)

**You Save:**
- Money - No per-token costs
- Privacy - Data never leaves your machine
- Latency - No network round-trips
- Reliability - Works offline
- Control - No rate limits or content policies

### vs Python Tools (llama.cpp, transformers, etc.)

**You Get:**
- Simplicity - Single binary, no dependencies
- Performance - Rust speed with GPU acceleration
- Stability - No version conflicts or dependency hell
- Integration - Easy to script and automate

## How It Works

```
┌─────────────┐
│  Your Prompt│
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────┐
│  Tokenization (HuggingFace)      │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  Model Loading (.safetensors)    │
│  - Memory-mapped for efficiency  │
│  - GPU acceleration (Metal/CUDA) │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  Inference (Candle)              │
│  - Forward pass with KV cache    │
│  - Temperature-based sampling    │
│  - Token-by-token generation     │
└──────┬───────────────────────────┘
       │
       ▼
┌─────────────┐
│  Output Text│
└─────────────┘
```

## Technical Details

### Ollama-Compatible API (partial)

When running `influence serve`, Influence also exposes a small subset of Ollama-compatible endpoints. This is intended to make it easier to integrate with tools that already speak Ollama, while keeping Influence’s internal callflow minimal.

**Supported:**
- `POST /api/generate`
  - Non-stream: returns JSON
  - Stream: returns `application/x-ndjson` (one JSON object per line)
- `POST /api/embeddings` (BERT embeddings only)
- `POST /api/tags` (returns the currently served model name)

**Notes / limitations:**
- The `model` field is currently accepted but not used to dynamically switch models (Influence serves one loaded model).
- Some Ollama fields are ignored for now; only a small set of `options` is mapped.

### Model Requirements

Each model directory must contain:
- `config.json` - Model architecture and parameters
- `tokenizer.json` or `tokenizer_config.json` - Tokenizer
- `*.safetensors` - Model weights (memory-mapped)

### Supported Architectures

- OK Llama (meta-llama/Llama-2-7b-hf, TinyLlama)
- OK Mamba (mamba family configs)
- OK GraniteMoeHybrid (attention-only configs)
- OK Encoder-only embeddings: BERT (`influence embed ...`)
- X Mixture of Experts (MoE) models (not yet supported)
- X GraniteMoeHybrid configs containing Mamba layers (not supported by candle-transformers yet)

### Performance

**Optimizations:**
- KV Caching - Reuse computed tensors for faster generation
- Memory Mapping - Zero-copy model loading
- Streaming Output - Display tokens as they're generated
- GPU Acceleration - Metal support on macOS (enabled by default)
- Proper Token Spacing - Handles SentencePiece space markers correctly

### Metal Warmup (macOS)

On **macOS with Metal GPU**, the first few decode steps can be significantly slower due to Metal kernel compilation overhead. To mitigate this, Influence automatically runs a small warmup (default: 6 decode steps) during model load to pre-compile kernels and reduce visible latency for the first generated tokens.

- **Control warmup**: Set `INFLUENCE_WARMUP_TOKENS=0` to disable, or adjust the count (e.g., `INFLUENCE_WARMUP_TOKENS=10`).
- **When it helps**: Most noticeable with TinyLlama and similar models on Metal.
- **Trade-off**: Slightly longer model load time in exchange for faster first-token generation.

### KV Cache Behavior

Influence creates a **fresh KV cache** for each generation request:

- **Stateless generation**: Each `generate` or API call starts with a clean cache, ensuring predictable behavior.
- **No cross-request cache reuse**: Currently, KV cache is not persisted across requests or chat turns.
- **Memory efficient**: Cache is automatically freed after each generation completes.
- **Future enhancement**: Session-based cache reuse for multi-turn conversations is planned to reduce redundant prefill computation.

## Troubleshooting

### Model Not Found Error

```bash
# Error: Model directory not found
# Solution: Check the model path exists
ls ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0
```

### Missing Tokenizer Error

```bash
# Error: Tokenizer file not found
# Solution: Ensure these files exist in model directory:
# - tokenizer.json (or tokenizer_config.json)
# - config.json
# - *.safetensors files
```

### Unsupported Architecture Error

```bash
# Error: Unsupported model architecture (Mamba/MoE)
# Solution: Use a supported model like TinyLlama
influence download -m TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### Slow Generation on CPU

```bash
# CPU inference is slower. Options:
# 1. Use a smaller model (TinyLlama instead of Mistral-7B)
# 2. Reduce max-tokens
# 3. Build with Metal support (macOS):
cargo build --release --features metal
```

## Development

### Build with Debug Logging

```bash
RUST_LOG=influence=debug cargo run -- generate "Hello" \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0
```

### Run Tests

```bash
cargo test
```

## Roadmap

- [ ] CUDA support for NVIDIA GPUs
- [ ] Quantized model support (GGUF)
- [ ] Chat mode with conversation history
- [ ] Batch generation
- [ ] HTTP API server mode
- [ ] Top-k and nucleus sampling

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Built with:
- [Candle](https://github.com/huggingface/candle) - ML framework by HuggingFace
- [Tokenizers](https://github.com/huggingface/tokenizers) - Fast tokenization
- [Clap](https://github.com/clap-rs/clap) - CLI parsing
