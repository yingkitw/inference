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
# Install (requires Rust)
cargo install influence

# Or build from source
cargo build --release

# Search for a model
influence search "tinyllama" --limit 5

# Download a model (~1GB for TinyLlama)
influence download -m TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Generate text locally
influence generate "Explain quantum computing in simple terms" \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0
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

## Installation

### From Source (Recommended)

```bash
# Clone and build
git clone https://github.com/yourusername/influence.git
cd influence
cargo build --release

# The binary will be at target/release/influence
# Add to your PATH:
export PATH="$PATH:$(pwd)/target/release"
```

### Via Cargo (Coming Soon)

```bash
cargo install influence
```

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
# Basic generation
influence generate "What is machine learning?" \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0

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

### Model Requirements

Each model directory must contain:
- `config.json` - Model architecture and parameters
- `tokenizer.json` or `tokenizer_config.json` - Tokenizer
- `*.safetensors` - Model weights (memory-mapped)

### Supported Architectures

- OK Llama (meta-llama/Llama-2-7b-hf, TinyLlama)
- OK Mistral (mistralai/Mistral-7B-v0.1)
- OK Phi (microsoft/phi-2)
- OK Granite (pure transformer variants)
- X Mamba/Hybrid models (specialized implementation required)
- X MoE models (not yet supported)
- X Encoder-only models (BERT, etc. - not for generation)

### Performance

**Optimizations:**
- KV Caching - Reuse computed tensors for faster generation
- Memory Mapping - Zero-copy model loading
- Streaming Output - Display tokens as they're generated
- GPU Acceleration - Metal support on macOS (CUDA coming soon)

**Memory Usage:**
- TinyLlama (1B): ~2GB RAM
- Phi-2 (2.7B): ~4GB RAM
- Mistral-7B: ~14GB RAM
- Add model size for total memory requirement

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
