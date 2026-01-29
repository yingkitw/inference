# Influence

> Privacy-first local LLM inference - Download models from HuggingFace and run them entirely on your machine.

## Why Influence?

**The Problem:** Most LLM tools require cloud APIs, expensive subscriptions, or complex Python setups. Your data leaves your machine, you pay per token, and you're locked into someone else's infrastructure.

**The Solution:** Influence gives you:
- **Complete privacy** - All inference happens locally on your machine
- **No API costs** - Pay once (in compute) and use forever
- **No vendor lock-in** - Models are downloaded to your disk
- **Simplicity** - Single binary, no Python, no virtual environments
- **GPU acceleration** - Metal (macOS) and CUDA (Linux/Windows) support with auto-detection

## What Makes It Different?

| Feature | Influence | Ollama | vLLM | Cloud APIs (OpenAI, etc.) | Python Tools |
|---------|-----------|---------|------|---------------------------|--------------|
| Privacy | 100% local | 100% local | 100% local | Data sent to servers | Local but complex |
| Cost | Free (after download) | Free | Free | Pay per token | Free but complex setup |
| Setup | Single binary | Binary install | Python, pip, venv | API key required | Python, pip, venv |
| GPU Support | Metal (macOS), CUDA (Linux/Windows) | Metal/CUDA | CUDA only | Server-side | Hard to configure |
| Offline Use | Yes | Yes | Yes | No | Yes |
| Model Management | Built-in list & deploy commands | CLI | Manual | N/A | Manual setup |
| API Server | Built-in REST/SSE | Built-in REST | OpenAI-compatible API | N/A | Manual setup |

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

# Generate text locally (auto-detects GPU: Metal/CUDA or falls back to CPU)
./target/release/influence generate "Explain quantum computing in simple terms" \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0
```

## Usage Examples

### Example 1: Model Management

```bash
# List all downloaded models
influence list

# Search and download a model
influence search "tinyllama" --limit 3
influence download -m TinyLlama/TinyLlama-1.1B-Chat-v1.0

# List again to see your downloaded model
influence list
```

**Benefit:** Easy model management with clear visibility of what's installed.

### Example 2: Deploy and Serve

```bash
# Deploy a model as an API server
influence deploy \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
  --port 8080

# In another terminal, test the API
curl http://localhost:8080/health
```

**Benefit:** Quick deployment of local models as REST APIs for your applications.

### Example 3: Quick Question Answering

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
- **Model management with `list` and `deploy` commands**
- Local inference with Llama-architecture models (including TinyLlama, Llama 2/3)
- GPU acceleration (Metal on macOS, CUDA on Linux/Windows)
- Streaming generation with fresh KV cache per request
- **Enhanced interactive chat mode with slash commands and session persistence**
  - `/help`, `/clear`, `/save`, `/load`, `/history`, `/set`, `/quit`
  - Session save/load (JSON format)
  - Runtime parameter adjustment
- **CLI UX improvements with streaming markdown rendering and syntax highlighting**
- **Configuration management with `config` command**
- Web API server (REST + SSE streaming)
- Top-k, top-p sampling and temperature control
- Repetition penalty
- System prompt support
- Metal GPU warmup for reduced first-token latency
- Ollama-compatible API endpoints (`/api/generate`, `/api/embeddings`, `/api/tags`)
- GGUF model detection and metadata parsing (full inference coming soon)

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
- `cuda` - CUDA support for NVIDIA GPUs on Linux/Windows
- `gguf` - GGUF quantized model support (experimental)

**Build with GGUF support:**
```bash
cargo build --release --features gguf,metal
```

**Build without GPU:**
```bash
cargo build --release --no-default-features
```

## GGUF (Quantized) Model Support

Influence now supports GGUF format models for reduced memory usage. GGUF models offer significant memory savings through quantization, making it possible to run larger models on hardware with limited RAM.

### Current Implementation Status

**✅ Working Now:**
- GGUF file auto-detection (`.gguf` extension)
- Quantization format detection from filenames
- Metadata parsing (quantization type, file path)
- Architecture detection (prioritizes GGUF over safetensors)
- Comprehensive test coverage (7 GGUF-specific tests)

**🔧 In Development:**
- Full GGUF inference engine integration
- Text generation with GGUF models
- Streaming support for GGUF models

### Enabling GGUF Support

```bash
# Build with GGUF support
cargo build --release --features gguf

# With Metal GPU (macOS)
cargo build --release --features gguf,metal

# With CUDA GPU (Linux/Windows)
cargo build --release --features gguf,cuda

# Run GGUF-specific tests
cargo test --features gguf gguf
```

### Supported Quantization Formats

GGUF models come in various quantization formats that trade off quality for memory efficiency:

| Format | Bit Width | Memory (7B model) | Quality | Use Case |
|--------|-----------|-------------------|---------|----------|
| Q2_K | 2-bit | ~2.5 GB | Lower | Maximum compression |
| Q4_K | 4-bit | ~4 GB | Good | Balanced option |
| Q4_K_M | 4-bit | ~4 GB | Good | **Recommended** - Best balance |
| Q5_K | 5-bit | ~5 GB | Better | High quality |
| Q5_K_M | 5-bit | ~5 GB | Better | High quality mixed |
| Q6_K | 6-bit | ~6 GB | Near-original | Excellent quality |
| Q8_0 | 8-bit | ~8 GB | Minimal loss | Maximum quality |
| F16 | 16-bit | ~14 GB | Original | No compression |

### Where to Get GGUF Models

Popular sources for GGUF models:

1. **TheBloke** (HuggingFace) - Largest collection:
   - https://huggingface.co/TheBloke
   - Search: `TheBloke/Llama-2-7B-GGUF`, `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`

2. **MaziyarPanahi** (Quantized models):
   - https://huggingface.co/maziyarpanahi
   - Search: `maziyarpanahi/Llama-3-8B-GGUF`

3. **Bartowski** (High-quality conversions):
   - https://huggingface.co/bartowski
   - Search: `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`

Example download:
```bash
# Using wget
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

# Using huggingface-cli
pip install huggingface-hub
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf --local-dir ./models
```

### Using GGUF Models

```bash
# Place a .gguf file in your models directory
mkdir -p ./models
mv llama-2-7b.Q4_K_M.gguf ./models/

# GGUF files are auto-detected by their .gguf extension
influence generate "Explain quantum computing in simple terms" \
  --model-path ./models/llama-2-7b.Q4_K_M.gguf

# The quantization format is automatically detected from the filename
# and logged: "Detected quantization: Q4_K_M"
```

### Memory Comparison

For a 7B parameter model:

| Format | VRAM/RAM Required | File Size | Compression Ratio |
|--------|-------------------|-----------|-------------------|
| FP16 (safetensors) | ~14 GB | ~14 GB | 1x (baseline) |
| Q8_0 | ~8 GB | ~8 GB | ~1.75x |
| Q6_K | ~6 GB | ~6 GB | ~2.3x |
| Q4_K_M | ~4 GB | ~4 GB | **3.5x** |
| Q2_K | ~2.5 GB | ~2.5 GB | ~5.6x |

**Key Benefit:** Q4_K_M uses only **29%** of the memory required for FP16 while maintaining good quality!

### Technical Details

#### Detection Priority

When loading a model directory with both GGUF and safetensors files:
1. GGUF files (`.gguf`) are checked first
2. If found, `ModelArchitecture::LlamaQuantized` is detected
3. Falls back to `config.json` for safetensors models

This allows easy switching between formats by simply adding/removing GGUF files.

#### Quantization Format Detection

The quantization format is detected from the filename using these patterns:
- `q2_k` → Q2_K
- `q4_k_m` → Q4_K_M (checked before `q4_k`)
- `q4_k` → Q4_K
- `q5_k_m` → Q5_K_M (checked before `q5_k`)
- `q5_k` → Q5_K
- `q6_k` → Q6_K
- `q8_0` → Q8_0
- `f16` → F16
- Case-insensitive matching

#### Testing

GGUF functionality is thoroughly tested:
```bash
# Run all GGUF tests
cargo test --features gguf gguf

# Run specific test
cargo test --features gguf test_detect_quantization

# Test GGUF file detection
cargo test --features gguf test_detect_architecture_gguf_file

# Run all tests (including GGUF)
cargo test --features gguf
```

Current test coverage:
- ✅ Quantization format detection (11 formats)
- ✅ GGUF file detection
- ✅ Architecture detection with GGUF files
- ✅ Multiple GGUF files handling
- ✅ GGUF priority over safetensors
- ✅ Feature flag validation
- ✅ Backend metadata methods

**Note:** Full GGUF inference support is under active development. Currently, GGUF files are detected and their metadata is parsed. Complete generation support will be added in an upcoming release.

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

### `list` - List Downloaded Models

```bash
influence list [options]
```

**Examples:**

```bash
# List all models in the default models directory
influence list

# List models from a custom location
influence list --models-dir ~/my-models

# See model details: format, architecture, size
influence list
```

**Output shows:**
- Model name
- File path
- Format (SafeTensors, GGUF with quantization)
- Architecture (llama, mistral, etc.)
- Size on disk
- File count

**Options:**
- `-m, --models-dir <PATH>` - Custom models directory

### `deploy` - Deploy Model Server

```bash
influence deploy [options]
```

**Examples:**

```bash
# Deploy model with default settings (port 8080)
influence deploy --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0

# Deploy on custom port with Metal GPU
influence deploy \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
  --port 9000 \
  --device metal

# Deploy in background (detached mode)
influence deploy \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
  --detached
```

**Deploy starts:**
- REST API server on the specified port
- SSE streaming endpoint
- Health check endpoint
- Ollama-compatible API endpoints

**Options:**
- `-m, --model-path <PATH>` - Path to model directory (or use INFLUENCE_MODEL_PATH from .env)
- `-p, --port <PORT>` - Port to serve on (default: 8080)
- `-d, --device <DEVICE>` - Compute device: auto|cpu|metal|cuda (default: auto)
- `--device-index <N>` - GPU device index (default: 0)
- `--detached` - Run in background (detached from terminal)

**After deployment, test with:**
```bash
# Health check
curl http://localhost:8080/health

# Generate text
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 50}'

# Chat completion (streaming)
curl -X POST http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

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

### `chat` - Interactive Chat Mode

```bash
influence chat [options]
```

**Examples:**

```bash
# Start interactive chat
influence chat --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0

# Chat with a system prompt
influence chat \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
  --system "You are a helpful coding assistant"

# Chat with custom parameters
influence chat \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
  --temperature 0.8 \
  --max-tokens 256

# Load a previous chat session
influence chat \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
  --session my_chat.json

# Auto-save session on exit
influence chat \
  --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
  --save-on-exit my_chat.json
```

**Chat Features:**

Interactive chat mode includes powerful slash commands for session management:

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/clear` | Clear conversation history (keeps system prompt) |
| `/history` | Display all messages in current session |
| `/save <filename>` | Save conversation to a JSON file |
| `/load <filename>` | Load conversation from a JSON file |
| `/set <param> <value>` | Change parameters during chat |
| `/quit` or `/exit` | Exit chat mode |

**Runtime Parameter Adjustment:**

Change parameters mid-conversation using `/set`:

```bash
You: /set temperature 0.9
✓ Temperature set to 0.9

You: /set top_p 0.95
✓ Top-p set to 0.95

You: /set max_tokens 1024
✓ Max tokens set to 1024
```

Available parameters:
- `temperature` - Sampling creativity (0.0-2.0)
- `top_p` - Nucleus sampling threshold (0.0-1.0)
- `top_k` - Top-k sampling limit
- `repeat_penalty` - Repetition penalty (0.0-2.0)
- `max_tokens` - Maximum tokens per response

**Session Management:**

Save and resume conversations:

```bash
# During chat, save your session
You: /save project_discussion.json
✓ Conversation saved to: project_discussion.json

# Later, resume the session
$ influence chat \
    --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
    --session project_discussion.json
✓ Loaded session from: project_discussion.json
  Messages: 15
```

**Session File Format:**

Sessions are saved as JSON with complete conversation history:

```json
{
  "model_path": "./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0",
  "system_prompt": "You are a helpful assistant.",
  "messages": [
    {
      "role": "user",
      "content": "Hello!",
      "timestamp": "2026-01-29 14:30:15"
    },
    {
      "role": "assistant",
      "content": "Hello! How can I help you today?",
      "timestamp": "2026-01-29 14:30:18"
    }
  ],
  "created_at": "2026-01-29 14:30:00"
}
```

**Conversation History:**

- Maintains full conversation context across turns
- Automatically keeps last 10 turns (20 messages) to manage memory
- System prompt is preserved when clearing or truncating
- View full history with `/history` command

**Options:**
- `-m, --model-path <PATH>` - Path to model directory (required)
- `--system <PROMPT>` - System prompt to set conversation context
- `--max-tokens <N>` - Max tokens per response (default: 512)
- `--temperature <0.0-2.0>` - Sampling temperature (default: 0.7)
- `--top-p <0.0-1.0>` - Top-p sampling threshold (default: 0.9)
- `--top-k <N>` - Top-k sampling limit (default: disabled)
- `--repeat-penalty <0.0-2.0>` - Repetition penalty (default: 1.1)
- `-d, --device <DEVICE>` - Compute device: auto|cpu|metal|cuda (default: auto)
- `--device-index <N>` - GPU device index (default: 0)
- `--session <FILE>` - Load chat session from file on startup
- `--save-on-exit <FILE>` - Auto-save session to file on exit

### `config` - Show Configuration Settings

```bash
influence config
```

**Examples:**

```bash
# Show all current configuration settings
influence config
```

**What it displays:**

The `config` command shows all current configuration settings from environment variables and `.env` file:

- **Model Settings**
  - Model path
  - Output directory
  - Mirror URL

- **Generation Parameters**
  - Temperature
  - Top-p (nucleus sampling)
  - Top-k sampling
  - Repeat penalty
  - Max tokens

- **Device Settings**
  - Compute device (auto/cpu/metal/cuda)
  - Device index

- **Server Settings**
  - Port

It also displays a helpful reference of all available environment variables that can be set in your `.env` file.

**Options:**
None (takes no arguments)

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

### Completed
- [x] Streaming markdown renderer with syntax highlighting for code blocks
- [x] CLI UX improvements with formatted output
- [x] Configuration management with `config` command
- [x] Security hardening and retry logic
- [x] Comprehensive API documentation
- [x] Model management with `list` and `deploy` commands
- [x] Interactive chat mode with slash commands and session persistence
- [x] GGUF model detection and metadata parsing
- [x] Embeddings support (BERT encoder-only models)
- [x] Ollama-compatible API endpoints
- [x] Metal GPU warmup for reduced first-token latency

### In Progress
- [ ] Full GGUF inference engine integration (detection working, inference in development)

### Planned
- [ ] Batch generation
- [ ] More quantization formats support
- [ ] Session-based KV cache reuse for multi-turn conversations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Built with:
- [Candle](https://github.com/huggingface/candle) - ML framework by HuggingFace
- [Tokenizers](https://github.com/huggingface/tokenizers) - Fast tokenization
- [Clap](https://github.com/clap-rs/clap) - CLI parsing
