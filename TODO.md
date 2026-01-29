# TODO

## Completed (v0.1.4)

### Core Infrastructure
- [x] Create Cargo.toml with dependencies
- [x] Implement CLI with clap
- [x] Set up error handling with thiserror
- [x] Configure tracing/logging
- [x] Modularize large code files for better maintainability (src/local, src/output)

### Model Search
- [x] Add search command for HuggingFace models
- [x] Support filtering by author/organization
- [x] Add limit option for search results
- [x] Display model metadata (downloads, likes, pipeline tag)

### Model Download
- [x] Implement model download from HuggingFace mirrors
- [x] Dynamic file discovery from HuggingFace API
- [x] Fallback to hardcoded file lists
- [x] Progress tracking with indicatif
- [x] Error handling for gated models (403 errors)
- [x] Model existence validation before download
- [x] Support for custom mirrors and output directories
- [x] Handle both .safetensors and .bin model formats
- [x] Retry logic for failed downloads (MAX_RETRIES=3)

### Model Management
- [x] Add `list` command to show all downloaded models
- [x] Add `deploy` command to start API server
- [x] Display model details (format, architecture, size, file count)

### Local Inference
- [x] Implement local model loading with tokenizers
- [x] Load Llama models from .safetensors weights
- [x] Parse actual config.json for model parameters
- [x] Model architecture detection (Llama, Mistral, Mamba, GraniteMoeHybrid, BERT, Phi, Granite)
- [x] Forward pass with KV caching
- [x] Temperature-based token sampling
- [x] Streaming token generation with markdown rendering
- [x] EOS token detection
- [x] Detect unsupported architectures (Mamba/MoE in GraniteMoeHybrid)
- [x] Provide helpful error messages for incompatible models

### GPU Support
- [x] Add Metal GPU support for macOS
- [x] Add CUDA support for Linux/Windows
- [x] Add --device/--device-index flags for explicit backend selection
- [x] Metal GPU warmup for reduced first-token latency (INFLUENCE_WARMUP_TOKENS)
- [x] Auto-detect GPU (Metal > CUDA > CPU)

### Generation Features
- [x] Add system prompt support
- [x] Add top-p and top-k sampling options
- [x] Add repetition penalty control
- [x] Streaming markdown output with syntax highlighting for code blocks
- [x] CLI UX improvements with formatted output (colors, markdown, tables)

### Chat Mode
- [x] Add chat mode with conversation history
- [x] Add slash commands: `/help`, `/clear`, `/save`, `/load`, `/history`, `/set`, `/quit`
- [x] Session save/load (JSON format)
- [x] Runtime parameter adjustment via `/set`
- [x] Keep last 10 turns (20 messages) for memory management

### Web API
- [x] Add HTTP server for serving API (Axum framework)
- [x] REST + SSE streaming endpoints
- [x] Health check endpoint
- [x] Chat completions endpoint (OpenAI-compatible)
- [x] Security hardening with CORS and proper error handling
- [x] Comprehensive API documentation (OpenAPI/Swagger compatible)
- [x] Document REST/SSE endpoints with curl examples (docs/API.md)
- [x] Add integration tests for /v1/generate and /v1/generate_stream

### Ollama Compatibility
- [x] `POST /api/generate` (non-stream JSON response)
- [x] `POST /api/generate` with NDJSON streaming
- [x] Map Ollama `options` to Influence params (temperature, top_p, top_k, repeat_penalty, num_predict)
- [x] `POST /api/embeddings` for BERT embeddings
- [x] `POST /api/tags` for model metadata
- [x] Document Ollama-compat API in README

### Embeddings
- [x] Add `embed` command for encoder-only models
- [x] BERT embeddings support
- [x] Proper error messages for encoder-only models attempting text generation

### Configuration
- [x] Add environment variable configuration (.env file support via dotenvy)
- [x] Add `config` command to display current settings
- [x] Priority: CLI args > Env vars > Built-in defaults

### GGUF Support (Partial)
- [x] GGUF file auto-detection (.gguf extension)
- [x] Quantization format detection from filenames (Q2_K, Q4_K_M, Q8_0, etc.)
- [x] Metadata parsing (quantization type, file path)
- [x] Architecture detection (prioritizes GGUF over safetensors)
- [x] Comprehensive test coverage (7 GGUF-specific tests)
- [ ] Full GGUF inference engine integration (in development)

### Documentation
- [x] Create README.md
- [x] Create SPEC.md and ARCHITECTURE.md
- [x] Create CLAUDE.md for project guidance
- [x] Update README with comprehensive command reference
- [x] Document all commands with examples
- [x] Add GGUF usage guide
- [x] Document troubleshooting guide
- [x] Update Cargo.toml with proper metadata

### Testing
- [x] Add unit tests for core functionality (79 total tests)
- [x] CLI parsing tests
- [x] GGUF-specific tests (feature-gated)
- [x] Architecture detection tests
- [x] Build succeeds with cargo build
- [x] All tests pass with cargo test

## Current Limitations

### Architecture Support
**Text Generation Working:**
- [x] Llama (Llama 2/3, TinyLlama, Phi-3)
- [x] Mamba
- [x] GraniteMoeHybrid (attention-only configs)
- [x] Granite

**Detection Only (Generation Not Implemented):**
- [ ] Mistral (can be loaded but generation not implemented)
- [ ] Phi (detected but no loading function)

**Embeddings Only:**
- [x] BERT (encoder-only, use `embed` command)

**Not Supported:**
- [ ] Mixture of Experts (MoE) models
- [ ] GraniteMoeHybrid with Mamba layers
- [ ] GGUF quantized models (detection works, inference in development)

### Performance Limitations
- [ ] No batch generation
- [ ] Fresh KV cache per request (no session-based cache reuse)
- [ ] No model caching in memory between requests

## In Progress

### GGUF Inference
- [ ] Full GGUF inference engine integration
- [ ] GGUF streaming support
- [ ] GGUF embeddings support

## Future Enhancements

### Ollama-Inspired Commands
- [x] `show` - Display detailed model information (parameters, architecture, template details)
- [x] `rm` / `remove` - Remove a downloaded model from disk
- [x] `ps` - Show running model servers (similar to `ollama ps`)
- [x] `copy` - Create a copy of a model
- [x] `info` - Show detailed model metadata and capabilities
- [x] `verify` - Verify model integrity (checksum validation)

### Model Support
- [ ] Implement Mistral text generation
- [ ] Implement Phi loading and generation
- [ ] Test with additional model architectures
- [ ] Add more quantization formats support (beyond GGUF)
- [ ] Add embeddings support for RoBERTa/ALBERT

### Performance
- [ ] Session-based KV cache reuse for multi-turn conversations
- [ ] Implement batch generation
- [ ] Add model caching in memory
- [ ] Optimize tokenization performance
- [ ] Resume capability for interrupted downloads

### Features
- [ ] Add conversation search/filtering in chat mode
- [ ] Add multi-model support in API server
- [ ] Add streaming response for `config` command
- [ ] Add model metadata storage and indexing

### Developer Experience
- [x] Add integration tests with mock server
- [ ] Add benchmarking suite
- [ ] Add more comprehensive examples
- [ ] Add CONTRIBUTING.md
- [ ] Add API documentation for library use
- [ ] Add performance benchmarks

### Security
- [ ] Add API key authentication for server
- [ ] Add rate limiting
- [ ] Add request validation
- [ ] Add input sanitization

## Version History

### v0.1.4 (Current)
- Streaming markdown renderer with syntax highlighting
- CLI UX improvements with formatted output
- `config` command to display settings
- Security hardening and retry logic
- Reduced default warmup tokens
- Performance optimizations (removed Arc<Mutex> overhead)

### v0.1.3
- GGUF quantized model support (detection)
- Model management commands (`list`, `deploy`)
- Enhanced chat mode features
- Session persistence

### v0.1.2
- Embeddings support (BERT)
- Ollama-compatible API endpoints
- Environment variable configuration
- Metal GPU warmup

### v0.1.1
- Chat mode with conversation history
- API server (REST + SSE streaming)
- Advanced sampling options
- Model validation

### v0.1.0
- Initial release
- Model search and download
- Local inference with Llama models
- Metal GPU support
- Basic generation features
