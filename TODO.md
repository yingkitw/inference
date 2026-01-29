# TODO

## Completed (v0.1.0)

### Core Infrastructure
- [x] Create Cargo.toml with dependencies
- [x] Implement CLI with clap
- [x] Set up error handling with thiserror
- [x] Configure tracing/logging

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

### Local Inference
- [x] Implement local model loading with tokenizers
- [x] Load Llama models from .safetensors weights
- [x] Parse actual config.json for model parameters
- [x] Model architecture detection (Llama, Mistral, Phi, Granite)
- [x] Forward pass with KV caching
- [x] Temperature-based token sampling
- [x] Streaming token generation
- [x] EOS token detection
- [x] Detect unsupported architectures (Mamba/MoE)
- [x] Provide helpful error messages for incompatible models

### Documentation
- [x] Create README.md
- [x] Create SPEC.md and ARCHITECTURE.md
- [x] Create USAGE.md guide
- [x] Create MODELS.md guide
- [x] Create IMPLEMENTATION.md
- [x] Update Cargo.toml with proper metadata
- [x] Remove all emojis from code

### Testing
- [x] Add unit tests for core functionality
- [x] Build succeeds with cargo build
- [x] All tests pass with cargo test
- [x] CLI help commands work correctly

## Future Enhancements

### Model Support
- [x] Add Mistral model support (architecture-specific loading)
- [ ] Add Phi model support
- [x] Fix logits extraction and softmax dimension errors
- [x] Test with TinyLlama model - WORKING!
- [x] Add Metal GPU support for macOS - WORKING!
- [x] Fix token spacing in decoded output - WORKING!
- [ ] Test with additional model architectures
- [ ] Validate Mamba generation with a real HF Mamba model
- [ ] Validate GraniteMoeHybrid generation with an attention-only GraniteMoeHybrid config
- [ ] Add embeddings support for RoBERTa/ALBERT (or map them to supported candle models)
- [x] Add CUDA support for Linux/Windows
- [x] Add --device/--device-index flags for explicit backend selection
- [ ] Test local generation with an actual Mistral model from HuggingFace
- [ ] Implement quantization support
- [ ] Add model caching in memory
- [ ] Optimize tokenization performance

### Performance
- [x] Add GPU support (Metal)
- [x] Reduce first-token latency on Metal by warming up a few decode steps at model load (set INFLUENCE_WARMUP_TOKENS=0 to disable)
- [x] KV cache capabilities: Fresh cache per generation request (stateless, memory efficient)
- [ ] Session-based KV cache reuse for multi-turn conversations (future enhancement)
- [ ] Implement batch generation
- [ ] Add model caching in memory
- [ ] Optimize tokenization performance

### Ollama Compatibility
- [x] Add Ollama-compatible endpoint: `POST /api/generate` (non-stream JSON response)
- [x] Add Ollama-compatible streaming for `/api/generate` using NDJSON (one JSON object per line)
- [x] Map Ollama `options` fields to Influence generation params:
  - `temperature`, `top_p`, `top_k`, `repeat_penalty`, `num_predict`
- [x] Add Ollama-compatible endpoint: `POST /api/embeddings` mapped to `influence embed` behavior (BERT only for now)
- [x] Add minimal Ollama discovery endpoints (TBD): e.g. `/api/tags` for “loaded model” metadata
- [x] Document Ollama-compat API in README/SPEC (what is supported vs not)
- [x] Add integration tests for Ollama endpoints (streaming + non-stream)

### Features
- [x] Add chat mode with conversation history
- [x] Add system prompt support
- [x] Add top-p and top-k sampling options
- [x] Add repetition penalty control
- [x] Add environment variable configuration (.env file support)
- [x] Add HTTP server for serving API
- [x] Add model validation after download
- [x] Improve CLI output UX with markdown rendering and syntax highlighting
- [x] Add config command to display configuration settings in formatted table

### Web API
- [ ] Document REST/SSE endpoints and example curl commands
- [ ] Add integration tests for /v1/generate and /v1/generate_stream

### Developer Experience
- [ ] Add integration tests with mock server
- [ ] Add retry logic for failed downloads
- [ ] Add resume capability for interrupted downloads
- [ ] Add model metadata storage
- [ ] Add more comprehensive examples
- [ ] Add benchmarking suite

### Documentation
- [ ] Add CONTRIBUTING.md
- [ ] Add API documentation for library use
- [ ] Add performance benchmarks
- [ ] Add troubleshooting guide

## Known Limitations

### Unsupported Models
- Mixture of Experts (MoE) models
- GraniteMoeHybrid configs containing Mamba layers
- Encoder-only text generation (BERT, RoBERTa, ALBERT) (embeddings supported for BERT)
- Models requiring specialized implementations

### Current Limitations
- GPU support via Metal/CUDA available
- No batch generation
- Text generation supported for Llama + Mamba + GraniteMoeHybrid (attention-only)
- Mistral loading supported but generation not implemented
