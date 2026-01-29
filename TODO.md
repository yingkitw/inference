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
- [ ] Add Mistral model support (architecture-specific loading)
- [ ] Add Phi model support
- [x] Fix logits extraction and softmax dimension errors
- [x] Test with TinyLlama model - WORKING!
- [x] Add Metal GPU support for macOS - WORKING!
- [x] Fix token spacing in decoded output - WORKING!
- [ ] Test with additional model architectures
- [ ] Add CUDA support for Linux/Windows
- [ ] Implement quantization support
- [ ] Add model caching in memory
- [ ] Optimize tokenization performance

### Performance
- [x] Add GPU support (Metal)
- [ ] Implement batch generation
- [ ] Add model caching in memory
- [ ] Optimize tokenization performance

### Features
- [ ] Add chat mode with conversation history
- [ ] Add system prompt support
- [ ] Add top-p and top-k sampling options
- [ ] Add repetition penalty control
- [ ] Add configuration file support
- [ ] Add HTTP server for serving API
- [ ] Add model validation after download

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
- Mamba/Hybrid models (e.g., GraniteMoeHybrid)
- Mixture of Experts (MoE) models
- Encoder-only models (BERT, RoBERTa, ALBERT)
- Models requiring specialized implementations

### Current Limitations
- CPU inference only (GPU support planned)
- No conversation history/memory
- No batch generation
- Llama-architecture only for inference
- Argmax sampling only (no nucleus sampling)
