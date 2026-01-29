# Influence - Specification

## Overview

Influence is a Rust CLI application that enables downloading models from HuggingFace mirrors and serving them as an LLM influencer using IBM WatsonX.

## Architecture

### Components

1. **CLI Module** (`cli.rs`)
   - Command-line interface using Clap
   - Three main commands: download, serve, generate
   - Follows Typer-like interface design

2. **Download Module** (`download.rs`)
   - Downloads models from HuggingFace mirrors
   - Progress tracking with indicatif
   - Automatic directory management
   - Supports custom mirrors and output paths

3. **Influencer Module** (`influencer.rs`)
   - LLM service using WatsonX SDK
   - Trait-based design for testability
   - Streaming generation support
   - Granite model prompt formatting

4. **Error Module** (`error.rs`)
   - Centralized error handling
   - Custom error types with thiserror

### Design Principles

- **DRY**: No code duplication
- **KISS**: Simple, straightforward implementation
- **SoC**: Clear separation between CLI, download, and LLM logic
- **Test-friendly**: Trait-based design for easy mocking
- **Maintainability**: Modular structure with clear responsibilities

## API Design

### Commands

#### `download`
```bash
influence download -m <model> [-r <mirror>] [-o <output>]
```
Downloads a model from HuggingFace mirror.

#### `serve`
```bash
influence serve [-m <model-path>] [-p <port>]
```
Starts the influencer service.

#### `generate`
```bash
influence generate <prompt> [-m <model-path>] [--max-tokens <n>] [--temperature <t>]
```
Generates text using the LLM with streaming output.

## Configuration

Environment variables:
- `WATSONX_API_KEY` - WatsonX API key (required)
- `WATSONX_PROJECT_ID` - WatsonX project ID (required)
- `WATSONX_MODEL_ID` - Model ID (optional, default: ibm/granite-4-h-small)

## Data Flow

### Download Flow
1. Parse CLI arguments
2. Determine output directory
3. Create HTTP client
4. Download each model file with progress tracking
5. Save to local directory

### Generate Flow
1. Parse CLI arguments
2. Load configuration from environment
3. Create WatsonX client
4. Format prompt for Granite model
5. Stream generation results
6. Display output in real-time

## Model Support

Currently supports:
- IBM Granite models (with special prompt formatting)
- Generic HuggingFace models

Model files downloaded:
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `model.safetensors` (Granite) or `pytorch_model.bin` (others)

## Error Handling

Custom error types:
- `DownloadError` - Download failures
- `ModelNotFound` - Missing model
- `InvalidConfig` - Configuration issues
- `LlmError` - LLM generation errors
- Standard errors: `IoError`, `HttpError`, `JsonError`

## Testing Strategy

- Unit tests for individual functions
- Integration tests with mock servers
- Trait-based design allows easy mocking of LLM service
- Test coverage for error cases

## Future Enhancements

- HTTP API server for serving requests
- Model caching and validation
- Configuration file support
- Batch generation
- Resume capability for downloads
- More model format support
